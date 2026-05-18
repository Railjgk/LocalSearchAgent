"""
Payment Layer - 用户确认单个行程目的地后模拟付款。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.state import PlanState
from src.tools.mock_apis import pay_order


PAYMENT_REQUIRED_ACTIONS = {"order_activity_ticket"}


def _amount_from_result(result: Dict[str, Any]) -> float:
    amount = result.get("total_price", result.get("price", 0))
    try:
        return float(amount or 0)
    except (TypeError, ValueError):
        return 0.0


def _format_amount(amount: float) -> str:
    if amount == int(amount):
        return f"{int(amount)}元"
    return f"{amount:.2f}元"


def _index_actions(action_sequence: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed = {}
    for action in action_sequence:
        action_type = action.get("action_type") or action.get("action")
        step = action.get("step")
        indexed[f"{action_type}_{step}"] = action
    return indexed


def _timeline_lookup(selected_plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup = {}
    for item in selected_plan.get("timeline", []):
        poi_id = item.get("poi_id")
        if poi_id:
            lookup[poi_id] = item
    return lookup


def _build_payable_items(state: PlanState) -> List[Dict[str, Any]]:
    """只挑选行程目的地中明确需要支付的成功订单。"""
    tool_results = state.get("tool_results", {})
    action_by_key = _index_actions(state.get("action_sequence", []))
    timeline_by_poi = _timeline_lookup(state.get("selected_plan", {}))

    payable_items = []
    for key, value in tool_results.items():
        if not value.get("success"):
            continue

        action = action_by_key.get(key, {})
        action_type = value.get("action") or action.get("action_type")
        result = value.get("data", {})

        if action_type not in PAYMENT_REQUIRED_ACTIONS:
            continue
        if result.get("payment_required") is not True:
            continue

        poi_id = action.get("poi_id")
        timeline_item = timeline_by_poi.get(poi_id)
        if not timeline_item:
            continue

        order_id = result.get("order_id")
        amount = _amount_from_result(result)
        if not order_id or amount <= 0:
            continue

        payable_items.append(
            {
                "key": key,
                "action_type": action_type,
                "order_id": order_id,
                "poi_id": poi_id,
                "name": timeline_item.get("activity")
                or value.get("name")
                or result.get("poi_name")
                or "行程目的地",
                "time": (
                    result.get("time_slot")
                    or action.get("time")
                    or timeline_item.get("time", "")
                ),
                "amount": amount,
                "quantity": result.get("quantity"),
            }
        )

    return payable_items


def _should_use_dialog(state: PlanState) -> bool:
    return state.get("payment_ui_mode") == "dialog"


def _show_confirm_dialog(item: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        return False, f"无法加载弹窗组件: {exc}"

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        confirmed = messagebox.askyesno(
            "确认行程目的地",
            (
                f"是否确认这个行程目的地？\n\n"
                f"目的地：{item['name']}\n"
                f"时间：{item.get('time') or '待定'}\n"
                f"需支付：{_format_amount(item['amount'])}"
            ),
            parent=root,
        )
        root.destroy()
        return confirmed, ""
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        return False, f"确认窗口打开失败: {exc}"


def _show_payment_dialog(item: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        return False, f"无法加载支付窗口组件: {exc}"

    paid = {"value": False}

    try:
        root = tk.Tk()
        root.title("支付订单")
        root.attributes("-topmost", True)
        root.resizable(False, False)

        frame = ttk.Frame(root, padding=20)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="请确认支付", font=("", 14, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )
        ttk.Label(frame, text="目的地").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Label(frame, text=item["name"]).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="订单号").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Label(frame, text=item["order_id"]).grid(row=2, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="金额").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Label(frame, text=_format_amount(item["amount"])).grid(
            row=3, column=1, sticky="w", pady=4
        )

        def mark_paid() -> None:
            paid["value"] = True
            root.destroy()

        button_row = ttk.Frame(frame)
        button_row.grid(row=4, column=0, columnspan=2, sticky="e", pady=(16, 0))
        ttk.Button(button_row, text="取消", command=root.destroy).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(button_row, text="确认付款", command=mark_paid).grid(row=0, column=1)

        root.mainloop()
        return paid["value"], ""
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        return False, f"支付窗口打开失败: {exc}"


def _show_success_dialog(item: Dict[str, Any], payment_result: Dict[str, Any]) -> str:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        return f"无法加载支付成功弹窗: {exc}"

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showinfo(
            "支付成功",
            (
                f"支付成功！\n\n"
                f"目的地：{item['name']}\n"
                f"金额：{_format_amount(item['amount'])}\n"
                f"支付流水：{payment_result.get('payment_id', '')}"
            ),
            parent=root,
        )
        root.destroy()
        return ""
    except Exception as exc:  # pragma: no cover - depends on local GUI support
        return f"支付成功弹窗打开失败: {exc}"


def payment_layer_node(state: PlanState) -> Dict[str, Any]:
    """
    对行程里需要支付的单个目的地进行确认与模拟支付。
    """
    print("[11] Payment Layer: 检查是否需要支付...")

    execution_log = state.get("execution_log", [])
    payable_items = _build_payable_items(state)

    if not payable_items:
        execution_log.append("✅ Payment Layer: 行程中没有需要支付的目的地下单项")
        return {
            "payment_order": {},
            "payment_results": {},
            "payment_status": "not_required",
            "execution_log": execution_log,
        }

    payment_order = payable_items[0]
    execution_log.append(
        "💳 Payment Layer: 待支付目的地 "
        f"{payment_order['name']}，金额{_format_amount(payment_order['amount'])}"
    )

    if _should_use_dialog(state):
        confirmed, confirm_error = _show_confirm_dialog(payment_order)
        if confirm_error:
            execution_log.append(f"⚠️ Payment Layer: {confirm_error}，已切换为自动确认")
            confirmed = state.get("payment_auto_confirm", True)
    else:
        confirmed = state.get("payment_auto_confirm", True)

    if not confirmed:
        execution_log.append("⏸️ Payment Layer: 用户未确认目的地，支付已取消")
        return {
            "payment_order": payment_order,
            "payment_results": {},
            "payment_status": "cancelled",
            "need_confirm": True,
            "execution_log": execution_log,
        }

    if _should_use_dialog(state):
        paid_by_user, payment_error = _show_payment_dialog(payment_order)
        if payment_error:
            execution_log.append(f"⚠️ Payment Layer: {payment_error}，已切换为自动付款")
            paid_by_user = state.get("payment_auto_pay", True)
    else:
        paid_by_user = state.get("payment_auto_pay", True)

    if not paid_by_user:
        execution_log.append("⏸️ Payment Layer: 用户未付款，订单保持未支付")
        return {
            "payment_order": payment_order,
            "payment_results": {},
            "payment_status": "pending",
            "need_confirm": True,
            "execution_log": execution_log,
        }

    payment_result = pay_order(
        payment_order["order_id"],
        payment_order["amount"],
        method=state.get("payment_method", "mock_pay"),
    )

    if payment_result.get("success"):
        execution_log.append(
            "✅ Payment Layer: 支付成功 "
            f"{payment_order['name']}，流水号{payment_result.get('payment_id')}"
        )
        if _should_use_dialog(state):
            success_error = _show_success_dialog(payment_order, payment_result)
            if success_error:
                execution_log.append(f"⚠️ Payment Layer: {success_error}")

        return {
            "payment_order": payment_order,
            "payment_results": payment_result,
            "payment_status": "success",
            "need_confirm": False,
            "execution_log": execution_log,
        }

    execution_log.append(
        "❌ Payment Layer: 支付失败 "
        f"{payment_order['name']}，原因"
        f"{payment_result.get('message', payment_result.get('error', '未知错误'))}"
    )
    return {
        "payment_order": payment_order,
        "payment_results": payment_result,
        "payment_status": "failed",
        "need_confirm": True,
        "execution_log": execution_log,
    }
