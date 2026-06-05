"""
Tool Router - 决定调用哪些工具
C负责
"""

from src.state import PlanState
from typing import Dict, Any


def _build_name_lookup(timeline: list[dict]) -> dict[str, str]:
    """从 timeline 中提取 poi_id 到展示名称的映射。"""
    lookup = {}
    for item in timeline:
        poi_id = item.get("poi_id")
        activity = item.get("activity")
        if poi_id and activity:
            lookup[poi_id] = activity
    return lookup


def parse_time_slot(time_str: str) -> str:
    """将时间范围解析为开始时间，如 '14:00-16:00' -> '14:00'"""
    if not time_str:
        return ""
    if "-" in time_str:
        return time_str.split("-")[0].strip()
    return time_str


def _infer_supported_action_type(item: dict[str, Any]) -> str | None:
    """Infer the C action type for concrete B timeline nodes missing `type`."""
    activity_type = str(item.get("type") or "").strip().lower()
    poi_id = str(item.get("poi_id") or "").strip().lower()
    role = str(item.get("role") or item.get("itinerary_role") or "").strip().lower()
    text = " ".join(
        str(value or "")
        for value in (
            item.get("activity"),
            item.get("name"),
            role,
            " ".join(str(note) for note in item.get("notes", []) if note),
        )
    )

    if activity_type in {"eat", "restaurant", "cafe", "tea_house"}:
        return "reserve_restaurant"
    if activity_type in {"play", "amusement", "museum", "art", "activity"}:
        return "order_activity_ticket"

    if poi_id.startswith(("res_", "gaode_res_")):
        return "reserve_restaurant"
    if poi_id.startswith(("act_", "gaode_act_")):
        return "order_activity_ticket"

    restaurant_terms = ("餐", "饭", "食", "咖啡", "茶室", "茶馆", "火锅", "寿司")
    if "restaurant" in role or "dinner" in role or "lunch" in role or any(
        term in text for term in restaurant_terms
    ):
        return "reserve_restaurant"

    activity_terms = ("博物馆", "展览", "田子坊", "活动", "演出", "剧", "桌游", "密室")
    if "activity" in role or "exhibition" in role or any(
        term in text for term in activity_terms
    ):
        return "order_activity_ticket"

    return None


def tool_router_node(state: PlanState) -> Dict[str, Any]:
    """
    根据selected_plan生成action_sequence
    输出格式对齐B的要求：
    {
        "action_type": "reserve_restaurant" / "order_activity_ticket",
        "poi_id": "xxx",
        "time": "14:30",
        "people": 3,
        "quantity": 3,
        "notes": []
    }
    """
    print("🔧 [8] Tool Router: 解析方案，生成执行动作列表...")

    execution_log = state.get("execution_log", [])
    selected_plan = state.get("selected_plan", {})
    constraints = state.get("constraints", {})
    b_replan_request = state.get("b_replan_request") or selected_plan.get("b_replan_request")

    if selected_plan.get("plan_status") == "needs_ai_replan" or b_replan_request:
        execution_log.append(
            "[C] Tool Router skipped execution because B requested AI-guided replanning"
        )
        return {
            "action_sequence": [],
            "execution_log": execution_log
        }

    if selected_plan.get("plan_status") == "needs_rag_candidate_evidence":
        execution_log.append(
            "[C] Tool Router skipped execution because B plan is not executable yet"
        )
        return {
            "action_sequence": [],
            "execution_log": execution_log
        }

    execution_contract = selected_plan.get("execution_contract") or {}
    if (
        selected_plan.get("execution_ready") is False
        and selected_plan.get("execution_scope") != "partial"
        and execution_contract.get("ready") is False
    ):
        blocking_reasons = execution_contract.get("blocking_reasons") or []
        reason_text = "; ".join(str(reason) for reason in blocking_reasons if reason)
        if reason_text:
            execution_log.append(
                f"[C] Tool Router skipped execution because B execution contract is not ready: {reason_text}"
            )
        else:
            execution_log.append(
                "[C] Tool Router skipped execution because B execution contract is not ready"
            )
        return {
            "action_sequence": [],
            "execution_log": execution_log
        }

    if selected_plan.get("partial_missing_roles") and not selected_plan.get("action_hints"):
        execution_log.append(
            "[C] Tool Router skipped execution because partial B plan still has missing itinerary roles"
        )
        return {
            "action_sequence": [],
            "execution_log": execution_log
        }

    if selected_plan.get("execution_ready") is False or selected_plan.get("execution_scope") == "partial":
        execution_log.append(
            "[C] Tool Router emits supported actions for a partial B plan"
        )

    # 从constraints获取人数（如果没有则默认为3）
    people_count = constraints.get("people_count", 3)

    action_sequence = []

    # 从selected_plan中提取需要执行的动作
    timeline = selected_plan.get("timeline", [])
    action_hints = selected_plan.get("action_hints", [])

    if not timeline and not action_hints:
        execution_log.append("⚠️ Tool Router: 没有可执行方案，未生成执行动作")
        return {
            "action_sequence": action_sequence,
            "execution_log": execution_log
        }

    if action_hints:
        name_lookup = _build_name_lookup(timeline)
        for idx, hint in enumerate(action_hints):
            action = dict(hint)
            action_type = action.get("action_type", "")
            poi_id = action.get("poi_id", "")

            action["step"] = idx + 1
            action["time"] = parse_time_slot(action.get("time", ""))
            action["name"] = action.get("name") or name_lookup.get(poi_id, poi_id)

            if action_type == "reserve_restaurant":
                action.setdefault("people", people_count)
            elif action_type == "order_activity_ticket":
                action.setdefault("quantity", people_count)
            elif action_type == "reserve_lodging":
                action.setdefault("people_count", people_count)
                action.setdefault("room_count", 1)

            action_sequence.append(action)
    else:
        for idx, item in enumerate(timeline):
            poi_id = item.get("poi_id", "")
            activity_name = item.get("activity", "")
            time_str = item.get("time", "")
            if not poi_id:
                continue

            # 解析时间：将 "14:00-16:00" 转换为 "14:00"
            parsed_time = parse_time_slot(time_str)
            action_type = _infer_supported_action_type(item)

            # 根据节点类型或可识别 POI 前缀决定调用什么工具
            if action_type == "reserve_restaurant":
                action_sequence.append({
                    "step": idx + 1,
                    "action_type": "reserve_restaurant",
                    "poi_id": poi_id,
                    "time": parsed_time,
                    "people": people_count,
                    "name": activity_name,
                    "notes": ["child_seat"] if state.get("scene_type") == "family" else []
                })
            elif action_type == "order_activity_ticket":
                action_sequence.append({
                    "step": idx + 1,
                    "action_type": "order_activity_ticket",
                    "poi_id": poi_id,
                    "time": parsed_time,
                    "quantity": people_count,
                    "name": activity_name,
                    "notes": []
                })
            elif activity_type in ["lodging", "hotel"]:
                action_sequence.append({
                    "step": idx + 1,
                    "action_type": "reserve_lodging",
                    "poi_id": poi_id,
                    "time": parsed_time,
                    "check_in_date": item.get("check_in_date") or "2026-06-01",
                    "check_out_date": item.get("check_out_date") or "2026-06-02",
                    "room_count": 1,
                    "people_count": people_count,
                    "name": activity_name,
                    "notes": []
                })

    execution_log.append(f"✅ Tool Router: 生成{len(action_sequence)}个执行动作")

    # 打印动作列表以便调试
    for action in action_sequence:
        action_name = action.get('action_type', 'unknown')
        exec_log = f"   - 动作{action['step']}: {action_name}"
        if action.get('name'):
            exec_log += f" - {action['name']}"
        execution_log.append(exec_log)

    return {
        "action_sequence": action_sequence,
        "execution_log": execution_log
    }
