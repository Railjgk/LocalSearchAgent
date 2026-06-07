"""
运行入口 - 测试完整流程
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
sys.path.append(".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.state import PlanState
from src.graph import WORKFLOW_NODES, WORKFLOW_STEPS, build_graph
from src.initial_state import build_user_initial_state


NODE_NAMES = [name for name, _ in WORKFLOW_STEPS]

DEFAULT_USER_INPUT = "今天下午和老婆孩子出去玩，孩子5岁，老婆最近在减肥"


def build_family_initial_state() -> PlanState:
    """构造家庭场景初始状态。"""

    return {
        # ========== 输入 ==========
        "user_input": "今天下午和老婆孩子出去玩，孩子5岁，老婆最近在减肥",
        "scene_type": "family",  # "family" 或 "friends"

        # ========== A产出（用户理解与场景建模）==========
        "constraints": {
            "people_count": 3,
            "child_age": 5,
            "mom_diet": "low_calorie",
            "duration": 4,
            "start_time": "14:00",
            "location": "杨浦区",
            "max_distance_km": 20,
            "max_queue_time": 30,
            "budget": 500
        },
        "user_profile": {},
        "short_term_memory": [],
        "scenario_activities": ["亲子乐园", "轻食餐厅", "儿童剧场"],

        # ========== B产出（候选生成与方案决策）==========
        "candidates": [],
        "filtered_candidates": [],
        "filter_reasons": {},

        # B节点会根据 candidates -> filtered_candidates -> selected_plan 自动生成方案
        "selected_plan": {},
        "optimization_score": 0.0,
        "alternative_plans": [],
        "explanation_text": "",

        # ========== C产出（工具调用与执行闭环）- 初始空值 ==========
        "action_sequence": [],
        "raw_api_results": {},
        "execution_status": "pending",
        "tool_results": {},
        "payment_order": {},
        "payment_results": {},
        "payment_status": "not_required",
        "retry_history": [],
        "final_share_message": "",

        # ========== 控制字段 ==========
        "execution_log": [],
        "retry_count": 0,
        "need_confirm": False,
        "payment_ui_mode": "dialog",
        "payment_auto_confirm": True,
        "payment_auto_pay": True,
        "payment_method": "mock_pay"
    }


def build_initial_state(
    user_input: str | None = None,
    *,
    user_id: str = "u001",
    payment_ui_mode: str = "auto",
) -> PlanState:
    """Build a real-user initial state and let A-stage fill planning fields."""

    return build_user_initial_state(
        user_input or DEFAULT_USER_INPUT,
        user_id=user_id,
        payment_ui_mode=payment_ui_mode,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the WeekendFlow full pipeline.")
    parser.add_argument(
        "user_input",
        nargs="*",
        help="真实用户输入，例如：今天下午和朋友 citywalk，4个人，预算600。",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="从标准输入读取用户需求，适合 echo 或管道调用。",
    )
    parser.add_argument("--user-id", default="u001", help="记忆系统使用的用户 ID。")
    parser.add_argument(
        "--payment-ui-mode",
        choices=("auto", "dialog"),
        default="auto",
        help="支付确认模式；默认 auto 避免命令行运行时弹窗。",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="只打印结果，不写入 docs/run-status-test-* 记录。",
    )
    return parser.parse_args(argv)


def _resolve_user_input(args: argparse.Namespace) -> str:
    if args.stdin:
        text = sys.stdin.read().strip()
    else:
        text = " ".join(args.user_input).strip()
        if not text and sys.stdin.isatty():
            try:
                text = input("请输入本地生活需求（回车使用默认示例）：").strip()
            except EOFError:
                text = ""
    return text or DEFAULT_USER_INPUT


def _json_safe(value):
    """Convert runtime values into JSON serializable data for trace records."""

    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return repr(value)


def _state_summary(state: PlanState) -> dict:
    selected_plan = state.get("selected_plan") or {}
    timeline = selected_plan.get("timeline") or []
    tool_results = state.get("tool_results") or {}
    return {
        "scene_type": state.get("scene_type"),
        "constraints": state.get("constraints", {}),
        "scenario_activities": state.get("scenario_activities", []),
        "candidates_count": len(state.get("candidates") or []),
        "filtered_candidates_count": len(state.get("filtered_candidates") or []),
        "filter_reasons_count": len(state.get("filter_reasons") or {}),
        "selected_plan_id": selected_plan.get("plan_id") or selected_plan.get("id"),
        "timeline_count": len(timeline),
        "optimization_score": state.get("optimization_score"),
        "action_sequence_count": len(state.get("action_sequence") or []),
        "raw_api_result_keys": list((state.get("raw_api_results") or {}).keys()),
        "tool_result_keys": list(tool_results.keys()),
        "execution_status": state.get("execution_status"),
        "payment_status": state.get("payment_status"),
        "retry_count": state.get("retry_count"),
        "execution_log_count": len(state.get("execution_log") or []),
    }


def _write_trace_record(trace: dict) -> tuple[Path, Path]:
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = docs_dir / f"run-status-test-{timestamp}.json"
    md_path = docs_dir / f"run-status-test-{timestamp}.md"

    json_path.write_text(
        json.dumps(_json_safe(trace), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# run.py 完整运行状态测试记录",
        "",
        f"- 运行时间: {trace['run_at']}",
        f"- 场景: {trace['initial_state'].get('scene_type')}",
        f"- 用户输入: {trace['initial_state'].get('user_input')}",
        f"- 最终执行状态: {trace['final_summary'].get('execution_status')}",
        f"- 最终支付状态: {trace['final_summary'].get('payment_status')}",
        f"- 最终优化分: {trace['final_summary'].get('optimization_score')}",
        f"- 原始 JSON 记录: `{json_path.name}`",
        "",
        "## 模块状态",
        "",
    ]
    for index, step in enumerate(trace["steps"], start=1):
        summary = step["state_summary"]
        lines.extend(
            [
                f"### {index}. {step['node']}",
                "",
                f"- 输出字段: {', '.join(step['updates'].keys()) or '无'}",
                f"- candidates: {summary['candidates_count']} / filtered: {summary['filtered_candidates_count']}",
                (
                    f"- actions: {summary['action_sequence_count']} / tools: "
                    f"{', '.join(summary['tool_result_keys']) or '无'}"
                ),
                f"- execution_status: {summary['execution_status']} / payment_status: {summary['payment_status']}",
                "",
                "```json",
                json.dumps(_json_safe(step["updates"]), ensure_ascii=False, indent=2)[:20000],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## 最终分享消息",
            "",
            trace["final_state"].get("final_share_message") or "",
            "",
            "## 最终执行日志",
            "",
        ]
    )
    for log in trace["final_state"].get("execution_log", []):
        lines.append(f"- {log}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def run_with_trace(initial_state: PlanState) -> tuple[PlanState, list[dict]]:
    """Run the workflow and collect each node's updates plus merged state."""

    graph = build_graph()
    current_state: PlanState = dict(initial_state)
    steps: list[dict] = []

    try:
        stream = graph.stream(initial_state, stream_mode="updates")
        for event in stream:
            for node_name, updates in event.items():
                if updates:
                    current_state.update(updates)
                steps.append(
                    {
                        "node": node_name,
                        "updates": updates or {},
                        "state": dict(current_state),
                        "state_summary": _state_summary(current_state),
                    }
                )
        return current_state, steps
    except AttributeError:
        pass

    current_state = dict(initial_state)
    for node_name, node in zip(NODE_NAMES, WORKFLOW_NODES):
        updates = node(current_state) or {}
        current_state.update(updates)
        steps.append(
            {
                "node": node_name,
                "updates": updates,
                "state": dict(current_state),
                "state_summary": _state_summary(current_state),
            }
        )
    return current_state, steps


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    user_input = _resolve_user_input(args)
    initial_state = build_initial_state(
        user_input,
        user_id=args.user_id,
        payment_ui_mode=args.payment_ui_mode,
    )

    # 构建并运行图
    print("\n" + "="*60)
    print("🚀 美团AI黑客松 - 本地生活Agent启动")
    print(f"🧾 用户输入: {user_input}")
    print("="*60 + "\n")

    final_state, steps = run_with_trace(initial_state)

    trace = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "initial_state": initial_state,
        "steps": steps,
        "final_state": final_state,
        "final_summary": _state_summary(final_state),
    }
    md_path = None
    json_path = None
    if not args.no_trace:
        md_path, json_path = _write_trace_record(trace)

    # ========== 输出结果 ==========
    print("\n" + "="*60)
    print("📊 执行结果")
    print("="*60)

    print("\n📝 执行日志:")
    for log in final_state.get("execution_log", []):
        print(f"   {log}")

    print(f"\n📅 最终方案:")
    timeline = final_state.get("selected_plan", {}).get("timeline", [])
    for item in timeline:
        print(f"   {item.get('time')} - {item.get('activity')}")

    print(f"\n📊 执行状态: {final_state.get('execution_status')}")
    print(f"💳 支付状态: {final_state.get('payment_status')}")

    print(f"\n💬 分享消息:")
    print(f"   {final_state.get('final_share_message')}")

    # 打印工具执行结果详情（调试用）
    tool_results = final_state.get("tool_results", {})
    if tool_results:
        print(f"\n🔧 工具执行详情:")
        for key, value in tool_results.items():
            if isinstance(value, dict):
                status = "✅" if value.get("success", True) else "❌"
                message = (value.get("data") or {}).get("message", "")
                print(f"   {status} {key}: {value.get('name', '')} - {message}")
            else:
                print(f"   ✅ {key}: {type(value).__name__}")

    if md_path and json_path:
        print(f"\n🧾 状态记录: {md_path}")
        print(f"🧾 原始JSON: {json_path}")

    print("\n" + "="*60)
    print("✅ 执行完成")
    print("="*60)


def test_friends_scene():
    """测试朋友场景"""
    print("\n" + "=" * 60)
    print("👥 测试朋友场景")
    print("=" * 60)

    initial_state: PlanState = {
        "user_input": "下午和朋友出去玩，4个人",
        "scene_type": "friends",
        "constraints": {
            "people_count": 4,
            "duration": 4,
            "start_time": "14:00",
            "location": "黄浦区",
            "max_distance_km": 20,
            "max_queue_time": 30,
            "budget": 600
        },
        "user_profile": {},
        "short_term_memory": [],
        "scenario_activities": ["艺术展览", "网红餐厅", "桌游吧"],
        "candidates": [],
        "filtered_candidates": [],
        "filter_reasons": {},
        "selected_plan": {},
        "optimization_score": 0.0,
        "alternative_plans": [],
        "explanation_text": "",
        "action_sequence": [],
        "raw_api_results": {},
        "execution_status": "pending",
        "tool_results": {},
        "payment_order": {},
        "payment_results": {},
        "payment_status": "not_required",
        "retry_history": [],
        "final_share_message": "",
        "execution_log": [],
        "retry_count": 0,
        "need_confirm": False,
        "payment_ui_mode": "dialog",
        "payment_auto_confirm": True,
        "payment_auto_pay": True,
        "payment_method": "mock_pay"
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    print("\n📝 执行日志:")
    for log in final_state.get("execution_log", []):
        print(f"   {log}")

    print(f"\n💬 分享消息: {final_state.get('final_share_message')}")
    print(f"📊 执行状态: {final_state.get('execution_status')}")
    print(f"💳 支付状态: {final_state.get('payment_status')}")

if __name__ == "__main__":
    # 运行家庭场景
    main()

    # 可选：运行朋友场景测试
    # test_friends_scene()
