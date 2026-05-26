"""Run the full local planning backend from a user requirement.

Examples:
    py run_backend_cli.py "今天下午和老婆孩子出去玩，孩子5岁，老婆最近在减肥"
    echo "今晚和朋友吃火锅，4个人，预算600" | py run_backend_cli.py --stdin
    py run_backend_cli.py --json "周末想和对象轻松约会，吃得清淡一点"
"""

from __future__ import annotations

import warnings

warnings.simplefilter("ignore")

import argparse
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import sys
from typing import Any, Iterator

warnings.filterwarnings(
    "ignore",
    message=r".*allowed_objects.*",
)
warnings.filterwarnings(
    "ignore",
    category=Warning,
    module=r"langgraph\.cache\.base.*",
)
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except ModuleNotFoundError:
    pass

from src.graph import get_graph
from src.state import PlanState
from src.tools.execution_mock_api import STATE_DIR, reset_execution_state


DEFAULT_REQUIREMENT = "今天下午和老婆孩子出去玩，孩子5岁，老婆最近在减肥"
EXECUTION_STATE_FILES = (
    "availability_state.json",
    "reservation_state.json",
    "coupon_state.json",
    "order_state.json",
    "route_state.json",
    "execution_state.json",
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _build_initial_state(requirement: str, user_id: str) -> PlanState:
    return {
        "user_id": user_id,
        "user_input": requirement.strip(),
        "scene_type": "unknown",
        "constraints": {},
        "user_profile": {},
        "short_term_memory": [],
        "scenario_activities": [],
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
        "payment_ui_mode": "auto",
        "payment_auto_confirm": True,
        "payment_auto_pay": True,
        "payment_method": "mock_pay",
    }


def _read_requirement(args: argparse.Namespace) -> str:
    if args.stdin:
        text = sys.stdin.read().strip()
    else:
        text = " ".join(args.requirement).strip()
        if not text and sys.stdin.isatty():
            text = input("请输入本地生活需求：").strip()
    return text or DEFAULT_REQUIREMENT


@contextmanager
def _isolated_execution_state(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    snapshot: dict[Path, str | None] = {}
    for name in EXECUTION_STATE_FILES:
        path = STATE_DIR / name
        snapshot[path] = path.read_text(encoding="utf-8") if path.exists() else None

    reset_execution_state()
    try:
        yield
    finally:
        for path, content in snapshot.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")


def _invoke_backend(
    requirement: str,
    *,
    user_id: str,
    isolate_execution_state: bool,
    verbose: bool,
) -> PlanState:
    state = _build_initial_state(requirement, user_id)
    graph = get_graph()

    with _isolated_execution_state(isolate_execution_state):
        if verbose:
            return graph.invoke(state)
        captured = io.StringIO()
        with redirect_stdout(captured):
            return graph.invoke(state)


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _core_output(requirement: str, result: PlanState) -> dict[str, Any]:
    selected_plan = result.get("selected_plan") or {}
    return {
        "requirement": requirement,
        "scene_type": result.get("scene_type"),
        "constraints": result.get("constraints") or {},
        "selected_plan_id": selected_plan.get("plan_id") or selected_plan.get("id"),
        "optimization_score": result.get("optimization_score"),
        "timeline": _safe_list(selected_plan.get("timeline")),
        "explanation": result.get("explanation_text") or "",
        "action_sequence": _safe_list(result.get("action_sequence")),
        "execution_status": result.get("execution_status"),
        "payment_status": result.get("payment_status"),
        "retry_history": _safe_list(result.get("retry_history")),
        "final_share_message": result.get("final_share_message") or "",
        "alternative_plans": _safe_list(result.get("alternative_plans"))[:3],
    }


def _print_human_output(data: dict[str, Any]) -> None:
    print("=" * 72)
    print("本地生活规划结果")
    print("=" * 72)
    print(f"需求：{data['requirement']}")
    print(f"场景：{data.get('scene_type') or 'unknown'}")
    print(f"执行状态：{data.get('execution_status')} / 支付状态：{data.get('payment_status')}")
    if data.get("optimization_score") is not None:
        print(f"方案评分：{data['optimization_score']}")

    print("\n推荐行程：")
    timeline = data.get("timeline") or []
    if not timeline:
        print("  暂无可执行行程")
    for index, item in enumerate(timeline, start=1):
        time_text = item.get("time", "")
        activity = item.get("activity") or item.get("name") or item.get("poi_id", "")
        poi_id = item.get("poi_id", "")
        print(f"  {index}. {time_text}  {activity}  {poi_id}".rstrip())

    explanation = data.get("explanation")
    if explanation:
        print("\n推荐理由：")
        print(f"  {explanation}")

    actions = data.get("action_sequence") or []
    if actions:
        print("\n执行动作：")
        for action in actions:
            name = action.get("name") or action.get("poi_id") or action.get("addon_type") or ""
            print(
                "  - "
                f"{action.get('action_type')} "
                f"{name} "
                f"{action.get('time', '')}".rstrip()
            )

    share_message = data.get("final_share_message")
    if share_message:
        print("\n最终输出：")
        print(f"  {share_message}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full local planning backend.")
    parser.add_argument("requirement", nargs="*", help="用户需求，例如：今晚和朋友吃火锅，4个人，预算600")
    parser.add_argument("--stdin", action="store_true", help="从标准输入读取用户需求")
    parser.add_argument("--user-id", default="u001", help="用于记忆系统的用户 ID")
    parser.add_argument("--json", action="store_true", help="输出 JSON，方便前端或脚本接入")
    parser.add_argument("--verbose", action="store_true", help="显示各节点内部日志")
    parser.add_argument(
        "--persist-execution-state",
        action="store_true",
        help="保留 C 阶段 mock 执行状态文件；默认会隔离并恢复现场",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    requirement = _read_requirement(args)
    result = _invoke_backend(
        requirement,
        user_id=args.user_id,
        isolate_execution_state=not args.persist_execution_state,
        verbose=args.verbose,
    )
    output = _core_output(requirement, result)

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        _print_human_output(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
