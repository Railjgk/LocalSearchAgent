"""
Mock API Layer - 模拟API调用
C负责
"""

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, Dict

from src.state import PlanState
from src.tools.execution_mock_api import execution_commit


def _b_rag_data_dir(state: PlanState) -> str:
    metadata = state.get("b_poi_rag_metadata") or {}
    if isinstance(metadata, dict):
        raw = str(metadata.get("data_dir") or "").strip()
        if raw and Path(raw).exists():
            return raw
    return ""


@contextmanager
def _execution_fixture_env(state: PlanState):
    """Make C execution consume the same POI supply that B RAG used."""

    data_dir = _b_rag_data_dir(state)
    previous_c = os.environ.get("WF_C_MOCK_DATA_DIR")
    previous_mock = os.environ.get("WF_MOCK_DATA_DIR")
    if data_dir:
        os.environ["WF_C_MOCK_DATA_DIR"] = data_dir
        os.environ.setdefault("WF_MOCK_DATA_DIR", data_dir)
    try:
        yield
    finally:
        if previous_c is None:
            os.environ.pop("WF_C_MOCK_DATA_DIR", None)
        else:
            os.environ["WF_C_MOCK_DATA_DIR"] = previous_c
        if previous_mock is None:
            os.environ.pop("WF_MOCK_DATA_DIR", None)
        else:
            os.environ["WF_MOCK_DATA_DIR"] = previous_mock


def mock_api_layer_node(state: PlanState) -> Dict[str, Any]:
    """
    执行Tool Router生成的动作序列，调用对应的Mock API
    """
    print("📡 [9] Mock API Layer: 调用API...")

    execution_log = state.get("execution_log", [])
    action_sequence = state.get("action_sequence", [])
    selected_plan = state.get("selected_plan", {})
    raw_results = {}

    if not action_sequence:
        execution_log.append("⚠️ Mock API Layer: 没有可执行 action_hints")
        return {
            "raw_api_results": raw_results,
            "execution_commit_result": {},
            "execution_log": execution_log,
        }

    with _execution_fixture_env(state):
        commit_result = execution_commit(
            plan_id=selected_plan.get("plan_id"),
            user_id=state.get("user_id"),
            action_hints=action_sequence,
            execution_contract=selected_plan.get("execution_contract"),
        )

    for idx, step in enumerate(commit_result.get("steps", []), start=1):
        action_type = step.get("action_type")
        key = f"{action_type}_{idx}"
        name = next(
            (
                action.get("name", "")
                for action in action_sequence
                if action.get("step") == idx
            ),
            "",
        )

        result = {
            "success": bool(step.get("success")),
            "status": step.get("status"),
            "failure_reason": step.get("failure_reason"),
            "reservation_id": step.get("reservation_id"),
            "order_id": step.get("order_id"),
            "amount": step.get("amount"),
            "time": step.get("time"),
            "payment_required": False,
            "message": (
                f"{action_type} {step.get('status')}"
                if step.get("success")
                else step.get("failure_reason", "执行失败")
            ),
            "raw_step": step,
        }

        raw_results[key] = {
            "action": action_type,
            "name": name,
            "result": result,
        }

        if result.get("success"):
            execution_log.append(
                f"      ✅ {action_type} 成功: "
                f"{result.get('order_id') or result.get('reservation_id') or result.get('status')}"
            )
        else:
            execution_log.append(
                f"      ❌ {action_type} 失败: "
                f"{result.get('failure_reason', '未知错误')}"
            )

    execution_log.append(
        "✅ Mock API Layer: /execution/commit "
        f"{commit_result.get('overall_status')}"
    )

    return {
        "raw_api_results": raw_results,
        "execution_commit_result": commit_result,
        "retry_history": commit_result.get("retry_history", []),
        "execution_log": execution_log
    }
