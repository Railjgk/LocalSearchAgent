"""Small helpers shared by WeekendFlow demo nodes."""

from __future__ import annotations

from typing import Any

from src.state import PlanState


def append_log(state: PlanState, message: str) -> list[str]:
    execution_log = list(state.get("execution_log", []))
    execution_log.append(message)
    return execution_log


def merge_tool_results(state: PlanState, key: str, value: Any) -> dict[str, Any]:
    tool_results = dict(state.get("tool_results", {}))
    tool_results[key] = value
    return tool_results

