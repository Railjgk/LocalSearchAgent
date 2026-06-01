"""Deterministic observability trace for optional B-stage AI assists."""

from __future__ import annotations

import os
import re
from typing import Any

try:
    from src.state import PlanState
except ImportError:  # pragma: no cover
    PlanState = dict


AI_STAGE_FIELDS = (
    ("requirement_compiler", "b_ai_requirement_compiler"),
    ("semantic_hints", "b_ai_semantic_hints"),
    ("plan_critic", "b_ai_plan_critic"),
    ("explanation", "b_ai_explanation"),
    ("repair_planner", "b_ai_repair_plan"),
)

SECRET_ENV_KEYS = (
    "LONGCAT_API_KEY",
    "LONGCAT_APP_KEY",
    "GAODE_API_KEY",
    "AMAP_API_KEY",
    "AMAP_WEB_KEY",
)

SECRET_PATTERNS = (
    re.compile(r"ak_[A-Za-z0-9_-]{8,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _redact_string(value: str) -> str:
    redacted = value
    for key in SECRET_ENV_KEYS:
        secret = os.environ.get(key)
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, dict):
        return {key: _redact(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_redact(nested) for nested in value]
    return value


def _usage_total_tokens(metadata: dict[str, Any]) -> int:
    usage = metadata.get("usage")
    if not isinstance(usage, dict):
        return 0
    return _to_int(usage.get("total_tokens") or usage.get("total_token_count"), 0)


def _compact_stage(stage_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "stage": stage_name,
        "enabled": bool(metadata.get("enabled", True)),
        "provider": metadata.get("provider"),
        "model": metadata.get("model"),
        "success": metadata.get("success"),
        "fallback": bool(metadata.get("fallback", False)),
        "skipped": bool(metadata.get("skipped", False)),
        "usage": metadata.get("usage", {}),
    }

    for key in (
        "reason",
        "error_type",
        "finish_reason",
        "repair_strategy",
        "selected_after_critic",
        "next_best_action",
    ):
        if metadata.get(key) not in (None, "", []):
            compact[key] = metadata[key]

    if stage_name == "semantic_hints":
        hints = metadata.get("hints")
        if isinstance(hints, dict):
            compact["hint_confidence"] = hints.get("confidence")
            compact["hint_counts"] = {
                "soft_tags": len(_as_list(hints.get("soft_tags"))),
                "avoid_tags": len(_as_list(hints.get("avoid_tags"))),
                "activity_intent_tags": len(_as_list(hints.get("activity_intent_tags"))),
                "restaurant_intent_tags": len(_as_list(hints.get("restaurant_intent_tags"))),
            }

    if stage_name == "plan_critic":
        guardrails = metadata.get("guardrails")
        if isinstance(guardrails, dict):
            compact["top_k"] = guardrails.get("top_k")
            compact["max_score_delta"] = guardrails.get("max_score_delta")
            compact["max_risk_delta"] = guardrails.get("max_risk_delta")
        gate = metadata.get("gate")
        if isinstance(gate, dict):
            compact["gate"] = gate
        compact["adjustment_count"] = len(_as_list(metadata.get("applied_adjustments")))

    if stage_name == "requirement_compiler":
        contract = metadata.get("contract") or metadata.get("deterministic_contract")
        if isinstance(contract, dict):
            compact["hard_requirement_count"] = len(_as_list(contract.get("hard_requirements")))
            compact["needs_confirmation_count"] = len(_as_list(contract.get("needs_confirmation")))

    return _redact(compact)


def build_b_ai_trace(state: PlanState) -> dict[str, Any] | None:
    """Aggregate B AI metadata without adding another AI decision point."""

    stages: list[dict[str, Any]] = []
    total_tokens = 0
    fallback_stages: list[str] = []
    successful_stages: list[str] = []

    for stage_name, field_name in AI_STAGE_FIELDS:
        metadata = state.get(field_name)
        if not isinstance(metadata, dict) or not metadata:
            continue
        compact = _compact_stage(stage_name, metadata)
        stages.append(compact)
        total_tokens += _usage_total_tokens(metadata)
        if compact.get("fallback"):
            fallback_stages.append(stage_name)
        if compact.get("success") is True:
            successful_stages.append(stage_name)

    if not stages:
        return None

    trace = {
        "enabled_stage_count": len(stages),
        "successful_stages": successful_stages,
        "fallback_stages": fallback_stages,
        "total_tokens": total_tokens,
        "stages": stages,
        "decision_impact": {
            "requirement_contract_available": bool(state.get("b_requirement_contract")),
            "semantic_hints_applied": "semantic_hints" in successful_stages,
            "plan_critic_applied": "plan_critic" in successful_stages,
            "repair_plan_available": bool(state.get("b_repair_plan")),
            "selected_after_critic": (state.get("b_ai_plan_critic") or {}).get("selected_after_critic"),
            "repair_strategy": (state.get("b_repair_plan") or {}).get("repair_strategy"),
        },
    }
    return _redact(trace)


def b_ai_trace_node(state: PlanState) -> dict[str, Any]:
    """Attach a compact B AI observability trace when optional AI assists ran."""

    trace = build_b_ai_trace(state)
    if not trace:
        return {}

    execution_log = state.get("execution_log", [])
    execution_log.append(
        "[B] b_ai_trace_node recorded "
        f"{trace['enabled_stage_count']} optional AI assist stage(s)"
    )
    return {
        "b_ai_trace": trace,
        "execution_log": execution_log,
    }
