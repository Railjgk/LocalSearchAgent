"""Optional B-stage repair planner for C-stage execution failures.

The repair planner runs after C has returned an execution result. It does not
execute tools, invent POIs, or override C state. When enabled, it asks the LLM
to propose a bounded repair strategy using only selected-plan facts, C failure
details, and C-provided alternatives. Deterministic validators and C execution
remain the source of truth.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping

from src.nodes.longcat_client import (
    TRUTHY_VALUES,
    chat_completion,
    is_b_ai_enabled,
    load_longcat_config,
    sanitize_longcat_error,
)

try:
    from src.state import PlanState
except ImportError:  # pragma: no cover
    PlanState = dict


ALLOWED_REPAIR_STRATEGIES = {
    "retry_same_poi_new_slot",
    "replace_failed_node",
    "switch_to_alternative_plan",
    "ask_user_confirm",
    "abort",
}
DEFAULT_MAX_NOTES = 4

B_REPAIR_SYSTEM_PROMPT = (
    "You are WeekendFlow's B-stage repair planner. "
    "A has already provided user intent and memory-derived preferences. "
    "B has selected a plan, and C has attempted execution and returned a failure or partial result. "
    "Your task is to propose a local repair strategy that preserves useful successful nodes when possible. "
    "Use only the facts in the payload. Do not invent merchants, POIs, prices, distances, inventory, "
    "reservation ids, order ids, user memory, or C-stage execution results. "
    "Do not bypass hard constraints. "
    "Return JSON only with this schema: "
    "{\"repair_strategy\":\"retry_same_poi_new_slot|replace_failed_node|switch_to_alternative_plan|"
    "ask_user_confirm|abort\","
    "\"preserve_poi_ids\":[\"...\"],"
    "\"replace_failed_node\":{\"failed_poi_id\":\"...\",\"preferred_category\":\"...\","
    "\"constraints\":[\"...\"],\"reason\":\"...\"},"
    "\"time_adjustments\":[{\"action_step\":1,\"from_time\":\"17:30\",\"to_time\":\"18:30\","
    "\"reason\":\"...\"}],"
    "\"candidate_plan_ids\":[\"...\"],"
    "\"user_message\":\"...\","
    "\"confidence\":0.0,"
    "\"evidence\":[\"...\"]}. "
    "If C provides an alternative time slot, prefer retry_same_poi_new_slot. "
    "If the failed POI has no usable alternative, propose replace_failed_node. "
    "If the failure is schema/unknown id, propose ask_user_confirm or abort."
)


def _env_mapping(env: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return os.environ if env is None else env


def is_b_repair_planner_enabled(env: Mapping[str, str] | None = None) -> bool:
    env = _env_mapping(env)
    raw_value = env.get("WF_B_AI_REPAIR_PLANNER_ENABLED")
    if raw_value is None:
        return is_b_ai_enabled(env)
    return raw_value.strip().lower() in TRUTHY_VALUES


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _dedupe_keep_order(values: list[Any], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _parse_jsonish(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _execution_needs_repair(state: PlanState) -> bool:
    commit = state.get("execution_commit_result", {}) or {}
    execution_status = str(state.get("execution_status") or "").strip().lower()
    if commit.get("success") is True or commit.get("overall_status") == "completed":
        return False
    if execution_status in {"success", "completed"}:
        return False
    if commit.get("success") is False or commit.get("overall_status") in {"failed", "rejected", "partial"}:
        return True
    return execution_status in {"failed", "partial"}


def _compact_timeline_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": item.get("type"),
        "activity": item.get("activity"),
        "poi_id": item.get("poi_id"),
        "time": item.get("time"),
        "duration_min": item.get("duration_min"),
        "notes": _dedupe_keep_order(_as_list(item.get("notes")), limit=5),
    }


def _compact_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": action.get("step"),
        "action_type": action.get("action_type"),
        "poi_id": action.get("poi_id"),
        "merchant_id": action.get("merchant_id"),
        "product_id": action.get("product_id"),
        "deal_id": action.get("deal_id"),
        "time": action.get("time"),
        "name": action.get("name"),
        "people": action.get("people"),
        "quantity": action.get("quantity"),
        "notes": _dedupe_keep_order(_as_list(action.get("notes")), limit=5),
    }


def _compact_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_id": step.get("step_id"),
        "action_type": step.get("action_type"),
        "status": step.get("status"),
        "success": step.get("success"),
        "failure_reason": step.get("failure_reason"),
        "poi_id": step.get("poi_id"),
        "merchant_id": step.get("merchant_id"),
        "product_id": step.get("product_id"),
        "deal_id": step.get("deal_id"),
        "time": step.get("time"),
        "alternatives": _as_list(step.get("alternatives"))[:5],
    }


def _compact_alternative_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan_id": plan.get("plan_id"),
        "title": plan.get("title"),
        "dominant_dimension": plan.get("dominant_dimension"),
        "weighted_score": plan.get("weighted_score"),
        "total_price": plan.get("total_price"),
        "total_distance_km": plan.get("total_distance_km"),
        "tradeoff": plan.get("tradeoff"),
    }


def _known_poi_ids(state: PlanState) -> set[str]:
    ids: set[str] = set()
    selected_plan = state.get("selected_plan", {}) or {}
    for item in _as_list(selected_plan.get("timeline")) + _as_list(selected_plan.get("action_hints")):
        if isinstance(item, dict) and item.get("poi_id"):
            ids.add(str(item["poi_id"]))
    for action in _as_list(state.get("action_sequence")):
        if isinstance(action, dict) and action.get("poi_id"):
            ids.add(str(action["poi_id"]))
    for step in _as_list((state.get("execution_commit_result", {}) or {}).get("steps")):
        if isinstance(step, dict) and step.get("poi_id"):
            ids.add(str(step["poi_id"]))
    return ids


def _known_plan_ids(state: PlanState) -> set[str]:
    ids: set[str] = set()
    selected_plan = state.get("selected_plan", {}) or {}
    if selected_plan.get("plan_id"):
        ids.add(str(selected_plan["plan_id"]))
    for plan in _as_list(state.get("alternative_plans")):
        if isinstance(plan, dict) and plan.get("plan_id"):
            ids.add(str(plan["plan_id"]))
    return ids


def _allowed_alternative_times(state: PlanState) -> set[str]:
    times: set[str] = set()
    commit = state.get("execution_commit_result", {}) or {}
    for step in _as_list(commit.get("steps")):
        if not isinstance(step, dict):
            continue
        for alt in _as_list(step.get("alternatives")):
            if isinstance(alt, dict) and alt.get("time"):
                times.add(str(alt["time"]))
    for item in _as_list(commit.get("retry_history")) + _as_list(state.get("retry_history")):
        if isinstance(item, dict):
            for key in ("retry_time", "time"):
                if item.get(key):
                    times.add(str(item[key]))
            for alt in _as_list(item.get("alternatives")):
                if isinstance(alt, dict) and alt.get("time"):
                    times.add(str(alt["time"]))
    return times


def _compact_failure_context(state: PlanState) -> dict[str, Any]:
    commit = state.get("execution_commit_result", {}) or {}
    selected_plan = state.get("selected_plan", {}) or {}
    return {
        "execution_status": state.get("execution_status"),
        "commit": {
            "success": commit.get("success"),
            "overall_status": commit.get("overall_status"),
            "failed_step": commit.get("failed_step"),
            "failure_reason": commit.get("failure_reason"),
            "retry_history": _as_list(commit.get("retry_history"))[:5],
            "steps": [_compact_step(step) for step in _as_list(commit.get("steps")) if isinstance(step, dict)],
        },
        "selected_plan": {
            "plan_id": selected_plan.get("plan_id"),
            "title": selected_plan.get("title"),
            "scene_type": selected_plan.get("scene_type"),
            "total_price": selected_plan.get("total_price"),
            "total_distance_km": selected_plan.get("total_distance_km"),
            "timeline": [
                _compact_timeline_item(item)
                for item in _as_list(selected_plan.get("timeline"))
                if isinstance(item, dict)
            ],
            "risk_factors": _dedupe_keep_order(_as_list(selected_plan.get("risk_factors")), limit=6),
        },
        "action_sequence": [
            _compact_action(action)
            for action in _as_list(state.get("action_sequence"))
            if isinstance(action, dict)
        ],
        "alternative_plans": [
            _compact_alternative_plan(plan)
            for plan in _as_list(state.get("alternative_plans"))
            if isinstance(plan, dict)
        ][:3],
    }


def _looks_like_time(value: str) -> bool:
    return bool(re.match(r"^\d{1,2}:\d{2}$", value.strip()))


def _normalize_repair_plan(raw_payload: dict[str, Any], state: PlanState) -> dict[str, Any] | None:
    strategy = str(raw_payload.get("repair_strategy") or raw_payload.get("strategy") or "").strip()
    if strategy not in ALLOWED_REPAIR_STRATEGIES:
        strategy = "ask_user_confirm"

    known_pois = _known_poi_ids(state)
    known_plans = _known_plan_ids(state)
    allowed_times = _allowed_alternative_times(state)

    preserve_poi_ids = [
        poi_id
        for poi_id in _dedupe_keep_order(_as_list(raw_payload.get("preserve_poi_ids")), limit=4)
        if poi_id in known_pois
    ]

    raw_replace = raw_payload.get("replace_failed_node")
    replace_failed_node = {}
    if isinstance(raw_replace, dict):
        failed_poi_id = str(raw_replace.get("failed_poi_id") or "").strip()
        if failed_poi_id in known_pois:
            replace_failed_node = {
                "failed_poi_id": failed_poi_id,
                "preferred_category": str(raw_replace.get("preferred_category") or "").strip()[:80],
                "constraints": _dedupe_keep_order(_as_list(raw_replace.get("constraints")), limit=6),
                "reason": str(raw_replace.get("reason") or "").strip()[:180],
            }

    time_adjustments = []
    for raw_item in _as_list(raw_payload.get("time_adjustments")):
        if not isinstance(raw_item, dict):
            continue
        to_time = str(raw_item.get("to_time") or "").strip()
        if not allowed_times or to_time not in allowed_times:
            continue
        if not _looks_like_time(to_time):
            continue
        time_adjustments.append(
            {
                "action_step": _to_int(raw_item.get("action_step"), 0),
                "from_time": str(raw_item.get("from_time") or "").strip()[:20],
                "to_time": to_time,
                "reason": str(raw_item.get("reason") or "").strip()[:180],
            }
        )
        if len(time_adjustments) >= 3:
            break

    candidate_plan_ids = [
        plan_id
        for plan_id in _dedupe_keep_order(_as_list(raw_payload.get("candidate_plan_ids")), limit=3)
        if plan_id in known_plans
    ]

    confidence = round(_clamp(_to_float(raw_payload.get("confidence"), 0.5), 0.0, 1.0), 3)
    return {
        "repair_strategy": strategy,
        "preserve_poi_ids": preserve_poi_ids,
        "replace_failed_node": replace_failed_node,
        "time_adjustments": time_adjustments,
        "candidate_plan_ids": candidate_plan_ids,
        "user_message": str(raw_payload.get("user_message") or "").strip()[:220],
        "confidence": confidence,
        "evidence": _dedupe_keep_order(_as_list(raw_payload.get("evidence")), limit=DEFAULT_MAX_NOTES),
    }


def generate_b_repair_plan(state: PlanState) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    env = _env_mapping()
    if not _execution_needs_repair(state):
        return None, None
    if not is_b_repair_planner_enabled(env):
        return None, None

    config = load_longcat_config(env)
    if config is None:
        return None, {
            "enabled": True,
            "provider": "longcat",
            "success": False,
            "fallback": True,
            "reason": "missing_api_key",
        }

    payload = {
        "user_input": state.get("user_input"),
        "scene_type": state.get("scene_type"),
        "constraints": state.get("constraints", {}) or {},
        "user_profile": state.get("user_profile", {}) or {},
        "scenario_activities": state.get("scenario_activities", []) or [],
        "failure_context": _compact_failure_context(state),
        "guardrails": {
            "known_poi_ids": sorted(_known_poi_ids(state)),
            "known_plan_ids": sorted(_known_plan_ids(state)),
            "allowed_alternative_times": sorted(_allowed_alternative_times(state)),
            "allowed_repair_strategies": sorted(ALLOWED_REPAIR_STRATEGIES),
        },
    }
    messages = [
        {"role": "system", "content": B_REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]

    try:
        response = chat_completion(messages, config=config)
        repair_plan = _normalize_repair_plan(_parse_jsonish(response["content"]), state)
    except Exception as exc:
        return None, {
            "enabled": True,
            "provider": "longcat",
            "model": config.model,
            "success": False,
            "fallback": True,
            "error_type": type(exc).__name__,
            "error": sanitize_longcat_error(exc)[:300],
        }

    metadata = {
        "enabled": True,
        "provider": "longcat",
        "model": response.get("model") or config.model,
        "success": True,
        "fallback": False,
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage", {}),
        "repair_strategy": (repair_plan or {}).get("repair_strategy"),
    }
    return repair_plan, metadata


def repair_planner_node(state: PlanState) -> dict[str, Any]:
    """Generate a bounded B repair recommendation after C execution failure."""

    if not _execution_needs_repair(state):
        return {}

    execution_log = state.get("execution_log", [])
    repair_plan, metadata = generate_b_repair_plan(state)

    updates: dict[str, Any] = {}
    if repair_plan:
        updates["b_repair_plan"] = repair_plan
        execution_log.append(
            "[B] repair_planner_node proposed "
            f"{repair_plan.get('repair_strategy')} after C execution failure"
        )
    if metadata:
        updates["b_ai_repair_plan"] = metadata
        if not metadata.get("success"):
            execution_log.append("[B] repair_planner_node kept C failure state; LLM repair unavailable")
    if updates:
        updates["execution_log"] = execution_log
    return updates
