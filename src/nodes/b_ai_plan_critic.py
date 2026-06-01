"""Optional LLM critic for B-stage top-K plan reranking.

The critic is intentionally narrow: deterministic B planning still generates,
filters, and scores plans first. The LLM only reviews a small top-K set and can
apply bounded soft rerank/risk adjustments. It must not invent merchants,
prices, distances, inventory, user memory, or C-stage execution results.
"""

from __future__ import annotations

import json
import os
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
except ImportError:  # pragma: no cover - tests may import without full state
    PlanState = dict


DEFAULT_TOP_K = 5
DEFAULT_MAX_SCORE_DELTA = 0.035
DEFAULT_MAX_RISK_DELTA = 0.08
DEFAULT_SKIP_MIN_SCORE_GAP = 0.055
DEFAULT_SKIP_MAX_RISK = 0.18
DEFAULT_SKIP_MAX_RISK_FACTORS = 0
MIN_CONFIDENCE_TO_APPLY = 0.35

B_PLAN_CRITIC_SYSTEM_PROMPT = (
    "You are WeekendFlow's B-stage plan critic. "
    "A has already supplied user intent and memory-derived preferences. "
    "B has already generated and scored feasible top-K plans. "
    "Your job is to identify subtle local-life fit issues among these candidates, "
    "such as family comfort, low cognitive load, route friction, social fit, diet fit, "
    "atmosphere mismatch, peak-time risk, and whether the activity and restaurant feel coherent. "
    "Do not invent merchants, prices, ratings, distances, queue times, inventory, memory, or C-stage execution results. "
    "Do not override hard constraints. "
    "Return JSON only with this schema: "
    "{\"candidate_adjustments\":[{\"plan_id\":\"...\",\"score_delta\":0.0,"
    "\"risk_delta\":0.0,\"confidence\":0.0,\"reasons\":[\"...\"],\"evidence\":[\"...\"]}],"
    "\"global_notes\":[\"...\"]}. "
    "score_delta is a small rerank hint, not an absolute score. Use values between -0.03 and 0.03. "
    "risk_delta is a small risk adjustment between -0.04 and 0.06."
)


def _env_mapping(env: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return os.environ if env is None else env


def is_b_plan_critic_enabled(env: Mapping[str, str] | None = None) -> bool:
    env = _env_mapping(env)
    raw_value = env.get("WF_B_AI_PLAN_CRITIC_ENABLED")
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


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_float(env: Mapping[str, str], key: str, default: float) -> float:
    return _to_float(env.get(key), default)


def _read_int(env: Mapping[str, str], key: str, default: int) -> int:
    return _to_int(env.get(key), default)


def _read_bool(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw_value = env.get(key)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in TRUTHY_VALUES


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


def _compact_tags(tags: Any, *, limit: int = 10) -> list[str]:
    flattened: list[Any] = []
    if isinstance(tags, dict):
        for nested in tags.values():
            flattened.extend(_as_list(nested))
    else:
        flattened.extend(_as_list(tags))
    return _dedupe_keep_order(flattened, limit=limit)


def _compact_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "poi_id": node.get("poi_id"),
        "name": node.get("name"),
        "type": node.get("type"),
        "category": node.get("category") or node.get("restaurant_category") or node.get("primary_category"),
        "tags": _compact_tags(node.get("tags"), limit=12),
        "rating": node.get("rating"),
        "price": node.get("price"),
        "queue_time_min": node.get("queue_time_min"),
        "business_hours": node.get("business_hours"),
        "review_keywords": _dedupe_keep_order(_as_list(node.get("review_keywords")), limit=4),
        "recommended_dishes": _dedupe_keep_order(_as_list(node.get("recommended_dishes")), limit=4),
        "service_facilities": _dedupe_keep_order(_as_list(node.get("service_facilities")), limit=5),
        "decision_profile": node.get("decision_profile"),
    }


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    plan = candidate.get("plan", {}) or {}
    route = plan.get("route", {}) or {}
    budget = plan.get("budget", {}) or {}
    availability = plan.get("availability", {}) or {}
    return {
        "plan_id": plan.get("plan_id"),
        "weighted_score": candidate.get("weighted_score"),
        "objective_vector": candidate.get("objective_vector", {}),
        "risk_factors": _dedupe_keep_order(_as_list(candidate.get("risk_factors")), limit=6),
        "route": {
            "total_distance_km": route.get("total_distance_km"),
            "total_travel_time_min": route.get("total_travel_time_min"),
            "traffic_status": route.get("traffic_status"),
        },
        "budget": {
            "total_price": budget.get("total_price"),
        },
        "availability": {
            "all_available": availability.get("all_available"),
            "max_queue_time_min": availability.get("max_queue_time_min"),
        },
        "estimated_duration_min": plan.get("estimated_duration_min"),
        "tags": _compact_tags(plan.get("tags"), limit=16),
        "nodes": [_compact_node(node) for node in _as_list(plan.get("nodes")) if isinstance(node, dict)],
        "why_selected": candidate.get("why_selected", {}),
    }


def _candidate_plan_id(candidate: dict[str, Any]) -> str:
    return str((candidate.get("plan") or {}).get("plan_id") or "").strip()


def _candidate_score(candidate: dict[str, Any]) -> float:
    return _to_float(candidate.get("weighted_score"), 0.0)


def _candidate_risk(candidate: dict[str, Any]) -> float:
    objective_vector = candidate.get("objective_vector", {}) or {}
    return _clamp(_to_float(objective_vector.get("risk"), 0.0), 0.0, 1.0)


def _plan_critic_gate(
    top_candidates: list[dict[str, Any]],
    env: Mapping[str, str],
) -> tuple[bool, dict[str, Any]]:
    """Skip the LLM critic when deterministic B already has a clear winner."""

    if _read_bool(env, "WF_B_AI_PLAN_CRITIC_ALWAYS", False):
        return False, {"reason": "forced", "always": True}

    if len(top_candidates) < 2:
        return True, {
            "reason": "single_candidate",
            "top_k": len(top_candidates),
        }

    top = top_candidates[0]
    second = top_candidates[1]
    top_score = _candidate_score(top)
    second_score = _candidate_score(second)
    score_gap = max(0.0, top_score - second_score)
    top_risk = _candidate_risk(top)
    top_risk_factor_count = len(_as_list(top.get("risk_factors")))

    min_gap = max(
        0.0,
        _read_float(env, "WF_B_AI_PLAN_CRITIC_SKIP_MIN_SCORE_GAP", DEFAULT_SKIP_MIN_SCORE_GAP),
    )
    max_low_risk = _clamp(
        _read_float(env, "WF_B_AI_PLAN_CRITIC_SKIP_MAX_RISK", DEFAULT_SKIP_MAX_RISK),
        0.0,
        1.0,
    )
    max_risk_factors = max(
        0,
        _read_int(env, "WF_B_AI_PLAN_CRITIC_SKIP_MAX_RISK_FACTORS", DEFAULT_SKIP_MAX_RISK_FACTORS),
    )

    should_skip = (
        score_gap >= min_gap
        and top_risk <= max_low_risk
        and top_risk_factor_count <= max_risk_factors
    )
    return should_skip, {
        "reason": "clear_deterministic_winner" if should_skip else "ambiguous_or_risky",
        "top_plan_id": _candidate_plan_id(top),
        "runner_up_plan_id": _candidate_plan_id(second),
        "score_gap": round(score_gap, 4),
        "top_risk": round(top_risk, 4),
        "top_risk_factor_count": top_risk_factor_count,
        "thresholds": {
            "min_score_gap": min_gap,
            "max_low_risk": max_low_risk,
            "max_risk_factors": max_risk_factors,
        },
    }


def _normalize_adjustments(
    raw_payload: dict[str, Any],
    *,
    known_plan_ids: set[str],
    max_score_delta: float,
    max_risk_delta: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_adjustments = raw_payload.get("candidate_adjustments")
    if raw_adjustments is None:
        raw_adjustments = raw_payload.get("adjustments")

    normalized: list[dict[str, Any]] = []
    for item in _as_list(raw_adjustments):
        if not isinstance(item, dict):
            continue
        plan_id = str(item.get("plan_id") or "").strip()
        if not plan_id or plan_id not in known_plan_ids:
            continue

        confidence = _clamp(_to_float(item.get("confidence"), 0.5), 0.0, 1.0)
        if confidence < MIN_CONFIDENCE_TO_APPLY:
            continue

        score_delta = _clamp(
            _to_float(item.get("score_delta"), 0.0),
            -abs(max_score_delta),
            abs(max_score_delta),
        )
        risk_delta = _clamp(
            _to_float(item.get("risk_delta"), 0.0),
            -abs(max_risk_delta) / 2.0,
            abs(max_risk_delta),
        )

        normalized.append(
            {
                "plan_id": plan_id,
                "score_delta": round(score_delta * confidence, 4),
                "risk_delta": round(risk_delta * confidence, 4),
                "confidence": round(confidence, 3),
                "reasons": _dedupe_keep_order(_as_list(item.get("reasons")), limit=3),
                "evidence": _dedupe_keep_order(_as_list(item.get("evidence")), limit=3),
            }
        )

    return normalized, _dedupe_keep_order(_as_list(raw_payload.get("global_notes")), limit=4)


def generate_b_plan_critic(
    state: PlanState,
    scored_candidates: list[dict[str, Any]],
    *,
    weights: dict[str, float],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    env = _env_mapping()
    if not is_b_plan_critic_enabled(env):
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

    top_k = max(1, _read_int(env, "WF_B_AI_PLAN_CRITIC_TOP_K", DEFAULT_TOP_K))
    max_score_delta = abs(_read_float(env, "WF_B_AI_PLAN_CRITIC_MAX_SCORE_DELTA", DEFAULT_MAX_SCORE_DELTA))
    max_risk_delta = abs(_read_float(env, "WF_B_AI_PLAN_CRITIC_MAX_RISK_DELTA", DEFAULT_MAX_RISK_DELTA))
    top_candidates = scored_candidates[:top_k]
    known_plan_ids = {
        _candidate_plan_id(candidate)
        for candidate in top_candidates
    }
    known_plan_ids.discard("")
    if not known_plan_ids:
        return None, None

    should_skip, gate = _plan_critic_gate(top_candidates, env)
    if should_skip:
        return None, {
            "enabled": True,
            "provider": "longcat",
            "success": False,
            "fallback": False,
            "skipped": True,
            "reason": gate.get("reason"),
            "gate": gate,
            "guardrails": {
                "max_score_delta": max_score_delta,
                "max_risk_delta": max_risk_delta,
                "top_k": len(top_candidates),
            },
        }

    payload = {
        "user_input": state.get("user_input"),
        "scene_type": state.get("scene_type"),
        "constraints": state.get("constraints", {}) or {},
        "user_profile": state.get("user_profile", {}) or {},
        "scenario_activities": state.get("scenario_activities", []) or [],
        "weights": weights,
        "guardrails": {
            "max_score_delta": max_score_delta,
            "max_risk_delta": max_risk_delta,
            "top_k": len(top_candidates),
        },
        "candidates": [_compact_candidate(candidate) for candidate in top_candidates],
    }
    messages = [
        {"role": "system", "content": B_PLAN_CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]

    try:
        response = chat_completion(messages, config=config)
        raw_payload = _parse_jsonish(response["content"])
        adjustments, global_notes = _normalize_adjustments(
            raw_payload,
            known_plan_ids=known_plan_ids,
            max_score_delta=max_score_delta,
            max_risk_delta=max_risk_delta,
        )
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

    return adjustments, {
        "enabled": True,
        "provider": "longcat",
        "model": response.get("model") or config.model,
        "success": True,
        "fallback": False,
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage", {}),
        "guardrails": {
            "max_score_delta": max_score_delta,
            "max_risk_delta": max_risk_delta,
            "top_k": len(top_candidates),
        },
        "adjustments": adjustments,
        "global_notes": global_notes,
    }


def apply_b_plan_critic(
    state: PlanState,
    scored_candidates: list[dict[str, Any]],
    *,
    weights: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    adjustments, metadata = generate_b_plan_critic(state, scored_candidates, weights=weights)
    if not adjustments:
        return scored_candidates, metadata

    by_plan_id = {adjustment["plan_id"]: adjustment for adjustment in adjustments}
    risk_weight = abs(_to_float(weights.get("risk"), 0.0))
    updated_candidates: list[dict[str, Any]] = []
    applied_adjustments: list[dict[str, Any]] = []

    for candidate in scored_candidates:
        plan = candidate.get("plan", {}) or {}
        plan_id = str(plan.get("plan_id") or "").strip()
        adjustment = by_plan_id.get(plan_id)
        if not adjustment:
            updated_candidates.append(candidate)
            continue

        updated = dict(candidate)
        objective_vector = dict(updated.get("objective_vector", {}) or {})
        score_breakdown = dict(updated.get("score_breakdown", {}) or {})
        risk_factors = list(updated.get("risk_factors", []) or [])

        old_score = _to_float(updated.get("weighted_score"), 0.0)
        old_risk = _to_float(objective_vector.get("risk"), 0.0)
        risk_delta = _to_float(adjustment.get("risk_delta"), 0.0)
        score_delta = _to_float(adjustment.get("score_delta"), 0.0)
        new_risk = _clamp(old_risk + risk_delta, 0.0, 1.0)
        risk_delta_applied = new_risk - old_risk
        new_score = _clamp(old_score + score_delta - risk_delta_applied * risk_weight, 0.0, 1.0)

        objective_vector["risk"] = round(new_risk, 3)
        score_breakdown["risk"] = round(new_risk * risk_weight, 3)
        if risk_delta_applied > 0 and adjustment.get("reasons"):
            risk_factors.extend(f"AI critic: {reason}" for reason in adjustment["reasons"])

        applied = {
            "plan_id": plan_id,
            "from_score": round(old_score, 4),
            "to_score": round(new_score, 4),
            "score_delta": round(score_delta, 4),
            "risk_delta": round(risk_delta_applied, 4),
            "confidence": adjustment.get("confidence"),
            "reasons": adjustment.get("reasons", []),
            "evidence": adjustment.get("evidence", []),
        }

        updated["weighted_score"] = round(new_score, 4)
        updated["objective_vector"] = objective_vector
        updated["score_breakdown"] = score_breakdown
        updated["risk_factors"] = _dedupe_keep_order(risk_factors, limit=10)
        updated["ai_critic_adjustment"] = applied
        applied_adjustments.append(applied)
        updated_candidates.append(updated)

    updated_candidates.sort(key=lambda item: item.get("weighted_score", 0.0), reverse=True)
    if metadata is not None:
        metadata = dict(metadata)
        metadata["applied_adjustments"] = applied_adjustments
        metadata["selected_after_critic"] = (updated_candidates[0].get("plan") or {}).get("plan_id") if updated_candidates else None
    return updated_candidates, metadata
