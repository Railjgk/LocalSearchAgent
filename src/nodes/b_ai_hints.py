"""Optional LLM semantic hints for B-stage candidate generation."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

from src.nodes.b_utils import expand_preference_tags
from src.nodes.longcat_client import (
    TRUTHY_VALUES,
    chat_completion,
    is_b_ai_enabled,
    load_longcat_config,
    sanitize_longcat_error,
)

try:
    from src.state import PlanState
except ImportError:
    PlanState = dict


CANONICAL_B_HINT_TAGS = {
    "atmosphere",
    "board_game",
    "budget",
    "chat_friendly",
    "city_limited",
    "citywalk",
    "coffee",
    "crowded_mall",
    "date_friendly",
    "dine_in",
    "escape_room",
    "family_friendly",
    "group_friendly",
    "healing",
    "high_protein",
    "hotpot",
    "indoor",
    "kid_friendly",
    "light_food",
    "local_culture",
    "local_experience",
    "local_market",
    "low_calorie",
    "low_intensity",
    "low_oil",
    "low_sugar",
    "micro_vacation",
    "nearby",
    "no_queue",
    "outdoor",
    "photogenic",
    "quiet",
    "relaxation",
    "ritual",
    "romantic",
    "social",
    "spa",
    "sports",
    "trust_evidence",
    "value_for_money",
    "vegetable_rich",
    "wellness",
}

B_SEMANTIC_HINT_SYSTEM_PROMPT = (
    "You are WeekendFlow's B-stage semantic hint generator. "
    "Your output helps candidate recall and ranking, but hard constraints are handled elsewhere. "
    "Read the user's local-life planning request and return JSON only. "
    "Allowed keys: soft_tags, avoid_tags, activity_intent_tags, restaurant_intent_tags, "
    "route_priority, budget_priority, confidence, evidence. "
    "Use short canonical English tags. Do not invent prices, distances, ratings, merchants, or availability. "
    "Do not output hard constraints."
)


def _env_mapping(env: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return os.environ if env is None else env


def is_b_semantic_hints_enabled(env: Mapping[str, str] | None = None) -> bool:
    env = _env_mapping(env)
    raw_value = env.get("WF_B_AI_SEMANTIC_HINTS_ENABLED")
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


def _normalize_hint_tags(values: Any, *, limit: int = 10) -> list[str]:
    normalized: list[str] = []
    for value in _as_list(values):
        expanded = expand_preference_tags(value)
        expanded.append(str(value).strip())
        for tag in expanded:
            tag = str(tag).strip()
            if tag in CANONICAL_B_HINT_TAGS:
                normalized.append(tag)
    return _dedupe_keep_order(normalized, limit=limit)


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


def _normalize_hints(raw_hints: dict[str, Any]) -> dict[str, Any]:
    confidence = raw_hints.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    hints = {
        "soft_tags": _normalize_hint_tags(raw_hints.get("soft_tags"), limit=12),
        "avoid_tags": _normalize_hint_tags(raw_hints.get("avoid_tags"), limit=6),
        "activity_intent_tags": _normalize_hint_tags(raw_hints.get("activity_intent_tags"), limit=8),
        "restaurant_intent_tags": _normalize_hint_tags(raw_hints.get("restaurant_intent_tags"), limit=8),
        "confidence": round(confidence, 3),
    }

    route_priority = str(raw_hints.get("route_priority") or "").strip().lower()
    if route_priority in {"nearby", "short_distance", "low_transfer", "balanced", "not_important"}:
        hints["route_priority"] = route_priority

    budget_priority = str(raw_hints.get("budget_priority") or "").strip().lower()
    if budget_priority in {"strict", "balanced", "flexible", "not_important"}:
        hints["budget_priority"] = budget_priority

    evidence = _dedupe_keep_order(_as_list(raw_hints.get("evidence")), limit=4)
    if evidence:
        hints["evidence"] = evidence

    return hints


def generate_b_semantic_hints(state: PlanState) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not is_b_semantic_hints_enabled():
        return None, None

    config = load_longcat_config()
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
        "allowed_tags": sorted(CANONICAL_B_HINT_TAGS),
    }
    messages = [
        {"role": "system", "content": B_SEMANTIC_HINT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]

    try:
        response = chat_completion(messages, config=config)
        hints = _normalize_hints(_parse_jsonish(response["content"]))
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
        "hints": hints,
    }
    return hints, metadata


def _merge_list_field(payload: dict[str, Any], key: str, additions: list[str], *, limit: int = 16) -> None:
    if not additions:
        return
    payload[key] = _dedupe_keep_order(_as_list(payload.get(key)) + additions, limit=limit)


def apply_b_semantic_hints(
    state: PlanState,
    *,
    constraints: dict[str, Any],
    user_profile: dict[str, Any],
    scenario_activities: list[str],
) -> tuple[dict[str, Any], dict[str, Any], list[str], dict[str, Any] | None]:
    hints, metadata = generate_b_semantic_hints(state)
    if not hints:
        return constraints, user_profile, scenario_activities, metadata

    enhanced_constraints = dict(constraints)
    enhanced_user_profile = dict(user_profile)
    planning_preferences = dict(enhanced_constraints.get("planning_preferences") or {})

    soft_tags = _dedupe_keep_order(
        hints.get("soft_tags", [])
        + hints.get("activity_intent_tags", [])
        + hints.get("restaurant_intent_tags", []),
        limit=18,
    )
    _merge_list_field(enhanced_constraints, "soft_tags", soft_tags, limit=24)
    _merge_list_field(enhanced_constraints, "avoid", hints.get("avoid_tags", []), limit=12)
    _merge_list_field(enhanced_user_profile, "avoid", hints.get("avoid_tags", []), limit=12)

    _merge_list_field(
        planning_preferences,
        "activity_type",
        hints.get("activity_intent_tags", []),
        limit=12,
    )
    _merge_list_field(
        planning_preferences,
        "food_type",
        hints.get("restaurant_intent_tags", []),
        limit=12,
    )
    if planning_preferences:
        enhanced_constraints["planning_preferences"] = planning_preferences

    enhanced_scenario_activities = _dedupe_keep_order(
        scenario_activities
        + hints.get("soft_tags", [])
        + hints.get("activity_intent_tags", [])
        + hints.get("restaurant_intent_tags", []),
        limit=24,
    )

    return enhanced_constraints, enhanced_user_profile, enhanced_scenario_activities, metadata
