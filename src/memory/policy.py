"""Memory projection policy for planner constraints."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.memory.schema import MEMORY_POLICY
from src.memory.utils import to_int
from src.state import ValueMemoryItem


def companion_roles(constraints: dict[str, Any]) -> set[str]:
    companions = constraints.get("companions", []) or []
    return {
        str(item.get("role"))
        for item in companions
        if isinstance(item, dict) and item.get("role")
    }


def has_child_request(constraints: dict[str, Any]) -> bool:
    roles = companion_roles(constraints)
    return "child" in roles or constraints.get("child_age") not in (None, "")


def has_spouse_request(constraints: dict[str, Any]) -> bool:
    roles = companion_roles(constraints)
    return bool(roles.intersection({"wife", "partner", "spouse"})) or constraints.get(
        "mom_diet"
    ) not in (None, "")


def current_child_age(constraints: dict[str, Any], memory: dict[str, Any]) -> Any:
    if constraints.get("child_age") not in (None, ""):
        return constraints.get("child_age")
    return memory.get("companion_profile", {}).get("child", {}).get("age")


def current_mom_diet(constraints: dict[str, Any], memory: dict[str, Any]) -> str | None:
    if constraints.get("mom_diet") not in (None, ""):
        return constraints.get("mom_diet")

    spouse_profile = memory.get("companion_profile", {}).get("wife", {})
    if spouse_profile.get("state") == "dieting":
        return "low_calorie"

    return None


def select_active_value_memory(
    constraints: dict[str, Any],
    memory: dict[str, Any],
) -> list[ValueMemoryItem]:
    value_memory = memory.get("value_profile", [])
    has_child = has_child_request(constraints)
    has_spouse = has_spouse_request(constraints)
    mom_diet = current_mom_diet(constraints, memory) if has_spouse else None
    active_value_ids = {"convenience", "cost_sensitivity"}

    if has_child or constraints.get("scene") == "family":
        active_value_ids.add("family_care")

    soft_tags = constraints.get("soft_tags", []) or []
    if mom_diet == "low_calorie" or any(
        tag in soft_tags for tag in ("low_calorie", "light_food")
    ):
        active_value_ids.add("health")

    return [item for item in value_memory if item.get("value_id") in active_value_ids]


def build_user_profile(
    constraints: dict[str, Any],
    memory: dict[str, Any],
    active_value_memory: list[ValueMemoryItem],
    retrieved_memories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    preference_profile = memory.get("preference_profile", {})
    planning_preferences = constraints.get("planning_preferences", {}) or {}
    food_preference = list(preference_profile.get("food", []) or [])
    food_preference.extend(planning_preferences.get("food_type", []) or [])
    activity_preference = list(preference_profile.get("activity", []) or [])
    activity_preference.extend(planning_preferences.get("activity_type", []) or [])

    return {
        "user_id": memory.get("user_id"),
        "stable_profile": memory.get("stable_profile", {}),
        "companion_profile": memory.get("companion_profile", {}),
        "companions": constraints.get("companions", []) or [],
        "people_count": constraints.get("people_count"),
        "food_preference": sorted(set(food_preference)),
        "activity_preference": sorted(set(activity_preference)),
        "avoid": constraints.get("avoid", []) or [],
        "value_profile": active_value_memory,
        "retrieved_memory_ids": [
            item.get("memory_id") for item in (retrieved_memories or [])
        ],
    }


def apply_value_memory(
    constraints: dict[str, Any],
    memory: dict[str, Any],
) -> dict[str, Any]:
    """Merge memory into calculable planner constraints.

    Current user input remains authoritative. Memory fills defaults, adds
    preferences, and exposes value weights for later ranking.
    """

    merged = deepcopy(constraints)
    merged.setdefault("avoid", [])
    merged.setdefault("hard_tags", [])
    merged.setdefault("soft_tags", [])
    merged.setdefault("companions", [])

    preference_profile = memory.get("preference_profile", {})
    for avoided in preference_profile.get("avoid", []):
        if avoided not in merged["avoid"]:
            merged["avoid"].append(avoided)

    child_profile = memory.get("companion_profile", {}).get("child", {})
    if has_child_request(merged):
        child_age = current_child_age(merged, memory)
        merged["child_age"] = child_age
    else:
        child_age = None
        merged["child_age"] = None

    child_age_value = to_int(child_age)
    if child_age_value is not None and child_profile and child_age_value <= 6:
        if "kid_friendly" not in merged["hard_tags"]:
            merged["hard_tags"].append("kid_friendly")

    if has_spouse_request(merged):
        mom_diet = current_mom_diet(merged, memory)
        if mom_diet:
            merged["mom_diet"] = mom_diet
    else:
        mom_diet = None
        merged["mom_diet"] = None

    if mom_diet == "low_calorie":
        for tag in ("low_calorie", "light_food"):
            if tag not in merged["soft_tags"]:
                merged["soft_tags"].append(tag)

    defaults = memory.get("derived_defaults", {})
    if merged.get("max_distance_km") in (None, ""):
        merged["max_distance_km"] = defaults.get(
            "max_distance_km",
            15.0,
        )
    if merged.get("transport_mode") in (None, "", "unknown"):
        merged["transport_mode"] = memory.get("stable_profile", {}).get(
            "default_transport",
            "unknown",
        )

    value_memory = select_active_value_memory(merged, memory)
    value_weights = {
        item["value_id"]: round(float(item["score"]), 2) for item in value_memory
    }
    value_confidence = {
        item["value_id"]: round(float(item["confidence"]), 2) for item in value_memory
    }
    merged["value_weights"] = value_weights
    merged["value_confidence"] = value_confidence
    merged["score_weights"] = {
        "group_fit": 0.30 + 0.10 * value_weights.get("family_care", 0.0),
        "availability": 0.20 + 0.05 * value_weights.get("convenience", 0.0),
        "route": 0.20 + 0.05 * value_weights.get("convenience", 0.0),
        "health": 0.10 + 0.10 * value_weights.get("health", 0.0),
        "budget": 0.10 + 0.05 * value_weights.get("cost_sensitivity", 0.0),
        "experience": 0.10,
    }
    merged["max_queue_time_min"] = (
        defaults.get("max_queue_time_min", 15)
        if "long_queue" in merged["avoid"]
        else 30
    )
    merged["max_queue_time"] = merged["max_queue_time_min"]
    merged["active_value_ids"] = sorted(value_weights)
    merged["avoid"] = sorted(set(merged["avoid"]))
    merged["hard_tags"] = sorted(set(merged["hard_tags"]))
    merged["soft_tags"] = sorted(set(merged["soft_tags"]))
    merged["memory_policy"] = MEMORY_POLICY
    return merged
