"""Value-aware memory manager for the WeekendFlow A-stage demo."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.nodes._utils import append_log, merge_tool_results
from src.state import PlanState, ValueMemoryItem


DEFAULT_MEMORY: dict[str, Any] = {
    "user_id": "u001",
    "stable_profile": {
        "home_area": "unknown",
        "consumption_level": "middle",
        "default_transport": "drive_or_taxi",
    },
    "companion_profile": {
        "child": {
            "age": 5,
            "needs": ["kid_friendly", "low_intensity"],
            "confidence": 0.9,
            "source": "historical_profile",
        },
        "wife": {
            "state": "dieting",
            "needs": ["low_calorie", "light_food"],
            "ttl": "short_term",
            "confidence": 0.8,
            "source": "recent_user_input",
        },
    },
    "preference_profile": {
        "food": ["light_food", "japanese"],
        "activity": ["indoor", "parent_child", "light_activity"],
        "avoid": ["long_queue", "crowded_mall"],
    },
    "history_feedback": [
        {
            "plan_id": "p001",
            "positive": ["kid_happy"],
            "negative": ["too_crowded", "too_far"],
        },
    ],
    "derived_defaults": {
        "max_distance_km": 8.0,
        "max_queue_time_min": 15,
        "preferred_duration_hours": [4, 6],
    },
}


DEFAULT_VALUE_MEMORY: list[ValueMemoryItem] = [
    {
        "value_id": "family_care",
        "label": "儿童优先和家庭舒适",
        "score": 0.92,
        "confidence": 0.9,
        "ttl": "long_term",
        "source": "historical_profile",
        "planning_effect": "increase group_fit and require kid_friendly activities",
        "evidence": ["常与5岁孩子同行", "历史反馈偏好低强度活动"],
    },
    {
        "value_id": "health",
        "label": "健康饮食",
        "score": 0.82,
        "confidence": 0.78,
        "ttl": "short_term",
        "source": "recent_user_input",
        "planning_effect": "prefer low_calorie and light_food restaurants",
        "evidence": ["妻子处于减脂状态", "偏好轻食或日料"],
    },
    {
        "value_id": "convenience",
        "label": "少排队和少折腾",
        "score": 0.76,
        "confidence": 0.84,
        "ttl": "long_term",
        "source": "history_feedback",
        "planning_effect": "penalize long_queue, crowded_mall and far routes",
        "evidence": ["用户历史负反馈: too_crowded", "当前输入包含别太远"],
    },
    {
        "value_id": "cost_sensitivity",
        "label": "中等预算",
        "score": 0.45,
        "confidence": 0.55,
        "ttl": "long_term",
        "source": "stable_profile",
        "planning_effect": "keep budget reasonable but do not force cheapest option",
        "evidence": ["消费层级为 middle"],
    },
]


def load_memory(user_id: str | None) -> dict[str, Any]:
    """Load demo memory for a user."""

    memory = deepcopy(DEFAULT_MEMORY)
    if user_id:
        memory["user_id"] = user_id
    memory["value_profile"] = deepcopy(DEFAULT_VALUE_MEMORY)
    return memory


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

    preference_profile = memory.get("preference_profile", {})
    for avoided in preference_profile.get("avoid", []):
        if avoided not in merged["avoid"]:
            merged["avoid"].append(avoided)

    child_profile = memory.get("companion_profile", {}).get("child")
    if child_profile and child_profile.get("age", 99) <= 6:
        if "kid_friendly" not in merged["hard_tags"]:
            merged["hard_tags"].append("kid_friendly")

    wife_profile = memory.get("companion_profile", {}).get("wife")
    if wife_profile and wife_profile.get("state") == "dieting":
        for tag in ("low_calorie", "light_food"):
            if tag not in merged["soft_tags"]:
                merged["soft_tags"].append(tag)

    defaults = memory.get("derived_defaults", {})
    if merged.get("distance_preference") != "nearby":
        merged["max_distance_km"] = defaults.get(
            "max_distance_km",
            merged.get("max_distance_km", 15.0),
        )
    if merged.get("transport_mode") == "unknown":
        merged["transport_mode"] = memory.get("stable_profile", {}).get(
            "default_transport",
            "unknown",
        )

    value_memory = memory.get("value_profile", [])
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
    merged["avoid"] = sorted(set(merged["avoid"]))
    merged["hard_tags"] = sorted(set(merged["hard_tags"]))
    merged["soft_tags"] = sorted(set(merged["soft_tags"]))
    merged["memory_policy"] = "explicit_current_input_first"
    return merged


def memory_manager_node(state: PlanState) -> dict[str, Any]:
    constraints = state.get("constraints", {})
    memory = load_memory(state.get("user_id"))
    merged_constraints = apply_value_memory(constraints, memory)
    short_term_memory = list(state.get("short_term_memory", []))
    if state.get("user_input"):
        short_term_memory.append(state["user_input"])
    value_memory = memory["value_profile"]
    tool_payload = {
        "memory": memory,
        "merged_constraints": merged_constraints,
    }
    return {
        "constraints": merged_constraints,
        "memory": memory,
        "user_profile": memory,
        "value_memory": value_memory,
        "short_term_memory": short_term_memory,
        "tool_results": merge_tool_results(state, "memory_manager", tool_payload),
        "execution_log": append_log(
            state,
            "[memory_manager] loaded user profile and applied value memory",
        ),
    }
