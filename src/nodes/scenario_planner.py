"""Scenario planner for the A-stage planning handoff."""

from __future__ import annotations

from typing import Any

from src.nodes._utils import append_log, merge_tool_results
from src.state import PlanState


SCENE_ACTIVITY_HINTS: dict[str, list[str]] = {
    "family": ["亲子乐园", "低强度室内活动", "轻食餐厅", "少排队", "附近"],
    "friends": ["朋友聚会", "室内活动", "特色餐厅", "社交体验", "附近"],
    "couple": ["约会活动", "氛围餐厅", "轻松活动", "附近"],
    "low_budget": ["附近", "平价活动", "平价餐厅", "少排队"],
    "solo": ["轻松活动", "附近餐厅", "低排队"],
}

SCENE_TEMPLATES: dict[str, dict[str, Any]] = {
    "family": {
        "scene_type": "family",
        "poi_mix": ["activity", "restaurant"],
        "route_pattern": ["start", "kid_friendly_activity", "restaurant"],
        "pace": "relaxed",
    },
    "friends": {
        "scene_type": "friends",
        "poi_mix": ["activity", "restaurant"],
        "route_pattern": ["start", "social_activity", "restaurant"],
        "pace": "flexible",
    },
    "couple": {
        "scene_type": "couple",
        "poi_mix": ["activity", "restaurant"],
        "route_pattern": ["start", "date_activity", "restaurant"],
        "pace": "relaxed",
    },
    "low_budget": {
        "scene_type": "low_budget",
        "poi_mix": ["activity", "restaurant"],
        "route_pattern": ["start", "budget_activity", "budget_restaurant"],
        "pace": "compact",
    },
    "solo": {
        "scene_type": "solo",
        "poi_mix": ["activity", "restaurant"],
        "route_pattern": ["start", "light_activity", "restaurant"],
        "pace": "relaxed",
    },
}


def _dedupe(values: list[Any]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _infer_scene_type(intent: dict[str, Any], constraints: dict[str, Any]) -> str:
    scene = constraints.get("scene") or intent.get("scene") or "solo"
    budget = constraints.get("budget")
    budget_sensitivity = (
        intent.get("budget", {}).get("sensitivity")
        if isinstance(intent.get("budget"), dict)
        else None
    )

    if scene in {"family", "friends", "couple"}:
        return scene

    if budget_sensitivity == "high":
        return "low_budget"

    try:
        if budget is not None and float(budget) <= 300:
            return "low_budget"
    except (TypeError, ValueError):
        pass

    return scene if scene in SCENE_ACTIVITY_HINTS else "solo"


def _activity_hints_from_intent(intent: dict[str, Any]) -> list[str]:
    planning_preferences = intent.get("planning_preferences", {}) or {}
    activity_types = planning_preferences.get("activity_type", []) or []
    food_types = planning_preferences.get("food_type", []) or []
    hints = []

    for item in activity_types:
        if item == "parent_child":
            hints.append("亲子乐园")
        elif item == "light_activity":
            hints.append("低强度室内活动")
        elif item == "group_activity":
            hints.append("朋友聚会")
        elif item == "date_activity":
            hints.append("约会活动")
        elif item == "budget_activity":
            hints.append("平价活动")
        else:
            hints.append(str(item))

    for item in food_types:
        if item in {"low_calorie", "light_food"}:
            hints.append("轻食餐厅")
        elif item == "budget":
            hints.append("平价餐厅")
        else:
            hints.append(str(item))

    return hints


def _activity_hints_from_constraints(constraints: dict[str, Any]) -> list[str]:
    hints = []

    hard_tags = constraints.get("hard_tags", []) or []
    soft_tags = constraints.get("soft_tags", []) or []
    avoid = constraints.get("avoid", []) or []

    if "kid_friendly" in hard_tags:
        hints.append("亲子乐园")
    if "low_intensity" in soft_tags:
        hints.append("低强度室内活动")
    if "low_calorie" in soft_tags or "light_food" in soft_tags:
        hints.append("轻食餐厅")
    if "long_queue" in avoid:
        hints.append("少排队")
    if constraints.get("distance_preference") == "nearby":
        hints.append("附近")
    if constraints.get("budget") is not None:
        try:
            if float(constraints["budget"]) <= 300:
                hints.extend(["平价活动", "平价餐厅"])
        except (TypeError, ValueError):
            pass

    return hints


def build_scenario_plan(state: PlanState) -> dict[str, Any]:
    """Build the explicit A-to-B scenario handoff."""

    intent = state.get("intent", {}) or {}
    constraints = state.get("constraints", {}) or {}

    if state.get("need_confirm") or intent.get("task_type") == "clarify_request":
        return {
            "scene_type": "unknown",
            "scenario_activities": [],
            "scenario_template": {
                "scene_type": "unknown",
                "poi_mix": [],
                "route_pattern": [],
                "pace": "unknown",
            },
            "route_pattern_hints": {
                "should_search": False,
                "reason": "missing_user_input",
            },
        }

    scene_type = _infer_scene_type(intent, constraints)
    base_template = dict(SCENE_TEMPLATES.get(scene_type, SCENE_TEMPLATES["solo"]))

    scenario_activities = _dedupe(
        SCENE_ACTIVITY_HINTS.get(scene_type, [])
        + _activity_hints_from_intent(intent)
        + _activity_hints_from_constraints(constraints)
    )

    if not scenario_activities:
        scenario_activities = list(SCENE_ACTIVITY_HINTS["solo"])

    duration_range = constraints.get("duration_range") or [3, 6]
    route_pattern_hints = {
        "should_search": True,
        "max_distance_km": constraints.get("max_distance_km", 8.0),
        "max_queue_time_min": constraints.get("max_queue_time_min"),
        "transport_mode": constraints.get("transport_mode", "unknown"),
        "time_window": constraints.get("time_window", "unspecified"),
        "duration_range": duration_range,
        "prefer_nearby": constraints.get("distance_preference") == "nearby",
        "search_terms": scenario_activities,
    }

    scenario_template = {
        **base_template,
        "time_window": constraints.get("time_window", "unspecified"),
        "duration_range": duration_range,
        "required_tags": constraints.get("hard_tags", []) or [],
        "preferred_tags": constraints.get("soft_tags", []) or [],
        "avoid": constraints.get("avoid", []) or [],
    }

    return {
        "scene_type": scene_type,
        "scenario_activities": scenario_activities,
        "scenario_template": scenario_template,
        "route_pattern_hints": route_pattern_hints,
    }


def scenario_planner_node(state: PlanState) -> dict[str, Any]:
    scenario_plan = build_scenario_plan(state)
    constraints = dict(state.get("constraints", {}) or {})
    constraints["scene"] = scenario_plan["scene_type"]
    constraints["scenario_activities"] = scenario_plan["scenario_activities"]
    constraints["scenario_template"] = scenario_plan["scenario_template"]
    constraints["route_pattern_hints"] = scenario_plan["route_pattern_hints"]

    return {
        **scenario_plan,
        "constraints": constraints,
        "tool_results": merge_tool_results(state, "scenario_planner", scenario_plan),
        "execution_log": append_log(
            state,
            "[scenario_planner] built explicit scenario activities and route hints",
        ),
    }
