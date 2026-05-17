"""Scenario planner for the A-stage planning handoff."""

from __future__ import annotations

from typing import Any

from src.nodes._utils import append_log, merge_tool_results
from src.nodes.taxonomy import (
    CHINESE_SCENARIO_SUBTYPE_LABELS,
    SCENE_DEFAULT_TAGS,
    build_scenario_facets,
    canonicalize_tags,
    dedupe,
    infer_scenario_subtype,
    localize_facets,
    to_chinese_tags,
)
from src.state import PlanState


SCENE_ACTIVITY_HINTS: dict[str, list[str]] = {
    "family": SCENE_DEFAULT_TAGS["family"],
    "friends": SCENE_DEFAULT_TAGS["friends"],
    "couple": SCENE_DEFAULT_TAGS["couple"],
    "low_budget": SCENE_DEFAULT_TAGS["low_budget"],
    "solo": SCENE_DEFAULT_TAGS["solo"],
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
    return dedupe(values)


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
    hints = []

    for key in (
        "activity_type",
        "food_type",
        "emotion_type",
        "atmosphere_type",
        "experience_type",
        "restaurant_type",
    ):
        hints.extend(canonicalize_tags(planning_preferences.get(key)))

    return hints


def _activity_hints_from_constraints(constraints: dict[str, Any]) -> list[str]:
    hints = []

    hard_tags = canonicalize_tags(constraints.get("hard_tags", []))
    soft_tags = canonicalize_tags(constraints.get("soft_tags", []))
    avoid = canonicalize_tags(constraints.get("avoid", []))

    hints.extend(hard_tags)
    hints.extend(soft_tags)
    if constraints.get("distance_preference") == "nearby":
        hints.append("nearby")
    if constraints.get("budget") is not None:
        try:
            if float(constraints["budget"]) <= 300:
                hints.extend(["budget", "low_budget", "value_for_money"])
        except (TypeError, ValueError):
            pass

    return [tag for tag in hints if tag not in avoid]


def build_scenario_plan(state: PlanState) -> dict[str, Any]:
    """Build the explicit A-to-B scenario handoff."""

    intent = state.get("intent", {}) or {}
    constraints = state.get("constraints", {}) or {}

    if state.get("need_confirm") or intent.get("task_type") == "clarify_request":
        return {
            "scene_type": "unknown",
            "scenario_subtype": "unknown",
            "scenario_facets": {},
            "scenario_activities": [],
            "scenario_template": {
                "scene_type": "unknown",
                "scenario_subtype": "unknown",
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
    scenario_activities = canonicalize_tags(scenario_activities)

    if not scenario_activities:
        scenario_activities = list(SCENE_ACTIVITY_HINTS["solo"])

    duration_range = constraints.get("duration_range") or [3, 6]
    facet_tags = scenario_activities + canonicalize_tags(constraints.get("avoid", []))
    scenario_subtype = infer_scenario_subtype(scene_type, scenario_activities, constraints)
    scenario_subtype_label = CHINESE_SCENARIO_SUBTYPE_LABELS.get(
        scenario_subtype,
        scenario_subtype,
    )
    scenario_facets = localize_facets(build_scenario_facets(scene_type, facet_tags, constraints))
    scenario_activities_cn = to_chinese_tags(scenario_activities)
    route_pattern_hints = {
        "should_search": True,
        "route_origin": constraints.get("route_origin"),
        "route_mode": constraints.get("route_mode", constraints.get("transport_mode", "unknown")),
        "city": constraints.get("city"),
        "location": constraints.get("location", {}),
        "max_distance_km": constraints.get("max_distance_km", 8.0),
        "max_queue_time_min": constraints.get("max_queue_time_min"),
        "transport_mode": constraints.get("transport_mode", "unknown"),
        "time_window": constraints.get("time_window", "unspecified"),
        "start_time": constraints.get("start_time"),
        "duration_range": duration_range,
        "prefer_nearby": constraints.get("distance_preference") == "nearby",
        "search_terms": scenario_activities_cn,
    }

    scenario_template = {
        **base_template,
        "scenario_subtype": scenario_subtype,
        "scenario_subtype_label": scenario_subtype_label,
        "scenario_facets": scenario_facets,
        "time_window": constraints.get("time_window", "unspecified"),
        "start_time": constraints.get("start_time"),
        "duration_range": duration_range,
        "required_tags": to_chinese_tags(constraints.get("hard_tags", [])),
        "preferred_tags": to_chinese_tags(constraints.get("soft_tags", [])),
        "avoid": to_chinese_tags(constraints.get("avoid", [])),
    }

    return {
        "scene_type": scene_type,
        "scenario_subtype": scenario_subtype,
        "scenario_subtype_label": scenario_subtype_label,
        "scenario_facets": scenario_facets,
        "scenario_activities": scenario_activities_cn,
        "scenario_template": scenario_template,
        "route_pattern_hints": route_pattern_hints,
    }


def scenario_planner_node(state: PlanState) -> dict[str, Any]:
    scenario_plan = build_scenario_plan(state)
    constraints = dict(state.get("constraints", {}) or {})
    constraints["scene"] = scenario_plan["scene_type"]
    constraints["scenario_subtype"] = scenario_plan["scenario_subtype"]
    constraints["scenario_subtype_label"] = scenario_plan["scenario_subtype_label"]
    constraints["scenario_facets"] = scenario_plan["scenario_facets"]
    constraints["scenario_activities"] = scenario_plan["scenario_activities"]
    constraints["scenario_template"] = scenario_plan["scenario_template"]
    constraints["route_pattern_hints"] = scenario_plan["route_pattern_hints"]
    constraints["hard_tags"] = to_chinese_tags(constraints.get("hard_tags", []))
    constraints["soft_tags"] = to_chinese_tags(constraints.get("soft_tags", []))
    constraints["avoid"] = to_chinese_tags(constraints.get("avoid", []))

    return {
        **scenario_plan,
        "constraints": constraints,
        "tool_results": merge_tool_results(state, "scenario_planner", scenario_plan),
        "execution_log": append_log(
            state,
            "[scenario_planner] built explicit scenario activities and route hints",
        ),
    }
