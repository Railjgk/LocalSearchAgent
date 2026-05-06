"""Prompt-wrapped intent parser for the WeekendFlow A-stage demo."""

from __future__ import annotations

import re
from typing import Any

from src.nodes._utils import append_log, merge_tool_results
from src.state import PlanState


INTENT_PARSER_PROMPT = """你是 WeekendFlow 的 Intent Parser。
请把用户的本地生活需求解析为 JSON intent，字段包含:
task_type, goal, scene, time, people, location, budget,
planning_preferences, constraints, missing_slots, confidence。
同时把隐含表达映射为 planning tags:
- 老婆/妻子减肥 -> low_calorie, light_food
- 孩子小/5岁 -> kid_friendly, low_intensity
- 别太远 -> nearby, max_distance_km
- 周末/下午 -> today_afternoon 或 weekend
只输出结构化 JSON，不输出解释。
"""


def _extract_age(text: str) -> int | None:
    match = re.search(r"(\d{1,2})\s*岁", text)
    if not match:
        return None
    age = int(match.group(1))
    return age if 0 < age < 18 else None


def _extract_budget(text: str) -> int | None:
    match = re.search(r"(?:预算|人均|别超过|不超过)\s*(\d{2,5})", text)
    if match:
        return int(match.group(1))
    if any(word in text for word in ("省钱", "便宜", "预算别太高", "别太贵")):
        return 300
    return None


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def build_intent_prompt(user_input: str) -> str:
    """Build the actual prompt text used by the simple parser."""

    return f"{INTENT_PARSER_PROMPT}\n用户输入: {user_input}\nJSON:"


def parse_intent(user_input: str) -> dict[str, Any]:
    """Parse a natural language local-life request into a structured intent."""

    text = user_input.strip()
    child_age = _extract_age(text)
    people: list[dict[str, Any]] = [{"role": "self", "needs": []}]
    avoid: list[str] = []
    hard_tags: list[str] = []
    soft_tags: list[str] = []
    activity_type: list[str] = []
    food_type: list[str] = []
    confidence: dict[str, float] = {}

    if _contains_any(text, ("老婆", "妻子", "太太", "媳妇")):
        wife_needs = []
        wife_state = None
        if _contains_any(text, ("减肥", "控卡", "低脂", "少油")):
            wife_state = "dieting"
            wife_needs.extend(["low_calorie", "light_food"])
            soft_tags.extend(["low_calorie", "light_food"])
            food_type.extend(["low_calorie", "light_food"])
            confidence["wife_dieting"] = 0.9
        people.append(
            {
                "role": "wife",
                "state": wife_state,
                "needs": wife_needs or ["comfortable"],
            }
        )

    if "孩子" in text or "小孩" in text or child_age is not None:
        child_needs = ["kid_friendly"]
        activity_type.append("parent_child")
        if child_age is not None and child_age <= 6:
            child_needs.append("low_intensity")
            hard_tags.append("kid_friendly")
            soft_tags.append("low_intensity")
            activity_type.append("light_activity")
            confidence["child_age"] = 0.95
        people.append({"role": "child", "age": child_age, "needs": child_needs})

    if any(
        word in text
        for word in ("别太远", "别离家太远", "不太远", "附近", "近一点", "离家近")
    ):
        distance_preference = "nearby"
        max_distance_km = 8.0
        avoid.append("too_far")
    else:
        distance_preference = "flexible"
        max_distance_km = 15.0

    if _contains_any(text, ("下午", "午后")):
        time_window = "today_afternoon" if "今天" in text else "afternoon"
        duration_range = [4, 6]
        confidence["time_window"] = 0.85 if "今天" in text else 0.7
    elif _contains_any(text, ("晚上", "今晚")):
        time_window = "tonight"
        duration_range = [2, 4]
        confidence["time_window"] = 0.85
    elif _contains_any(text, ("周末", "星期六", "星期天")):
        time_window = "weekend"
        duration_range = [4, 8]
        confidence["time_window"] = 0.8
    else:
        time_window = "unspecified"
        duration_range = [3, 6]
        confidence["time_window"] = 0.35

    if _contains_any(text, ("排队", "等位", "人多")):
        avoid.extend(["long_queue", "crowded"])
    else:
        avoid.append("long_queue")

    budget = _extract_budget(text)
    scene = (
        "family"
        if any(item["role"] in {"wife", "child"} for item in people)
        else "solo"
    )
    people_count = len(people)
    confidence["scene"] = 0.92 if scene == "family" else 0.7
    confidence["distance"] = 0.85 if distance_preference == "nearby" else 0.45

    origin = "home" if _contains_any(text, ("离家", "家附近", "从家")) else "unknown"
    missing_slots = []
    if budget is None:
        missing_slots.append("budget")
    if origin == "unknown":
        missing_slots.append("exact_origin")
    missing_slots.append("transport_mode")

    return {
        "task_type": "local_life_plan",
        "goal": "安排一次本地生活出行计划",
        "scene": scene,
        "time": {
            "window": time_window,
            "duration_range": duration_range,
            "start_time": None,
            "end_time": None,
        },
        "people": people,
        "location": {
            "origin": origin,
            "distance_preference": distance_preference,
            "max_distance_km": max_distance_km,
            "transport_mode": "unknown",
        },
        "budget": {
            "amount": budget,
            "type": "total" if budget is not None else None,
            "sensitivity": (
                "high" if budget is not None and budget <= 300 else "unknown"
            ),
        },
        "planning_preferences": {
            "activity_type": sorted(set(activity_type)) or ["relaxed_activity"],
            "food_type": sorted(set(food_type)),
            "pace": "relaxed",
        },
        "constraints": {
            "hard": sorted(set(hard_tags)),
            "soft": sorted(set(soft_tags)),
            "avoid": sorted(set(avoid)),
        },
        "people_count": people_count,
        "missing_slots": missing_slots,
        "confidence": confidence,
        "raw_text": text,
    }


def constraints_from_intent(intent: dict[str, Any]) -> dict[str, Any]:
    """Flatten the intent into fields expected by downstream planning modules."""

    return {
        "task_type": intent["task_type"],
        "scene": intent["scene"],
        "time_window": intent["time"]["window"],
        "duration_range": intent["time"]["duration_range"],
        "companions": [item for item in intent["people"] if item["role"] != "self"],
        "people_count": intent["people_count"],
        "origin": intent["location"]["origin"],
        "distance_preference": intent["location"]["distance_preference"],
        "max_distance_km": intent["location"]["max_distance_km"],
        "transport_mode": intent["location"]["transport_mode"],
        "budget": intent["budget"]["amount"],
        "hard_tags": intent["constraints"]["hard"],
        "soft_tags": intent["constraints"]["soft"],
        "avoid": intent["constraints"]["avoid"],
        "planning_preferences": intent["planning_preferences"],
        "missing_slots": intent["missing_slots"],
        "confidence": intent["confidence"],
        "raw_text": intent["raw_text"],
    }


def intent_parser_node(state: PlanState) -> dict[str, Any]:
    user_input = state.get("user_input", "")
    prompt = build_intent_prompt(user_input)
    intent = parse_intent(user_input)
    constraints = constraints_from_intent(intent)
    return {
        "intent": intent,
        "constraints": constraints,
        "scene_type": intent["scene"],
        "tool_results": merge_tool_results(state, "intent_parser_prompt", prompt),
        "execution_log": append_log(
            state,
            "[intent_parser] parsed user input into structured intent",
        ),
    }
