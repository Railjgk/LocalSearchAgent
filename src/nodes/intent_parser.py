"""Prompt-wrapped intent parser for the WeekendFlow A-stage demo."""

from __future__ import annotations

import re
from typing import Any

from src.nodes._utils import append_log, merge_tool_results
from src.nodes.taxonomy import (
    canonicalize_tags,
    dedupe,
    tags_by_category,
    tags_from_text,
    to_chinese_tags,
)
from src.state import PlanState


INTENT_PARSER_PROMPT = """你是 WeekendFlow 的 Intent Parser。
请把用户的本地生活需求解析为 JSON intent，字段包含:
task_type, goal, scene, time, people, location, budget,
planning_preferences, constraints, missing_slots, confidence。
同时把隐含表达映射为 planning tags:
- 老婆/妻子减肥 -> low_calorie, light_food
- 孩子小/5岁 -> kid_friendly, low_intensity
- 别太远 -> nearby, max_distance_km
- 堂食/订座 -> dine_in, reservation_needed
- 微度假/放松/仪式感 -> micro_vacation, relaxation, ritual
- 周末/下午 -> today_afternoon 或 weekend
只输出结构化 JSON，不输出解释。
"""


CHINESE_NUMBER_MAP = {
    "一": 1,
    "二": 2,
    "两": 2,
    "俩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def normalize_user_input(raw_input: Any) -> str:
    """Normalize supported user input shapes into a single text string."""

    if raw_input is None:
        return ""

    if isinstance(raw_input, str):
        return raw_input.strip()

    if isinstance(raw_input, dict):
        for key in ("content", "text", "input", "query", "user_input"):
            value = raw_input.get(key)
            if value:
                return normalize_user_input(value)
        return ""

    if isinstance(raw_input, (list, tuple)):
        for item in reversed(raw_input):
            if isinstance(item, dict):
                role = str(item.get("role", item.get("type", ""))).lower()
                if role and role not in {"user", "human"}:
                    continue
            text = normalize_user_input(item)
            if text:
                return text
        return ""

    return str(raw_input).strip()


def _extract_age(text: str) -> int | None:
    match = re.search(r"(\d{1,2})\s*岁", text)
    if not match:
        return None
    age = int(match.group(1))
    return age if 0 < age < 18 else None


def _extract_budget(text: str) -> tuple[int | None, str | None]:
    per_person_match = re.search(r"(?:人均|每人|一人)\s*(\d{2,5})", text)
    if per_person_match:
        return int(per_person_match.group(1)), "per_person"

    match = re.search(r"(?:总预算|预算|总共|一共|别超过|不超过)\s*(\d{2,5})", text)
    if match:
        return int(match.group(1)), "total"

    if any(word in text for word in ("省钱", "便宜", "预算别太高", "别太贵")):
        return 300, "total"

    return None, None


def _extract_people_count(text: str) -> int | None:
    match = re.search(r"(?<!孩子)(\d{1,2})\s*(?:个人|人|位)", text)
    if match:
        count = int(match.group(1))
        return count if 0 < count <= 20 else None

    match = re.search(r"([一二两俩三四五六七八九十])\s*(?:个人|人|位)", text)
    if match:
        return CHINESE_NUMBER_MAP.get(match.group(1))

    return None


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _extract_start_time(text: str) -> str | None:
    match = re.search(r"(\d{1,2})\s*[:：]\s*(\d{1,2})", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    match = re.search(
        r"(上午|早上|中午|下午|晚上|今晚)?\s*(\d{1,2})\s*点(?:半|(\d{1,2})分?)?", text
    )
    if not match:
        return None

    period = match.group(1) or ""
    hour = int(match.group(2))
    minute = 30 if "半" in match.group(0) else int(match.group(3) or 0)

    if period in {"下午", "晚上", "今晚"} and hour < 12:
        hour += 12
    elif period == "中午" and hour < 11:
        hour += 12

    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return f"{hour:02d}:{minute:02d}"

    return None


def _extract_route_mode(text: str) -> str:
    if _contains_any(text, ("开车", "自驾", "打车")):
        return "driving"
    if _contains_any(text, ("步行", "走路")):
        return "walking"
    if _contains_any(text, ("公交", "地铁")):
        return "transit"
    if _contains_any(text, ("骑车", "单车")):
        return "bicycling"
    return "unknown"


def _extract_city(text: str) -> str | None:
    for city in ("上海", "北京", "广州", "深圳", "杭州", "成都", "南京", "苏州"):
        if city in text:
            return city
    return None


def _extract_location_origin(text: str) -> str:
    coordinate = re.search(r"(\d{2,3}\.\d+)\s*,\s*(\d{1,2}\.\d+)", text)
    if coordinate:
        return f"{coordinate.group(1)},{coordinate.group(2)}"

    match = re.search(r"从([^，。,.]{2,24}?)(?:出发|附近|开始|走|开车)", text)
    if match:
        return match.group(1).strip()

    match = re.search(r"([^，。,.]{2,18}?)(?:附近|周边)", text)
    if match:
        return match.group(1).strip()

    if _contains_any(text, ("离家", "家附近", "从家")):
        return "home"

    return "unknown"


def _extend_unique(target: list[str], values: Any) -> None:
    target[:] = dedupe(target + canonicalize_tags(values))


def build_intent_prompt(user_input: str) -> str:
    """Build the actual prompt text used by the simple parser."""

    return f"{INTENT_PARSER_PROMPT}\n用户输入: {user_input}\nJSON:"


def parse_intent(user_input: str) -> dict[str, Any]:
    """Parse a natural language local-life request into a structured intent."""

    text = normalize_user_input(user_input)
    if not text:
        return {
            "task_type": "clarify_request",
            "goal": "等待用户提供本地生活需求",
            "scene": "unknown",
            "time": {
                "window": "unspecified",
                "duration_range": [0, 0],
                "start_time": None,
                "end_time": None,
            },
            "people": [],
            "location": {
                "origin": "unknown",
                "distance_preference": "unknown",
                "max_distance_km": None,
                "transport_mode": "unknown",
            },
            "budget": {
                "amount": None,
                "type": None,
                "sensitivity": "unknown",
            },
            "planning_preferences": {
                "activity_type": [],
                "food_type": [],
                "emotion_type": [],
                "atmosphere_type": [],
                "experience_type": [],
                "restaurant_type": [],
                "pace": "unknown",
            },
            "constraints": {
                "hard": [],
                "soft": [],
                "avoid": [],
            },
            "people_count": 0,
            "missing_slots": ["user_input"],
            "confidence": {"user_input": 0.0},
            "raw_text": "",
        }

    child_age = _extract_age(text)
    explicit_people_count = _extract_people_count(text)
    people: list[dict[str, Any]] = [{"role": "self", "needs": []}]
    avoid: list[str] = []
    hard_tags: list[str] = []
    soft_tags: list[str] = []
    activity_type: list[str] = []
    food_type: list[str] = []
    emotion_type: list[str] = []
    atmosphere_type: list[str] = []
    experience_type: list[str] = []
    restaurant_type: list[str] = []
    confidence: dict[str, float] = {}
    text_tags = tags_from_text(text)
    text_groups = tags_by_category(text_tags)
    _extend_unique(activity_type, text_groups["activity"])
    _extend_unique(food_type, text_groups["food"])
    _extend_unique(emotion_type, text_groups["emotion"])
    _extend_unique(
        atmosphere_type,
        [
            tag
            for tag in text_groups["emotion"]
            if tag in {"quiet", "atmosphere", "romantic"}
        ],
    )
    _extend_unique(
        restaurant_type,
        [tag for tag in text_groups["food"] if tag in {"dine_in", "takeaway_only"}],
    )
    _extend_unique(
        experience_type,
        [
            tag
            for tag in text_tags
            if tag in {"hands_on_parent_child", "local_discovery", "local_culture"}
        ],
    )

    spouse_present = _contains_any(
        text,
        ("老婆", "妻子", "太太", "媳妇"),
    )
    partner_present = _contains_any(
        text, ("对象", "情侣", "约会", "女朋友", "男朋友", "伴侣", "爱人")
    )
    friends_present = _contains_any(
        text,
        ("朋友", "同事", "同学", "哥们", "闺蜜", "伙伴"),
    )
    couple_present = partner_present

    if spouse_present:
        wife_needs = []
        wife_state = None
        if _contains_any(
            text, ("减肥", "减脂", "控卡", "低脂", "少油", "低卡", "清淡")
        ):
            wife_state = "dieting"
            wife_needs.extend(["低卡", "轻食"])
            _extend_unique(soft_tags, ["low_calorie", "light_food"])
            _extend_unique(food_type, ["low_calorie", "light_food"])
            confidence["wife_dieting"] = 0.9
        people.append(
            {
                "role": "wife",
                "state": wife_state,
                "needs": wife_needs or ["comfortable"],
            }
        )

    if partner_present and not spouse_present:
        people.append(
            {
                "role": "partner",
                "needs": ["comfortable", "atmosphere"],
            }
        )
        _extend_unique(activity_type, ["date_activity"])
        _extend_unique(soft_tags, ["romantic", "atmosphere"])
        _extend_unique(atmosphere_type, ["romantic", "atmosphere"])
        confidence["couple"] = 0.82

    if friends_present:
        people.append(
            {
                "role": "friends",
                "count": max(1, (explicit_people_count or 2) - 1),
                "needs": ["group_friendly", "social"],
            }
        )
        _extend_unique(activity_type, ["group_activity"])
        _extend_unique(soft_tags, ["group_friendly", "social"])
        confidence["friends"] = 0.85

    if "孩子" in text or "小孩" in text or child_age is not None:
        child_needs = ["儿童友好"]
        _extend_unique(activity_type, ["parent_child"])
        if child_age is not None and child_age <= 6:
            child_needs.append("低强度")
            _extend_unique(hard_tags, ["kid_friendly"])
            _extend_unique(soft_tags, ["low_intensity"])
            _extend_unique(activity_type, ["light_activity"])
            confidence["child_age"] = 0.95
        people.append({"role": "child", "age": child_age, "needs": child_needs})

    if any(
        word in text
        for word in ("别太远", "别离家太远", "不太远", "附近", "近一点", "离家近")
    ):
        distance_preference = "nearby"
        max_distance_km = 8.0
        _extend_unique(soft_tags, ["nearby"])
        _extend_unique(avoid, ["too_far"])
    else:
        distance_preference = "flexible"
        max_distance_km = 15.0

    if _contains_any(text, ("跨区也可以", "远一点也行", "跨区")):
        distance_preference = "cross_area_ok"
        max_distance_km = 20.0
        _extend_unique(soft_tags, ["cross_area_ok"])

    start_time = _extract_start_time(text)
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
        _extend_unique(avoid, ["long_queue", "crowded"])
    else:
        _extend_unique(avoid, ["long_queue"])

    if _contains_any(text, ("商场", "人挤", "人少点")):
        _extend_unique(avoid, ["crowded_mall"])

    if _contains_any(text, ("不要大油", "不想高热量", "高热量", "大油")):
        _extend_unique(avoid, ["high_calorie"])

    if _contains_any(text, ("外带", "打包")) and _contains_any(
        text, ("不要外带", "只要堂食")
    ):
        _extend_unique(avoid, ["takeaway_only"])

    if _contains_any(text, ("踩雷", "靠谱", "评价")):
        _extend_unique(avoid, ["few_reviews", "new_merchant"])

    budget, budget_type = _extract_budget(text)
    low_budget_request = any(
        word in text for word in ("省钱", "便宜", "预算有限", "平价", "低预算")
    )
    if low_budget_request:
        _extend_unique(activity_type, ["budget"])
        _extend_unique(soft_tags, ["budget", "low_budget", "value_for_money"])

    if budget_type == "per_person":
        _extend_unique(soft_tags, ["per_person_budget"])
    elif budget_type == "total":
        _extend_unique(soft_tags, ["total_budget"])

    route_mode = _extract_route_mode(text)
    if route_mode != "unknown":
        _extend_unique(soft_tags, [route_mode])

    route_origin = None
    location_origin = _extract_location_origin(text)
    if re.fullmatch(r"\d{2,3}\.\d+,\d{1,2}\.\d+", location_origin):
        route_origin = location_origin

    city = _extract_city(text)
    if "上海" in text or city is None:
        city = city or "上海"

    ritual_need = "ritual" in text_tags
    if ritual_need:
        _extend_unique(soft_tags, ["ritual"])

    for tag in text_tags:
        category = tags_by_category([tag])
        if category["risk"]:
            continue
        if tag in {"dine_in"}:
            _extend_unique(hard_tags, [tag])
        elif tag in {"takeaway_only"}:
            _extend_unique(soft_tags, [tag])
        else:
            _extend_unique(soft_tags, [tag])

    has_child = any(item["role"] == "child" for item in people)
    if has_child:
        scene = "family"
    elif friends_present:
        scene = "friends"
    elif spouse_present or couple_present:
        scene = "couple"
    elif low_budget_request:
        scene = "low_budget"
    else:
        scene = "solo"

    people_count = explicit_people_count or len(people)
    confidence["scene"] = 0.92 if scene == "family" else 0.7
    confidence["distance"] = 0.85 if distance_preference == "nearby" else 0.45

    origin = location_origin
    missing_slots = []
    if budget is None:
        missing_slots.append("budget")
    if origin == "unknown":
        missing_slots.append("exact_origin")
    if route_mode == "unknown":
        missing_slots.append("transport_mode")

    display_activity_type = to_chinese_tags(activity_type) or ["轻量活动"]
    if (
        "group_activity" in activity_type
        and "group_activity" not in display_activity_type
    ):
        display_activity_type.append("group_activity")

    return {
        "task_type": "local_life_plan",
        "goal": "安排一次本地生活出行计划",
        "scene": scene,
        "time": {
            "window": time_window,
            "duration_range": duration_range,
            "start_time": start_time,
            "end_time": None,
        },
        "people": people,
        "location": {
            "origin": origin,
            "route_origin": route_origin,
            "distance_preference": distance_preference,
            "max_distance_km": max_distance_km,
            "transport_mode": route_mode,
            "route_mode": route_mode,
            "city": city,
        },
        "budget": {
            "amount": budget,
            "type": budget_type,
            "sensitivity": (
                "high"
                if low_budget_request or (budget is not None and budget <= 300)
                else "unknown"
            ),
        },
        "planning_preferences": {
            "activity_type": display_activity_type,
            "food_type": to_chinese_tags(food_type),
            "emotion_type": to_chinese_tags(emotion_type),
            "atmosphere_type": to_chinese_tags(atmosphere_type),
            "experience_type": to_chinese_tags(experience_type),
            "restaurant_type": to_chinese_tags(restaurant_type),
            "pace": "relaxed",
        },
        "constraints": {
            "hard": to_chinese_tags(hard_tags),
            "soft": to_chinese_tags(soft_tags),
            "avoid": to_chinese_tags(avoid),
        },
        "people_count": people_count,
        "ritual_need": ritual_need,
        "emotion_need": to_chinese_tags(emotion_type),
        "missing_slots": missing_slots,
        "confidence": confidence,
        "raw_text": text,
    }


def constraints_from_intent(intent: dict[str, Any]) -> dict[str, Any]:
    """Flatten the intent into fields expected by downstream planning modules."""

    companions = [item for item in intent["people"] if item["role"] != "self"]
    child = next((item for item in companions if item.get("role") == "child"), {})
    spouse = next(
        (item for item in companions if item.get("role") in {"wife", "partner"}),
        {},
    )
    needs = canonicalize_tags(spouse.get("needs", []))
    mom_diet = (
        "low_calorie"
        if spouse.get("state") == "dieting"
        or "low_calorie" in needs
        or "light_food" in needs
        else None
    )
    location = intent.get("location", {}) or {}
    budget = intent.get("budget", {}) or {}
    time_info = intent.get("time", {}) or {}
    avoid = intent["constraints"]["avoid"]
    normalized_avoid = canonicalize_tags(avoid)
    max_queue_time_min = 15 if "long_queue" in normalized_avoid else None

    return {
        "task_type": intent["task_type"],
        "scene": intent["scene"],
        "time_window": time_info["window"],
        "duration_range": time_info["duration_range"],
        "start_time": time_info.get("start_time"),
        "companions": companions,
        "people_count": intent["people_count"],
        "child_age": child.get("age"),
        "mom_diet": mom_diet,
        "origin": location["origin"],
        "route_origin": location.get("route_origin"),
        "location": {"origin": location["origin"]}
        if location.get("route_origin") is None
        else {},
        "distance_preference": location["distance_preference"],
        "max_distance_km": location["max_distance_km"],
        "transport_mode": location["transport_mode"],
        "route_mode": location.get("route_mode", location["transport_mode"]),
        "city": location.get("city"),
        "budget": budget["amount"],
        "budget_type": budget.get("type"),
        "max_queue_time_min": max_queue_time_min,
        "hard_tags": intent["constraints"]["hard"],
        "soft_tags": intent["constraints"]["soft"],
        "avoid": avoid,
        "planning_preferences": intent["planning_preferences"],
        "ritual_need": intent.get("ritual_need", False),
        "emotion_need": intent.get("emotion_need", []),
        "missing_slots": intent["missing_slots"],
        "confidence": intent["confidence"],
        "raw_text": intent["raw_text"],
    }


def intent_parser_node(state: PlanState) -> dict[str, Any]:
    raw_input = state.get(
        "user_input",
        state.get("messages", state.get("input", state.get("query", ""))),
    )
    user_input = normalize_user_input(raw_input)
    prompt = build_intent_prompt(user_input)
    intent = parse_intent(user_input)
    constraints = constraints_from_intent(intent)
    return {
        "user_input": user_input,
        "intent": intent,
        "constraints": constraints,
        "scene_type": intent["scene"],
        "need_confirm": intent["task_type"] == "clarify_request",
        "tool_results": merge_tool_results(state, "intent_parser_prompt", prompt),
        "execution_log": append_log(
            state,
            "[intent_parser] parsed user input into structured intent",
        ),
    }
