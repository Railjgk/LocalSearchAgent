"""Prompt-wrapped intent parser for the WeekendFlow A-stage demo."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping

from src.nodes._utils import append_log, merge_tool_results
from src.nodes.longcat_client import (
    DEFAULT_LONGCAT_BASE_URL,
    DEFAULT_LONGCAT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    TRUTHY_VALUES,
    LongCatConfig,
    chat_completion,
    sanitize_longcat_error,
)
from src.nodes.taxonomy import (
    SCENE_TYPES,
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
只使用下列解析关键词做槽位抽取和 planning tags，不要补充商家、价格、距离、库存或预约结果:
people: family, wife, partner, child, friends, group_activity, group_friendly, social
activity: parent_child, kid_friendly, low_intensity, indoor, outdoor, citywalk, local_market, micro_vacation, wellness
food: low_calorie, light_food, healthy, japanese, hotpot, bbq, dine_in, takeaway_only
emotion: relaxation, healing, ritual, quiet, atmosphere, lively, romantic, comfortable, novelty
route: nearby, short_distance, same_area, cross_area_ok, driving, walking, transit, bicycling
budget: budget, low_budget, value_for_money, per_person_budget, total_budget
risk: long_queue, crowded_mall, crowded, high_calorie, too_far
execution: bookable, ticket_required, reservation_needed, walk_in_ok, has_inventory, has_time_slot
只输出结构化 JSON，不输出解释。
"""

A_LLM_INTENT_SYSTEM_PROMPT = """你是 WeekendFlow A 阶段的 Intent Parser。
你的任务是把用户真实自然语言请求解析成稳定 JSON，供下游 B/C 阶段直接消费。

必须只返回 JSON object，不要 Markdown，不要解释。
JSON schema:
{
  "task_type": "local_life_plan" | "clarify_request",
  "goal": string,
  "scene": "family" | "friends" | "couple" | "low_budget" | "solo" | "unknown",
  "time": {
    "window": string,
    "duration_range": [number, number],
    "start_time": "HH:MM" | null,
    "end_time": "HH:MM" | null
  },
  "people": [
    {
      "role": "self" | "wife" | "partner" | "child" | "friends",
      "age": number | null,
      "state": string | null,
      "needs": [string]
    }
  ],
  "location": {
    "origin": string,
    "route_origin": string | null,
    "distance_preference": "nearby" | "flexible" | "cross_area_ok" | "unknown",
    "max_distance_km": number | null,
    "transport_mode": "driving" | "walking" | "transit" | "bicycling" | "unknown",
    "route_mode": "driving" | "walking" | "transit" | "bicycling" | "unknown",
    "city": string | null
  },
  "budget": {"amount": number | null, "type": "total" | "per_person" | null, "sensitivity": string},
  "planning_preferences": {
    "activity_type": [string],
    "food_type": [string],
    "emotion_type": [string],
    "atmosphere_type": [string],
    "experience_type": [string],
    "restaurant_type": [string],
    "pace": string
  },
  "constraints": {"hard": [string], "soft": [string], "avoid": [string]},
  "people_count": number,
  "ritual_need": boolean,
  "emotion_need": [string],
  "missing_slots": [string],
  "confidence": object,
  "raw_text": string
}

解析关键词只允许来自下列集合，中文原词也可以保留在 raw_text:
people: family, wife, partner, child, friends, group_activity, group_friendly, social
activity: parent_child, kid_friendly, low_intensity, indoor, outdoor, citywalk, local_market, micro_vacation, wellness
food: low_calorie, light_food, healthy, japanese, hotpot, bbq, dine_in, takeaway_only
emotion: relaxation, healing, ritual, quiet, atmosphere, lively, romantic, comfortable, novelty
route: nearby, short_distance, same_area, cross_area_ok, driving, walking, transit, bicycling
budget: budget, low_budget, value_for_money, per_person_budget, total_budget
risk: long_queue, crowded_mall, crowded, high_calorie, too_far
execution: bookable, ticket_required, reservation_needed, walk_in_ok, has_inventory, has_time_slot
不要编造商家、价格、距离、库存或预约结果。
"""

A_LLM_ENABLE_ENV_KEYS = ("WF_A_LLM_ENABLED", "WF_A_AI_ENABLED")
A_LLM_API_FORMAT = "openai"
A_LLM_PROVIDER = "longcat"


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


def _env_mapping(env: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _read_float(env: Mapping[str, str], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        raw_value = env.get(key)
        if not raw_value:
            continue
        try:
            return float(raw_value)
        except ValueError:
            continue
    return default


def _read_int(env: Mapping[str, str], keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        raw_value = env.get(key)
        if not raw_value:
            continue
        try:
            return int(raw_value)
        except ValueError:
            continue
    return default


def is_a_llm_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether A-stage LLM parsing was explicitly enabled."""

    env = _env_mapping(env)
    for key in A_LLM_ENABLE_ENV_KEYS:
        if key in env:
            return env.get(key, "").strip().lower() in TRUTHY_VALUES
    return False


def load_a_llm_config(env: Mapping[str, str] | None = None) -> LongCatConfig | None:
    """Load A-stage OpenAI-compatible LongCat config only when enabled."""

    env = _env_mapping(env)
    if not is_a_llm_enabled(env):
        return None

    api_key = (
        env.get("WF_A_LLM_API_KEY")
        or env.get("WF_A_LLM_APP_KEY")
        or env.get("LONGCAT_API_KEY")
        or env.get("LONGCAT_APP_KEY")
        or ""
    ).strip()
    if not api_key:
        return None

    base_url = (
        env.get("WF_A_LLM_BASE_URL")
        or env.get("LONGCAT_BASE_URL")
        or DEFAULT_LONGCAT_BASE_URL
    ).strip().rstrip("/")
    model = (env.get("WF_A_LLM_MODEL") or env.get("LONGCAT_MODEL") or DEFAULT_LONGCAT_MODEL).strip()

    return LongCatConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=_read_float(
            env,
            ("WF_A_LLM_TIMEOUT_SECONDS", "LONGCAT_TIMEOUT_SECONDS"),
            DEFAULT_TIMEOUT_SECONDS,
        ),
        max_tokens=_read_int(
            env,
            ("WF_A_LLM_MAX_TOKENS", "LONGCAT_MAX_TOKENS"),
            max(DEFAULT_MAX_TOKENS, 900),
        ),
        temperature=_read_float(
            env,
            ("WF_A_LLM_TEMPERATURE", "LONGCAT_TEMPERATURE"),
            DEFAULT_TEMPERATURE,
        ),
    )


def _sanitize_a_llm_error(error: BaseException, env: Mapping[str, str] | None = None) -> str:
    text = sanitize_longcat_error(error, env=env)
    env = _env_mapping(env)
    for key_name in ("WF_A_LLM_API_KEY", "WF_A_LLM_APP_KEY"):
        key_value = env.get(key_name)
        if key_value:
            text = text.replace(key_value, "<redacted>")
    return text


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


def build_llm_intent_messages(user_input: str, baseline_intent: dict[str, Any]) -> list[dict[str, str]]:
    """Build chat messages for the optional A-stage LLM parser."""

    payload = {
        "user_input": user_input,
        "baseline_intent": baseline_intent,
        "allowed_scene_types": sorted(SCENE_TYPES | {"unknown"}),
    }
    return [
        {"role": "system", "content": A_LLM_INTENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def _strip_json_fence(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_jsonish(content: str) -> dict[str, Any]:
    cleaned = _strip_json_fence(content)
    candidates = [cleaned]
    if "{" in cleaned and "}" in cleaned:
        candidates.append(cleaned[cleaned.find("{") : cleaned.rfind("}") + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _as_str_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        raw_values = values
    else:
        raw_values = [values]
    return dedupe([str(value).strip() for value in raw_values if str(value).strip()])


def _as_optional_str(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_optional_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_int(value: Any, default: int | None = None) -> int | None:
    number = _as_optional_float(value)
    if number is None:
        return default
    return int(number)


def _merge_chinese_tags(*values: Any) -> list[str]:
    merged: list[str] = []
    for value in values:
        merged.extend(_as_str_list(value))
    return to_chinese_tags(merged)


def _duration_range(raw_value: Any, fallback: list[Any]) -> list[Any]:
    values = raw_value if isinstance(raw_value, list) else fallback
    if not isinstance(values, list) or len(values) < 2:
        return fallback
    start = _as_optional_float(values[0])
    end = _as_optional_float(values[1])
    if start is None or end is None:
        return fallback
    if start > end:
        start, end = end, start
    if start == int(start) and end == int(end):
        return [int(start), int(end)]
    return [start, end]


def _normalize_scene(raw_scene: Any, fallback: str) -> str:
    scene = str(raw_scene or "").strip()
    aliases = {
        "亲子": "family",
        "家庭": "family",
        "朋友": "friends",
        "多人": "friends",
        "情侣": "couple",
        "约会": "couple",
        "低预算": "low_budget",
        "省钱": "low_budget",
        "单人": "solo",
        "独自": "solo",
    }
    if scene in SCENE_TYPES or scene == "unknown":
        return scene
    return aliases.get(scene, fallback)


def _normalize_people(raw_people: Any, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(raw_people, list):
        return fallback

    role_aliases = {
        "老婆": "wife",
        "妻子": "wife",
        "太太": "wife",
        "对象": "partner",
        "伴侣": "partner",
        "女朋友": "partner",
        "男朋友": "partner",
        "孩子": "child",
        "小孩": "child",
        "朋友": "friends",
        "同事": "friends",
    }
    people: list[dict[str, Any]] = []
    for item in raw_people:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        role = role_aliases.get(role, role)
        if role not in {"self", "wife", "partner", "child", "friends"}:
            continue
        normalized = {
            "role": role,
            "needs": _as_str_list(item.get("needs")),
        }
        age = _as_optional_int(item.get("age"))
        if age is not None:
            normalized["age"] = age
        state = _as_optional_str(item.get("state"))
        if state:
            normalized["state"] = state
        count = _as_optional_int(item.get("count"))
        if count is not None:
            normalized["count"] = count
        people.append(normalized)

    if not people:
        return fallback
    if not any(item.get("role") == "self" for item in people):
        people.insert(0, {"role": "self", "needs": []})
    return people


def _normalize_confidence(raw_confidence: Any, fallback: dict[str, Any]) -> dict[str, float]:
    confidence: dict[str, float] = {}
    if isinstance(fallback, dict):
        for key, value in fallback.items():
            number = _as_optional_float(value)
            if number is not None:
                confidence[str(key)] = max(0.0, min(1.0, number))
    if isinstance(raw_confidence, dict):
        for key, value in raw_confidence.items():
            number = _as_optional_float(value)
            if number is not None:
                confidence[str(key)] = max(0.0, min(1.0, number))
    return confidence


def _normalize_llm_intent(
    raw_intent: dict[str, Any],
    baseline_intent: dict[str, Any],
    user_input: str,
) -> dict[str, Any]:
    """Coerce an LLM response into the deterministic intent schema."""

    intent = dict(baseline_intent)

    task_type = str(raw_intent.get("task_type") or intent["task_type"]).strip()
    intent["task_type"] = task_type if task_type in {"local_life_plan", "clarify_request"} else intent["task_type"]
    intent["goal"] = _as_optional_str(raw_intent.get("goal"), intent.get("goal")) or intent["goal"]
    intent["scene"] = _normalize_scene(raw_intent.get("scene"), intent["scene"])

    raw_time = raw_intent.get("time") if isinstance(raw_intent.get("time"), dict) else {}
    baseline_time = intent.get("time", {}) or {}
    intent["time"] = {
        "window": _as_optional_str(raw_time.get("window"), baseline_time.get("window")) or "unspecified",
        "duration_range": _duration_range(raw_time.get("duration_range"), baseline_time.get("duration_range", [3, 6])),
        "start_time": _as_optional_str(raw_time.get("start_time"), baseline_time.get("start_time")),
        "end_time": _as_optional_str(raw_time.get("end_time"), baseline_time.get("end_time")),
    }

    raw_location = raw_intent.get("location") if isinstance(raw_intent.get("location"), dict) else {}
    baseline_location = intent.get("location", {}) or {}
    route_mode = _as_optional_str(
        raw_location.get("route_mode") or raw_location.get("transport_mode"),
        baseline_location.get("route_mode") or baseline_location.get("transport_mode"),
    )
    if route_mode not in {"driving", "walking", "transit", "bicycling", "unknown"}:
        route_mode = baseline_location.get("route_mode") or baseline_location.get("transport_mode") or "unknown"
    intent["location"] = {
        "origin": _as_optional_str(raw_location.get("origin"), baseline_location.get("origin")) or "unknown",
        "route_origin": _as_optional_str(raw_location.get("route_origin"), baseline_location.get("route_origin")),
        "distance_preference": _as_optional_str(
            raw_location.get("distance_preference"),
            baseline_location.get("distance_preference"),
        )
        or "unknown",
        "max_distance_km": _as_optional_float(
            raw_location.get("max_distance_km"),
            baseline_location.get("max_distance_km"),
        ),
        "transport_mode": route_mode,
        "route_mode": route_mode,
        "city": _as_optional_str(raw_location.get("city"), baseline_location.get("city")),
    }

    raw_budget = raw_intent.get("budget") if isinstance(raw_intent.get("budget"), dict) else {}
    baseline_budget = intent.get("budget", {}) or {}
    budget_type = _as_optional_str(raw_budget.get("type"), baseline_budget.get("type"))
    if budget_type not in {"total", "per_person", None}:
        budget_type = baseline_budget.get("type")
    intent["budget"] = {
        "amount": _as_optional_int(raw_budget.get("amount"), baseline_budget.get("amount")),
        "type": budget_type,
        "sensitivity": _as_optional_str(raw_budget.get("sensitivity"), baseline_budget.get("sensitivity"))
        or "unknown",
    }

    raw_preferences = (
        raw_intent.get("planning_preferences")
        if isinstance(raw_intent.get("planning_preferences"), dict)
        else {}
    )
    baseline_preferences = intent.get("planning_preferences", {}) or {}
    preference_keys = (
        "activity_type",
        "food_type",
        "emotion_type",
        "atmosphere_type",
        "experience_type",
        "restaurant_type",
    )
    planning_preferences = {}
    for key in preference_keys:
        planning_preferences[key] = _merge_chinese_tags(
            baseline_preferences.get(key, []),
            raw_preferences.get(key, []),
        )
    planning_preferences["pace"] = _as_optional_str(
        raw_preferences.get("pace"),
        baseline_preferences.get("pace"),
    ) or "relaxed"
    intent["planning_preferences"] = planning_preferences

    raw_constraints = raw_intent.get("constraints") if isinstance(raw_intent.get("constraints"), dict) else {}
    baseline_constraints = intent.get("constraints", {}) or {}
    intent["constraints"] = {
        "hard": _merge_chinese_tags(
            baseline_constraints.get("hard", []),
            raw_constraints.get("hard", raw_constraints.get("hard_tags", [])),
        ),
        "soft": _merge_chinese_tags(
            baseline_constraints.get("soft", []),
            raw_constraints.get("soft", raw_constraints.get("soft_tags", [])),
        ),
        "avoid": _merge_chinese_tags(
            baseline_constraints.get("avoid", []),
            raw_constraints.get("avoid", raw_constraints.get("avoid_tags", [])),
        ),
    }

    intent["people"] = _normalize_people(raw_intent.get("people"), intent.get("people", []))
    people_count = _as_optional_int(raw_intent.get("people_count"), intent.get("people_count"))
    intent["people_count"] = max(0, people_count or 0)
    intent["ritual_need"] = bool(raw_intent.get("ritual_need", intent.get("ritual_need", False)))
    intent["emotion_need"] = _merge_chinese_tags(
        intent.get("emotion_need", []),
        raw_intent.get("emotion_need", []),
    )
    if "missing_slots" in raw_intent:
        intent["missing_slots"] = _as_str_list(raw_intent.get("missing_slots"))
    else:
        intent["missing_slots"] = _as_str_list(intent.get("missing_slots"))
    intent["confidence"] = _normalize_confidence(raw_intent.get("confidence"), intent.get("confidence", {}))
    intent["raw_text"] = _as_optional_str(raw_intent.get("raw_text"), user_input) or user_input
    intent["parse_source"] = "llm"
    return intent


def maybe_parse_intent_with_llm(
    user_input: str,
    baseline_intent: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Optionally parse intent with LLM, falling back to the deterministic parser."""

    if not is_a_llm_enabled():
        return baseline_intent, None

    config = load_a_llm_config()
    if config is None:
        return baseline_intent, {
            "enabled": True,
            "provider": A_LLM_PROVIDER,
            "api_format": A_LLM_API_FORMAT,
            "success": False,
            "fallback": True,
            "reason": "missing_api_key",
        }

    messages = build_llm_intent_messages(user_input, baseline_intent)
    try:
        response = chat_completion(messages, config=config)
        raw_intent = _parse_jsonish(response["content"])
        if not raw_intent:
            raise ValueError("A-stage LLM response did not contain a JSON object")
        intent = _normalize_llm_intent(raw_intent, baseline_intent, user_input)
    except Exception as exc:
        return baseline_intent, {
            "enabled": True,
            "provider": A_LLM_PROVIDER,
            "api_format": A_LLM_API_FORMAT,
            "model": config.model,
            "base_url": config.base_url,
            "success": False,
            "fallback": True,
            "error_type": type(exc).__name__,
            "error": _sanitize_a_llm_error(exc)[:300],
        }

    return intent, {
        "enabled": True,
        "provider": A_LLM_PROVIDER,
        "api_format": A_LLM_API_FORMAT,
        "base_url": config.base_url,
        "model": response.get("model") or config.model,
        "success": True,
        "fallback": False,
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage", {}),
    }


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
        [
            tag
            for tag in text_groups["food"]
            if tag in {"dine_in", "takeaway_only", "japanese", "hotpot", "bbq"}
        ],
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
    user_input = normalize_user_input(state.get("user_input"))
    if not user_input:
        for fallback_key in ("messages", "input", "query"):
            user_input = normalize_user_input(state.get(fallback_key))
            if user_input:
                break
    prompt = build_intent_prompt(user_input)
    baseline_intent = parse_intent(user_input)
    intent, llm_metadata = maybe_parse_intent_with_llm(user_input, baseline_intent)
    constraints = constraints_from_intent(intent)
    tool_results = merge_tool_results(state, "intent_parser_prompt", prompt)
    result = {
        "user_input": user_input,
        "intent": intent,
        "constraints": constraints,
        "scene_type": intent["scene"],
        "need_confirm": intent["task_type"] == "clarify_request",
        "tool_results": tool_results,
        "execution_log": append_log(
            state,
            "[intent_parser] parsed user input into structured intent"
            if not llm_metadata
            else (
                "[intent_parser] parsed user input with LongCat OpenAI-format LLM"
                if llm_metadata.get("success")
                else "[intent_parser] used deterministic parser after LongCat fallback"
            ),
        ),
    }
    if llm_metadata:
        result["a_llm_intent"] = llm_metadata
    return result
