# -*- coding: utf-8 -*-
import re
from collections import Counter
from typing import Any


CHINESE_TAG_MAPPING = {
    # A-stage canonical / intermediate tags
    "parent_child": ["kid_friendly", "family_friendly"],
    "light_activity": "low_intensity",
    "group_activity": ["group_friendly", "social"],
    "date_activity": ["romantic", "atmosphere"],
    "budget_activity": "budget",
    "budget_restaurant": "budget",
    "healthy": ["low_calorie", "light_food"],
    "relaxed": "low_intensity",
    "comfortable": "low_intensity",
    "dine_in": "dine_in",
    "nearby": "nearby",
    "too_far": "nearby",
    "long_queue": "long_queue",
    "crowded": "crowded_mall",
    "crowded_mall": "crowded_mall",
    "high_calorie": "high_calorie",
    "takeaway_only": "takeaway_only",

    # ??/??
    "??": "kid_friendly",
    "????": "kid_friendly",
    "??": "kid_friendly",
    "????": "kid_friendly",
    "??": "kid_friendly",
    "??": "kid_friendly",
    "????": "kid_friendly",
    "??": "kid_friendly",
    "??": "family_friendly",
    "????": "family_friendly",
    "????": "family_friendly",

    # ??/??
    "???": "low_intensity",
    "????": "low_intensity",
    "??": "low_intensity",
    "???": "low_intensity",
    "??": "low_intensity",
    "??": "low_intensity",
    "???": "low_intensity",

    # ??
    "??": ["low_calorie", "light_food"],
    "????": ["low_calorie", "light_food"],
    "??": ["low_calorie", "light_food"],
    "??": ["low_calorie", "light_food"],
    "??": ["low_calorie", "light_food"],
    "???": ["low_calorie", "light_food"],
    "??": ["low_calorie", "light_food"],
    "??": ["low_calorie", "light_food"],
    "??": ["low_calorie", "light_food"],
    "??": ["low_calorie", "light_food"],
    "??": ["low_calorie", "light_food"],
    "??": ["low_calorie", "light_food"],
    "??": ["low_calorie", "light_food"],
    "??": ["japanese", "light_food"],
    "???": ["japanese", "light_food"],
    "????": "fresh_ingredients",
    "????": "vegetable_rich",

    # ??/??
    "??": "indoor",
    "??": "indoor",
    "??": "indoor",
    "????": "indoor",

    # ??
    "???": "budget",
    "??": "budget",
    "??": "budget",
    "??": "budget",
    "???": "budget",

    # ??
    "??": "nearby",
    "???": "nearby",
    "?": "nearby",
    "??": "nearby",

    # ??
    "???": "no_queue",
    "???": "no_queue",
    "???": "no_queue",
    "???": "long_queue",
    "???": "long_queue",
    "???": "long_queue",
    "???": "long_queue",

    # ??/??
    "??": "group_friendly",
    "????": ["group_friendly", "social"],
    "????": ["group_friendly", "social"],
    "??": "group_friendly",
    "??": "social",
    "??": "romantic",
    "????": ["romantic", "atmosphere"],
    "??": "romantic",
    "??": "atmosphere",
    "???": "atmosphere",
    "????": "local_culture",
    "??": "local_culture",
    "????": "local_culture",
    "??": "local_market",
    "????": "local_market",
    "Citywalk": "citywalk",
    "??": "hotpot",
    "??": "social",
    "??": "social",
    "??": "dine_in",
    "??": "dine_in",
    "????": "dine_in",
    "??": "takeaway_only",
    "???": "takeaway_only",
    "???": "high_calorie",
    "??": "high_calorie",
    "??": "crowded_mall",
    "??": "crowded_mall",
    "????": "crowded_mall",
    "????": "crowded_mall",
    "??": "trust_evidence",
    "??": "trust_evidence",
    "???": "few_reviews",
    "??": "new_merchant",
    "???": "new_merchant",
    "???": "holiday",

    # ??/???
    "??": "relaxation",
    "??": "relaxation",
    "??": "healing",
    "??": "healing",
    "??": ["wellness", "healing", "relaxation"],
    "??": ["spa", "wellness", "relaxation"],
    "???": "ritual",
    "??": "quiet",
    "??": "comfortable",
    "???": "novelty",
    "????": "local_discovery",
    "???": "micro_vacation",
    "????": "micro_vacation",
    "????": "micro_vacation",

    # ??/??
    "????": "budget",
    "???": "value_for_money",
    "????": "per_person_budget",
    "???": "total_budget",
    "???": "bookable",
    "???": "ticket_required",
    "????": "walk_in_ok",
    "???": "has_inventory",
    "?????": "has_time_slot",
    "?????": "child_seat_available",
}


SCENE_TEMPLATES = {
    "family": ["activity", "transition", "restaurant"],
    "friends": ["activity", "transition", "restaurant"],
    "couple": ["activity", "transition", "restaurant"],
    "low_budget": ["activity", "transition", "restaurant"],
    "solo": ["activity", "transition", "restaurant"],
}


def _flatten_tags(mapped: Any) -> list[str]:
    if not mapped:
        return []
    if isinstance(mapped, list):
        return [str(x).strip() for x in mapped if str(x).strip()]
    return [str(mapped).strip()]


def _as_list(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    if isinstance(values, set):
        return list(values)
    return [values]


def _dedupe(values: list[Any]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _taxonomy_expand(values: Any) -> list[str]:
    """
    Use A-stage taxonomy helpers when they are present, while keeping B usable
    before PR #30 lands.
    """
    try:
        from . import taxonomy  # type: ignore
    except Exception:
        return []

    canonicalize_tags = getattr(taxonomy, "canonicalize_tags", None)
    if callable(canonicalize_tags):
        try:
            return [str(item).strip() for item in _as_list(canonicalize_tags(values)) if str(item).strip()]
        except Exception:
            return []

    return []


def normalize_scene_type(scene_type: Any) -> str:
    value = str(scene_type or "family").strip().lower()
    aliases = {
        "single": "solo",
        "individual": "solo",
        "alone": "solo",
        "friend": "friends",
        "dating": "couple",
    }
    return aliases.get(value, value or "family")


def _extract_companions(constraints: dict | None, user_profile: dict | None = None) -> list[dict]:
    constraints = constraints or {}
    user_profile = user_profile or {}

    companions = _as_list(constraints.get("companions"))
    if companions:
        return [item for item in companions if isinstance(item, dict)]

    people = _as_list(constraints.get("people"))
    if people:
        return [item for item in people if isinstance(item, dict) and item.get("role") != "self"]

    companion_profile = user_profile.get("companion_profile", {})
    derived = []
    if isinstance(companion_profile, dict):
        for role, payload in companion_profile.items():
            if not isinstance(payload, dict):
                continue
            item = {"role": role}
            item.update(payload)
            derived.append(item)
    return derived


def expand_preference_tags(values: Any) -> list[str]:
    """
    ??????????????? tag ??????? tags?
    ?? str / list / tuple / set / None?
    """
    expanded: list[str] = []
    expanded.extend(_taxonomy_expand(values))

    for raw_value in _as_list(values):
        if raw_value is None:
            continue

        value = str(raw_value).strip()
        if not value:
            continue

        # ?????? tag / ??????????
        expanded.append(value)

        # ????
        if value in CHINESE_TAG_MAPPING:
            expanded.extend(_flatten_tags(CHINESE_TAG_MAPPING[value]))

        # ??????? ???????? -> kid_friendly
        for keyword, mapped in CHINESE_TAG_MAPPING.items():
            if keyword in value:
                expanded.extend(_flatten_tags(mapped))

    return _dedupe(expanded)


def collect_tag_fields(payload: dict | None, *field_names: str) -> list[str]:
    """Collect canonical and localized tag variants from A/B handoff payloads."""

    payload = payload or {}
    values: list[Any] = []
    for field_name in field_names:
        values.extend(_as_list(payload.get(field_name)))
        values.extend(_as_list(payload.get(f"{field_name}_cn")))
    return expand_preference_tags(values)


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_child_age(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            value = match.group(0)
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def parse_duration_range(value: Any) -> list[int]:
    """
    ???
    - None -> [240, 360]
    - [4, 6] ?? -> [240, 360]
    - [240, 360] ?? -> [240, 360]
    - "4-6" -> [240, 360]
    """
    default_range = [240, 360]

    if value is None or value == "":
        return default_range

    if isinstance(value, str):
        cleaned = value.replace("??", "").replace("h", "").replace("H", "").strip()
        if "-" in cleaned:
            parts = cleaned.split("-")
        elif "," in cleaned:
            parts = cleaned.split(",")
        else:
            return default_range

        try:
            numbers = [float(parts[0]), float(parts[1])]
        except (TypeError, ValueError, IndexError):
            return default_range

    elif isinstance(value, (int, float)):
        numbers = [float(value), float(value) + 2]
    elif isinstance(value, (list, tuple)):
        if len(value) < 2:
            return default_range
        try:
            numbers = [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return default_range
    else:
        return default_range

    if numbers[0] > numbers[1]:
        numbers = [numbers[1], numbers[0]]

    # ???? 24?????
    if max(numbers) <= 24:
        return [int(numbers[0] * 60), int(numbers[1] * 60)]

    return [int(numbers[0]), int(numbers[1])]


def normalize(value: float, minimum: float, maximum: float) -> float:
    value = to_float(value, minimum)
    minimum = to_float(minimum, 0.0)
    maximum = to_float(maximum, 1.0)

    if maximum <= minimum:
        return 0.0

    normalized = (value - minimum) / (maximum - minimum)
    return max(0.0, min(1.0, normalized))


def safe_match_count(values: Any, tags: Any) -> int:
    """
    ????? tags ??????
    ??????????? tag ??????????
    """
    normalized_values = expand_preference_tags(values)
    normalized_tags = expand_preference_tags(tags) + [
        str(tag).strip() for tag in _as_list(tags) if str(tag).strip()
    ]

    if not normalized_values or not normalized_tags:
        return 0

    count = 0
    for value in normalized_values:
        for tag in normalized_tags:
            if value == tag or value in tag or tag in value:
                count += 1
                break

    return count


def get_scene_template(scene_type: str) -> list[str]:
    return SCENE_TEMPLATES.get(normalize_scene_type(scene_type), SCENE_TEMPLATES["family"])


def collect_preference_sources(
    constraints: dict | None,
    user_profile: dict | None = None,
    scenario_activities: Any = None,
) -> list[str]:
    constraints = constraints or {}
    user_profile = user_profile or {}

    preference_sources: list[str] = []
    planning_preferences = constraints.get("planning_preferences", {}) or {}
    preference_profile = user_profile.get("preference_profile", {}) or {}

    for key in (
        "activity_type",
        "food_type",
        "emotion_type",
        "atmosphere_type",
        "experience_type",
        "restaurant_type",
    ):
        preference_sources.extend(_as_list(planning_preferences.get(key)))

    preference_sources.extend(_as_list(planning_preferences.get("pace")))
    preference_sources.extend(collect_tag_fields(constraints, "hard_tags", "soft_tags", "hard", "soft"))
    preference_sources.extend(_as_list(scenario_activities))

    for key in ("food_preference", "activity_preference"):
        preference_sources.extend(_as_list(user_profile.get(key)))
    preference_sources.extend(_as_list(user_profile.get("emotion_need")))

    preference_sources.extend(_as_list(preference_profile.get("food")))
    preference_sources.extend(_as_list(preference_profile.get("activity")))
    preference_sources.extend(_as_list(preference_profile.get("emotion")))

    if constraints.get("ritual_need"):
        preference_sources.append("ritual")

    return expand_preference_tags(preference_sources)


def derive_scenario_activities(
    constraints: dict | None,
    user_profile: dict | None = None,
    scenario_activities: Any = None,
) -> list[str]:
    explicit = [str(item).strip() for item in _as_list(scenario_activities) if str(item).strip()]
    if explicit:
        return explicit

    collected = collect_preference_sources(constraints, user_profile, scenario_activities)
    seen = set()
    result = []
    for item in collected:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def get_people_count(constraints: dict | None, user_profile: dict | None = None) -> int:
    constraints = constraints or {}
    user_profile = user_profile or {}

    direct = constraints.get("people_count", user_profile.get("people_count"))
    if direct not in (None, ""):
        return max(1, int(to_float(direct, 1)))

    companions = _extract_companions(constraints, user_profile)
    if companions:
        return len(companions) + 1

    return 1


def get_constraint_config(constraints: dict | None) -> dict[str, Any]:
    return get_constraint_config_with_profile(constraints, None)


def get_constraint_config_with_profile(
    constraints: dict | None,
    user_profile: dict | None = None,
) -> dict[str, Any]:
    constraints = constraints or {}
    user_profile = user_profile or {}
    raw_duration = constraints.get("duration_range")
    if raw_duration in (None, ""):
        raw_duration = constraints.get("duration")

    mom_diet = constraints.get("mom_diet")
    companions = _extract_companions(constraints, user_profile)
    child_age = parse_child_age(constraints.get("child_age"))

    if child_age is None:
        for item in companions:
            if item.get("role") == "child":
                child_age = parse_child_age(item.get("age"))
                if child_age is not None:
                    break

    soft_tags = collect_tag_fields(constraints, "soft_tags", "soft")
    hard_tags = collect_tag_fields(constraints, "hard_tags", "hard")
    planning_preferences = constraints.get("planning_preferences", {}) or {}
    planning_food_tags = expand_preference_tags(planning_preferences.get("food_type"))

    if mom_diet in (None, ""):
        for item in companions:
            role = str(item.get("role", "")).lower()
            state = str(item.get("state", "")).lower()
            needs = expand_preference_tags(item.get("needs"))
            if role == "wife" and (
                state == "dieting"
                or "low_calorie" in needs
                or "light_food" in needs
            ):
                mom_diet = "low_calorie"
                break

    if mom_diet in (None, "") and (
        "low_calorie" in soft_tags
        or "light_food" in soft_tags
        or "low_calorie" in hard_tags
        or "light_food" in hard_tags
        or "low_calorie" in planning_food_tags
        or "light_food" in planning_food_tags
    ):
        mom_diet = "low_calorie"

    if isinstance(mom_diet, str):
        lowered = mom_diet.strip().lower()
        if lowered in {"??", "??", "??", "low_cal", "low_calorie", "light_food", "dieting"}:
            mom_diet = "low_calorie"

    max_queue_time = constraints.get("max_queue_time")
    if max_queue_time in (None, ""):
        max_queue_time = constraints.get("max_queue_time_min")

    people_count = get_people_count(constraints, user_profile)
    budget = to_float(constraints.get("budget"), 500.0)
    if constraints.get("budget_type") == "per_person":
        budget *= people_count

    return {
        "max_distance_km": to_float(constraints.get("max_distance_km"), 8.0),
        "max_queue_time": to_float(max_queue_time, 30.0),
        "duration_range": parse_duration_range(raw_duration),
        "budget": budget,
        "child_age": child_age,
        "mom_diet": mom_diet,
        "people_count": people_count,
    }


def generate_relaxation_suggestions(filter_reasons: dict, constraints: dict | None) -> list[str]:
    """
    ?????????????????
    filter_reasons ????? _summary / _relaxation_suggestions ??? key?????????
    """
    constraints = constraints or {}
    reason_values = [
        reason for plan_id, reason in filter_reasons.items()
        if not str(plan_id).startswith("_") and isinstance(reason, str)
    ]

    reason_counts = Counter(reason_values)
    suggestions = []

    max_distance = to_float(constraints.get("max_distance_km"), 8.0)
    max_queue = to_float(constraints.get("max_queue_time"), 30.0)
    budget = to_float(constraints.get("budget"), 500.0)

    if reason_counts.get("???????????", 0) > 0:
        suggestions.append(f"??????? {max_distance:g} ????? {max_distance + 2:g} ??")

    if reason_counts.get("??????", 0) > 0:
        suggestions.append(f"????????? {max_queue:g} ????? {max_queue + 10:g} ??")

    if reason_counts.get("?????????", 0) > 0:
        suggestions.append(f"????? {budget:g} ???? {int(budget * 1.2)} ???")

    if reason_counts.get("????????????", 0) > 0:
        suggestions.append("???????????????????????")

    if reason_counts.get("???????????", 0) > 0:
        suggestions.append("????????????????????????????")

    if reason_counts.get("??????????", 0) > 0:
        suggestions.append("?????????????????????????")

    if reason_counts.get("??????????", 0) > 0:
        suggestions.append("??????????????????")

    if not suggestions:
        suggestions.append("??????????????????????")

    return suggestions[:3]


def build_filter_summary(total_candidates: int, valid_candidates: int, filter_reasons: dict) -> dict[str, Any]:
    reason_values = [
        reason for plan_id, reason in filter_reasons.items()
        if not str(plan_id).startswith("_") and isinstance(reason, str)
    ]

    reason_counts = dict(Counter(reason_values))

    return {
        "total_candidates": total_candidates,
        "valid_candidates": valid_candidates,
        "invalid_candidates": total_candidates - valid_candidates,
        "reason_counts": reason_counts,
    }
