# -*- coding: utf-8 -*-
import re
from collections import Counter
from typing import Any

from .taxonomy import CANONICAL_BY_CHINESE, CHINESE_TAG_LABELS, TRIGGER_TAGS

CANONICAL_ALIAS_TAGS = {
    "parent_child": ["kid_friendly", "family_friendly"],
    "light_activity": ["low_intensity"],
    "group_activity": ["group_friendly", "social"],
    "date_activity": ["romantic", "atmosphere"],
    "budget_activity": ["budget"],
    "budget_restaurant": ["budget"],
    "healthy": ["low_calorie", "light_food"],
    "relaxed": ["low_intensity", "relaxation"],
    "comfortable": ["low_intensity"],
    "nearby": ["nearby", "short_distance"],
    "dine_in": ["dine_in"],
    "too_far": ["nearby"],
    "long_queue": ["long_queue"],
    "crowded": ["crowded", "crowded_mall"],
    "crowded_mall": ["crowded_mall"],
    "high_calorie": ["high_calorie"],
    "takeaway_only": ["takeaway_only"],
}

CHINESE_TAG_MAPPING = {**TRIGGER_TAGS, **CANONICAL_ALIAS_TAGS}

CHINESE_VALUE_LABELS = {
    "activity": "活动",
    "restaurant": "餐厅",
    "indoor": "室内",
    "outdoor": "户外",
    "nearby": "附近",
    "short_distance": "距离近",
    "low_intensity": "低强度",
    "kid_friendly": "儿童友好",
    "family_friendly": "家庭友好",
    "child_seat": "儿童椅",
    "coupon_available": "有优惠券",
    "budget": "平价",
    "mid": "中等价位",
    "mid_high": "中高档",
    "low": "低",
    "medium": "中",
    "high": "高",
    "dine_in": "堂食",
    "takeaway": "外带",
    "takeaway_only": "仅外带",
    "before_1h_free": "提前1小时可退",
    "anytime_refund_before_use": "未使用随时可退",
    "indoor_safe": "室内安全",
    "weather_sensitive": "天气敏感",
    "restaurant_category": "餐饮类目",
    "light_food": "轻食",
    "salad_light_food": "沙拉轻食",
    "japanese_light_food": "日料轻食",
    "vegetarian_light_food": "素食轻食",
    "family_bistro": "家庭餐厅",
    "regional_home_cuisine": "本帮家常菜",
    "local_cuisine": "本地菜",
    "barbecue": "烤肉",
    "bbq": "烤肉",
    "hotpot": "火锅",
    "fried_chicken": "炸鸡小吃",
    "fast_food": "快餐",
    "parent_child_activity": "亲子活动",
    "indoor_playground": "亲子乐园",
    "handcraft": "手作体验",
    "art_experience": "艺术体验",
    "museum": "博物馆展览",
    "citywalk": "城市漫步",
    "local_market": "本地市集",
    "sports": "运动体验",
    "escape_room": "密室桌游",
    "micro_vacation": "近场放松",
    "wellness": "康养放松",
    "spa": "SPA",
    "low_calorie": "低卡",
    "healthy": "健康",
    "low_oil": "少油",
    "low_sugar": "低糖",
    "high_protein": "高蛋白",
    "vegetable_rich": "蔬菜丰富",
    "high_calorie": "高热量",
    "high_oil": "油脂偏高",
    "high_sodium": "钠含量偏高",
    "long_queue": "排队久",
    "crowded_mall": "商场拥挤",
    "smoke_smell": "油烟味",
    "limited_seats": "座位有限",
    "lively": "热闹",
    "popular": "热门",
    "quiet": "安静",
    "clean": "干净",
    "simple": "清爽简洁",
    "warm": "温暖",
    "local": "本地感",
    "social": "社交",
    "group_friendly": "多人友好",
    "date_friendly": "约会友好",
    "romantic": "浪漫",
    "relaxation": "放松",
    "healing": "疗愈",
    "ritual": "仪式感",
    "photogenic": "适合拍照",
    "cultural": "文化感",
    "educational": "教育启发",
    "creative": "创意",
    "local_experience": "本地体验",
    "local_culture": "本地文化",
    "city_limited": "城市限定",
}


SCENE_TEMPLATES = {
    "family": ["activity", "transition", "restaurant"],
    "friends": ["activity", "transition", "restaurant"],
    "couple": ["activity", "transition", "restaurant"],
    "low_budget": ["activity", "transition", "restaurant"],
    "solo": ["activity", "transition", "restaurant"],
}

CITY_ADCODE_BY_NAME = {
    "上海": "310000",
    "上海市": "310000",
    "shanghai": "310000",
    "Shanghai": "310000",
    "北京": "110000",
    "北京市": "110000",
    "beijing": "110000",
    "Beijing": "110000",
    "广州": "440100",
    "广州市": "440100",
    "深圳": "440300",
    "深圳市": "440300",
    "杭州": "330100",
    "杭州市": "330100",
    "成都": "510100",
    "成都市": "510100",
    "南京": "320100",
    "南京市": "320100",
    "苏州": "320500",
    "苏州市": "320500",
    "青岛": "370200",
    "青岛市": "370200",
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


def normalize_city_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text in CITY_ADCODE_BY_NAME:
        return text[:-1] if text.endswith("市") and len(text) > 2 else text
    if text.endswith("市"):
        text = text[:-1]
    return text


def destination_city_from_constraints(constraints: dict | None) -> str:
    constraints = constraints or {}
    location = constraints.get("location")
    location = location if isinstance(location, dict) else {}
    for key in ("destination_city", "trip_city", "city"):
        city = normalize_city_name(constraints.get(key))
        if city:
            return city
    for key in ("destination_city", "trip_city", "city"):
        city = normalize_city_name(location.get(key))
        if city:
            return city
    return ""


def current_city_from_constraints(constraints: dict | None) -> str:
    constraints = constraints or {}
    location = constraints.get("location")
    location = location if isinstance(location, dict) else {}
    return (
        normalize_city_name(constraints.get("current_city"))
        or normalize_city_name(location.get("current_city"))
    )


def is_cross_city_trip(constraints: dict | None) -> bool:
    current_city = current_city_from_constraints(constraints)
    destination_city = destination_city_from_constraints(constraints)
    return bool(current_city and destination_city and current_city != destination_city)


def _flatten_city_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        values: list[Any] = []
        for nested in value.values():
            values.extend(_flatten_city_values(nested))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for nested in value:
            values.extend(_flatten_city_values(nested))
        return values
    return [value]


def _parse_lng_lat(value: Any) -> tuple[float, float] | None:
    text = str(value or "").strip()
    if "," not in text:
        return None
    left, right = text.split(",", 1)
    try:
        return float(left), float(right)
    except ValueError:
        return None


def _looks_like_shanghai_coordinate(value: Any) -> bool:
    coord = _parse_lng_lat(value)
    if not coord:
        return False
    lng, lat = coord
    return 120.8 <= lng <= 122.2 and 30.6 <= lat <= 31.9


def item_matches_destination_city(item: dict | None, destination_city: Any) -> bool:
    """Return whether a POI can truthfully belong to the requested trip city."""

    city = normalize_city_name(destination_city)
    if not city or not isinstance(item, dict):
        return True

    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    nested_raw = raw.get("raw") if isinstance(raw.get("raw"), dict) else {}
    strict_city_values: list[Any] = []
    for source in (item, raw, nested_raw):
        for key in ("city", "cityname", "pname", "province"):
            strict_city_values.extend(_flatten_city_values(source.get(key)))

    explicit_text = " ".join(
        str(value) for value in strict_city_values if str(value or "").strip()
    )
    known_cities = {
        normalize_city_name(name)
        for name in CITY_ADCODE_BY_NAME
        if normalize_city_name(name)
    }
    mentioned_cities = {known for known in known_cities if known and known in explicit_text}
    if mentioned_cities:
        return city in mentioned_cities

    address_values: list[Any] = []
    for source in (item, raw, nested_raw):
        for key in ("address",):
            address_values.extend(_flatten_city_values(source.get(key)))
    address_text = " ".join(str(value) for value in address_values if str(value or "").strip())
    address_cities = {
        known
        for known in known_cities
        if known and (f"{known}市" in address_text or f"{known}省" in address_text)
    }
    if address_cities:
        return city in address_cities

    coordinate_values = [
        item.get("coordinates"),
        item.get("location"),
        item.get("longitude_latitude"),
        raw.get("location"),
        nested_raw.get("location"),
    ]
    if city != "上海" and any(_looks_like_shanghai_coordinate(value) for value in coordinate_values):
        return False
    return True


def weather_city_from_constraints(constraints: dict | None, *, default: str = "310000") -> str:
    constraints = constraints or {}
    location = constraints.get("location")
    location = location if isinstance(location, dict) else {}
    for key in ("weather_adcode", "city_adcode", "adcode"):
        value = constraints.get(key)
        if value not in (None, ""):
            return str(value).strip()
    for key in ("weather_adcode", "city_adcode", "adcode"):
        value = location.get(key)
        if value not in (None, ""):
            return str(value).strip()

    city = destination_city_from_constraints(constraints)
    return CITY_ADCODE_BY_NAME.get(city, city or default)


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

    explicit_people_count = constraints.get("people_count")
    if explicit_people_count not in (None, "") and to_float(explicit_people_count, 1) <= 1:
        return []

    if (
        explicit_people_count in (None, "")
        and constraints.get("child_age") in (None, "")
        and constraints.get("scene_type") not in {"family", "parent_child"}
        and constraints.get("scene") not in {"family", "parent_child"}
    ):
        return []

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
    将中文偏好、自然语言片段、英文 tag 展开为稳定匹配 tokens。
    B 侧新逻辑优先使用中文语义层；这里保留英文 token 兼容旧数据和 eval。
    支持 str / list / tuple / set / None。
    """
    expanded: list[str] = []

    for raw_value in _as_list(values):
        if raw_value is None:
            continue

        value = str(raw_value).strip()
        if not value:
            continue

        # 保留原始英文 tag / 原始词，便于精确匹配
        expanded.append(value)

        # 完全命中
        if value in CHINESE_TAG_MAPPING:
            expanded.extend(_flatten_tags(CHINESE_TAG_MAPPING[value]))
        if value in CANONICAL_BY_CHINESE:
            expanded.append(CANONICAL_BY_CHINESE[value])

        # 子串命中，例如 “想找亲子乐园” -> kid_friendly
        for keyword, mapped in CHINESE_TAG_MAPPING.items():
            if keyword in value:
                expanded.extend(_flatten_tags(mapped))

    # 去重，但保持稳定顺序
    seen = set()
    result = []
    for item in expanded:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def to_chinese_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return CHINESE_VALUE_LABELS.get(text, CHINESE_TAG_LABELS.get(text, text))


def to_chinese_tags(values: Any) -> list[str]:
    translated: list[str] = []
    for value in _as_list(values):
        if isinstance(value, dict):
            for nested_value in value.values():
                translated.extend(to_chinese_tags(nested_value))
            continue
        text = to_chinese_value(value)
        if text:
            translated.append(text)
    seen = set()
    result = []
    for item in translated:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def to_chinese_tag_groups(groups: Any) -> Any:
    if isinstance(groups, dict):
        return {
            str(key): to_chinese_tags(value)
            for key, value in groups.items()
        }
    return to_chinese_tags(groups)


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
    支持：
    - None -> [240, 360]
    - [4, 6] 小时 -> [240, 360]
    - [240, 360] 分钟 -> [240, 360]
    - "4-6" -> [240, 360]
    """
    default_range = [240, 360]

    if value is None or value == "":
        return default_range

    if isinstance(value, str):
        cleaned = value.replace("小时", "").replace("h", "").replace("H", "").strip()
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

    # 小于等于 24，视为小时
    if max(numbers) <= 24:
        return [int(numbers[0] * 60), int(numbers[1] * 60)]

    return [int(numbers[0]), int(numbers[1])]


def _contains_explicit_diet_text(value: Any) -> bool:
    terms = (
        "减肥",
        "减脂",
        "低卡",
        "低糖",
        "少油",
        "轻食",
        "清淡",
        "健康餐",
        "健身餐",
        "沙拉",
        "素食",
        "diet",
        "dieting",
        "low_calorie",
    )
    values = []
    if isinstance(value, dict):
        values.extend(value.values())
    elif isinstance(value, (list, tuple, set)):
        values.extend(value)
    else:
        values.append(value)
    text = " ".join(str(item or "").lower() for item in values)
    return any(term.lower() in text for term in terms)


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
    统计偏好与 tags 的匹配数量。
    支持中文偏好映射、英文 tag 精确匹配、子串匹配。
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
    preference_sources.extend(
        collect_tag_fields(constraints, "hard_tags", "soft_tags", "hard", "soft")
    )
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
    explicit_diet_context = any(
        _contains_explicit_diet_text(value)
        for value in (
            constraints.get("raw_text"),
            constraints.get("user_input"),
            constraints.get("query"),
            constraints.get("mom_diet"),
            planning_preferences.get("food_type"),
        )
    )

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
        or "low_calorie" in hard_tags
        or "low_calorie" in planning_food_tags
        or (
            explicit_diet_context
            and (
                "light_food" in soft_tags
                or "light_food" in hard_tags
                or "light_food" in planning_food_tags
            )
        )
    ):
        mom_diet = "low_calorie"

    if isinstance(mom_diet, str):
        lowered = mom_diet.strip().lower()
        if lowered in {"减肥", "低卡", "轻食", "low_cal", "low_calorie", "light_food", "dieting"}:
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
    根据过滤原因给出可解释的放宽建议。
    filter_reasons 中可能含有 _summary / _relaxation_suggestions 等特殊 key，因此会自动跳过。
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

    if reason_counts.get("距离超过用户可接受范围", 0) > 0:
        suggestions.append(f"可将最大距离从 {max_distance:g} 公里放宽到 {max_distance + 2:g} 公里")

    if reason_counts.get("排队时间过长", 0) > 0:
        suggestions.append(f"可将最大排队时间从 {max_queue:g} 分钟放宽到 {max_queue + 10:g} 分钟")

    if reason_counts.get("预算超出可接受上限", 0) > 0:
        suggestions.append(f"可将预算从 {budget:g} 元放宽到 {int(budget * 1.2)} 元左右")

    if reason_counts.get("时长不满足用户的时间范围", 0) > 0:
        suggestions.append("可将活动时长范围适当放宽，或允许更短的轻量行程")

    if reason_counts.get("不满足低龄儿童友好要求", 0) > 0:
        suggestions.append("可扩大活动类型，允许低强度室内活动或亲子友好度中等的活动")

    if reason_counts.get("不符合低卡或轻食需求", 0) > 0:
        suggestions.append("可允许家庭友好餐厅，并在订座备注中加入少油少盐需求")

    if reason_counts.get("活动或餐厅当前不可用", 0) > 0:
        suggestions.append("可更换相近时间段或选择同类型备选商家")

    if not suggestions:
        suggestions.append("可适当放宽距离、排队时间、预算或活动类型限制")

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
