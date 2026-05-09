# -*- coding: utf-8 -*-
import re
from collections import Counter
from typing import Any


CHINESE_TAG_MAPPING = {
    # 亲子/家庭
    "亲子": "kid_friendly",
    "亲子乐园": "kid_friendly",
    "儿童": "kid_friendly",
    "小孩": "kid_friendly",
    "孩子": "kid_friendly",
    "低龄儿童": "kid_friendly",
    "宝宝": "kid_friendly",
    "家庭": "family_friendly",
    "家庭友好": "family_friendly",
    "亲子友好": "family_friendly",

    # 强度/节奏
    "低强度": "low_intensity",
    "不累": "low_intensity",
    "别太累": "low_intensity",
    "轻松": "low_intensity",
    "休闲": "low_intensity",
    "慢节奏": "low_intensity",

    # 饮食
    "轻食": ["low_calorie", "light_food"],
    "轻食餐厅": ["low_calorie", "light_food"],
    "低卡": ["low_calorie", "light_food"],
    "减肥": ["low_calorie", "light_food"],
    "健康餐": ["low_calorie", "light_food"],
    "少油": ["low_calorie", "light_food"],
    "低油": ["low_calorie", "light_food"],
    "清淡": ["low_calorie", "light_food"],

    # 天气/空间
    "室内": "indoor",
    "下雨": "indoor",
    "雨天": "indoor",
    "不怕淋雨": "indoor",

    # 预算
    "预算低": "budget",
    "便宜": "budget",
    "省钱": "budget",
    "平价": "budget",
    "低预算": "budget",

    # 距离
    "附近": "nearby",
    "别太远": "nearby",
    "近": "nearby",
    "不远": "nearby",

    # 等待
    "不排队": "no_queue",
    "别排队": "no_queue",
    "少排队": "no_queue",

    # 朋友/情侣
    "朋友": "group_friendly",
    "聚会": "group_friendly",
    "社交": "social",
    "情侣": "romantic",
    "约会": "romantic",
    "氛围": "atmosphere",
}


SCENE_TEMPLATES = {
    "family": ["activity", "transition", "restaurant"],
    "friends": ["activity", "transition", "restaurant"],
    "couple": ["activity", "transition", "restaurant"],
    "low_budget": ["activity", "transition", "restaurant"],
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


def expand_preference_tags(values: Any) -> list[str]:
    """
    将中文偏好、自然语言片段、英文 tag 统一展开为英文 tags。
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
    return SCENE_TEMPLATES.get(scene_type, SCENE_TEMPLATES["family"])


def get_constraint_config(constraints: dict | None) -> dict[str, Any]:
    constraints = constraints or {}
    raw_duration = constraints.get("duration_range")
    if raw_duration in (None, ""):
        raw_duration = constraints.get("duration")

    mom_diet = constraints.get("mom_diet")
    if isinstance(mom_diet, str):
        lowered = mom_diet.strip().lower()
        if lowered in {"减肥", "低卡", "轻食", "low_cal", "low_calorie", "light_food"}:
            mom_diet = "low_calorie"

    return {
        "max_distance_km": to_float(constraints.get("max_distance_km"), 8.0),
        "max_queue_time": to_float(constraints.get("max_queue_time"), 30.0),
        "duration_range": parse_duration_range(raw_duration),
        "budget": to_float(constraints.get("budget"), 500.0),
        "child_age": parse_child_age(constraints.get("child_age")),
        "mom_diet": mom_diet,
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
