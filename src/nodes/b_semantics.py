# -*- coding: utf-8 -*-
"""Chinese-first semantic helpers for B-stage supply matching.

The product handoff and Gaode-backed mock supply are moving toward Chinese
labels.  This module keeps Chinese terms as the primary planning signal and
uses legacy English tags only as auxiliary indexes for older fixtures.
"""

from __future__ import annotations

from typing import Any


B_SEMANTIC_GROUPS: dict[str, dict[str, list[str]]] = {
    "烤肉": {
        "primary": [
            "烤肉",
            "烧烤",
            "烤串",
            "羊肉串",
            "肉串",
            "串烧",
            "炭火",
            "炭烤",
            "炭火烤肉",
            "日式烧肉",
            "日式烤肉",
            "韩式烤肉",
            "韩式烧肉",
            "自助烤肉",
            "巴西烤肉",
            "巴西牛排",
            "铁板烧",
        ],
        "auxiliary": [
            "bbq",
            "barbecue",
            "grill",
            "grilled_meat",
            "skewer",
            "skewers",
            "charcoal_grill",
            "yakiniku",
            "japanese_bbq",
            "korean_bbq",
            "meat",
        ],
    },
    "火锅": {
        "primary": ["火锅", "涮锅", "牛油锅", "铜锅", "潮汕牛肉火锅", "串串香", "冒菜"],
        "auxiliary": ["hotpot", "shabu", "social_hotpot"],
    },
    "轻食": {
        "primary": [
            "轻食",
            "低卡",
            "沙拉",
            "健康餐",
            "减脂餐",
            "健身餐",
            "能量碗",
            "日料轻食",
            "寿司",
            "刺身",
            "茶泡饭",
        ],
        "auxiliary": [
            "light_food",
            "low_calorie",
            "healthy",
            "salad_light_food",
            "japanese_light_food",
            "low_oil",
            "low_sugar",
            "high_protein",
            "vegetable_rich",
        ],
    },
    "日料": {
        "primary": ["日料", "日本料理", "寿司", "刺身", "茶泡饭", "烧鸟", "居酒屋", "日式烧肉", "日式烤肉"],
        "auxiliary": ["japanese", "japanese_light_food", "sushi", "izakaya", "yakiniku"],
    },
    "本帮菜": {
        "primary": ["本帮菜", "上海菜", "江浙菜", "家常菜", "地方菜", "本帮家常菜"],
        "auxiliary": ["regional_home_cuisine", "local_cuisine", "home_cuisine"],
    },
    "亲子餐厅": {
        "primary": ["亲子餐厅", "家庭餐厅", "儿童餐", "儿童椅", "宝宝椅", "带娃友好"],
        "auxiliary": ["family_bistro", "family_friendly", "kid_friendly", "child_seat"],
    },
    "炸鸡小吃": {
        "primary": ["炸鸡", "汉堡", "小吃", "炸串", "快餐", "外带"],
        "auxiliary": ["fried_chicken", "fast_food", "takeaway", "snack"],
    },
    "亲子活动": {
        "primary": ["亲子", "亲子活动", "儿童乐园", "室内乐园", "儿童友好", "适合孩子", "适龄"],
        "auxiliary": ["kid_friendly", "family_friendly", "parent_child", "indoor_playground"],
    },
    "手作体验": {
        "primary": ["手作", "手工", "陶艺", "diy", "DIY", "手工课", "画室", "编织"],
        "auxiliary": ["handcraft", "art_experience", "hands_on"],
    },
    "城市漫步": {
        "primary": ["citywalk", "Citywalk", "城市漫步", "散步", "街区", "小众路线", "逛街"],
        "auxiliary": ["citywalk", "local_culture", "local_experience"],
    },
    "近场放松": {
        "primary": ["轻松", "放松", "疗愈", "安静", "低强度", "温泉", "按摩", "spa", "SPA"],
        "auxiliary": ["relaxation", "healing", "low_intensity", "wellness", "spa"],
    },
}


B_RESTAURANT_INTENT_GROUPS = {
    "烤肉",
    "火锅",
    "轻食",
    "日料",
    "本帮菜",
    "亲子餐厅",
    "炸鸡小吃",
}

B_ACTIVITY_INTENT_GROUPS = {
    "亲子活动",
    "手作体验",
    "城市漫步",
    "近场放松",
}

B_SEMANTIC_TEXT_FIELDS = (
    "name",
    "category",
    "sub_category",
    "experience_type",
    "restaurant_category",
    "primary_category",
    "primary_keyword",
    "gaode_keyword",
    "gaode_type",
    "tags",
    "tag_groups",
    "health_tags",
    "menu_health_options",
    "emotion_tags",
    "atmosphere_tags",
    "local_character_tags",
    "local_flavor_tags",
    "risk_flags",
    "review_keywords",
    "signature_dishes",
    "recommended_dishes",
    "dish_tags",
    "package_options",
    "promotion_highlights",
    "decision_profile",
    "dietary_options",
    "business_hours",
)

B_SEMANTIC_FIELD_WEIGHTS = {
    "restaurant_category": 3.0,
    "category": 2.8,
    "primary_category": 2.8,
    "primary_keyword": 2.5,
    "gaode_keyword": 2.5,
    "name": 2.4,
    "tags": 2.0,
    "tag_groups": 2.0,
    "signature_dishes": 1.8,
    "recommended_dishes": 1.8,
    "dish_tags": 1.6,
    "review_keywords": 1.4,
    "package_options": 1.2,
    "promotion_highlights": 1.2,
    "decision_profile": 1.1,
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _dedupe_keep_order(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def normalize_semantic_text(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _has_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def flatten_semantic_values(value: Any) -> list[str]:
    """Flatten nested POI fields while preserving Chinese business terms."""

    values: list[str] = []
    if value is None:
        return values
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if isinstance(nested_value, (list, tuple, set, dict)):
                values.extend(flatten_semantic_values(nested_value))
            elif nested_value not in (None, ""):
                values.append(str(nested_value).strip())
            if key not in {"raw", "source_queries"} and not isinstance(nested_value, (dict, list, tuple, set)):
                values.append(str(key).strip())
        return values
    if isinstance(value, (list, tuple, set)):
        for item in value:
            values.extend(flatten_semantic_values(item))
        return values
    text = str(value).strip()
    return [text] if text else []


def semantic_group_for_term(value: Any) -> str | None:
    text = normalize_semantic_text(value)
    if not text:
        return None
    for group, payload in B_SEMANTIC_GROUPS.items():
        for term in payload.get("primary", []):
            normalized = normalize_semantic_text(term)
            if not normalized:
                continue
            if normalized == text:
                return group
            if (_has_cjk(normalized) or _has_cjk(text)) and (normalized in text or text in normalized):
                return group
        for term in payload.get("auxiliary", []):
            normalized = normalize_semantic_text(term)
            if normalized and normalized == text:
                return group
    return None


def b_semantic_terms(values: Any, *, include_auxiliary: bool = True) -> list[str]:
    """Expand user/supply text into Chinese-first B semantic terms."""

    expanded: list[str] = []
    for raw_value in _as_list(values):
        for value in flatten_semantic_values(raw_value):
            expanded.append(value)
            group = semantic_group_for_term(value)
            if not group:
                continue
            payload = B_SEMANTIC_GROUPS[group]
            expanded.extend(payload.get("primary", []))
            if include_auxiliary:
                expanded.extend(payload.get("auxiliary", []))
    return _dedupe_keep_order(expanded)


def semantic_terms_for_groups(groups: set[str], *, include_auxiliary: bool = True) -> list[str]:
    terms: list[str] = []
    for group in groups:
        payload = B_SEMANTIC_GROUPS.get(group)
        if not payload:
            continue
        terms.append(group)
        terms.extend(payload.get("primary", []))
        if include_auxiliary:
            terms.extend(payload.get("auxiliary", []))
    return _dedupe_keep_order(terms)


def semantic_groups_in_values(values: Any) -> set[str]:
    groups: set[str] = set()
    for term in b_semantic_terms(values, include_auxiliary=True):
        group = semantic_group_for_term(term)
        if group:
            groups.add(group)
    return groups


def semantic_groups_for_item(
    item: dict[str, Any],
    *,
    fields: tuple[str, ...] = B_SEMANTIC_TEXT_FIELDS,
) -> set[str]:
    groups: set[str] = set()
    for field_name in fields:
        for value in flatten_semantic_values(item.get(field_name)):
            group = semantic_group_for_term(value)
            if group:
                groups.add(group)
    return groups


def semantic_match_score(
    query_terms: Any,
    item: dict[str, Any],
    *,
    fields: tuple[str, ...] = B_SEMANTIC_TEXT_FIELDS,
) -> float:
    """Return a soft Chinese semantic match score for a query against a POI."""

    terms = b_semantic_terms(query_terms, include_auxiliary=True)
    if not terms or not item:
        return 0.0

    normalized_terms = [
        normalize_semantic_text(term)
        for term in terms
        if len(normalize_semantic_text(term)) >= 2
    ]
    if not normalized_terms:
        return 0.0

    best = 0.0
    matched_groups: set[str] = set()
    item_groups = semantic_groups_for_item(item, fields=fields)

    for field_name in fields:
        field_weight = B_SEMANTIC_FIELD_WEIGHTS.get(field_name, 1.0)
        for raw_value in flatten_semantic_values(item.get(field_name)):
            value = normalize_semantic_text(raw_value)
            if not value:
                continue
            for term in normalized_terms:
                if term == value:
                    best = max(best, field_weight + 1.0)
                elif term in value:
                    best = max(best, field_weight + 0.55)
                elif value in term and len(value) >= 2:
                    best = max(best, field_weight + 0.35)

    for group in semantic_groups_in_values(query_terms):
        if group in item_groups:
            matched_groups.add(group)

    if matched_groups:
        best = max(best, 2.2 + min(1.0, 0.25 * len(matched_groups)))

    return round(best, 3)


def direct_text_match_score(
    query_terms: Any,
    item: dict[str, Any],
    *,
    fields: tuple[str, ...] = B_SEMANTIC_TEXT_FIELDS,
) -> float:
    """Score direct lexical matches before expanding to broad semantic groups."""

    terms = [
        normalize_semantic_text(term)
        for term in flatten_semantic_values(query_terms)
        if len(normalize_semantic_text(term)) >= 2
    ]
    if not terms or not item:
        return 0.0

    best = 0.0
    for field_name in fields:
        field_weight = B_SEMANTIC_FIELD_WEIGHTS.get(field_name, 1.0)
        for raw_value in flatten_semantic_values(item.get(field_name)):
            value = normalize_semantic_text(raw_value)
            if not value:
                continue
            for term in terms:
                if term == value:
                    best = max(best, field_weight + 1.2)
                elif term in value:
                    best = max(best, field_weight + 0.85)
                elif value in term and len(value) >= 2:
                    best = max(best, field_weight + 0.45)
    return round(best, 3)


def has_semantic_group(values: Any, group: str) -> bool:
    return group in semantic_groups_in_values(values)


def has_item_semantic_group(item: dict[str, Any], group: str) -> bool:
    return group in semantic_groups_for_item(item)
