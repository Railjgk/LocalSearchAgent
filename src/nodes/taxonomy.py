"""Chinese-first scenario taxonomy with legacy English tag compatibility.

产品侧数据计划迁移为中文格式。该模块暂时保留英文 canonical tag 字典，
主要用于兼容当前仍为英文形式的 mock POI、历史 memory、eval case 和 B 侧
打分逻辑，并提供中英文标签互转能力。
"""

from __future__ import annotations

from typing import Any


SCENE_TYPES = {"family", "friends", "couple", "low_budget", "solo"}


TAG_CATEGORIES: dict[str, str] = {
    # people and relationship
    "parent_child": "people",
    "kid_friendly": "people",
    "low_age_child": "people",
    "family_friendly": "people",
    "couple": "people",
    "date_activity": "people",
    "romantic": "people",
    "group_activity": "people",
    "group_friendly": "people",
    "social": "people",
    # activity
    "light_activity": "activity",
    "low_intensity": "activity",
    "indoor": "activity",
    "outdoor": "activity",
    "handcraft": "activity",
    "hands_on_parent_child": "activity",
    "art_experience": "activity",
    "museum": "activity",
    "amusement": "activity",
    "citywalk": "activity",
    "local_market": "activity",
    "local_culture": "activity",
    "wellness": "activity",
    "micro_vacation": "activity",
    # food
    "low_calorie": "food",
    "light_food": "food",
    "low_oil": "food",
    "low_sugar": "food",
    "healthy": "food",
    "fresh_ingredients": "food",
    "vegetable_rich": "food",
    "japanese": "food",
    "hotpot": "food",
    "regional_home_cuisine": "food",
    "local_flavor": "food",
    "dine_in": "food",
    "takeaway_only": "food",
    # emotion and experience
    "relaxation": "emotion",
    "healing": "emotion",
    "ritual": "emotion",
    "quiet": "emotion",
    "atmosphere": "emotion",
    "comfortable": "emotion",
    "novelty": "emotion",
    "local_discovery": "emotion",
    # route and space
    "nearby": "route",
    "short_distance": "route",
    "same_area": "route",
    "cross_area_ok": "route",
    "driving": "route",
    "walking": "route",
    "transit": "route",
    "bicycling": "route",
    "easy_parking": "route",
    # budget
    "budget": "budget",
    "low_budget": "budget",
    "value_for_money": "budget",
    "premium_ok": "budget",
    "per_person_budget": "budget",
    "total_budget": "budget",
    # risk
    "long_queue": "risk",
    "crowded_mall": "risk",
    "crowded": "risk",
    "high_calorie": "risk",
    "few_reviews": "risk",
    "new_merchant": "risk",
    "high_intensity": "risk",
    "too_far": "risk",
    # execution
    "bookable": "execution",
    "ticket_required": "execution",
    "reservation_needed": "execution",
    "walk_in_ok": "execution",
    "has_inventory": "execution",
    "has_time_slot": "execution",
    "child_seat_available": "execution",
}

CANONICAL_TAGS = set(TAG_CATEGORIES)


CHINESE_TAG_LABELS: dict[str, str] = {
    # people and relationship
    "parent_child": "亲子",
    "kid_friendly": "儿童友好",
    "low_age_child": "低龄儿童",
    "family_friendly": "家庭友好",
    "couple": "情侣",
    "date_activity": "约会活动",
    "romantic": "浪漫",
    "group_activity": "多人活动",
    "group_friendly": "多人友好",
    "social": "社交",
    # activity
    "light_activity": "轻量活动",
    "low_intensity": "低强度",
    "indoor": "室内",
    "outdoor": "户外",
    "handcraft": "手作",
    "hands_on_parent_child": "亲子动手体验",
    "art_experience": "艺术体验",
    "museum": "博物馆展览",
    "amusement": "游乐",
    "citywalk": "Citywalk",
    "local_market": "本地市集",
    "local_culture": "本地文化",
    "wellness": "康养",
    "micro_vacation": "微度假",
    # food
    "low_calorie": "低卡",
    "light_food": "轻食",
    "low_oil": "少油",
    "low_sugar": "少糖",
    "healthy": "健康",
    "fresh_ingredients": "食材新鲜",
    "vegetable_rich": "蔬菜丰富",
    "japanese": "日料",
    "hotpot": "火锅",
    "regional_home_cuisine": "本帮家常菜",
    "local_flavor": "本地口味",
    "dine_in": "堂食",
    "takeaway_only": "仅外带",
    # emotion and experience
    "relaxation": "放松",
    "healing": "疗愈",
    "ritual": "仪式感",
    "quiet": "安静",
    "atmosphere": "氛围感",
    "comfortable": "舒适",
    "novelty": "新鲜感",
    "local_discovery": "本地探索",
    # route and space
    "nearby": "附近",
    "short_distance": "短距离",
    "same_area": "同区",
    "cross_area_ok": "可跨区",
    "driving": "开车",
    "walking": "步行",
    "transit": "公共交通",
    "bicycling": "骑行",
    "easy_parking": "停车方便",
    # budget
    "budget": "预算敏感",
    "low_budget": "低预算",
    "value_for_money": "性价比",
    "premium_ok": "可接受高价",
    "per_person_budget": "人均预算",
    "total_budget": "总预算",
    # risk
    "long_queue": "排队久",
    "crowded_mall": "商场拥挤",
    "crowded": "人多拥挤",
    "high_calorie": "高热量",
    "few_reviews": "评价少",
    "new_merchant": "新商户",
    "high_intensity": "高强度",
    "too_far": "太远",
    # execution
    "bookable": "可预约",
    "ticket_required": "需买票",
    "reservation_needed": "需要订座",
    "walk_in_ok": "可直接去",
    "has_inventory": "有库存",
    "has_time_slot": "有可用时段",
    "child_seat_available": "有儿童座椅",
}

CANONICAL_BY_CHINESE = {value: key for key, value in CHINESE_TAG_LABELS.items()}

CHINESE_SCENARIO_SUBTYPE_LABELS = {
    "family_parent_child_light": "亲子轻松活动",
    "family_rainy_indoor": "雨天室内亲子",
    "family_health_food": "亲子健康餐饮",
    "family_hands_on": "亲子动手体验",
    "friends_social_activity": "朋友社交活动",
    "friends_group_meal": "朋友聚餐",
    "friends_local_explore": "朋友本地探索",
    "couple_date_atmosphere": "情侣氛围约会",
    "couple_micro_vacation": "情侣微度假",
    "couple_ritual": "情侣仪式感",
    "budget_nearby_light": "附近低预算轻量方案",
    "budget_food_first": "低预算餐饮优先",
    "solo_relaxation": "单人放松",
    "solo_local_explore": "单人本地探索",
    "solo_light_meal": "单人轻食",
    "unknown": "未知",
}


TRIGGER_TAGS: dict[str, list[str]] = {
    # people and relationship
    "亲子": ["parent_child", "kid_friendly"],
    "亲子乐园": ["parent_child", "kid_friendly"],
    "儿童": ["kid_friendly"],
    "小孩": ["kid_friendly"],
    "孩子": ["parent_child", "kid_friendly"],
    "低龄儿童": ["kid_friendly", "low_age_child"],
    "宝宝": ["kid_friendly", "low_age_child"],
    "家庭": ["family_friendly"],
    "家庭友好": ["family_friendly"],
    "亲子友好": ["family_friendly"],
    "朋友": ["group_activity", "group_friendly", "social"],
    "聚会": ["group_activity", "group_friendly", "social"],
    "社交": ["social"],
    "同事": ["group_activity", "group_friendly", "social"],
    "同学": ["group_activity", "group_friendly", "social"],
    "闺蜜": ["group_activity", "group_friendly", "social"],
    "情侣": ["date_activity", "romantic", "atmosphere"],
    "约会": ["date_activity", "romantic", "atmosphere"],
    "对象": ["date_activity", "romantic", "atmosphere"],
    "女朋友": ["date_activity", "romantic", "atmosphere"],
    "男朋友": ["date_activity", "romantic", "atmosphere"],
    "氛围": ["atmosphere"],
    # activity
    "低强度": ["light_activity", "low_intensity"],
    "不累": ["light_activity", "low_intensity"],
    "别太累": ["light_activity", "low_intensity"],
    "轻松": ["light_activity", "low_intensity", "relaxation"],
    "休闲": ["light_activity", "relaxation"],
    "慢节奏": ["low_intensity", "relaxation"],
    "室内": ["indoor"],
    "下雨": ["indoor"],
    "雨天": ["indoor"],
    "手作": ["handcraft", "hands_on_parent_child"],
    "陶艺": ["handcraft", "hands_on_parent_child", "art_experience"],
    "画画": ["handcraft", "hands_on_parent_child", "art_experience"],
    "绘画": ["handcraft", "hands_on_parent_child", "art_experience"],
    "展览": ["museum", "art_experience"],
    "博物馆": ["museum"],
    "游乐": ["amusement"],
    "citywalk": ["citywalk", "local_discovery"],
    "Citywalk": ["citywalk", "local_discovery"],
    "逛街": ["citywalk", "local_discovery"],
    "本地生活": ["local_culture", "local_discovery"],
    "市集": ["local_market", "local_culture"],
    "本地文化": ["local_culture", "local_discovery"],
    "微度假": ["micro_vacation", "wellness", "relaxation"],
    "温泉": ["micro_vacation", "wellness", "healing"],
    "康养": ["wellness", "healing"],
    # food
    "轻食": ["low_calorie", "light_food"],
    "轻食餐厅": ["low_calorie", "light_food"],
    "低卡": ["low_calorie", "light_food"],
    "减肥": ["low_calorie", "light_food"],
    "减脂": ["low_calorie", "light_food"],
    "控卡": ["low_calorie", "light_food"],
    "低脂": ["low_calorie", "low_oil"],
    "健康餐": ["healthy", "low_calorie", "light_food"],
    "健康": ["healthy"],
    "少油": ["low_oil"],
    "低油": ["low_oil"],
    "清淡": ["light_food", "low_oil"],
    "清爽": ["light_food", "low_oil"],
    "少盐": ["light_food"],
    "少糖": ["low_sugar"],
    "低糖": ["low_sugar"],
    "食材好": ["fresh_ingredients"],
    "有机": ["fresh_ingredients", "healthy"],
    "蔬菜": ["vegetable_rich"],
    "日料": ["japanese", "light_food"],
    "日本菜": ["japanese", "light_food"],
    "火锅": ["hotpot", "social"],
    "本帮菜": ["regional_home_cuisine", "local_flavor"],
    "本地口味": ["regional_home_cuisine", "local_flavor"],
    "家常菜": ["regional_home_cuisine"],
    "堂食": ["dine_in"],
    "订座": ["dine_in", "reservation_needed"],
    "预约": ["bookable", "reservation_needed"],
    "外带": ["takeaway_only"],
    "打包": ["takeaway_only"],
    # emotion
    "放松": ["relaxation"],
    "疗愈": ["healing"],
    "治愈": ["healing"],
    "仪式感": ["ritual"],
    "安静": ["quiet"],
    "浪漫": ["romantic"],
    "舒服": ["comfortable"],
    "舒适": ["comfortable"],
    "新鲜感": ["novelty"],
    # route and budget
    "附近": ["nearby", "short_distance"],
    "别太远": ["nearby", "short_distance"],
    "不太远": ["nearby", "short_distance"],
    "近一点": ["nearby", "short_distance"],
    "离家近": ["nearby", "short_distance"],
    "同区": ["same_area"],
    "跨区": ["cross_area_ok"],
    "开车": ["driving"],
    "自驾": ["driving"],
    "打车": ["driving"],
    "步行": ["walking"],
    "走路": ["walking"],
    "公交": ["transit"],
    "地铁": ["transit"],
    "骑车": ["bicycling"],
    "停车": ["easy_parking"],
    "预算低": ["budget", "low_budget"],
    "便宜": ["budget", "low_budget", "value_for_money"],
    "省钱": ["budget", "low_budget", "value_for_money"],
    "平价": ["budget", "low_budget", "value_for_money"],
    "低预算": ["budget", "low_budget"],
    "人均": ["per_person_budget"],
    "总预算": ["total_budget"],
    # risk
    "不排队": ["long_queue"],
    "别排队": ["long_queue"],
    "少排队": ["long_queue"],
    "排队": ["long_queue"],
    "等位": ["long_queue"],
    "人多": ["crowded"],
    "人少点": ["crowded"],
    "人挤": ["crowded_mall"],
    "商场": ["crowded_mall"],
    "高热量": ["high_calorie"],
    "大油": ["high_calorie", "low_oil"],
    "踩雷": ["few_reviews", "new_merchant"],
    "靠谱": ["few_reviews", "new_merchant"],
}


SCENE_DEFAULT_TAGS: dict[str, list[str]] = {
    "family": ["parent_child", "kid_friendly", "indoor", "light_activity", "low_intensity"],
    "friends": ["group_activity", "group_friendly", "social", "indoor"],
    "couple": ["date_activity", "romantic", "atmosphere", "relaxation"],
    "low_budget": ["budget", "low_budget", "value_for_money", "nearby"],
    "solo": ["light_activity", "relaxation", "nearby"],
}


def dedupe(values: list[Any]) -> list[str]:
    seen = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def as_list(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    if isinstance(values, set):
        return list(values)
    return [values]


def tags_from_text(text: str) -> list[str]:
    tags: list[str] = []
    for keyword, mapped_tags in TRIGGER_TAGS.items():
        if keyword in text:
            tags.extend(mapped_tags)
    return dedupe(tags)


def canonicalize_tags(values: Any, *, preserve_unknown: bool = False) -> list[str]:
    tags: list[str] = []
    for raw_value in as_list(values):
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if not value:
            continue
        if value in CANONICAL_TAGS:
            tags.append(value)
        elif value in CANONICAL_BY_CHINESE:
            tags.append(CANONICAL_BY_CHINESE[value])
        elif preserve_unknown:
            tags.append(value)
        for keyword, mapped_tags in TRIGGER_TAGS.items():
            if keyword in value:
                tags.extend(mapped_tags)
    return dedupe(tags)


def to_chinese_tag(value: Any, *, preserve_unknown: bool = True) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text in CHINESE_TAG_LABELS:
        return CHINESE_TAG_LABELS[text]
    if text in CANONICAL_BY_CHINESE:
        return text

    tags = canonicalize_tags(text)
    if tags:
        return CHINESE_TAG_LABELS.get(tags[0], tags[0])
    return text if preserve_unknown else ""


def to_chinese_tags(values: Any, *, preserve_unknown: bool = False) -> list[str]:
    tags = canonicalize_tags(values, preserve_unknown=preserve_unknown)
    localized = [CHINESE_TAG_LABELS.get(tag, str(tag)) for tag in tags]
    if preserve_unknown:
        for raw_value in as_list(values):
            text = str(raw_value or "").strip()
            if text and text not in CANONICAL_TAGS and text not in CANONICAL_BY_CHINESE:
                localized.append(text)
    return dedupe(localized)


def localize_facets(facets: dict[str, list[str]]) -> dict[str, list[str]]:
    localized: dict[str, list[str]] = {}
    for key, values in facets.items():
        localized[key] = to_chinese_tags(values, preserve_unknown=True)
    return localized


def tags_by_category(tags: Any) -> dict[str, list[str]]:
    grouped = {
        "people": [],
        "activity": [],
        "food": [],
        "emotion": [],
        "route": [],
        "budget": [],
        "risk": [],
        "execution": [],
    }
    for tag in canonicalize_tags(tags):
        category = TAG_CATEGORIES.get(tag)
        if category:
            grouped[category].append(tag)
    return {key: dedupe(value) for key, value in grouped.items()}


def infer_scenario_subtype(scene_type: str, tags: Any, constraints: dict | None = None) -> str:
    constraints = constraints or {}
    tag_set = set(canonicalize_tags(tags))
    text = str(constraints.get("raw_text") or "")

    if scene_type == "family":
        if {"handcraft", "hands_on_parent_child", "art_experience"} & tag_set:
            return "family_hands_on"
        if "indoor" in tag_set and ("下雨" in text or "雨天" in text):
            return "family_rainy_indoor"
        if {"low_calorie", "light_food", "healthy"} & tag_set:
            return "family_health_food"
        return "family_parent_child_light"

    if scene_type == "friends":
        if {"citywalk", "local_market", "local_culture"} & tag_set:
            return "friends_local_explore"
        if {"hotpot", "regional_home_cuisine", "local_flavor"} & tag_set:
            return "friends_group_meal"
        return "friends_social_activity"

    if scene_type == "couple":
        if {"micro_vacation", "wellness", "healing"} & tag_set:
            return "couple_micro_vacation"
        if "ritual" in tag_set or constraints.get("ritual_need"):
            return "couple_ritual"
        return "couple_date_atmosphere"

    if scene_type == "low_budget":
        return "budget_food_first" if "dine_in" in tag_set else "budget_nearby_light"

    if scene_type == "solo":
        if {"citywalk", "local_market", "local_culture"} & tag_set:
            return "solo_local_explore"
        if {"low_calorie", "light_food", "dine_in"} & tag_set:
            return "solo_light_meal"
        return "solo_relaxation"

    return "unknown"


def build_scenario_facets(scene_type: str, tags: Any, constraints: dict | None = None) -> dict[str, list[str]]:
    constraints = constraints or {}
    grouped = tags_by_category(tags)
    relationship = [scene_type] if scene_type in SCENE_TYPES else []
    companions = constraints.get("companions") or []
    for item in companions:
        if isinstance(item, dict) and item.get("role"):
            relationship.append(str(item["role"]))

    return {
        "relationship": dedupe(relationship + grouped["people"]),
        "motive": dedupe(grouped["emotion"] + grouped["budget"]),
        "activity_mode": grouped["activity"],
        "food_mode": grouped["food"],
        "route_mode": grouped["route"],
        "risk_posture": grouped["risk"],
        "execution_mode": grouped["execution"],
    }
