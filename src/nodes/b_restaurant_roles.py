"""Restaurant role classification for B candidate ranking."""
from __future__ import annotations

from .b_semantics import semantic_groups_for_item, semantic_groups_in_values


RESTAURANT_ROLE_FIELDS = (
    "name",
    "category",
    "restaurant_category",
    "primary_category",
    "primary_keyword",
    "gaode_keyword",
    "gaode_type",
)

CAFE_DESSERT_PREFERENCE_PHRASES = (
    "咖啡",
    "下午茶",
    "甜品",
    "小坐",
    "不想吃正餐",
    "不吃正餐",
    "不想正餐",
)


def restaurant_role(item: dict) -> str:
    category_groups = semantic_groups_for_item(item, fields=RESTAURANT_ROLE_FIELDS)
    if "咖啡甜品" in category_groups:
        return "cafe_dessert"
    if "轻食" in category_groups:
        return "light_meal"

    service_mode = str(item.get("service_mode") or "")
    category = str(item.get("restaurant_category") or item.get("category") or "")
    if "饮品" in category or "甜品" in category or "咖啡" in category or "下午茶" in category:
        return "cafe_dessert"
    if service_mode in {"咖啡小坐", "下午茶", "轻食简餐"}:
        return "cafe_dessert" if service_mode in {"咖啡小坐", "下午茶"} else "light_meal"
    return "full_meal"


def preferred_restaurant_role_from_values(
    raw_preference_values: list,
    *,
    mom_diet: str | None = None,
    raw_text: str | None = None,
) -> str | None:
    groups = semantic_groups_in_values(raw_preference_values)
    request_text = str(raw_text or "")
    if "咖啡甜品" in groups or any(phrase in request_text for phrase in CAFE_DESSERT_PREFERENCE_PHRASES):
        return "cafe_dessert"
    if "轻食" in groups or str(mom_diet or "").lower() == "low_calorie":
        return "light_meal"
    return None


def restaurant_role_score(item: dict, preferred_role: str | None) -> float:
    if not preferred_role:
        return 0.0
    actual_role = restaurant_role(item)
    if preferred_role == "cafe_dessert":
        if actual_role == "cafe_dessert":
            return 10.0
        if actual_role == "light_meal":
            return -2.0
        return -7.0
    if preferred_role == "light_meal":
        if actual_role == "light_meal":
            return 7.0
        if actual_role == "cafe_dessert":
            return 2.0
        return -5.0
    return 0.0
