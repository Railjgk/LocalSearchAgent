"""Local cuisine guardrails for B restaurant intent matching."""
from __future__ import annotations

from .b_semantics import flatten_semantic_values, normalize_semantic_text
from .b_text_match import item_matches_terms


RESTAURANT_CUISINE_IDENTITY_FIELDS = (
    "name",
    "category",
    "sub_category",
    "gaode_type",
)

LOCAL_SHANGHAI_FOOD_INTENT_TERMS = ("本帮", "本帮菜", "上海菜", "江浙", "江浙菜", "沪菜", "小笼", "生煎", "汤包")
LOCAL_SHANGHAI_FOOD_CONFLICT_TERMS = (
    "日本料理",
    "日料",
    "日式",
    "寿司",
    "刺身",
    "鮨",
    "和食",
    "居酒屋",
    "韩国料理",
    "韩餐",
    "西餐",
    "意大利",
    "泰国菜",
    "越南菜",
)


def intent_requires_local_shanghai_food(intent: dict) -> bool:
    text = normalize_semantic_text(
        " ".join(str(value) for value in flatten_semantic_values(intent.get("search_terms")))
    )
    return any(term in text for term in LOCAL_SHANGHAI_FOOD_INTENT_TERMS)


def item_has_local_shanghai_food_identity(item: dict) -> bool:
    return item_matches_terms(
        item,
        LOCAL_SHANGHAI_FOOD_INTENT_TERMS,
        fields=RESTAURANT_CUISINE_IDENTITY_FIELDS,
    )


def item_conflicts_with_local_shanghai_food(item: dict) -> bool:
    return item_matches_terms(
        item,
        LOCAL_SHANGHAI_FOOD_CONFLICT_TERMS,
        fields=RESTAURANT_CUISINE_IDENTITY_FIELDS,
    ) and not item_has_local_shanghai_food_identity(item)
