from __future__ import annotations

from src.nodes.b_local_food_guardrails import (
    intent_requires_local_shanghai_food,
    item_conflicts_with_local_shanghai_food,
    item_has_local_shanghai_food_identity,
)


def test_intent_requires_local_shanghai_food_from_search_terms() -> None:
    intent = {"search_terms": ["晚上想吃本帮菜", "小笼包"]}

    assert intent_requires_local_shanghai_food(intent) is True


def test_item_has_local_shanghai_food_identity_from_cuisine_fields() -> None:
    item = {"name": "上海小笼馆", "category": "本帮菜", "tags": ["晚餐"]}

    assert item_has_local_shanghai_food_identity(item) is True


def test_item_conflicts_with_local_shanghai_food_for_foreign_cuisine() -> None:
    item = {"name": "日式寿司小馆", "category": "日本料理"}

    assert item_conflicts_with_local_shanghai_food(item) is True


def test_local_shanghai_food_identity_wins_over_conflict_terms() -> None:
    item = {"name": "本帮小笼日料融合", "category": "本帮菜"}

    assert item_has_local_shanghai_food_identity(item) is True
    assert item_conflicts_with_local_shanghai_food(item) is False
