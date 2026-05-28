from __future__ import annotations

from src.nodes.b_semantics import semantic_group_for_term, semantic_groups_in_values
from src.nodes.candidate_generator import _explicit_restaurant_requirements


def test_exact_chinese_food_intent_wins_before_substring_expansion():
    assert semantic_group_for_term("日料") == "日料"
    assert semantic_groups_in_values(["日料"]) == {"日料"}


def test_explicit_restaurant_requirements_keep_fine_grained_japanese_intent():
    requirements = _explicit_restaurant_requirements(
        {"planning_preferences": {"food_type": ["日料"]}}
    )

    assert "日料" in requirements
    assert "japanese" in requirements
    assert "火锅" not in requirements
    assert "烤肉" not in requirements
    assert "barbecue" not in requirements


def test_vegetarian_intent_maps_to_light_food_family():
    assert semantic_group_for_term("素食") == "轻食"
    assert semantic_groups_in_values(["素食"]) == {"轻食"}
