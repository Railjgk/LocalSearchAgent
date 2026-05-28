from __future__ import annotations

from src.nodes.b_semantics import semantic_group_for_term, semantic_groups_in_values
from src.nodes.candidate_generator import (
    _explicit_activity_requirements,
    _explicit_restaurant_requirements,
)
from src.nodes.plan_optimizer import _build_plan_title


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


def test_coffee_and_dessert_intents_are_restaurant_requirements():
    requirements = _explicit_restaurant_requirements(
        {"planning_preferences": {"food_type": ["咖啡", "下午茶"]}}
    )

    assert "咖啡甜品" in requirements
    assert "coffee" in requirements
    assert "afternoon_tea" in requirements
    assert "火锅" not in requirements
    assert "烤肉" not in requirements


def test_museum_and_exhibition_intents_are_activity_requirements():
    requirements = _explicit_activity_requirements(
        {"planning_preferences": {"activity_type": ["看展", "博物馆"]}}
    )

    assert "博物馆展览" in requirements
    assert "museum" in requirements
    assert "exhibition" in requirements
    assert "密室桌游" not in requirements


def test_board_game_and_escape_room_intents_are_activity_requirements():
    requirements = _explicit_activity_requirements(
        {"planning_preferences": {"activity_type": ["桌游", "密室逃脱"]}}
    )

    assert "密室桌游" in requirements
    assert "board_game" in requirements
    assert "escape_room" in requirements
    assert "博物馆展览" not in requirements


def test_plan_title_uses_selected_supply_semantics():
    assert _build_plan_title(
        "solo",
        {"category": "博物馆展览", "tags": ["室内"]},
        {"restaurant_category": "咖啡甜品", "tags": ["咖啡"]},
        None,
    ) == "看展咖啡放松计划"

    assert _build_plan_title(
        "friends",
        {"category": "密室桌游", "tags": ["社交"]},
        {"restaurant_category": "火锅", "tags": ["火锅"]},
        None,
    ) == "桌游火锅朋友聚会计划"

    assert _build_plan_title(
        "solo",
        {"category": "近场放松", "tags": ["低强度"]},
        {"restaurant_category": "轻食", "tags": ["轻食"]},
        None,
    ) == "一个人轻松探索计划"
