from __future__ import annotations

from src.nodes.b_restaurant_roles import (
    preferred_restaurant_role_from_values,
    restaurant_role,
    restaurant_role_score,
)


def test_restaurant_role_detects_cafe_light_meal_and_full_meal() -> None:
    assert restaurant_role({"restaurant_category": "咖啡甜品"}) == "cafe_dessert"
    assert restaurant_role({"service_mode": "轻食简餐"}) == "light_meal"
    assert restaurant_role({"restaurant_category": "本帮菜"}) == "full_meal"


def test_preferred_restaurant_role_uses_request_text_and_diet() -> None:
    assert preferred_restaurant_role_from_values([], raw_text="下午想找个地方喝咖啡小坐") == "cafe_dessert"
    assert preferred_restaurant_role_from_values([], mom_diet="low_calorie") == "light_meal"
    assert preferred_restaurant_role_from_values(["轻食"], raw_text="随便吃点") == "light_meal"
    assert preferred_restaurant_role_from_values(["火锅"], raw_text="朋友聚餐") is None


def test_restaurant_role_score_matches_existing_weighting() -> None:
    cafe = {"restaurant_category": "咖啡"}
    light = {"service_mode": "轻食简餐"}
    meal = {"restaurant_category": "本帮菜"}

    assert restaurant_role_score(cafe, "cafe_dessert") == 10.0
    assert restaurant_role_score(light, "cafe_dessert") == -2.0
    assert restaurant_role_score(meal, "cafe_dessert") == -7.0
    assert restaurant_role_score(light, "light_meal") == 7.0
    assert restaurant_role_score(cafe, "light_meal") == 2.0
    assert restaurant_role_score(meal, "light_meal") == -5.0
