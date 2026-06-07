from __future__ import annotations

from src.nodes.b_weather_scoring import is_indoor_safe_item, weather_candidate_bonus


def _collect_tags(item: dict) -> list[str]:
    tags = item.get("tags") or []
    return list(tags) if isinstance(tags, list) else []


def test_weather_bonus_prefers_indoor_safe_activity_in_rain() -> None:
    weather_context = {
        "available": True,
        "prefer_indoor": True,
        "condition_tags": ["rainy"],
    }
    activity = {
        "type": "activity",
        "category": "handcraft",
        "tags": ["indoor"],
        "weather_sensitivity": "indoor_safe",
    }

    assert is_indoor_safe_item(activity, collect_tags=_collect_tags) is True
    assert weather_candidate_bonus(activity, weather_context, collect_tags=_collect_tags) == 5.0


def test_weather_bonus_penalizes_outdoor_high_sensitivity_when_indoor_preferred() -> None:
    weather_context = {
        "available": True,
        "prefer_indoor": True,
        "condition_tags": ["rainy"],
    }
    activity = {
        "type": "activity",
        "category": "citywalk",
        "tags": ["outdoor"],
        "weather_sensitivity": "high",
    }

    assert weather_candidate_bonus(activity, weather_context, collect_tags=_collect_tags) == -11.0


def test_weather_bonus_ignores_restaurants_and_unavailable_weather() -> None:
    restaurant = {"type": "restaurant", "tags": ["indoor"]}
    activity = {"type": "activity", "tags": ["indoor"]}

    assert weather_candidate_bonus(restaurant, {"available": True}, collect_tags=_collect_tags) == 0.0
    assert weather_candidate_bonus(activity, {"available": False}, collect_tags=_collect_tags) == 0.0
