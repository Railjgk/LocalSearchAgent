from __future__ import annotations

from src.nodes.b_sequence_policy import (
    SEQUENCE_ACTIVITY_THEN_RESTAURANT,
    SEQUENCE_RESTAURANT_THEN_ACTIVITY,
    sequence_preference,
)


def test_sequence_preference_defaults_to_activity_then_restaurant() -> None:
    assert sequence_preference({}) == SEQUENCE_ACTIVITY_THEN_RESTAURANT


def test_sequence_preference_respects_explicit_value() -> None:
    assert sequence_preference(
        {"sequence_preference": SEQUENCE_RESTAURANT_THEN_ACTIVITY}
    ) == SEQUENCE_RESTAURANT_THEN_ACTIVITY


def test_sequence_preference_detects_meal_first_from_raw_text() -> None:
    assert sequence_preference({"raw_text": "我们吃完火锅再去唱歌"}) == SEQUENCE_RESTAURANT_THEN_ACTIVITY
    assert sequence_preference({"raw_text": "饭后找个地方散步"}) == SEQUENCE_RESTAURANT_THEN_ACTIVITY
