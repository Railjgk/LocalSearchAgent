"""Sequence preference policy for activity/restaurant plans."""
from __future__ import annotations


SEQUENCE_ACTIVITY_THEN_RESTAURANT = "activity_then_restaurant"
SEQUENCE_RESTAURANT_THEN_ACTIVITY = "restaurant_then_activity"
RESTAURANT_THEN_ACTIVITY_PHRASES = (
    "吃完",
    "饭后",
    "餐后",
    "用餐后",
    "吃完饭",
    "吃完火锅",
)


def sequence_preference(constraints: dict | None) -> str:
    constraints = constraints or {}
    explicit_sequence = str(constraints.get("sequence_preference") or "").strip()
    if explicit_sequence in {
        SEQUENCE_ACTIVITY_THEN_RESTAURANT,
        SEQUENCE_RESTAURANT_THEN_ACTIVITY,
    }:
        return explicit_sequence

    raw_text = str(constraints.get("raw_text") or "")
    if raw_text and any(phrase in raw_text for phrase in RESTAURANT_THEN_ACTIVITY_PHRASES):
        return SEQUENCE_RESTAURANT_THEN_ACTIVITY
    return SEQUENCE_ACTIVITY_THEN_RESTAURANT
