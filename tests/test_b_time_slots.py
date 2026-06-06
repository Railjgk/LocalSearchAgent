from __future__ import annotations

from src.nodes import b_candidate_policy
from src.nodes.b_time_slots import (
    available_slot_minutes,
    choose_node_start_time,
    format_itinerary_time_range,
    pick_time_slots,
    pick_time_slots_restaurant_first,
    slot_to_minutes,
    time_to_minutes,
)


def _clear_policy(monkeypatch) -> None:
    monkeypatch.delenv("WF_PLANNER_POLICY_PATH", raising=False)
    b_candidate_policy.load_policy_config.cache_clear()


def test_slot_to_minutes_handles_basic_and_invalid_values() -> None:
    assert slot_to_minutes("14:30") == 14 * 60 + 30
    assert slot_to_minutes("") == -1
    assert slot_to_minutes("bad") == -1


def test_multiday_time_helpers_parse_and_format_ranges() -> None:
    assert time_to_minutes("次日 09:30", default=0, day=2) == 24 * 60 + 9 * 60 + 30
    assert time_to_minutes("bad", default=123, day=2) == 123
    assert format_itinerary_time_range(23 * 60 + 30, 24 * 60 + 30) == "23:30-次日00:30"


def test_available_slot_minutes_and_node_start_choice() -> None:
    item = {
        "available_slots": [{"time": "10:00"}, {"time": "10:00"}],
        "reservation_slots": ["11:30"],
    }

    assert available_slot_minutes(item, day=2) == [2040, 2130]
    assert choose_node_start_time(item, desired_start=2020, earliest_start=2000, day=2) == 2040
    assert choose_node_start_time(item, desired_start=2200, earliest_start=2000, day=2) == 2200


def test_pick_time_slots_minimizes_gap_for_activity_then_restaurant(monkeypatch) -> None:
    _clear_policy(monkeypatch)
    activity = {
        "duration_min": 90,
        "available_slots": [{"time": "14:00"}, {"time": "15:00"}],
    }
    restaurant = {
        "duration_min": 60,
        "available_slots": [{"time": "16:30"}, {"time": "17:30"}],
    }

    assert pick_time_slots(activity, restaurant, {"start_time": "14:00"}) == ("14:00", "16:30")


def test_pick_time_slots_restaurant_first(monkeypatch) -> None:
    _clear_policy(monkeypatch)
    activity = {
        "duration_min": 90,
        "available_slots": [{"time": "14:30"}, {"time": "16:00"}],
    }
    restaurant = {
        "duration_min": 60,
        "available_slots": [{"time": "12:30"}, {"time": "13:00"}],
    }

    assert pick_time_slots_restaurant_first(activity, restaurant, {"start_time": "12:00"}) == ("14:30", "13:00")


def test_pick_time_slots_returns_none_when_restaurant_cannot_follow(monkeypatch) -> None:
    _clear_policy(monkeypatch)
    activity = {"duration_min": 120, "available_slots": [{"time": "16:00"}]}
    restaurant = {"duration_min": 60, "available_slots": [{"time": "17:00"}]}

    assert pick_time_slots(activity, restaurant, {"start_time": "14:00"}) == ("16:00", None)
