from __future__ import annotations

from src.nodes import b_candidate_policy
from src.nodes.b_time_slots import (
    pick_time_slots,
    pick_time_slots_restaurant_first,
    slot_to_minutes,
)


def _clear_policy(monkeypatch) -> None:
    monkeypatch.delenv("WF_PLANNER_POLICY_PATH", raising=False)
    b_candidate_policy.load_policy_config.cache_clear()


def test_slot_to_minutes_handles_basic_and_invalid_values() -> None:
    assert slot_to_minutes("14:30") == 14 * 60 + 30
    assert slot_to_minutes("") == -1
    assert slot_to_minutes("bad") == -1


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
