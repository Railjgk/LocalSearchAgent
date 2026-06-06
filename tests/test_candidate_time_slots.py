from src.nodes.candidate_generator import (
    _pick_time_slots,
    _pick_time_slots_restaurant_first,
)


def test_activity_then_restaurant_does_not_fallback_before_explicit_start():
    activity = {
        "available_slots": [{"time": "14:00"}, {"time": "15:30"}],
        "duration_min": 90,
    }
    restaurant = {
        "available_slots": [{"time": "17:30"}],
        "duration_min": 90,
    }

    assert _pick_time_slots(activity, restaurant, {"start_time": "16:20"}) == (
        None,
        None,
    )


def test_activity_then_restaurant_respects_explicit_end_time():
    activity = {
        "available_slots": [{"time": "16:30"}],
        "duration_min": 60,
    }
    restaurant = {
        "available_slots": [{"time": "18:00"}],
        "duration_min": 90,
    }

    assert _pick_time_slots(
        activity,
        restaurant,
        {"start_time": "16:20", "end_time": "19:00"},
    ) == (None, None)


def test_restaurant_then_activity_does_not_fallback_before_explicit_start():
    activity = {
        "available_slots": [{"time": "16:30"}],
        "duration_min": 60,
    }
    restaurant = {
        "available_slots": [{"time": "14:00"}, {"time": "15:00"}],
        "duration_min": 60,
    }

    assert _pick_time_slots_restaurant_first(
        activity,
        restaurant,
        {"start_time": "15:30"},
    ) == (None, None)
