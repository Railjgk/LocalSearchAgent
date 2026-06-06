"""Time slot selection for B activity/restaurant plans."""
from __future__ import annotations

from .b_candidate_policy import time_slot_bool, transition_buffer_min


def slot_to_minutes(slot: str) -> int:
    if not slot or ":" not in str(slot):
        return -1
    hour, minute = str(slot).split(":", 1)
    return int(hour) * 60 + int(minute)


def _available_slot_times(item: dict) -> list[str]:
    return sorted(
        [slot.get("time") for slot in item.get("available_slots", []) if slot.get("time")],
        key=slot_to_minutes,
    )


def pick_time_slots(activity: dict, restaurant: dict, constraints: dict) -> tuple[str | None, str | None]:
    start_time = str(constraints.get("start_time") or "14:00")
    start_minutes = slot_to_minutes(start_time)
    buffer_min = transition_buffer_min()
    prefer_earliest_activity = time_slot_bool("prefer_earliest_valid_activity_slot", True)
    prefer_earliest_restaurant = time_slot_bool("prefer_earliest_valid_restaurant_slot", True)
    minimize_transition_gap = time_slot_bool("minimize_transition_gap", True)

    activity_slots = _available_slot_times(activity)
    restaurant_slots = _available_slot_times(restaurant)

    valid_activity_slots = [slot for slot in activity_slots if slot_to_minutes(slot) >= start_minutes]
    if minimize_transition_gap:
        valid_pairs: list[tuple[int, int, int, str, str]] = []
        activity_pool = valid_activity_slots or activity_slots
        for activity_slot in activity_pool:
            activity_start_minutes = slot_to_minutes(activity_slot)
            if activity_start_minutes < 0:
                continue
            activity_end_minutes = activity_start_minutes + int(activity.get("duration_min", 0))
            min_restaurant_minutes = activity_end_minutes + buffer_min
            for restaurant_slot in restaurant_slots:
                restaurant_start_minutes = slot_to_minutes(restaurant_slot)
                if restaurant_start_minutes < min_restaurant_minutes:
                    continue
                transition_gap = restaurant_start_minutes - activity_end_minutes
                valid_pairs.append(
                    (
                        transition_gap,
                        activity_start_minutes,
                        restaurant_start_minutes,
                        activity_slot,
                        restaurant_slot,
                    )
                )

        if valid_pairs:
            if prefer_earliest_activity and prefer_earliest_restaurant:
                valid_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
            elif prefer_earliest_activity:
                valid_pairs.sort(key=lambda item: (item[0], item[1], -item[2]))
            elif prefer_earliest_restaurant:
                valid_pairs.sort(key=lambda item: (item[0], -item[1], item[2]))
            else:
                valid_pairs.sort(key=lambda item: (item[0], -item[1], -item[2]))
            _, _, _, activity_start, restaurant_start = valid_pairs[0]
            return activity_start, restaurant_start

    if prefer_earliest_activity:
        activity_start = valid_activity_slots[0] if valid_activity_slots else (activity_slots[0] if activity_slots else None)
    else:
        activity_start = valid_activity_slots[-1] if valid_activity_slots else (activity_slots[-1] if activity_slots else None)

    if activity_start is None:
        return None, None

    min_restaurant_minutes = slot_to_minutes(activity_start) + int(activity.get("duration_min", 0)) + buffer_min
    valid_restaurant_slots = [slot for slot in restaurant_slots if slot_to_minutes(slot) >= min_restaurant_minutes]
    if prefer_earliest_restaurant:
        restaurant_start = valid_restaurant_slots[0] if valid_restaurant_slots else None
    else:
        restaurant_start = valid_restaurant_slots[-1] if valid_restaurant_slots else None

    return activity_start, restaurant_start


def pick_time_slots_restaurant_first(
    activity: dict,
    restaurant: dict,
    constraints: dict,
) -> tuple[str | None, str | None]:
    start_time = str(constraints.get("start_time") or "14:00")
    start_minutes = slot_to_minutes(start_time)
    buffer_min = transition_buffer_min()
    prefer_earliest_activity = time_slot_bool("prefer_earliest_valid_activity_slot", True)
    prefer_earliest_restaurant = time_slot_bool("prefer_earliest_valid_restaurant_slot", True)

    activity_slots = _available_slot_times(activity)
    restaurant_slots = _available_slot_times(restaurant)

    restaurant_pool = [
        slot for slot in restaurant_slots if slot_to_minutes(slot) >= start_minutes
    ] or restaurant_slots
    valid_pairs: list[tuple[int, int, int, str, str]] = []
    for restaurant_slot in restaurant_pool:
        restaurant_start_minutes = slot_to_minutes(restaurant_slot)
        if restaurant_start_minutes < 0:
            continue
        restaurant_end_minutes = restaurant_start_minutes + int(restaurant.get("duration_min", 0))
        min_activity_minutes = restaurant_end_minutes + buffer_min
        for activity_slot in activity_slots:
            activity_start_minutes = slot_to_minutes(activity_slot)
            if activity_start_minutes < min_activity_minutes:
                continue
            transition_gap = activity_start_minutes - restaurant_end_minutes
            valid_pairs.append(
                (
                    transition_gap,
                    restaurant_start_minutes,
                    activity_start_minutes,
                    activity_slot,
                    restaurant_slot,
                )
            )

    if not valid_pairs:
        return None, None

    if prefer_earliest_restaurant and prefer_earliest_activity:
        valid_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    elif prefer_earliest_restaurant:
        valid_pairs.sort(key=lambda item: (item[0], item[1], -item[2]))
    elif prefer_earliest_activity:
        valid_pairs.sort(key=lambda item: (item[0], -item[1], item[2]))
    else:
        valid_pairs.sort(key=lambda item: (item[0], -item[1], -item[2]))
    _, _, _, activity_start, restaurant_start = valid_pairs[0]
    return activity_start, restaurant_start
