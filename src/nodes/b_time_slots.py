"""Time slot selection for B activity/restaurant plans."""
from __future__ import annotations

from .b_candidate_policy import time_slot_bool, transition_buffer_min


def slot_to_minutes(slot: str) -> int:
    if not slot or ":" not in str(slot):
        return -1
    hour, minute = str(slot).split(":", 1)
    return int(hour) * 60 + int(minute)


def time_to_minutes(value: object, *, default: int, day: int = 1) -> int:
    text = str(value or "")
    token = ""
    for char in text:
        if char.isdigit() or char == ":":
            token += char
        elif token:
            break
    if ":" not in token:
        return default
    try:
        hour, minute = token.split(":", 1)
        return (max(1, day) - 1) * 1440 + int(hour) * 60 + int(minute)
    except (TypeError, ValueError):
        return default


def format_itinerary_time(total_minutes: int) -> str:
    minute_of_day = total_minutes % 1440
    hour, minute = divmod(minute_of_day, 60)
    return f"{hour:02d}:{minute:02d}"


def format_itinerary_time_range(start_minutes: int, end_minutes: int) -> str:
    start_text = format_itinerary_time(start_minutes)
    end_text = format_itinerary_time(end_minutes)
    if end_minutes // 1440 > start_minutes // 1440:
        return f"{start_text}-次日{end_text}"
    return f"{start_text}-{end_text}"


def available_slot_minutes(item: dict, day: int) -> list[int]:
    deal_slots: list[int] = []
    for deal in item.get("deals", []) or []:
        if not isinstance(deal, dict):
            continue
        for raw_time in deal.get("valid_time", []) or []:
            minutes = time_to_minutes(raw_time, default=-1, day=day)
            if minutes >= 0:
                deal_slots.append(minutes)

    operational_slots: list[int] = []
    for field_name in ("available_slots", "reservation_slots"):
        for slot in item.get(field_name, []) or []:
            if isinstance(slot, dict):
                raw_time = slot.get("time")
            else:
                raw_time = slot
            minutes = time_to_minutes(raw_time, default=-1, day=day)
            if minutes >= 0:
                operational_slots.append(minutes)

    deal_slot_set = set(deal_slots)
    operational_slot_set = set(operational_slots)
    if deal_slot_set and operational_slot_set:
        intersection = deal_slot_set.intersection(operational_slot_set)
        if intersection:
            return sorted(intersection)
        return sorted(deal_slot_set)
    if deal_slot_set:
        return sorted(deal_slot_set)
    return sorted(operational_slot_set)


def choose_node_start_time(
    item: dict,
    *,
    desired_start: int,
    earliest_start: int,
    day: int,
) -> int:
    target_start = max(desired_start, earliest_start)
    valid_slots = [slot for slot in available_slot_minutes(item, day) if slot >= target_start]
    if valid_slots:
        return valid_slots[0]
    return target_start


def _available_slot_times(item: dict) -> list[str]:
    return sorted(
        [slot.get("time") for slot in item.get("available_slots", []) if slot.get("time")],
        key=slot_to_minutes,
    )


def pick_time_slots(activity: dict, restaurant: dict, constraints: dict) -> tuple[str | None, str | None]:
    explicit_start = constraints.get("start_time") not in (None, "")
    explicit_end = constraints.get("end_time") not in (None, "")
    start_time = str(constraints.get("start_time") or "14:00")
    start_minutes = slot_to_minutes(start_time)
    end_minutes = slot_to_minutes(str(constraints.get("end_time"))) if explicit_end else -1
    buffer_min = transition_buffer_min()
    prefer_earliest_activity = time_slot_bool("prefer_earliest_valid_activity_slot", True)
    prefer_earliest_restaurant = time_slot_bool("prefer_earliest_valid_restaurant_slot", True)
    minimize_transition_gap = time_slot_bool("minimize_transition_gap", True)

    activity_slots = _available_slot_times(activity)
    restaurant_slots = _available_slot_times(restaurant)

    valid_activity_slots = [slot for slot in activity_slots if slot_to_minutes(slot) >= start_minutes]
    if minimize_transition_gap:
        valid_pairs: list[tuple[int, int, int, str, str]] = []
        activity_pool = valid_activity_slots if explicit_start else (valid_activity_slots or activity_slots)
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
                restaurant_end_minutes = restaurant_start_minutes + int(restaurant.get("duration_min", 0))
                if explicit_end and end_minutes >= start_minutes and restaurant_end_minutes > end_minutes:
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

    if explicit_start and not valid_activity_slots:
        return None, None

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
    if restaurant_start is None:
        return None, None
    if explicit_end and end_minutes >= start_minutes:
        restaurant_end_minutes = slot_to_minutes(restaurant_start) + int(restaurant.get("duration_min", 0))
        if restaurant_end_minutes > end_minutes:
            return None, None

    return activity_start, restaurant_start


def pick_time_slots_restaurant_first(
    activity: dict,
    restaurant: dict,
    constraints: dict,
) -> tuple[str | None, str | None]:
    explicit_start = constraints.get("start_time") not in (None, "")
    explicit_end = constraints.get("end_time") not in (None, "")
    start_time = str(constraints.get("start_time") or "14:00")
    start_minutes = slot_to_minutes(start_time)
    end_minutes = slot_to_minutes(str(constraints.get("end_time"))) if explicit_end else -1
    buffer_min = transition_buffer_min()
    prefer_earliest_activity = time_slot_bool("prefer_earliest_valid_activity_slot", True)
    prefer_earliest_restaurant = time_slot_bool("prefer_earliest_valid_restaurant_slot", True)

    activity_slots = _available_slot_times(activity)
    restaurant_slots = _available_slot_times(restaurant)

    valid_restaurant_slots = [
        slot for slot in restaurant_slots if slot_to_minutes(slot) >= start_minutes
    ]
    restaurant_pool = valid_restaurant_slots if explicit_start else (valid_restaurant_slots or restaurant_slots)
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
            activity_end_minutes = activity_start_minutes + int(activity.get("duration_min", 0))
            if explicit_end and end_minutes >= start_minutes and activity_end_minutes > end_minutes:
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
