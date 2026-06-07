"""Weather-aware candidate scoring for B planning."""
from __future__ import annotations

from collections.abc import Callable

from .b_utils import expand_preference_tags


OUTDOOR_ACTIVITY_CATEGORIES = {"citywalk", "local_market", "sports"}
INDOOR_SAFE_TAGS = {"indoor", "museum", "handcraft", "indoor_playground", "escape_room"}


def weather_tags(weather_context: dict | None) -> set[str]:
    weather_context = weather_context or {}
    tags = set(str(tag) for tag in weather_context.get("condition_tags", []) or [])
    tags.update(str(tag) for tag in weather_context.get("risk_tags", []) or [])
    return tags


def item_weather_sensitivity(item: dict) -> str:
    return str(item.get("weather_sensitivity") or "").strip().lower()


def is_indoor_safe_item(
    item: dict,
    tags: set[str] | None = None,
    *,
    collect_tags: Callable[[dict], list[str]] | None = None,
) -> bool:
    if tags is None:
        tags = set(collect_tags(item) if collect_tags else [])
    category = str(item.get("category") or item.get("experience_type") or "").strip()
    sensitivity = item_weather_sensitivity(item)
    return (
        sensitivity == "indoor_safe"
        or bool(item.get("indoor_backup"))
        or bool(tags.intersection(INDOOR_SAFE_TAGS))
        or category in {"museum", "handcraft", "indoor_playground", "escape_room", "micro_vacation"}
    )


def weather_candidate_bonus(
    item: dict,
    weather_context: dict | None,
    *,
    collect_tags: Callable[[dict], list[str]] | None = None,
) -> float:
    if not weather_context or not weather_context.get("available"):
        return 0.0

    if item.get("type") != "activity":
        return 0.0

    tags = set(expand_preference_tags(collect_tags(item) if collect_tags else []))
    category = str(item.get("category") or item.get("experience_type") or "").strip()
    sensitivity = item_weather_sensitivity(item)
    tags_from_weather = weather_tags(weather_context)
    prefer_indoor = bool(weather_context.get("prefer_indoor"))
    indoor_safe = is_indoor_safe_item(item, tags, collect_tags=collect_tags)
    outdoor_like = category in OUTDOOR_ACTIVITY_CATEGORIES or "outdoor" in tags

    bonus = 0.0
    if prefer_indoor:
        if indoor_safe:
            bonus += 5.0
        if sensitivity == "medium":
            bonus -= 2.5
        elif sensitivity == "high":
            bonus -= 6.0
        if outdoor_like and not item.get("indoor_backup"):
            bonus -= 5.0

    if "hot" in tags_from_weather:
        if indoor_safe:
            bonus += 2.0
        if category in {"sports", "citywalk"} and not item.get("indoor_backup"):
            bonus -= 4.0

    if "comfortable" in tags_from_weather and category in OUTDOOR_ACTIVITY_CATEGORIES:
        bonus += 2.0

    return bonus
