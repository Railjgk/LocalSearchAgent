from __future__ import annotations

from typing import Any


try:
    from src.tools.mock_apis import search_activities, search_restaurants
except ImportError:  # pragma: no cover - standalone B smoke tests may not have C code
    search_activities = None
    search_restaurants = None


def _normalize_slots(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for item in items or []:
        copied = dict(item)
        copied["available_slots"] = copied.get("available_slots", []) or []
        normalized.append(copied)
    return normalized


def fetch_activity_candidates(
    *,
    constraints: dict | None = None,
    scene_type: str = "family",
    scenario_activities: list[str] | None = None,
) -> list[dict[str, Any]]:
    if search_activities is None:
        return []

    constraints = constraints or {}
    scenario_activities = scenario_activities or []
    raw_tags = " ".join(str(x) for x in scenario_activities)
    child_age = constraints.get("child_age")

    try:
        result = search_activities(
            radius=int(float(constraints.get("max_distance_km", 8)) * 1000),
            kid_friendly=scene_type == "family" or child_age not in (None, ""),
            low_intensity=(
                ("低强度" in raw_tags)
                or ("轻松" in raw_tags)
                or ("low_intensity" in raw_tags)
                or ("light_activity" in raw_tags)
            ),
            indoor=("室内" in raw_tags) or ("下雨" in raw_tags) or ("indoor" in raw_tags),
            limit=10,
        )
    except Exception:
        return []

    return _normalize_slots(result.get("items", []))


def fetch_restaurant_candidates(
    *,
    constraints: dict | None = None,
    scene_type: str = "family",
    scenario_activities: list[str] | None = None,
) -> list[dict[str, Any]]:
    if search_restaurants is None:
        return []

    constraints = constraints or {}
    scenario_activities = scenario_activities or []
    raw_tags = " ".join(str(x) for x in scenario_activities)
    mom_diet = str(constraints.get("mom_diet") or "").lower()

    try:
        result = search_restaurants(
            radius=int(float(constraints.get("max_distance_km", 8)) * 1000),
            low_calorie=(
                (mom_diet == "low_calorie")
                or ("轻食" in raw_tags)
                or ("低卡" in raw_tags)
                or ("low_calorie" in raw_tags)
                or ("light_food" in raw_tags)
            ),
            family_friendly=scene_type == "family",
            limit=10,
        )
    except Exception:
        return []

    return _normalize_slots(result.get("items", []))
