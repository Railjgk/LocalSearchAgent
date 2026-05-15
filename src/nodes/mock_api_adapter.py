from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - policy file is optional at runtime
    yaml = None

try:
    from src.tools.mock_apis import search_activities, search_restaurants
except ImportError:  # pragma: no cover - standalone B smoke tests may not have C code
    search_activities = None
    search_restaurants = None

from .b_utils import expand_preference_tags, to_float


DEFAULT_SOURCE_ORDER = ("local_supply_mock", "c_mock_api")
DEFAULT_MOCK_DATA_DIR = Path(__file__).resolve().parents[2] / "experiments" / "mock_data"
DEFAULT_ACTIVITY_KEYWORDS = "亲子 活动 手作 博物馆"
DEFAULT_RESTAURANT_KEYWORDS = "轻食 餐厅 简餐"


def _policy_path() -> Path:
    override = os.environ.get("WF_PLANNER_POLICY_PATH", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "experiments" / "planner_policy.yaml"


@lru_cache(maxsize=8)
def _load_policy_config(policy_path_key: str) -> dict:
    if yaml is None:
        return {}

    policy_path = Path(policy_path_key)
    if not policy_path.exists():
        return {}

    try:
        with policy_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _candidate_generation_config() -> dict:
    policy = _load_policy_config(str(_policy_path().resolve()))
    candidate_generation = policy.get("candidate_generation")
    return candidate_generation if isinstance(candidate_generation, dict) else {}


def _source_order() -> list[str]:
    raw_order = _candidate_generation_config().get("data_source_order", DEFAULT_SOURCE_ORDER)
    if not isinstance(raw_order, list):
        return list(DEFAULT_SOURCE_ORDER)
    return [str(item).strip() for item in raw_order if str(item).strip()]


def _mock_data_dir() -> Path:
    raw_dir = _candidate_generation_config().get("local_mock_dir")
    if not raw_dir:
        return DEFAULT_MOCK_DATA_DIR

    path = Path(str(raw_dir))
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


@lru_cache(maxsize=16)
def _load_json_file(path_key: str) -> Any:
    path = Path(path_key)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _as_list(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    if isinstance(values, set):
        return list(values)
    return [values]


def _flatten_tags(raw_tags: Any) -> list[str]:
    values: list[Any] = []
    if isinstance(raw_tags, dict):
        for item in raw_tags.values():
            values.extend(_as_list(item))
    else:
        values.extend(_as_list(raw_tags))

    result = []
    seen = set()
    for value in values:
        tag = str(value).strip()
        if tag and tag not in seen:
            seen.add(tag)
            result.append(tag)
    return result


def _normalize_slots(values: Any) -> list[dict[str, Any]]:
    normalized = []
    for item in _as_list(values):
        if isinstance(item, dict):
            copied = dict(item)
            if copied.get("time"):
                normalized.append(copied)
        elif item not in (None, ""):
            normalized.append({"time": str(item)})
    return normalized


def _normalize_poi(item: dict[str, Any], expected_type: str) -> dict[str, Any]:
    tags = _flatten_tags(item.get("tags"))
    tags.extend(tag for tag in expand_preference_tags(tags) if tag not in tags)

    available_slots = _normalize_slots(item.get("available_slots") or item.get("time_slots"))
    inventory_left = item.get("inventory_left")
    if inventory_left is None:
        inventory_left = item.get("remaining")

    available = item.get("available")
    if available is None:
        available = to_float(inventory_left, 1.0) > 0 if inventory_left is not None else True

    distance_km = item.get("distance_km")
    if distance_km is None and item.get("distance_m") is not None:
        distance_km = to_float(item.get("distance_m"), 0.0) / 1000.0

    poi_id = item.get("poi_id") or item.get("id") or item.get("amap_id")
    normalized = {
        "poi_id": str(poi_id),
        "name": item.get("name", str(poi_id)),
        "type": expected_type,
        "tags": tags,
        "price": to_float(item.get("price"), 0.0),
        "distance_km": round(to_float(distance_km, 8.0), 2),
        "duration_min": int(to_float(item.get("duration_min"), 90.0)),
        "rating": round(to_float(item.get("rating"), 4.0), 2),
        "queue_time_min": int(to_float(item.get("queue_time_min"), 0.0)),
        "available": bool(available),
        "available_slots": available_slots,
        "location": item.get("location") or item.get("address") or "",
        "source": item.get("source", "local_supply_mock"),
    }

    for optional_key in (
        "category",
        "amap_id",
        "coordinates",
        "deal_ids",
        "inventory_left",
        "refund_policy",
        "risk_flags",
        "commercial_features",
        "supply_notes",
    ):
        if optional_key in item:
            normalized[optional_key] = item[optional_key]

    return normalized


def _apply_availability_overlay(items: list[dict[str, Any]], expected_type: str) -> list[dict[str, Any]]:
    availability_path = _mock_data_dir() / "availability.json"
    raw = _load_json_file(str(availability_path.resolve()))
    overlays = raw if isinstance(raw, dict) else {}

    normalized = []
    for item in items:
        poi_id = item.get("poi_id")
        overlay = overlays.get(poi_id, {}) if poi_id else {}
        if isinstance(overlay, dict):
            merged = dict(item)
            merged.update({k: v for k, v in overlay.items() if k != "poi_id"})
        else:
            merged = item
        normalized.append(_normalize_poi(merged, expected_type))
    return normalized


def _load_local_supply(expected_type: str) -> list[dict[str, Any]]:
    file_name = "activities.json" if expected_type == "activity" else "restaurants.json"
    raw = _load_json_file(str((_mock_data_dir() / file_name).resolve()))
    if isinstance(raw, dict):
        items = raw.get("items", [])
    else:
        items = raw
    if not isinstance(items, list):
        return []
    return _apply_availability_overlay([item for item in items if isinstance(item, dict)], expected_type)


def _fetch_from_c_mock_api(
    *,
    expected_type: str,
    constraints: dict,
    scene_type: str,
    scenario_activities: list[str],
) -> list[dict[str, Any]]:
    raw_tags = " ".join(str(x) for x in scenario_activities)

    if expected_type == "activity":
        if search_activities is None:
            return []
        child_age = constraints.get("child_age")
        try:
            result = search_activities(
                radius=int(float(constraints.get("max_distance_km", 8)) * 1000),
                kid_friendly=scene_type == "family" or child_age not in (None, ""),
                low_intensity=("low_intensity" in raw_tags) or ("light_activity" in raw_tags),
                indoor=("indoor" in raw_tags) or ("rainy" in raw_tags),
                limit=10,
            )
        except Exception:
            return []
    else:
        if search_restaurants is None:
            return []
        mom_diet = str(constraints.get("mom_diet") or "").lower()
        try:
            result = search_restaurants(
                radius=int(float(constraints.get("max_distance_km", 8)) * 1000),
                low_calorie=(mom_diet == "low_calorie") or ("low_calorie" in raw_tags) or ("light_food" in raw_tags),
                family_friendly=scene_type == "family",
                limit=10,
            )
        except Exception:
            return []

    return [_normalize_poi(item, expected_type) for item in result.get("items", [])]


def _keywords_for_gaode(
    *,
    expected_type: str,
    scenario_activities: list[str],
    constraints: dict,
) -> str:
    planning_preferences = constraints.get("planning_preferences", {}) or {}
    if expected_type == "activity":
        values = _as_list(planning_preferences.get("activity_type")) + _as_list(scenario_activities)
        fallback = DEFAULT_ACTIVITY_KEYWORDS
    else:
        values = _as_list(planning_preferences.get("food_type")) + _as_list(scenario_activities)
        fallback = DEFAULT_RESTAURANT_KEYWORDS

    keywords = " ".join(str(item).strip() for item in values if str(item).strip())
    return keywords or fallback


def _normalize_gaode_poi(
    poi: dict[str, Any],
    expected_type: str,
    scenario_activities: list[str],
) -> dict[str, Any]:
    biz_ext = poi.get("biz_ext", {}) if isinstance(poi.get("biz_ext"), dict) else {}
    tags = expand_preference_tags(scenario_activities)
    if expected_type == "activity":
        tags.extend(["activity"])
    else:
        tags.extend(["restaurant"])

    distance_m = poi.get("distance")
    base = {
        "poi_id": poi.get("id"),
        "amap_id": poi.get("id"),
        "name": poi.get("name"),
        "type": expected_type,
        "tags": list(dict.fromkeys(tags)),
        "price": to_float(biz_ext.get("cost"), 0.0),
        "distance_m": distance_m,
        "duration_min": 90 if expected_type == "activity" else 60,
        "rating": to_float(biz_ext.get("rating"), 4.0),
        "queue_time_min": 0,
        "available": True,
        "available_slots": ["14:00", "15:30", "17:30", "18:30"],
        "location": poi.get("address") or "",
        "coordinates": poi.get("location"),
        "source": "gaode_poi_search",
        "raw": poi,
    }
    return _normalize_poi(base, expected_type)


def _fetch_from_gaode_poi(
    *,
    expected_type: str,
    constraints: dict,
    scenario_activities: list[str],
) -> list[dict[str, Any]]:
    if not os.getenv("GAODE_API_KEY"):
        return []

    try:
        from .poi_searcher import POISearcher
    except Exception:
        return []

    keywords = _keywords_for_gaode(
        expected_type=expected_type,
        scenario_activities=scenario_activities,
        constraints=constraints,
    )
    city = constraints.get("city") or constraints.get("adcode")
    offset = int(to_float(_candidate_generation_config().get("gaode_search_limit"), 10))

    try:
        results = POISearcher().search(
            keywords=keywords,
            city=city,
            citylimit=bool(city),
            offset=max(1, min(offset, 25)),
        )
    except Exception:
        return []

    return [_normalize_gaode_poi(item, expected_type, scenario_activities) for item in results]


def _fetch_candidates(
    *,
    expected_type: str,
    constraints: dict | None,
    scene_type: str,
    scenario_activities: list[str] | None,
) -> list[dict[str, Any]]:
    constraints = constraints or {}
    scenario_activities = scenario_activities or []

    for source in _source_order():
        if source in {"local_supply_mock", "local_mock"}:
            items = _load_local_supply(expected_type)
        elif source in {"gaode_poi_search", "gaode_poi", "amap_poi"}:
            items = _fetch_from_gaode_poi(
                expected_type=expected_type,
                constraints=constraints,
                scenario_activities=scenario_activities,
            )
        elif source in {"c_mock_api", "mock_api"}:
            items = _fetch_from_c_mock_api(
                expected_type=expected_type,
                constraints=constraints,
                scene_type=scene_type,
                scenario_activities=scenario_activities,
            )
        else:
            items = []

        if items:
            return items

    return []


def fetch_activity_candidates(
    *,
    constraints: dict | None = None,
    scene_type: str = "family",
    scenario_activities: list[str] | None = None,
) -> list[dict[str, Any]]:
    return _fetch_candidates(
        expected_type="activity",
        constraints=constraints,
        scene_type=scene_type,
        scenario_activities=scenario_activities,
    )


def fetch_restaurant_candidates(
    *,
    constraints: dict | None = None,
    scene_type: str = "family",
    scenario_activities: list[str] | None = None,
) -> list[dict[str, Any]]:
    return _fetch_candidates(
        expected_type="restaurant",
        constraints=constraints,
        scene_type=scene_type,
        scenario_activities=scenario_activities,
    )
