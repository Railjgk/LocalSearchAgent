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

from .b_utils import destination_city_from_constraints, expand_preference_tags, to_float
from .poi_cleaning import should_exclude_poi


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
    env_order = os.environ.get("WF_DATA_SOURCE_ORDER", "").strip()
    if env_order:
        return [item.strip() for item in env_order.split(",") if item.strip()]

    raw_order = _candidate_generation_config().get("data_source_order", DEFAULT_SOURCE_ORDER)
    if not isinstance(raw_order, list):
        return list(DEFAULT_SOURCE_ORDER)
    return [str(item).strip() for item in raw_order if str(item).strip()]


def _mock_data_dir() -> Path:
    env_dir = os.environ.get("WF_MOCK_DATA_DIR", "").strip()
    if env_dir:
        path = Path(env_dir)
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[2] / path

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


def _read_json_file_uncached(path: Path) -> Any:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _items_from_json(raw: Any) -> list[dict[str, Any]]:
    items = raw.get("items", []) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _group_items_by_poi(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        poi_id = item.get("poi_id")
        if poi_id:
            grouped.setdefault(str(poi_id), []).append(item)
    return grouped


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
    except Exception:
        return []
    return records


def _read_mock_records(mock_data_dir: Path, stem: str) -> list[dict[str, Any]]:
    json_path = mock_data_dir / f"{stem}.json"
    raw = _read_json_file_uncached(json_path)
    records = _items_from_json(raw)
    if records:
        return records

    jsonl_path = mock_data_dir / f"{stem}.jsonl"
    if jsonl_path.exists():
        return _read_jsonl_records(jsonl_path)

    for shard_dir_name in (f"{stem}_shards", f"{stem}_jsonl", f"{stem}.jsonl.d"):
        shard_dir = mock_data_dir / shard_dir_name
        if not shard_dir.exists():
            continue
        shard_records: list[dict[str, Any]] = []
        for shard in sorted(shard_dir.glob("*.jsonl")):
            shard_records.extend(_read_jsonl_records(shard))
        if shard_records:
            return shard_records

    return []


@lru_cache(maxsize=32)
def _read_mock_records_cached(
    mock_data_dir_key: str,
    stem: str,
    signature: str,
) -> tuple[dict[str, Any], ...]:
    del signature  # Signature is part of the cache key and invalidates stale reads.
    return tuple(_read_mock_records(Path(mock_data_dir_key), stem))


def _read_mock_records_with_cache(mock_data_dir: Path, stem: str) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in _read_mock_records_cached(
            str(mock_data_dir.resolve()),
            stem,
            _records_signature(mock_data_dir, stem),
        )
    ]


def _records_signature(mock_data_dir: Path, stem: str) -> str:
    parts: list[str] = []
    for path in (mock_data_dir / f"{stem}.json", mock_data_dir / f"{stem}.jsonl"):
        try:
            stat = path.stat()
        except OSError:
            continue
        parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")

    for shard_dir_name in (f"{stem}_shards", f"{stem}_jsonl", f"{stem}.jsonl.d"):
        shard_dir = mock_data_dir / shard_dir_name
        if not shard_dir.exists():
            continue
        for shard in sorted(shard_dir.glob("*.jsonl")):
            try:
                stat = shard.stat()
            except OSError:
                continue
            parts.append(f"{shard_dir.name}/{shard.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts) or f"{stem}:missing"


def _local_supply_cache_signature(mock_dir: Path, expected_type: str) -> str:
    supply_stem = "activities" if expected_type == "activity" else "restaurants"
    stems = (supply_stem, "availability", "deals", "products")
    return "|".join(_records_signature(mock_dir, stem) for stem in stems)


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
        "tag_groups": item.get("tags") if isinstance(item.get("tags"), dict) else {},
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
        "capacity_limit",
        "refund_policy",
        "risk_flags",
        "commercial_features",
        "supply_notes",
        "sub_category",
        "experience_type",
        "trust_score",
        "verified_reviews",
        "review_count",
        "source_evidence",
        "emotion_tags",
        "atmosphere_tags",
        "ritual_score",
        "local_character_tags",
        "city_limited",
        "citywalk_score",
        "restaurant_category",
        "avg_price_per_person",
        "category_price_band",
        "health_tags",
        "menu_health_options",
        "reservation_required",
        "failure_reason",
        "weather_sensitivity",
        "traffic_risk",
        "walking_time_min",
        "ugc_summary",
        "source_channel",
        "content_heat_score",
        "queue_time_by_period",
        "reservation_slots",
        "business_hours",
        "holiday_status",
        "operation_stability_score",
        "serving_speed_min",
        "service_mode",
        "dine_in_available",
        "takeaway_available",
        "delivery_available",
        "child_menu",
        "local_flavor_tags",
        "package_options",
        "review_breakdown",
        "review_keywords",
        "signature_dishes",
        "recommended_dishes",
        "dish_tags",
        "promotion_highlights",
        "baby_chair_available",
        "parking_fee_policy",
        "private_room_available",
        "noise_level",
        "spice_level",
        "booking_policy",
        "dietary_options",
        "allergen_notes",
        "decision_profile",
        "fulfillment_actions",
        "substitution_strategy",
        "peak_risk_profile",
        "suitable_age",
        "age_range",
        "service_facilities",
        "physical_intensity",
        "weather_plan",
        "mock_detail_sources",
        "cancel_policy",
        "hidden_cost_risk",
        "indoor_backup",
        "operator_license",
        "safety_level",
        "skill_level",
        "equipment_rental",
        "guide_available",
        "parking_available",
        "online_verify",
        "digital_ticketing",
        "low_sugar",
        "low_oil",
        "plant_based",
        "pickup_available",
        "seat_capacity",
        "business_format",
        "best_deal_price",
        "products",
        "deals",
        "merchant_id",
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


def _load_deals_by_poi() -> dict[str, list[dict[str, Any]]]:
    return _group_items_by_poi(_read_mock_records_with_cache(_mock_data_dir(), "deals"))


def _load_products_by_poi() -> dict[str, list[dict[str, Any]]]:
    return _group_items_by_poi(_read_mock_records_with_cache(_mock_data_dir(), "products"))


def _attach_supply_side_details(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deals_by_poi = _load_deals_by_poi()
    products_by_poi = _load_products_by_poi()
    enriched = []
    for item in items:
        copied = dict(item)
        poi_id = copied.get("poi_id")
        if poi_id:
            deals = deals_by_poi.get(str(poi_id), [])
            products = products_by_poi.get(str(poi_id), [])
            if deals:
                copied["deals"] = deals
                copied.setdefault("deal_ids", [deal.get("deal_id") for deal in deals if deal.get("deal_id")])
                copied["best_deal_price"] = min(
                    to_float(deal.get("sale_price"), copied.get("price", 0.0))
                    for deal in deals
                )
            if products:
                copied["products"] = products
                copied.setdefault("product_ids", [product.get("product_id") for product in products if product.get("product_id")])
        enriched.append(copied)
    return enriched


@lru_cache(maxsize=32)
def _load_local_supply_cached(
    mock_data_dir_key: str,
    expected_type: str,
    signature: str,
) -> tuple[dict[str, Any], ...]:
    del signature  # Part of the cache key; invalidates stale shard/file reads.
    mock_data_dir = Path(mock_data_dir_key)
    stem = "activities" if expected_type == "activity" else "restaurants"
    items = _read_mock_records_with_cache(mock_data_dir, stem)
    cleaned_items = [
        item
        for item in items
        if isinstance(item, dict) and not should_exclude_poi(item, expected_type)
    ]
    availability_raw = _read_json_file_uncached(mock_data_dir / "availability.json")
    overlays = availability_raw if isinstance(availability_raw, dict) else {}
    deals_by_poi = _group_items_by_poi(_read_mock_records_with_cache(mock_data_dir, "deals"))
    products_by_poi = _group_items_by_poi(_read_mock_records_with_cache(mock_data_dir, "products"))

    normalized = []
    for item in cleaned_items:
        poi_id = item.get("poi_id")
        overlay = overlays.get(poi_id, {}) if poi_id else {}
        if isinstance(overlay, dict):
            merged = dict(item)
            merged.update({k: v for k, v in overlay.items() if k != "poi_id"})
        else:
            merged = item

        copied = _normalize_poi(merged, expected_type)
        normalized_poi_id = copied.get("poi_id")
        if normalized_poi_id:
            deals = deals_by_poi.get(str(normalized_poi_id), [])
            products = products_by_poi.get(str(normalized_poi_id), [])
            if deals:
                copied["deals"] = deals
                copied.setdefault("deal_ids", [deal.get("deal_id") for deal in deals if deal.get("deal_id")])
                copied["best_deal_price"] = min(
                    to_float(deal.get("sale_price"), copied.get("price", 0.0))
                    for deal in deals
                )
            if products:
                copied["products"] = products
                copied.setdefault(
                    "product_ids",
                    [product.get("product_id") for product in products if product.get("product_id")],
                )
        normalized.append(copied)

    return tuple(normalized)


def _load_local_supply(expected_type: str) -> list[dict[str, Any]]:
    mock_data_dir = _mock_data_dir().resolve()
    signature = _local_supply_cache_signature(mock_data_dir, expected_type)
    return [
        dict(item)
        for item in _load_local_supply_cached(
            str(mock_data_dir),
            expected_type,
            signature,
        )
    ]


def _fetch_from_c_mock_api(
    *,
    expected_type: str,
    constraints: dict,
    scene_type: str,
    scenario_activities: list[str],
) -> list[dict[str, Any]]:
    expanded_tags = set(expand_preference_tags(scenario_activities))

    if expected_type == "activity":
        if search_activities is None:
            return []
        child_age = constraints.get("child_age")
        try:
            result = search_activities(
                radius=int(float(constraints.get("max_distance_km", 8)) * 1000),
                kid_friendly=scene_type == "family" or child_age not in (None, ""),
                low_intensity=bool({"low_intensity", "light_activity"} & expanded_tags),
                indoor=bool({"indoor", "rainy"} & expanded_tags),
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
                low_calorie=(mom_diet == "low_calorie")
                or bool({"low_calorie", "light_food"} & expanded_tags),
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
    city = destination_city_from_constraints(constraints) or constraints.get("adcode")
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

    return [
        _normalize_gaode_poi(item, expected_type, scenario_activities)
        for item in results
        if isinstance(item, dict) and not should_exclude_poi(item, expected_type)
    ]


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

    constraints = constraints or {}
    scenario_activities = scenario_activities or []
    raw_tags = " ".join(str(x) for x in scenario_activities)
    mom_diet = str(constraints.get("mom_diet") or "").lower()

    try:
        result = search_restaurants(
            radius=int(float(constraints.get("max_distance_km", 8)) * 1000),
            latitude=constraints.get("latitude"),
            longitude=constraints.get("longitude"),
            low_calorie=(mom_diet == "low_calorie") or ("轻食" in raw_tags) or ("低卡" in raw_tags),
            family_friendly=scene_type == "family",
            limit=10,
        )
    except Exception:
        return []

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
