#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate WeekendFlow B mock supply data quality.

The checker separates four concepts that should not be mixed:

- observed_fixture: real or stable place/supply facts
- rule_imputed_fixture: category-based business fields added by WeekendFlow
- execution_mock_state: availability/order/route-like mutable state
- b_derived_or_prior: scores or priors that B may consume today but should not
  be treated as verified facts

It is intentionally strict on structural errors and conservative on realism:
realism mismatches are warnings so we can improve the fixture iteratively.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nodes.b_utils import expand_preference_tags
from src.nodes.poi_cleaning import contains_closed_status, is_invalid_activity_poi

try:
    from src.nodes.taxonomy import canonicalize_tags
except Exception:  # pragma: no cover - taxonomy is optional for standalone data checks
    canonicalize_tags = None


def to_internal_key(value: Any) -> str:
    if str(value or "") in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[str(value)]
    if canonicalize_tags is not None:
        tags = canonicalize_tags(value, preserve_unknown=True)
        if tags:
            return str(tags[0])
    expanded = expand_preference_tags(value)
    if expanded:
        return str(expanded[0])
    return str(value or "unknown")

DEFAULT_MOCK_DIR = REPO_ROOT / "experiments" / "mock_data"


FIELD_SOURCE_RULES: dict[str, set[str]] = {
    "observed_fixture": {
        "poi_id",
        "amap_id",
        "merchant_id",
        "name",
        "type",
        "location",
        "address",
        "coordinates",
        "latitude",
        "longitude",
        "tel",
        "adcode",
        "citycode",
        "pcode",
        "source",
        "source_channel",
        "source_evidence",
        "gaode_keyword",
        "gaode_type",
        "gaode_typecode",
        "source_queries",
        "raw",
        "cover_photo_url",
        "photo_urls",
        "photos",
        "photo_source",
        "photo_enrichment",
    },
    "rule_imputed_fixture": {
        "category",
        "sub_category",
        "experience_type",
        "restaurant_category",
        "tags",
        "price",
        "distance_km",
        "avg_price_per_person",
        "category_price_band",
        "duration_min",
        "business_hours",
        "refund_policy",
        "cancel_policy",
        "hidden_cost_risk",
        "weather_sensitivity",
        "traffic_risk",
        "walking_time_min",
        "service_mode",
        "dine_in_available",
        "takeaway_available",
        "delivery_available",
        "seat_capacity",
        "business_format",
        "child_menu",
        "local_flavor_tags",
        "operator_license",
        "safety_level",
        "skill_level",
        "equipment_rental",
        "guide_available",
        "parking_available",
        "parking_fee_policy",
        "baby_chair_available",
        "private_room_available",
        "noise_level",
        "spice_level",
        "booking_policy",
        "dietary_options",
        "allergen_notes",
        "suitable_age",
        "age_range",
        "service_facilities",
        "physical_intensity",
        "weather_plan",
        "indoor_backup",
        "online_verify",
        "digital_ticketing",
        "low_sugar",
        "low_oil",
        "plant_based",
        "pickup_available",
        "supply_notes",
        "curation_note",
        "primary_category",
        "field_sources",
        "mock_detail_sources",
    },
    "research_prior_fixture": {
        "emotion_tags",
        "atmosphere_tags",
        "local_character_tags",
        "health_tags",
        "menu_health_options",
        "ugc_summary",
        "review_breakdown",
        "review_keywords",
        "signature_dishes",
        "recommended_dishes",
        "dish_tags",
        "promotion_highlights",
        "decision_profile",
        "substitution_strategy",
        "peak_risk_profile",
        "package_options",
        "beverage_pairings",
        "functional_food_tags",
    },
    "execution_mock_state": {
        "available",
        "available_slots",
        "inventory_left",
        "capacity_limit",
        "reservation_required",
        "failure_reason",
        "holiday_status",
        "queue_time_min",
        "queue_time_by_period",
        "reservation_slots",
        "serving_speed_min",
        "fulfillment_actions",
        "stock_location_type",
    },
    "transaction_fixture": {
        "deal_ids",
        "product_ids",
        "deal_id",
        "product_id",
        "deal_type",
        "coupon_type",
        "title",
        "sale_price",
        "original_price",
        "requires_reservation",
        "valid_time",
        "package_components",
        "target_persona",
        "family_ticket",
        "stock_limit_per_slot",
        "product_type",
        "inventory_model",
        "fulfillment_mode",
        "redemption_rate",
        "commission_rate",
    },
    "b_derived_or_prior": {
        "rating",
        "trust_score",
        "verified_reviews",
        "review_count",
        "content_heat_score",
        "operation_stability_score",
        "ritual_score",
        "city_limited",
        "citywalk_score",
        "best_deal_price",
        "risk_flags",
        "commercial_features",
        "products",
        "deals",
        "recent_order_count",
        "reservation_count",
        "service_risks",
        "business_capabilities",
        "merchant_type",
        "poi_ids",
        "gaode_fields_available",
        "meituan_mock_fields_owned",
    },
}

DERIVED_FIELDS_THAT_SHOULD_NOT_BE_VERIFIED = {
    "trust_score",
    "content_heat_score",
    "operation_stability_score",
    "ritual_score",
    "citywalk_score",
    "best_deal_price",
}

CORE_POI_FIELDS = {"poi_id", "name", "type", "tags", "price", "duration_min", "location"}
REAL_POI_HINT_FIELDS = {"amap_id", "coordinates", "source_evidence"}

EXPECTED_ACTIVITY_CATEGORIES = {
    "parent_child_activity",
    "citywalk",
    "sports",
    "local_market",
    "micro_vacation",
    "handcraft",
    "museum",
    "escape_room",
}
EXPECTED_RESTAURANT_CATEGORIES = {
    "japanese_light_food",
    "salad_light_food",
    "family_bistro",
    "regional_home_cuisine",
    "hotpot",
    "barbecue",
    "fried_chicken",
}

CATEGORY_ALIASES = {
    "亲子活动": "parent_child_activity",
    "城市漫步": "citywalk",
    "运动体验": "sports",
    "本地市集": "local_market",
    "近场放松": "micro_vacation",
    "手作体验": "handcraft",
    "博物馆展览": "museum",
    "密室桌游": "escape_room",
    "日料轻食": "japanese_light_food",
    "沙拉轻食": "salad_light_food",
    "家庭餐厅": "family_bistro",
    "本帮家常菜": "regional_home_cuisine",
    "火锅": "hotpot",
    "烤肉": "barbecue",
    "炸鸡小吃": "fried_chicken",
}


@dataclass
class Issue:
    severity: str
    code: str
    file: str
    item_id: str
    message: str


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
    return records


def load_mock_records(mock_dir: Path, stem: str) -> list[dict[str, Any]]:
    json_path = mock_dir / f"{stem}.json"
    if json_path.exists():
        return as_items(load_json(json_path))

    jsonl_path = mock_dir / f"{stem}.jsonl"
    if jsonl_path.exists():
        return load_jsonl_records(jsonl_path)

    for shard_dir_name in (f"{stem}_shards", f"{stem}_jsonl", f"{stem}.jsonl.d"):
        shard_dir = mock_dir / shard_dir_name
        if not shard_dir.exists():
            continue
        records: list[dict[str, Any]] = []
        for shard in sorted(shard_dir.glob("*.jsonl")):
            records.extend(load_jsonl_records(shard))
        if records:
            return records
    return []


def as_items(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        items = raw.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def flatten_tags(value: Any) -> list[str]:
    tags: list[str] = []
    if isinstance(value, dict):
        for group in value.values():
            tags.extend(flatten_tags(group))
    elif isinstance(value, list):
        tags.extend(str(item).strip() for item in value if str(item).strip())
    elif value not in (None, ""):
        tags.append(str(value).strip())
    return tags


def field_source(field_name: str) -> str:
    for source_name, fields in FIELD_SOURCE_RULES.items():
        if field_name in fields:
            return source_name
    return "unknown"


def add_issue(
    issues: list[Issue],
    severity: str,
    code: str,
    file: str,
    item_id: str,
    message: str,
) -> None:
    issues.append(Issue(severity, code, file, item_id, message))


def is_real_or_curated_gaode(item: dict[str, Any]) -> bool:
    source = str(item.get("source") or item.get("source_channel") or "").lower()
    amap_id = str(item.get("amap_id") or "").lower()
    if amap_id.startswith("mock_"):
        return "gaode" in source or "amap" in source
    return bool(amap_id) or "gaode" in source or "amap" in source


def validate_unique_ids(
    items: list[dict[str, Any]],
    *,
    id_field: str,
    file_name: str,
    issues: list[Issue],
) -> set[str]:
    ids: set[str] = set()
    seen: set[str] = set()
    for index, item in enumerate(items):
        item_id = str(item.get(id_field) or "")
        if not item_id:
            add_issue(issues, "error", "missing_id", file_name, f"row_{index}", f"missing {id_field}")
            continue
        if item_id in seen:
            add_issue(issues, "error", "duplicate_id", file_name, item_id, f"duplicate {id_field}")
        seen.add(item_id)
        ids.add(item_id)
    return ids


def validate_poi_item(
    item: dict[str, Any],
    *,
    file_name: str,
    expected_type: str,
    merchant_ids: set[str],
    product_ids: set[str],
    deal_ids: set[str],
    product_poi_ids: set[str],
    deal_poi_ids: set[str],
    availability_ids: set[str],
    issues: list[Issue],
) -> None:
    poi_id = str(item.get("poi_id") or "<missing>")
    missing_core = sorted(field for field in CORE_POI_FIELDS if item.get(field) in (None, "", []))
    if missing_core:
        add_issue(issues, "error", "missing_core_fields", file_name, poi_id, f"missing {missing_core}")

    if item.get("type") != expected_type:
        add_issue(issues, "error", "wrong_type", file_name, poi_id, f"expected type={expected_type}, got {item.get('type')}")

    if is_real_or_curated_gaode(item):
        missing_real = sorted(field for field in REAL_POI_HINT_FIELDS if item.get(field) in (None, "", []))
        if missing_real:
            add_issue(issues, "warn", "real_poi_missing_evidence", file_name, poi_id, f"real/gaode POI missing {missing_real}")

    merchant_id = item.get("merchant_id")
    if merchant_id and merchant_id not in merchant_ids:
        add_issue(issues, "warn", "merchant_link_missing", file_name, poi_id, f"merchant_id not found: {merchant_id}")

    for product_id in item.get("product_ids") or []:
        if str(product_id) not in product_ids:
            add_issue(issues, "warn", "product_id_missing", file_name, poi_id, f"product_id not found: {product_id}")
    for deal_id in item.get("deal_ids") or []:
        if str(deal_id) not in deal_ids:
            add_issue(issues, "warn", "deal_id_missing", file_name, poi_id, f"deal_id not found: {deal_id}")

    if poi_id not in product_poi_ids:
        add_issue(issues, "warn", "product_link_missing", file_name, poi_id, "no product record linked by poi_id")
    if poi_id not in deal_poi_ids:
        add_issue(issues, "warn", "deal_link_missing", file_name, poi_id, "no deal record linked by poi_id")
    if poi_id not in availability_ids:
        add_issue(issues, "warn", "availability_missing", file_name, poi_id, "no availability overlay")

    price = item.get("price")
    if isinstance(price, (int, float)) and price < 0:
        add_issue(issues, "error", "negative_price", file_name, poi_id, "price cannot be negative")

    if item.get("inventory_left") is not None and item.get("capacity_limit") is not None:
        try:
            if float(item["inventory_left"]) > float(item["capacity_limit"]):
                add_issue(issues, "warn", "inventory_exceeds_capacity", file_name, poi_id, "inventory_left > capacity_limit")
        except (TypeError, ValueError):
            add_issue(issues, "warn", "invalid_inventory_capacity", file_name, poi_id, "inventory/capacity should be numeric")

    if item.get("available") is False and not item.get("failure_reason"):
        add_issue(issues, "warn", "unavailable_without_reason", file_name, poi_id, "available=false should include failure_reason")

    if contains_closed_status(item) and item.get("available") is not False:
        add_issue(issues, "warn", "closed_poi_marked_available", file_name, poi_id, "closed/paused POI should be unavailable or excluded")

    if expected_type == "activity" and is_invalid_activity_poi(item):
        add_issue(issues, "warn", "food_service_activity_mismatch", file_name, poi_id, "pure food-service POI should not be used as an activity")

    for field in DERIVED_FIELDS_THAT_SHOULD_NOT_BE_VERIFIED:
        if field in item:
            add_issue(
                issues,
                "info",
                "materialized_prior_score",
                file_name,
                poi_id,
                f"{field} is a B prior/derived-like field; do not treat it as verified fact",
            )

    category = to_internal_key(item.get("category") or item.get("restaurant_category"))
    tags = set(expand_preference_tags(flatten_tags(item.get("tags"))))
    tags.update(expand_preference_tags(flatten_tags(item.get("health_tags"))))
    tags.update(expand_preference_tags(flatten_tags(item.get("menu_health_options"))))

    if category in {"hotpot", "fried_chicken", "bbq", "barbecue"} and "low_calorie" in tags and "high_calorie" not in tags:
        add_issue(issues, "warn", "health_logic_conflict", file_name, poi_id, f"{category} has low_calorie without high_calorie risk")

    if category in {"salad_light_food", "japanese_light_food", "light_food"} and not tags.intersection({"low_calorie", "light_food", "low_oil", "high_protein", "vegetable_rich"}):
        add_issue(issues, "warn", "missing_health_signal", file_name, poi_id, f"{category} should expose health/menu signals")

    if item.get("service_mode") == "takeaway_only" and item.get("dine_in_available") is not False:
        add_issue(issues, "warn", "takeaway_mode_inconsistent", file_name, poi_id, "takeaway_only should set dine_in_available=false")

    if expected_type == "activity" and item.get("category") in {"citywalk", "local_market"} and item.get("reservation_required") is True:
        add_issue(issues, "warn", "citywalk_reservation_suspicious", file_name, poi_id, "citywalk/local_market usually should not require reservation")


def validate_availability(
    availability: dict[str, Any],
    *,
    poi_ids: set[str],
    issues: list[Issue],
) -> None:
    for poi_id, overlay in availability.items():
        if poi_id not in poi_ids:
            add_issue(issues, "warn", "availability_orphan", "availability.json", poi_id, "availability id not found in activities/restaurants")
        if not isinstance(overlay, dict):
            add_issue(issues, "error", "availability_not_object", "availability.json", poi_id, "availability overlay must be object")
            continue
        if overlay.get("inventory_left") is not None and overlay.get("capacity_limit") is not None:
            try:
                if float(overlay["inventory_left"]) > float(overlay["capacity_limit"]):
                    add_issue(issues, "warn", "availability_inventory_exceeds_capacity", "availability.json", poi_id, "inventory_left > capacity_limit")
            except (TypeError, ValueError):
                add_issue(issues, "warn", "invalid_availability_inventory", "availability.json", poi_id, "inventory/capacity should be numeric")
        if overlay.get("available") is False and not overlay.get("failure_reason"):
            add_issue(issues, "warn", "availability_false_without_reason", "availability.json", poi_id, "available=false should include failure_reason")


def validate_products_deals(
    products: list[dict[str, Any]],
    deals: list[dict[str, Any]],
    *,
    poi_ids: set[str],
    merchant_ids: set[str],
    product_ids: set[str],
    issues: list[Issue],
) -> None:
    for product in products:
        product_id = str(product.get("product_id") or "<missing>")
        poi_id = product.get("poi_id")
        merchant_id = product.get("merchant_id")
        if poi_id not in poi_ids:
            add_issue(issues, "warn", "product_orphan_poi", "products.json", product_id, f"poi_id not found: {poi_id}")
        if merchant_id and merchant_id not in merchant_ids:
            add_issue(issues, "warn", "product_orphan_merchant", "products.json", product_id, f"merchant_id not found: {merchant_id}")
        if product.get("price") is not None:
            try:
                if float(product["price"]) < 0:
                    add_issue(issues, "error", "product_negative_price", "products.json", product_id, "product price cannot be negative")
            except (TypeError, ValueError):
                add_issue(issues, "warn", "product_invalid_price", "products.json", product_id, "product price should be numeric")

    for deal in deals:
        deal_id = str(deal.get("deal_id") or "<missing>")
        poi_id = deal.get("poi_id")
        product_id = deal.get("product_id")
        if poi_id not in poi_ids:
            add_issue(issues, "warn", "deal_orphan_poi", "deals.json", deal_id, f"poi_id not found: {poi_id}")
        if product_id and product_id not in product_ids:
            add_issue(issues, "warn", "deal_orphan_product", "deals.json", deal_id, f"product_id not found: {product_id}")
        sale_price = deal.get("sale_price")
        original_price = deal.get("original_price")
        try:
            if sale_price is not None and original_price is not None and float(sale_price) > float(original_price):
                add_issue(issues, "warn", "deal_price_inversion", "deals.json", deal_id, "sale_price > original_price")
        except (TypeError, ValueError):
            add_issue(issues, "warn", "deal_invalid_price", "deals.json", deal_id, "deal prices should be numeric")


def validate_routes(routes: dict[str, Any], *, poi_ids: set[str], issues: list[Issue]) -> None:
    overrides = routes.get("overrides", []) if isinstance(routes, dict) else []
    for index, route in enumerate(overrides):
        if not isinstance(route, dict):
            add_issue(issues, "warn", "route_override_not_object", "routes.json", f"overrides[{index}]", "route override should be object")
            continue
        from_id = route.get("from")
        to_id = route.get("to")
        for field_name, value in (("from", from_id), ("to", to_id)):
            if value in (None, "", "start_point"):
                continue
            if str(value) not in poi_ids:
                add_issue(issues, "warn", "route_orphan_poi", "routes.json", f"overrides[{index}]", f"{field_name} id not found: {value}")
        if route.get("distance_km") is None or route.get("travel_time_min") is None:
            add_issue(issues, "warn", "route_missing_distance_time", "routes.json", f"overrides[{index}]", "route should include distance_km and travel_time_min")


def summarize_field_sources(items_by_file: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per_file: dict[str, dict[str, int]] = {}
    unknown_fields: dict[str, list[str]] = {}
    for file_name, items in items_by_file.items():
        counter: Counter[str] = Counter()
        unknown = set()
        for item in items:
            for field_name in item:
                source = field_source(field_name)
                counter[source] += 1
                if source == "unknown":
                    unknown.add(field_name)
        per_file[file_name] = dict(counter)
        if unknown:
            unknown_fields[file_name] = sorted(unknown)
    return {"per_file": per_file, "unknown_fields": unknown_fields}


def build_report(mock_dir: Path) -> dict[str, Any]:
    issues: list[Issue] = []
    record_stems = ["activities", "restaurants", "merchants", "products", "deals"]
    required_json_files = ["availability.json", "routes.json"]

    raw: dict[str, Any] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    for stem in record_stems:
        file_name = f"{stem}.json"
        try:
            items = load_mock_records(mock_dir, stem)
        except Exception as exc:
            add_issue(issues, "error", "json_load_failed", file_name, file_name, str(exc))
            items = []
        if not items:
            add_issue(
                issues,
                "error",
                "missing_file",
                file_name,
                file_name,
                f"missing {stem}.json, {stem}.jsonl, or {stem}_shards/*.jsonl in {mock_dir}",
            )
        records[file_name] = items

    for file_name in required_json_files:
        path = mock_dir / file_name
        if not path.exists():
            add_issue(issues, "error", "missing_file", file_name, file_name, f"missing file: {path}")
            raw[file_name] = {}
            continue
        try:
            raw[file_name] = load_json(path)
        except Exception as exc:
            add_issue(issues, "error", "json_load_failed", file_name, file_name, str(exc))
            raw[file_name] = {}

    activities = records["activities.json"]
    restaurants = records["restaurants.json"]
    merchants = records["merchants.json"]
    products = records["products.json"]
    deals = records["deals.json"]
    availability = raw.get("availability.json") if isinstance(raw.get("availability.json"), dict) else {}
    routes = raw.get("routes.json") if isinstance(raw.get("routes.json"), dict) else {}

    activity_ids = validate_unique_ids(activities, id_field="poi_id", file_name="activities.json", issues=issues)
    restaurant_ids = validate_unique_ids(restaurants, id_field="poi_id", file_name="restaurants.json", issues=issues)
    merchant_ids = validate_unique_ids(merchants, id_field="merchant_id", file_name="merchants.json", issues=issues)
    product_ids = validate_unique_ids(products, id_field="product_id", file_name="products.json", issues=issues)
    deal_ids = validate_unique_ids(deals, id_field="deal_id", file_name="deals.json", issues=issues)

    poi_ids = activity_ids | restaurant_ids
    product_poi_ids = {str(item.get("poi_id")) for item in products if item.get("poi_id")}
    deal_poi_ids = {str(item.get("poi_id")) for item in deals if item.get("poi_id")}
    availability_ids = set(str(key) for key in availability)

    for item in activities:
        validate_poi_item(
            item,
            file_name="activities.json",
            expected_type="activity",
            merchant_ids=merchant_ids,
            product_ids=product_ids,
            deal_ids=deal_ids,
            product_poi_ids=product_poi_ids,
            deal_poi_ids=deal_poi_ids,
            availability_ids=availability_ids,
            issues=issues,
        )

    for item in restaurants:
        validate_poi_item(
            item,
            file_name="restaurants.json",
            expected_type="restaurant",
            merchant_ids=merchant_ids,
            product_ids=product_ids,
            deal_ids=deal_ids,
            product_poi_ids=product_poi_ids,
            deal_poi_ids=deal_poi_ids,
            availability_ids=availability_ids,
            issues=issues,
        )

    validate_availability(availability, poi_ids=poi_ids, issues=issues)
    validate_products_deals(products, deals, poi_ids=poi_ids, merchant_ids=merchant_ids, product_ids=product_ids, issues=issues)
    validate_routes(routes, poi_ids=poi_ids, issues=issues)

    activity_categories = Counter(str(item.get("category") or "unknown") for item in activities)
    restaurant_categories = Counter(str(item.get("restaurant_category") or item.get("category") or "unknown") for item in restaurants)
    activity_category_keys = {
        to_internal_key(item.get("category") or "unknown") for item in activities
    }
    restaurant_category_keys = {
        to_internal_key(item.get("restaurant_category") or item.get("category") or "unknown")
        for item in restaurants
    }

    missing_activity_categories = sorted(EXPECTED_ACTIVITY_CATEGORIES - activity_category_keys)
    missing_restaurant_categories = sorted(EXPECTED_RESTAURANT_CATEGORIES - restaurant_category_keys)
    for category in missing_activity_categories:
        add_issue(issues, "warn", "missing_activity_category", "activities.json", category, "expected category missing from activity supply")
    for category in missing_restaurant_categories:
        add_issue(issues, "warn", "missing_restaurant_category", "restaurants.json", category, "expected category missing from restaurant supply")

    items_by_file = {
        "activities.json": activities,
        "restaurants.json": restaurants,
        "merchants.json": merchants,
        "products.json": products,
        "deals.json": deals,
    }
    severity_counts = Counter(issue.severity for issue in issues)

    return {
        "mock_dir": str(mock_dir),
        "counts": {
            "activities": len(activities),
            "restaurants": len(restaurants),
            "merchants": len(merchants),
            "products": len(products),
            "deals": len(deals),
            "availability": len(availability),
            "route_overrides": len(routes.get("overrides", [])) if isinstance(routes, dict) else 0,
        },
        "coverage": {
            "activity_categories": dict(activity_categories),
            "restaurant_categories": dict(restaurant_categories),
            "missing_activity_categories": missing_activity_categories,
            "missing_restaurant_categories": missing_restaurant_categories,
        },
        "field_source_rules": {key: sorted(values) for key, values in FIELD_SOURCE_RULES.items()},
        "field_source_summary": summarize_field_sources(items_by_file),
        "issues": [asdict(issue) for issue in issues],
        "severity_counts": dict(severity_counts),
        "passed": severity_counts.get("error", 0) == 0,
    }


def print_text_report(report: dict[str, Any], max_issues: int, include_info: bool = False) -> None:
    print("=" * 80)
    print("WeekendFlow Mock Data Quality Check")
    print("=" * 80)
    print(f"mock_dir: {report['mock_dir']}")
    print(f"passed:   {report['passed']}")
    print(f"counts:   {json.dumps(report['counts'], ensure_ascii=False)}")
    print(f"issues:   {json.dumps(report['severity_counts'], ensure_ascii=False)}")
    print()

    print("Coverage")
    print("- activity categories:", json.dumps(report["coverage"]["activity_categories"], ensure_ascii=False))
    print("- restaurant categories:", json.dumps(report["coverage"]["restaurant_categories"], ensure_ascii=False))
    if report["coverage"]["missing_activity_categories"]:
        print("- missing activity categories:", ", ".join(report["coverage"]["missing_activity_categories"]))
    if report["coverage"]["missing_restaurant_categories"]:
        print("- missing restaurant categories:", ", ".join(report["coverage"]["missing_restaurant_categories"]))
    print()

    unknown_fields = report["field_source_summary"].get("unknown_fields", {})
    if unknown_fields:
        print("Unknown Fields")
        for file_name, fields in unknown_fields.items():
            print(f"- {file_name}: {', '.join(fields)}")
        print()

    print("Top Issues")
    severity_rank = {"error": 0, "warn": 1, "info": 2}
    issues = [
        issue for issue in report["issues"]
        if include_info or issue["severity"] != "info"
    ]
    issues = sorted(issues, key=lambda item: (severity_rank.get(item["severity"], 9), item["file"], item["item_id"], item["code"]))
    for issue in issues[:max_issues]:
        print(
            f"[{issue['severity'].upper()}] {issue['code']} "
            f"{issue['file']}::{issue['item_id']} - {issue['message']}"
        )
    if len(issues) > max_issues:
        print(f"... {len(issues) - max_issues} more displayed issue(s)")
    if not include_info and report["severity_counts"].get("info", 0):
        print(f"... {report['severity_counts']['info']} info note(s) hidden; use --include-info to display them")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check WeekendFlow mock data quality and field-source rules.")
    parser.add_argument("--mock-dir", type=Path, default=DEFAULT_MOCK_DIR)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--max-issues", type=int, default=80)
    parser.add_argument("--include-info", action="store_true", help="Show informational prior/derived-field notes in text output.")
    parser.add_argument("--fail-on-warn", action="store_true")
    args = parser.parse_args()

    report = build_report(args.mock_dir)

    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_text_report(report, args.max_issues, include_info=args.include_info)

    if report["severity_counts"].get("error", 0):
        return 1
    if args.fail_on_warn and report["severity_counts"].get("warn", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
