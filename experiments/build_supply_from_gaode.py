#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build an offline WeekendFlow supply seed from Gaode POI search.

This script intentionally writes to a separate output directory by default:

    experiments/mock_data/gaode_seed/

The output is meant to be inspected and selectively merged into the curated
mock files, not blindly used as production-ready supply. Gaode POI data gives
us place identity, address, coordinates, district, phone and sometimes rating
or price. WeekendFlow still owns the business fields that a map POI API usually
does not expose: inventory, queue, booking slots, deals, cancellation policy,
health menu options, family friendliness and commercial guardrails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nodes.poi_searcher import POISearcher
from src.nodes.route_planner import RoutePlanner


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "mock_data" / "gaode_seed"
DEFAULT_CITY = "上海"
DEFAULT_ACTIVITY_KEYWORDS = [
    "上海 亲子 陶艺",
    "上海 儿童 室内乐园",
    "上海 科学馆 亲子",
    "上海 Citywalk",
    "上海 周末 市集",
    "上海 飞盘 体验",
    "上海 温泉 康养",
]
DEFAULT_RESTAURANT_KEYWORDS = [
    "上海 轻食 日料",
    "上海 亲子餐厅",
    "上海 本帮菜 家庭餐厅",
    "上海 沙拉 轻食",
    "上海 火锅",
    "上海 炸鸡 外带",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Gaode POIs and enrich them into WeekendFlow offline supply mock files."
    )
    parser.add_argument("--city", default=DEFAULT_CITY, help="Gaode city name or adcode")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--activity-keyword", action="append", dest="activity_keywords")
    parser.add_argument("--restaurant-keyword", action="append", dest="restaurant_keywords")
    parser.add_argument("--pages", type=int, default=1, help="Pages per keyword")
    parser.add_argument("--offset", type=int, default=10, help="Gaode page size, max 25")
    parser.add_argument("--no-citylimit", action="store_true")
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--route-origin",
        default=None,
        help='Optional route origin coordinate, for example "121.4737,31.2304".',
    )
    parser.add_argument("--route-mode", default="driving", choices=["driving", "walking", "bicycling", "transit"])
    parser.add_argument("--route-city", default=DEFAULT_CITY, help="Required by Gaode transit routing.")
    parser.add_argument(
        "--route-limit",
        type=int,
        default=10,
        help="Maximum POIs to enrich with route data when --route-origin is set.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty report instead of failing when GAODE_API_KEY is missing.",
    )
    return parser.parse_args()


def require_api_key(args: argparse.Namespace) -> str | None:
    api_key = args.api_key or os.getenv("GAODE_API_KEY")
    if api_key and api_key != "自己的key":
        return api_key
    if args.allow_empty:
        return None
    raise SystemExit(
        "GAODE_API_KEY is missing. Set it before running, or pass --allow-empty "
        "to create an empty output report."
    )


def stable_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", value)
    return value.strip("_")[:48] or "poi"


def poi_identity(raw_poi: dict[str, Any], prefix: str) -> str:
    amap_id = raw_poi.get("id") or raw_poi.get("amap_id")
    if amap_id:
        return f"{prefix}_{amap_id}"
    return f"{prefix}_{stable_slug(raw_poi.get('name', 'unknown'))}"


def to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_activity_profile(keyword: str, poi: dict[str, Any]) -> dict[str, Any]:
    text = f"{keyword} {poi.get('name', '')} {poi.get('type', '')}"
    tags = {
        "functional": ["activity"],
        "aesthetic": [],
        "risk": [],
        "commercial": ["coupon_available"],
    }
    category = "leisure"
    sub_category = "general_activity"
    experience_type = "offline_experience"
    duration_min = 90
    price = 120.0
    reservation_required = True
    weather_sensitivity = "low"

    if any(token in text for token in ["亲子", "儿童", "孩子"]):
        category = "parent_child_activity"
        sub_category = "parent_child"
        experience_type = "hands_on_parent_child"
        tags["functional"].extend(["kid_friendly", "low_intensity", "family_friendly"])
        tags["aesthetic"].extend(["warm", "hands_on"])
        price = 160.0
    if any(token in text for token in ["室内", "乐园", "陶艺", "科学", "馆", "美术馆", "博物馆"]):
        tags["functional"].append("indoor")
        weather_sensitivity = "indoor_safe"
    if any(token in text for token in ["陶艺", "手作", "手工", "DIY", "diy"]):
        category = "handcraft"
        sub_category = "creative_handcraft"
        experience_type = "hands_on_workshop"
        tags["functional"].extend(["indoor", "hands_on", "date_friendly"])
        tags["aesthetic"].extend(["creative", "ritual", "warm"])
        duration_min = 120
        price = 168.0
    if any(token in text for token in ["博物馆", "美术馆", "科技馆", "科学馆", "展览"]):
        category = "museum"
        sub_category = "culture_exhibition"
        experience_type = "culture_learning"
        tags["functional"].extend(["indoor", "local_experience", "educational"])
        tags["aesthetic"].extend(["cultural", "quiet", "photogenic"])
        reservation_required = False
        duration_min = 120
        price = 60.0
    if any(
        token in text
        for token in [
            "Citywalk",
            "citywalk",
            "城市漫步",
            "市集",
            "街区",
            "武康路",
            "安福路",
            "愚园路",
            "田子坊",
            "新天地",
            "外滩",
            "衡山路",
        ]
    ):
        category = "citywalk" if "Citywalk" in text or "citywalk" in text else "local_market"
        sub_category = "local_experience"
        experience_type = "city_limited_local_experience"
        tags["functional"].extend(["nearby", "budget", "local_experience"])
        tags["aesthetic"].extend(["city_limited", "photogenic", "cultural"])
        tags["risk"].append("weather_sensitive")
        reservation_required = False
        weather_sensitivity = "medium"
        price = 50.0
    if any(token in text for token in ["飞盘", "运动", "骑行", "攀岩", "球馆", "公园"]):
        category = "sports"
        sub_category = "social_sports"
        experience_type = "social_sports"
        tags["functional"].extend(["sports", "social", "group_friendly"])
        tags["risk"].extend(["weather_sensitive", "not_low_intensity"])
        weather_sensitivity = "high"
        price = 88.0
    if any(token in text for token in ["温泉", "康养", "SPA", "spa"]):
        category = "micro_vacation"
        sub_category = "wellness_spa"
        experience_type = "wellness_micro_vacation"
        tags["functional"].extend(["low_intensity", "relaxation", "date_friendly"])
        tags["aesthetic"].extend(["healing", "quiet", "ritual"])
        duration_min = 150
        price = 238.0

    return {
        "category": category,
        "sub_category": sub_category,
        "experience_type": experience_type,
        "tags": dedupe_tag_groups(tags),
        "price": price,
        "duration_min": duration_min,
        "reservation_required": reservation_required,
        "weather_sensitivity": weather_sensitivity,
    }


def infer_restaurant_profile(keyword: str, poi: dict[str, Any]) -> dict[str, Any]:
    text = f"{keyword} {poi.get('name', '')} {poi.get('type', '')}"
    tags = {
        "functional": ["restaurant"],
        "aesthetic": [],
        "risk": [],
        "commercial": ["coupon_available"],
    }
    restaurant_category = "restaurant"
    avg_price_per_person = 80.0
    duration_min = 80
    health_tags: list[str] = []
    menu_health_options: list[str] = []
    service_mode = "dine_in"
    dine_in_available = True
    reservation_required = True

    if any(token in text for token in ["轻食", "沙拉", "低卡", "健康"]):
        restaurant_category = "salad_light_food" if "沙拉" in text else "light_food"
        tags["functional"].extend(["low_calorie", "light_food", "healthy"])
        tags["aesthetic"].extend(["clean", "simple"])
        health_tags.extend(["low_calorie", "low_oil", "low_sugar", "vegetable_rich"])
        menu_health_options.extend(["低油主食", "低糖饮品", "蔬菜增量"])
        avg_price_per_person = 65.0
    if "日料" in text or "日本" in text:
        restaurant_category = "japanese_light_food"
        tags["functional"].extend(["japanese", "low_calorie", "light_food"])
        health_tags.extend(["low_oil", "high_protein"])
        menu_health_options.extend(["刺身饭", "少油茶泡饭"])
        avg_price_per_person = 110.0
    if any(token in text for token in ["亲子", "家庭", "儿童"]):
        restaurant_category = "family_bistro"
        tags["functional"].extend(["family_friendly", "kid_friendly", "child_seat"])
        menu_health_options.extend(["儿童餐", "蒸蛋"])
        avg_price_per_person = 85.0
    if any(token in text for token in ["本帮", "地方菜", "家常"]):
        restaurant_category = "regional_home_cuisine"
        tags["functional"].extend(["local_cuisine", "family_friendly"])
        tags["aesthetic"].extend(["local", "warm"])
        health_tags.extend(["low_oil_available", "fresh_ingredients"])
        menu_health_options.extend(["少油清蒸菜", "时令蔬菜"])
        avg_price_per_person = 90.0
    if "火锅" in text:
        restaurant_category = "hotpot"
        tags["functional"].extend(["hotpot", "social"])
        tags["aesthetic"].extend(["lively", "popular"])
        tags["risk"].extend(["high_calorie", "long_queue"])
        health_tags.extend(["high_calorie", "high_sodium"])
        menu_health_options.extend(["清汤锅底", "低糖茶饮"])
        avg_price_per_person = 120.0
        duration_min = 120
    if any(token in text for token in ["炸鸡", "外带", "小吃"]):
        restaurant_category = "fried_chicken"
        tags["functional"].extend(["budget", "fast_food", "takeaway"])
        tags["risk"].extend(["high_calorie", "limited_seats"])
        health_tags.extend(["high_calorie", "high_oil"])
        menu_health_options.append("无糖茶饮")
        avg_price_per_person = 45.0
        duration_min = 35
        reservation_required = False
        if "外带" in text:
            service_mode = "takeaway_only"
            dine_in_available = False

    return {
        "restaurant_category": restaurant_category,
        "avg_price_per_person": avg_price_per_person,
        "category_price_band": price_band(avg_price_per_person),
        "tags": dedupe_tag_groups(tags),
        "price": avg_price_per_person * 2,
        "duration_min": duration_min,
        "health_tags": sorted(set(health_tags)),
        "menu_health_options": sorted(set(menu_health_options)),
        "service_mode": service_mode,
        "dine_in_available": dine_in_available,
        "reservation_required": reservation_required,
    }


def dedupe_tag_groups(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    result = {}
    for key, values in groups.items():
        seen = set()
        deduped = []
        for value in values:
            if value not in seen:
                seen.add(value)
                deduped.append(value)
        result[key] = deduped
    return result


def price_band(avg_price: float) -> str:
    if avg_price <= 60:
        return "budget"
    if avg_price <= 120:
        return "mid"
    return "mid_high"


def parse_coordinates(location: str | None) -> tuple[float | None, float | None]:
    if not location or "," not in location:
        return None, None
    lng, lat = location.split(",", 1)
    return to_float(lng, None), to_float(lat, None)


def build_common_fields(
    poi: dict[str, Any],
    *,
    expected_type: str,
    keyword: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    biz_ext = poi.get("biz_ext", {}) if isinstance(poi.get("biz_ext"), dict) else {}
    amap_id = poi.get("id")
    prefix = "gaode_act" if expected_type == "activity" else "gaode_res"
    poi_id = poi_identity(poi, prefix)
    lng, lat = parse_coordinates(poi.get("location"))
    rating = to_float(biz_ext.get("rating"), 4.2)
    price = to_float(biz_ext.get("cost"), profile.get("price", 120.0))

    base = {
        "poi_id": poi_id,
        "amap_id": amap_id,
        "merchant_id": f"m_{poi_id}",
        "name": poi.get("name") or poi_id,
        "type": expected_type,
        "tags": profile["tags"],
        "price": round(price, 2),
        "distance_km": 3.0,
        "duration_min": profile["duration_min"],
        "rating": round(rating, 2),
        "queue_time_min": 10,
        "available": True,
        "available_slots": default_slots(expected_type),
        "inventory_left": 20 if expected_type == "restaurant" else 30,
        "capacity_limit": 30 if expected_type == "restaurant" else 60,
        "reservation_required": profile["reservation_required"],
        "location": poi.get("address") or "",
        "coordinates": poi.get("location"),
        "latitude": lat,
        "longitude": lng,
        "source": "gaode_seed_enriched",
        "source_channel": "gaode_poi_search",
        "source_evidence": [
            f"Gaode POI keyword={keyword}; amap_id={amap_id}; adcode={poi.get('adcode')}"
        ],
        "gaode_keyword": keyword,
        "gaode_type": poi.get("type"),
        "adcode": poi.get("adcode"),
        "citycode": poi.get("citycode"),
        "tel": poi.get("tel"),
        "trust_score": 0.74 if not rating else min(0.92, 0.55 + rating / 10),
        "verified_reviews": 80,
        "review_count": 120,
        "content_heat_score": 0.55,
        "operation_stability_score": 0.82,
        "business_hours": {"weekday": "10:00-21:00", "weekend": "10:00-22:00"},
        "holiday_status": {"open": True, "reservation_required": profile["reservation_required"], "peak_risk": "medium"},
        "refund_policy": "before_1h_free" if profile["reservation_required"] else "anytime_refund_before_use",
        "cancel_policy": "before_1h_free" if profile["reservation_required"] else "anytime_refund_before_use",
        "hidden_cost_risk": "low",
        "commercial_features": {
            "coupon_available": True,
            "commission_rate": 0.06,
            "ad_boost": 0.0,
            "redemption_rate": 0.6,
        },
        "raw": poi,
    }
    base.update({k: v for k, v in profile.items() if k != "tags"})
    return base


def default_slots(expected_type: str) -> list[dict[str, Any]]:
    if expected_type == "activity":
        return [
            {"time": "14:00", "inventory_left": 30},
            {"time": "15:30", "inventory_left": 24},
            {"time": "17:00", "inventory_left": 18},
        ]
    return [
        {"time": "17:30", "inventory_left": 20},
        {"time": "18:30", "inventory_left": 16},
        {"time": "19:30", "inventory_left": 12},
    ]


def make_deal(item: dict[str, Any], expected_type: str) -> dict[str, Any]:
    deal_id = f"deal_{item['poi_id']}"
    product_id = f"prod_{item['poi_id']}"
    return {
        "deal_id": deal_id,
        "poi_id": item["poi_id"],
        "product_id": product_id,
        "deal_type": "activity_ticket" if expected_type == "activity" else "meal_coupon",
        "coupon_type": item.get("category") or item.get("restaurant_category", expected_type),
        "title": f"{item['name']}体验券" if expected_type == "activity" else f"{item['name']}双人套餐",
        "sale_price": item["price"],
        "original_price": round(item["price"] * 1.2, 2),
        "requires_reservation": item.get("reservation_required", True),
        "refund_policy": item.get("refund_policy", "before_1h_free"),
        "cancel_policy": item.get("cancel_policy", "before_1h_free"),
        "valid_time": [slot["time"] for slot in item.get("available_slots", []) if slot.get("time")],
        "package_components": package_components(item, expected_type),
        "target_persona": target_persona(item),
        "family_ticket": "family_friendly" in flatten_tags(item.get("tags", {})),
        "redemption_rate": 0.6,
        "commission_rate": 0.06,
        "stock_limit_per_slot": item.get("capacity_limit", 30),
        "hidden_cost_risk": item.get("hidden_cost_risk", "low"),
    }


def make_product(item: dict[str, Any], expected_type: str) -> dict[str, Any]:
    product_id = f"prod_{item['poi_id']}"
    return {
        "product_id": product_id,
        "poi_id": item["poi_id"],
        "merchant_id": item["merchant_id"],
        "product_type": "activity_ticket" if expected_type == "activity" else "meal_package",
        "name": f"{item['name']}产品",
        "target_persona": target_persona(item),
        "price": item["price"],
        "duration_min": item.get("duration_min"),
        "requires_reservation": item.get("reservation_required", True),
        "inventory_model": "slot_capacity" if expected_type == "activity" else "table_slot",
        "package_components": package_components(item, expected_type),
        "health_tags": item.get("health_tags", []),
        "refund_policy": item.get("refund_policy"),
        "fulfillment_mode": "onsite_verify" if expected_type == "activity" else (
            "takeaway_only" if not item.get("dine_in_available", True) else "dine_in_reservation"
        ),
    }


def make_merchant(item: dict[str, Any], expected_type: str) -> dict[str, Any]:
    return {
        "merchant_id": item["merchant_id"],
        "name": item["name"],
        "merchant_type": "activity_provider" if expected_type == "activity" else "restaurant",
        "poi_ids": [item["poi_id"]],
        "source_channel": "gaode_poi_search",
        "trust_score": item.get("trust_score", 0.75),
        "review_count": item.get("review_count", 120),
        "recent_order_count": 40,
        "reservation_count": 20 if item.get("reservation_required") else 0,
        "operation_stability_score": item.get("operation_stability_score", 0.82),
        "business_capabilities": merchant_capabilities(item, expected_type),
        "service_risks": item.get("tags", {}).get("risk", []),
        "gaode_fields_available": ["id", "name", "address", "location", "tel", "adcode", "biz_ext"],
        "meituan_mock_fields_owned": ["inventory", "deal", "refund_policy", "queue", "reservation_slots"],
    }


def make_availability(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": item.get("available", True),
        "available_slots": item.get("available_slots", []),
        "queue_time_min": item.get("queue_time_min", 0),
        "inventory_left": item.get("inventory_left", 0),
        "capacity_limit": item.get("capacity_limit", 0),
        "reservation_required": item.get("reservation_required", True),
        "failure_reason": None,
        "holiday_status": item.get("holiday_status"),
    }


def build_route_overlays(
    planner: RoutePlanner,
    items: list[dict[str, Any]],
    *,
    origin: str,
    mode: str,
    city: str,
    limit: int,
) -> dict[str, Any]:
    route_overlays: dict[str, Any] = {}
    calls_left = max(0, limit)
    for item in items:
        if calls_left <= 0:
            break
        destination = item.get("coordinates")
        if not destination:
            continue

        try:
            kwargs = {"city": city} if mode == "transit" else {}
            route = planner.plan(origin, destination, mode=mode, **kwargs)
            calls_left -= 1
        except Exception as exc:  # Keep seed generation resilient to route misses.
            route = {"feasible": False, "reason": str(exc), "mode": mode}

        duration_seconds = route.get("duration", 0) or 0
        distance_meters = route.get("distance", 0) or 0
        route_overlays[item["poi_id"]] = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "feasible": route.get("feasible", False),
            "duration_min": round(duration_seconds / 60, 1) if duration_seconds else None,
            "duration_text": route.get("duration_text"),
            "distance_km": round(distance_meters / 1000, 2) if distance_meters else None,
            "distance_text": route.get("distance_text"),
            "segments": route.get("segments"),
            "steps": route.get("steps"),
            "walking_distance_m": route.get("walking_distance"),
            "traffic_lights": route.get("traffic_lights"),
            "tolls": route.get("tolls"),
            "nightflag": route.get("nightflag"),
            "reason": route.get("reason"),
            "source": "gaode_route_planner",
        }
    return route_overlays


def flatten_tags(tag_groups: dict[str, Any]) -> list[str]:
    values = []
    if isinstance(tag_groups, dict):
        for group in tag_groups.values():
            if isinstance(group, list):
                values.extend(group)
            elif group:
                values.append(str(group))
    return values


def target_persona(item: dict[str, Any]) -> list[str]:
    tags = flatten_tags(item.get("tags", {}))
    personas = []
    for tag, persona in [
        ("kid_friendly", "family"),
        ("family_friendly", "family"),
        ("low_calorie", "low_calorie"),
        ("local_experience", "local_culture"),
        ("social", "friends"),
        ("relaxation", "relaxation"),
        ("hotpot", "social"),
        ("takeaway", "takeaway"),
    ]:
        if tag in tags and persona not in personas:
            personas.append(persona)
    return personas or ["general"]


def package_components(item: dict[str, Any], expected_type: str) -> list[str]:
    if expected_type == "activity":
        return ["预约名额", "现场体验", "基础服务"]
    if item.get("menu_health_options"):
        return list(item["menu_health_options"])[:3]
    return ["双人主食", "饮品", "基础服务"]


def merchant_capabilities(item: dict[str, Any], expected_type: str) -> list[str]:
    caps = ["reservation" if item.get("reservation_required") else "walk_in"]
    if expected_type == "restaurant":
        if item.get("dine_in_available", True):
            caps.append("dine_in")
        if item.get("takeaway_available", True):
            caps.append("takeaway")
        if item.get("delivery_available", True):
            caps.append("delivery")
    else:
        caps.append("onsite_verify")
    return caps


def dedupe_pois(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        key = item.get("amap_id") or item.get("name")
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def fetch_group(
    searcher: POISearcher,
    *,
    keywords: list[str],
    expected_type: str,
    city: str,
    citylimit: bool,
    pages: int,
    offset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_records = []
    enriched = []
    for keyword in keywords:
        for page in range(1, pages + 1):
            pois = searcher.search(
                keywords=keyword,
                city=city,
                citylimit=citylimit,
                page=page,
                offset=offset,
            )
            # Some Gaode v5 responses may return more rows than requested by the
            # wrapper's offset parameter. Keep seed generation bounded locally so
            # smoke runs do not produce unexpectedly large mock sets.
            pois = pois[:offset]
            raw_records.append({"keyword": keyword, "page": page, "pois": pois})
            for poi in pois:
                profile = (
                    infer_activity_profile(keyword, poi)
                    if expected_type == "activity"
                    else infer_restaurant_profile(keyword, poi)
                )
                item = build_common_fields(
                    poi,
                    expected_type=expected_type,
                    keyword=keyword,
                    profile=profile,
                )
                enriched.append(item)
    return raw_records, dedupe_pois(enriched)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_readme(output_dir: Path, report: dict[str, Any]) -> None:
    text = f"""# Gaode Seed Supply

Generated at: {report['generated_at']}

City: {report['city']}

This directory is an offline seed built from Gaode POI search and WeekendFlow
business enrichment rules. It should be inspected before merging into the
curated `experiments/mock_data` files.

Files:

- `gaode_raw_pois.json`: Raw grouped POI search results by keyword/page.
- `activities.json`: B-normalized activity candidates enriched from Gaode POIs.
- `restaurants.json`: B-normalized restaurant candidates enriched from Gaode POIs.
- `availability.json`: Synthetic slot/capacity overlays keyed by `poi_id`.
- `deals.json`: Synthetic deal/coupon/package records.
- `products.json`: Synthetic product/package records.
- `merchants.json`: Merchant capability/trust records.
- `routes.json`: Optional route overlays from a supplied origin to selected POIs.
- `build_report.json`: Counts, keywords and run metadata.

Important: Gaode supplies base place facts. Inventory, packages, booking,
health suitability and commercial data are generated by local enrichment rules.
Route overlays are only generated when `--route-origin` is supplied, so search
and route quotas can be controlled independently.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    api_key = require_api_key(args)
    activity_keywords = args.activity_keywords or DEFAULT_ACTIVITY_KEYWORDS
    restaurant_keywords = args.restaurant_keywords or DEFAULT_RESTAURANT_KEYWORDS
    output_dir = args.output_dir

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "city": args.city,
        "output_dir": str(output_dir),
        "activity_keywords": activity_keywords,
        "restaurant_keywords": restaurant_keywords,
        "has_api_key": bool(api_key),
    }

    if not api_key:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.update({"activity_count": 0, "restaurant_count": 0, "error": "GAODE_API_KEY missing"})
        write_json(output_dir / "build_report.json", report)
        write_readme(output_dir, report)
        print("GAODE_API_KEY missing; wrote empty report because --allow-empty was set.")
        return 0

    searcher = POISearcher(api_key=api_key)
    citylimit = not args.no_citylimit

    activity_raw, activities = fetch_group(
        searcher,
        keywords=activity_keywords,
        expected_type="activity",
        city=args.city,
        citylimit=citylimit,
        pages=max(1, args.pages),
        offset=max(1, min(args.offset, 25)),
    )
    restaurant_raw, restaurants = fetch_group(
        searcher,
        keywords=restaurant_keywords,
        expected_type="restaurant",
        city=args.city,
        citylimit=citylimit,
        pages=max(1, args.pages),
        offset=max(1, min(args.offset, 25)),
    )

    all_items = activities + restaurants
    deals = [make_deal(item, item["type"]) for item in all_items]
    products = [make_product(item, item["type"]) for item in all_items]
    merchants = [make_merchant(item, item["type"]) for item in all_items]
    availability = {item["poi_id"]: make_availability(item) for item in all_items}
    route_overlays = {}

    if args.route_origin:
        planner = RoutePlanner(api_key=api_key)
        route_overlays = build_route_overlays(
            planner,
            all_items,
            origin=args.route_origin,
            mode=args.route_mode,
            city=args.route_city,
            limit=args.route_limit,
        )

    for item in all_items:
        item["deal_ids"] = [f"deal_{item['poi_id']}"]
        item["product_ids"] = [f"prod_{item['poi_id']}"]

    report.update(
        {
            "activity_count": len(activities),
            "restaurant_count": len(restaurants),
            "deal_count": len(deals),
            "product_count": len(products),
            "merchant_count": len(merchants),
            "route_count": len(route_overlays),
            "route_origin": args.route_origin,
            "route_mode": args.route_mode if args.route_origin else None,
        }
    )

    write_json(output_dir / "gaode_raw_pois.json", {"activities": activity_raw, "restaurants": restaurant_raw})
    write_json(output_dir / "activities.json", activities)
    write_json(output_dir / "restaurants.json", restaurants)
    write_json(output_dir / "availability.json", availability)
    write_json(output_dir / "deals.json", deals)
    write_json(output_dir / "products.json", products)
    write_json(output_dir / "merchants.json", merchants)
    write_json(output_dir / "routes.json", route_overlays)
    write_json(output_dir / "build_report.json", report)
    write_readme(output_dir, report)

    print(
        f"Wrote {len(activities)} activities, {len(restaurants)} restaurants, "
        f"{len(deals)} deals to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
