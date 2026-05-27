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
from src.nodes.poi_cleaning import should_exclude_poi
from src.nodes.route_planner import RoutePlanner
from src.nodes.b_utils import (
    expand_preference_tags,
    to_chinese_tag_groups,
    to_chinese_tags,
    to_chinese_value,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "mock_data" / "gaode_seed"
DEFAULT_CITY = "上海"
CHINESE_SCALAR_FIELDS = {
    "category",
    "sub_category",
    "experience_type",
    "restaurant_category",
    "category_price_band",
    "weather_sensitivity",
    "traffic_risk",
    "service_mode",
    "business_format",
    "hidden_cost_risk",
    "refund_policy",
    "cancel_policy",
}
CHINESE_LIST_FIELDS = {
    "health_tags",
    "menu_health_options",
    "emotion_tags",
    "atmosphere_tags",
    "local_character_tags",
    "local_flavor_tags",
    "risk_flags",
    "service_risks",
    "business_capabilities",
    "target_persona",
    "package_components",
}
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
    parser.add_argument("--activity-keywords-file", type=Path, default=None)
    parser.add_argument("--restaurant-keywords-file", type=Path, default=None)
    parser.add_argument("--pages", type=int, default=1, help="Pages per keyword")
    parser.add_argument("--offset", type=int, default=10, help="Gaode page size, max 25")
    parser.add_argument(
        "--max-search-calls",
        type=int,
        default=None,
        help=(
            "Hard cap for POI keyword-search calls in this run. "
            "Use this to stay below the daily Gaode quota."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful records already saved in the output directory and skip those API calls.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop on the first POI search error. By default errors are recorded and later calls continue.",
    )
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


def read_keyword_file(path: Path | None) -> list[str]:
    if not path:
        return []
    if not path.exists():
        raise SystemExit(f"Keyword file not found: {path}")
    keywords = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        keywords.append(line)
    return keywords


def redact_secret(value: Any) -> str:
    text = str(value)
    return re.sub(r"([?&]key=)[^&\s]+", r"\1<redacted>", text)


def redact_json_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: redact_json_secrets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_json_secrets(item) for item in value]
    if isinstance(value, str):
        return redact_secret(value)
    return value


def resolve_keywords(
    keyword_file: Path | None,
    cli_keywords: list[str] | None,
    default_keywords: list[str],
) -> list[str]:
    if keyword_file is not None:
        return read_keyword_file(keyword_file)
    return cli_keywords or default_keywords


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
            "集市",
            "夜市",
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
    if any(token in text for token in ["密室", "密室逃脱", "剧本杀", "推理馆", "沉浸式推理", "escape room"]):
        category = "escape_room"
        sub_category = "immersive_puzzle"
        experience_type = "immersive_social_game"
        tags["functional"].extend(["indoor", "social", "group_friendly", "reservation_recommended"])
        tags["aesthetic"].extend(["immersive", "story_driven"])
        tags["risk"].append("time_slot_sensitive")
        duration_min = 120
        price = 168.0
    if any(token in text for token in ["飞盘", "运动", "骑行", "攀岩", "球馆", "公园"]):
        category = "sports"
        sub_category = "social_sports"
        experience_type = "social_sports"
        tags["functional"].extend(["sports", "social", "group_friendly"])
        tags["risk"].extend(["weather_sensitive", "not_low_intensity"])
        weather_sensitivity = "high"
        price = 88.0
    if any(
        token in text
        for token in [
            "温泉",
            "康养",
            "SPA",
            "spa",
            "疗愈",
            "芳疗",
            "瑜伽",
            "冥想",
            "茶空间",
            "双人放松",
            "微度假",
            "近场度假",
        ]
    ):
        category = "micro_vacation"
        sub_category = "wellness_spa"
        experience_type = "wellness_micro_vacation"
        tags["functional"].extend(
            ["low_intensity", "relaxation", "date_friendly", "micro_vacation", "wellness", "spa"]
        )
        tags["aesthetic"].extend(["healing", "quiet", "ritual", "wellness_micro_vacation"])
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
    if any(token in text for token in ["烤肉", "烧烤", "炭烤", "韩式烤肉", "巴西牛排"]):
        restaurant_category = "barbecue"
        tags["functional"].extend(["barbecue", "meat", "social"])
        tags["aesthetic"].extend(["lively", "popular"])
        tags["risk"].extend(["high_calorie", "long_queue", "smoke_smell"])
        health_tags.extend(["high_calorie", "high_oil"])
        menu_health_options.extend(["生菜包肉", "蔬菜拼盘", "无糖茶饮"])
        avg_price_per_person = 125.0
        duration_min = 110
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


def apply_chinese_first_contract(item: dict[str, Any]) -> dict[str, Any]:
    """Persist semantic mock fields in Chinese; B can normalize them internally."""

    if "tags" in item:
        item["tags"] = to_chinese_tag_groups(item["tags"])
    for field_name in CHINESE_SCALAR_FIELDS:
        if field_name in item and item[field_name] not in (None, ""):
            item[field_name] = to_chinese_value(item[field_name])
    for field_name in CHINESE_LIST_FIELDS:
        if field_name in item:
            item[field_name] = to_chinese_tags(item[field_name])
    return item


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
    base.update(build_rich_mock_overlay(base, expected_type=expected_type, keyword=keyword, profile=profile))
    return apply_chinese_first_contract(base)


def stable_bucket(*parts: Any, modulo: int = 100) -> int:
    text = "|".join(str(part or "") for part in parts)
    return sum(ord(ch) for ch in text) % modulo


def score_near(rating: float, salt: str, *, low: float = 3.8, high: float = 4.9) -> float:
    offset = (stable_bucket(salt, modulo=9) - 4) * 0.04
    return round(max(low, min(high, rating + offset)), 1)


def build_rich_mock_overlay(
    item: dict[str, Any],
    *,
    expected_type: str,
    keyword: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    if expected_type == "restaurant":
        return build_restaurant_detail_mock(item, keyword=keyword, profile=profile)
    return build_activity_detail_mock(item, keyword=keyword, profile=profile)


def build_restaurant_detail_mock(item: dict[str, Any], *, keyword: str, profile: dict[str, Any]) -> dict[str, Any]:
    name = str(item.get("name") or "")
    rating = float(item.get("rating") or 4.2)
    category = str(profile.get("restaurant_category") or "restaurant")
    dishes = restaurant_dish_profile(category, keyword, name)
    review_breakdown = {
        "口味": score_near(rating, name + "taste"),
        "服务": score_near(rating, name + "service"),
        "环境": score_near(rating, name + "environment"),
        "食材": score_near(rating, name + "ingredient"),
    }
    family_friendly = category == "family_bistro" or any(token in f"{keyword} {name}" for token in ["亲子", "儿童", "家庭", "带娃"])
    light_food = category in {"light_food", "salad_light_food", "japanese_light_food"}
    high_queue = category in {"hotpot", "barbecue"} or stable_bucket(name, "queue", modulo=10) >= 7
    avg_price = float(profile.get("avg_price_per_person") or item.get("price") or 80)
    baby_chair_available = family_friendly or stable_bucket(name, "baby", modulo=10) >= 6
    parking_available = stable_bucket(name, "parking", modulo=10) >= 4
    can_reserve = bool(profile.get("reservation_required", True))
    package_options = restaurant_package_options(
        item,
        category=category,
        dishes=dishes,
        avg_price_per_person=avg_price,
        family_friendly=family_friendly,
        light_food=light_food,
    )

    return {
        "review_breakdown": review_breakdown,
        "review_keywords": restaurant_review_keywords(category, light_food=light_food, family_friendly=family_friendly),
        "ugc_summary": restaurant_ugc_summary(category, dishes, high_queue=high_queue, light_food=light_food),
        "signature_dishes": dishes["signature"],
        "recommended_dishes": dishes["recommended"],
        "dish_tags": dishes["tags"],
        "package_options": package_options,
        "promotion_highlights": promotion_highlights(package_options, category),
        "baby_chair_available": baby_chair_available,
        "child_menu": family_friendly,
        "parking_available": parking_available,
        "parking_fee_policy": "商场停车可抵扣" if parking_available and stable_bucket(name, "mall", modulo=2) else ("附近有收费停车场" if parking_available else "停车不稳定"),
        "reservation_slots": reservation_slots(item, can_reserve=can_reserve),
        "queue_time_by_period": queue_time_by_period(category, base_queue=int(item.get("queue_time_min") or 10)),
        "serving_speed_min": serving_speed_min(category),
        "seat_capacity": seat_capacity(category, family_friendly=family_friendly),
        "private_room_available": category in {"regional_home_cuisine", "hotpot"} and stable_bucket(name, "room", modulo=10) >= 5,
        "noise_level": noise_level(category),
        "spice_level": spice_level(category),
        "booking_policy": "建议提前订座" if can_reserve else "到店排队或外带为主",
        "dietary_options": dietary_options(category, light_food=light_food),
        "allergen_notes": allergen_notes(category),
        "decision_profile": restaurant_decision_profile(
            category,
            family_friendly=family_friendly,
            light_food=light_food,
            high_queue=high_queue,
        ),
        "fulfillment_actions": restaurant_fulfillment_actions(item, category=category, has_coupon=True),
        "substitution_strategy": restaurant_substitution_strategy(category),
        "peak_risk_profile": peak_risk_profile(category, high_queue=high_queue),
        "mock_detail_sources": {
            "review_breakdown": "rating_based_rule_imputation",
            "dishes": "category_keyword_rule_imputation",
            "packages": "price_category_rule_imputation",
            "facilities": "category_probability_rule_imputation",
        },
    }


def build_activity_detail_mock(item: dict[str, Any], *, keyword: str, profile: dict[str, Any]) -> dict[str, Any]:
    name = str(item.get("name") or "")
    rating = float(item.get("rating") or 4.2)
    category = str(profile.get("category") or "leisure")
    family_activity = category == "parent_child_activity" or any(token in f"{keyword} {name}" for token in ["亲子", "儿童", "孩子"])
    indoor = any(token in f"{keyword} {name}" for token in ["室内", "馆", "手作", "陶艺", "博物馆", "美术馆", "密室", "桌游"])
    package_options = activity_package_options(item, category=category, family_activity=family_activity)
    return {
        "review_breakdown": {
            "体验": score_near(rating, name + "experience"),
            "服务": score_near(rating, name + "service"),
            "环境": score_near(rating, name + "environment"),
            "安全": score_near(rating, name + "safety"),
        },
        "review_keywords": activity_review_keywords(category, family_activity=family_activity, indoor=indoor),
        "ugc_summary": activity_ugc_summary(category, family_activity=family_activity, indoor=indoor),
        "package_options": package_options,
        "promotion_highlights": promotion_highlights(package_options, category),
        "parking_available": stable_bucket(name, "parking", modulo=10) >= 5,
        "parking_fee_policy": "附近商业体停车" if stable_bucket(name, "parking", modulo=10) >= 5 else "建议地铁/打车前往",
        "suitable_age": "3-8岁" if family_activity else "成人/朋友同行",
        "age_range": {"min": 3, "max": 8} if family_activity else {"min": 12, "max": 60},
        "indoor_backup": indoor,
        "guide_available": category in {"museum", "handcraft", "parent_child_activity"},
        "equipment_rental": category in {"sports", "handcraft"},
        "reservation_slots": reservation_slots(item, can_reserve=bool(profile.get("reservation_required", True))),
        "queue_time_by_period": queue_time_by_period(category, base_queue=int(item.get("queue_time_min") or 10)),
        "service_facilities": activity_facilities(category, family_activity=family_activity, indoor=indoor),
        "physical_intensity": physical_intensity(category, family_activity=family_activity),
        "weather_plan": weather_plan(category, indoor=indoor),
        "decision_profile": activity_decision_profile(category, family_activity=family_activity, indoor=indoor),
        "fulfillment_actions": activity_fulfillment_actions(item, category=category),
        "substitution_strategy": activity_substitution_strategy(category),
        "peak_risk_profile": peak_risk_profile(category, high_queue=False),
        "mock_detail_sources": {
            "review_breakdown": "rating_based_rule_imputation",
            "packages": "category_rule_imputation",
            "facilities": "category_probability_rule_imputation",
        },
    }


def restaurant_dish_profile(category: str, keyword: str, name: str) -> dict[str, list[str]]:
    text = f"{category} {keyword} {name}"
    if category == "barbecue" or any(token in text for token in ["烤肉", "烧烤", "炭烤"]):
        return {
            "signature": ["招牌牛五花", "秘制横膈膜", "生菜包肉"],
            "recommended": ["烤肉拼盘", "蔬菜拼盘", "无糖乌龙茶"],
            "tags": ["肉食", "适合朋友聚餐", "热闹"],
        }
    if category == "hotpot" or "火锅" in text:
        return {
            "signature": ["鸳鸯锅底", "鲜切牛肉", "手打虾滑"],
            "recommended": ["清汤锅底", "菌菇拼盘", "低糖茶饮"],
            "tags": ["聚餐", "热闹", "可选清淡锅底"],
        }
    if category in {"light_food", "salad_light_food"} or any(token in text for token in ["轻食", "沙拉", "健身餐", "低卡"]):
        return {
            "signature": ["鸡胸牛油果沙拉", "低脂能量碗", "贝果三明治"],
            "recommended": ["少酱沙拉", "低糖酸奶碗", "黑咖啡"],
            "tags": ["低卡", "清爽", "适合减脂"],
        }
    if category == "japanese_light_food" or any(token in text for token in ["日料", "日本料理", "寿司"]):
        return {
            "signature": ["刺身饭", "寿司拼盘", "茶泡饭"],
            "recommended": ["少油茶泡饭", "烤鱼定食", "味噌汤"],
            "tags": ["高蛋白", "清淡", "约会"],
        }
    if category == "family_bistro" or any(token in text for token in ["亲子", "家庭", "儿童"]):
        return {
            "signature": ["儿童蒸蛋", "番茄肉酱意面", "家庭分享拼盘"],
            "recommended": ["儿童餐", "少油炒时蔬", "玉米汁"],
            "tags": ["儿童友好", "家庭聚餐", "口味温和"],
        }
    if category == "regional_home_cuisine" or any(token in text for token in ["本帮", "家常", "江浙"]):
        return {
            "signature": ["本帮红烧肉", "清蒸时蔬", "葱油拌面"],
            "recommended": ["少油清蒸鱼", "时令蔬菜", "老上海点心"],
            "tags": ["家常", "本地风味", "适合家庭"],
        }
    if category == "fried_chicken" or any(token in text for token in ["炸鸡", "汉堡", "小吃"]):
        return {
            "signature": ["招牌炸鸡", "蜂蜜芥末鸡块", "薯条"],
            "recommended": ["无糖茶饮", "小份炸鸡", "外带套餐"],
            "tags": ["快餐", "高热量", "适合外带"],
        }
    return {
        "signature": ["招牌套餐", "时令主菜", "人气饮品"],
        "recommended": ["双人套餐", "清爽饮品", "少油可备注"],
        "tags": ["日常餐饮", "可团购", "适合就近安排"],
    }


def restaurant_review_keywords(category: str, *, light_food: bool, family_friendly: bool) -> list[str]:
    if light_food:
        return ["清爽", "低负担", "出餐快", "适合减脂"]
    if family_friendly:
        return ["带娃方便", "服务耐心", "口味温和", "座位宽敞"]
    if category in {"hotpot", "barbecue"}:
        return ["人气高", "适合聚餐", "晚高峰排队", "口味稳定"]
    if category == "regional_home_cuisine":
        return ["家常口味", "适合家庭", "本地特色", "可少油备注"]
    return ["口味稳定", "位置方便", "套餐划算", "服务正常"]


def restaurant_ugc_summary(category: str, dishes: dict[str, list[str]], *, high_queue: bool, light_food: bool) -> str:
    if light_food:
        return f"网友常点{dishes['recommended'][0]}，评价集中在清爽、分量适中、适合减脂期。"
    if high_queue:
        return f"网友推荐{dishes['signature'][0]}，晚高峰人气较高，建议提前订座或错峰到店。"
    return f"网友推荐{dishes['signature'][0]}，整体评价偏稳定，适合放入低决策成本方案。"


def restaurant_package_options(
    item: dict[str, Any],
    *,
    category: str,
    dishes: dict[str, list[str]],
    avg_price_per_person: float,
    family_friendly: bool,
    light_food: bool,
) -> list[dict[str, Any]]:
    two_person_price = round(avg_price_per_person * (1.72 if not light_food else 1.55), 2)
    family_price = round(avg_price_per_person * (2.85 if family_friendly else 2.45), 2)
    options = [
        {
            "name": "双人精选套餐",
            "people_count": 2,
            "sale_price": two_person_price,
            "original_price": round(two_person_price * 1.22, 2),
            "components": [dishes["signature"][0], dishes["recommended"][0], "饮品/小食"],
            "coupon_available": True,
            "reservation_required": item.get("reservation_required", True),
        },
        {
            "name": "工作日错峰券",
            "people_count": 1,
            "sale_price": round(avg_price_per_person * 0.82, 2),
            "original_price": round(avg_price_per_person, 2),
            "components": [dishes["recommended"][0], "指定饮品"],
            "coupon_available": True,
            "reservation_required": False,
        },
    ]
    if family_friendly:
        options.append(
            {
                "name": "家庭三人套餐",
                "people_count": 3,
                "sale_price": family_price,
                "original_price": round(family_price * 1.18, 2),
                "components": [dishes["signature"][0], "儿童餐", "少油时蔬"],
                "coupon_available": True,
                "reservation_required": True,
            }
        )
    return options


def activity_package_options(item: dict[str, Any], *, category: str, family_activity: bool) -> list[dict[str, Any]]:
    price = float(item.get("price") or 120)
    options = [
        {
            "name": "单人体验票",
            "people_count": 1,
            "sale_price": round(price, 2),
            "original_price": round(price * 1.15, 2),
            "components": ["预约名额", "现场体验", "基础材料/服务"],
            "coupon_available": True,
            "reservation_required": item.get("reservation_required", True),
        }
    ]
    if family_activity:
        options.append(
            {
                "name": "亲子陪伴套票",
                "people_count": 2,
                "sale_price": round(price * 1.65, 2),
                "original_price": round(price * 1.95, 2),
                "components": ["儿童体验名额", "成人陪同", "基础材料"],
                "coupon_available": True,
                "reservation_required": True,
            }
        )
    if category in {"handcraft", "museum", "micro_vacation", "sports"}:
        options.append(
            {
                "name": "双人轻体验套餐",
                "people_count": 2,
                "sale_price": round(price * 1.75, 2),
                "original_price": round(price * 2.05, 2),
                "components": ["双人预约名额", "基础服务", "到店核销"],
                "coupon_available": True,
                "reservation_required": item.get("reservation_required", True),
            }
        )
    return options


def promotion_highlights(package_options: list[dict[str, Any]], category: str) -> list[str]:
    highlights = ["支持团购券" if any(option.get("coupon_available") for option in package_options) else "暂无可用券"]
    if len(package_options) >= 2:
        highlights.append("有错峰优惠")
    if category in {"light_food", "salad_light_food", "japanese_light_food"}:
        highlights.append("可备注少油少酱")
    if category in {"family_bistro", "parent_child_activity"}:
        highlights.append("适合亲子同行")
    return highlights


def reservation_slots(item: dict[str, Any], *, can_reserve: bool) -> list[dict[str, Any]]:
    slots = []
    for slot in item.get("available_slots", []) or []:
        time = slot.get("time")
        if not time:
            continue
        slots.append(
            {
                "time": time,
                "reservable": can_reserve,
                "inventory_left": slot.get("inventory_left", item.get("inventory_left", 0)),
                "deposit_required": False,
            }
        )
    return slots


def queue_time_by_period(category: str, *, base_queue: int) -> dict[str, int]:
    multiplier = 1.6 if category in {"hotpot", "barbecue"} else 1.2
    return {
        "午市": max(5, int(base_queue * 0.8)),
        "下午": max(0, int(base_queue * 0.5)),
        "晚高峰": max(10, int(base_queue * multiplier)),
        "周末": max(15, int(base_queue * (multiplier + 0.4))),
    }


def serving_speed_min(category: str) -> int:
    if category in {"fried_chicken", "light_food", "salad_light_food"}:
        return 12
    if category in {"hotpot", "barbecue"}:
        return 25
    return 18


def seat_capacity(category: str, *, family_friendly: bool) -> int:
    if family_friendly:
        return 72
    if category in {"hotpot", "barbecue", "regional_home_cuisine"}:
        return 96
    if category in {"light_food", "salad_light_food"}:
        return 36
    return 58


def noise_level(category: str) -> str:
    if category in {"hotpot", "barbecue", "fried_chicken"}:
        return "偏热闹"
    if category in {"light_food", "salad_light_food", "japanese_light_food"}:
        return "较安静"
    return "适中"


def spice_level(category: str) -> str:
    if category == "hotpot":
        return "可选清淡/中辣"
    if category in {"barbecue", "fried_chicken"}:
        return "可选微辣"
    return "默认清淡"


def activity_review_keywords(category: str, *, family_activity: bool, indoor: bool) -> list[str]:
    if family_activity:
        return ["适合小朋友", "低强度", "安全感强", "陪伴感"]
    if category == "handcraft":
        return ["动手体验", "适合约会", "成品可带走", "节奏轻松"]
    if category == "citywalk":
        return ["小众路线", "拍照友好", "本地感", "天气敏感"]
    if category == "sports":
        return ["朋友局", "轻运动", "需要预约", "装备方便"]
    if indoor:
        return ["室内", "雨天可去", "路线简单", "停留舒适"]
    return ["周末感", "轻松", "可临时安排", "本地体验"]


def activity_ugc_summary(category: str, *, family_activity: bool, indoor: bool) -> str:
    if family_activity:
        return "评价集中在适龄、安全、低强度，适合作为带娃半日计划的活动节点。"
    if category == "handcraft":
        return "网友反馈体验节奏轻松，适合需要一点仪式感但不想太累的安排。"
    if category == "citywalk":
        return "网友常把它作为顺路逛店/拍照节点，天气好时体验更稳定。"
    if indoor:
        return "室内属性较强，适合作为雨天或高温天气的备选节点。"
    return "整体反馈偏轻松，适合放入低心智负担的周末方案。"


def activity_facilities(category: str, *, family_activity: bool, indoor: bool) -> list[str]:
    facilities = ["可预约", "到店核销"]
    if indoor:
        facilities.append("室内空间")
    if family_activity:
        facilities.extend(["亲子友好", "基础安全提示"])
    if category == "handcraft":
        facilities.append("材料包")
    if category == "sports":
        facilities.append("装备租赁")
    return facilities


def dietary_options(category: str, *, light_food: bool) -> dict[str, bool]:
    base = {
        "可少油": True,
        "可少盐": category not in {"fried_chicken"},
        "有低糖饮品": True,
        "有素食/蔬菜选项": category not in {"barbecue"},
        "有高蛋白选项": category in {"light_food", "salad_light_food", "japanese_light_food", "barbecue", "hotpot"},
        "减脂期友好": light_food or category in {"japanese_light_food"},
    }
    if category in {"hotpot", "barbecue", "fried_chicken"}:
        base["减脂期友好"] = False
    return base


def allergen_notes(category: str) -> list[str]:
    if category in {"japanese_light_food"}:
        return ["可能含海鲜", "可备注不放芥末"]
    if category in {"light_food", "salad_light_food"}:
        return ["沙拉酱可分装", "坚果配料需提前确认"]
    if category == "hotpot":
        return ["锅底辣度需确认", "海鲜/牛羊肉过敏需避开"]
    if category == "fried_chicken":
        return ["油炸食品", "可能含麸质"]
    return ["具体过敏原需到店确认"]


def restaurant_decision_profile(
    category: str,
    *,
    family_friendly: bool,
    light_food: bool,
    high_queue: bool,
) -> dict[str, Any]:
    good_for = ["就近吃饭", "可执行餐饮节点"]
    avoid_if = []
    if family_friendly:
        good_for.extend(["带娃", "家庭同行"])
    if light_food:
        good_for.extend(["减脂期", "低负担饮食"])
    if category in {"hotpot", "barbecue"}:
        good_for.append("朋友聚餐")
        avoid_if.extend(["明确低卡优先", "不想沾味道"])
    if high_queue:
        avoid_if.append("完全不能等待")
    return {
        "适合": good_for,
        "不适合": avoid_if or ["无明显硬性避雷"],
        "推荐理由": "餐饮属性和当前场景匹配，可作为方案中的履约节点。",
        "B侧使用": ["饮食约束", "排队风险", "预算", "订座可行性"],
    }


def activity_decision_profile(category: str, *, family_activity: bool, indoor: bool) -> dict[str, Any]:
    good_for = ["周末短时活动", "本地生活体验"]
    avoid_if = []
    if family_activity:
        good_for.extend(["亲子同行", "低强度陪伴"])
    if indoor:
        good_for.append("雨天/高温备用")
    if category == "sports":
        good_for.append("朋友局")
        avoid_if.append("完全不想运动")
    if category == "citywalk":
        good_for.append("路线型体验")
        avoid_if.append("下雨或高温")
    return {
        "适合": good_for,
        "不适合": avoid_if or ["无明显硬性避雷"],
        "推荐理由": "可作为餐前/餐后活动节点，帮助方案形成完整时间线。",
        "B侧使用": ["活动强度", "天气风险", "同行人适配", "路线衔接"],
    }


def restaurant_fulfillment_actions(item: dict[str, Any], *, category: str, has_coupon: bool) -> list[dict[str, str]]:
    actions = [
        {"action": "reservation/check", "target": item["poi_id"], "reason": "确认目标时间是否可订座"},
        {"action": "queue/check", "target": item["poi_id"], "reason": "确认晚高峰排队风险"},
    ]
    if has_coupon:
        actions.append({"action": "coupon/check", "target": item["poi_id"], "reason": "确认套餐券是否可买可核销"})
    if category in {"light_food", "salad_light_food", "japanese_light_food"}:
        actions.append({"action": "note/request", "target": item["poi_id"], "reason": "备注少油少酱/低糖饮品"})
    return actions


def activity_fulfillment_actions(item: dict[str, Any], *, category: str) -> list[dict[str, str]]:
    actions = [
        {"action": "availability/check", "target": item["poi_id"], "reason": "确认目标场次库存"},
        {"action": "ticket/lock", "target": item["poi_id"], "reason": "用户确认后锁定名额"},
    ]
    if category in {"citywalk", "sports"}:
        actions.append({"action": "weather/check", "target": item["poi_id"], "reason": "确认天气是否影响体验"})
    return actions


def restaurant_substitution_strategy(category: str) -> dict[str, Any]:
    replacements = {
        "light_food": ["沙拉轻食", "日料轻食", "简餐"],
        "salad_light_food": ["轻食", "日料轻食", "简餐"],
        "japanese_light_food": ["轻食", "本帮家常菜", "简餐"],
        "family_bistro": ["家庭餐厅", "本帮家常菜", "商场餐厅"],
        "hotpot": ["本帮家常菜", "烤肉", "餐厅"],
        "barbecue": ["本帮家常菜", "火锅", "餐厅"],
        "fried_chicken": ["小吃快餐", "轻食", "餐厅"],
    }
    return {
        "可替换类目": replacements.get(category, ["餐厅", "本帮家常菜", "轻食"]),
        "保留节点": "活动节点优先保留，仅替换餐厅",
        "替换触发": ["满座", "排队过长", "低卡需求冲突", "距离过远"],
    }


def activity_substitution_strategy(category: str) -> dict[str, Any]:
    replacements = {
        "parent_child_activity": ["亲子活动", "博物馆展览", "手作体验"],
        "handcraft": ["手作体验", "博物馆展览", "近场放松"],
        "citywalk": ["本地市集", "咖啡甜品", "博物馆展览"],
        "sports": ["运动体验", "密室桌游", "近场放松"],
        "micro_vacation": ["近场放松", "茶空间", "手作体验"],
    }
    return {
        "可替换类目": replacements.get(category, ["休闲活动", "博物馆展览", "近场放松"]),
        "保留节点": "餐厅可保留，替换同区域活动",
        "替换触发": ["无票", "天气不适合", "距离过远", "同行人不适配"],
    }


def peak_risk_profile(category: str, *, high_queue: bool) -> dict[str, Any]:
    if category in {"hotpot", "barbecue"} or high_queue:
        return {
            "高峰时段": ["18:00-20:00", "周末晚间"],
            "主要风险": ["排队久", "座位紧张"],
            "规避策略": ["提前订座", "错峰到店", "保留附近备选"],
        }
    if category in {"citywalk", "sports"}:
        return {
            "高峰时段": ["周末下午"],
            "主要风险": ["天气影响", "人流拥挤"],
            "规避策略": ["准备室内备选", "缩短路线"],
        }
    return {
        "高峰时段": ["周末下午/晚间"],
        "主要风险": ["库存变化", "临时排队"],
        "规避策略": ["执行前复查库存", "保留同区域备选"],
    }


def physical_intensity(category: str, *, family_activity: bool) -> str:
    if family_activity:
        return "低强度"
    if category == "sports":
        return "中等强度"
    if category in {"citywalk"}:
        return "轻中强度"
    return "低强度"


def weather_plan(category: str, *, indoor: bool) -> dict[str, Any]:
    if indoor:
        return {"天气敏感度": "低", "雨天方案": "可照常执行", "高温方案": "可照常执行"}
    if category in {"citywalk", "sports"}:
        return {"天气敏感度": "高", "雨天方案": "切换室内活动", "高温方案": "缩短户外停留"}
    return {"天气敏感度": "中", "雨天方案": "执行前复查", "高温方案": "增加室内休息点"}


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
    item_tags = expand_preference_tags(flatten_tags(item.get("tags", {})))
    return {
        "deal_id": deal_id,
        "poi_id": item["poi_id"],
        "product_id": product_id,
        "deal_type": to_chinese_value("activity_ticket" if expected_type == "activity" else "meal_coupon"),
        "coupon_type": item.get("category") or item.get("restaurant_category", to_chinese_value(expected_type)),
        "title": f"{item['name']}体验券" if expected_type == "activity" else f"{item['name']}双人套餐",
        "sale_price": item["price"],
        "original_price": round(item["price"] * 1.2, 2),
        "requires_reservation": item.get("reservation_required", True),
        "refund_policy": to_chinese_value(item.get("refund_policy", "before_1h_free")),
        "cancel_policy": to_chinese_value(item.get("cancel_policy", "before_1h_free")),
        "valid_time": [slot["time"] for slot in item.get("available_slots", []) if slot.get("time")],
        "package_components": package_components(item, expected_type),
        "target_persona": target_persona(item),
        "family_ticket": "family_friendly" in item_tags,
        "redemption_rate": 0.6,
        "commission_rate": 0.06,
        "stock_limit_per_slot": item.get("capacity_limit", 30),
        "hidden_cost_risk": to_chinese_value(item.get("hidden_cost_risk", "low")),
    }


def make_product(item: dict[str, Any], expected_type: str) -> dict[str, Any]:
    product_id = f"prod_{item['poi_id']}"
    return {
        "product_id": product_id,
        "poi_id": item["poi_id"],
        "merchant_id": item["merchant_id"],
        "product_type": to_chinese_value("activity_ticket" if expected_type == "activity" else "meal_package"),
        "name": f"{item['name']}产品",
        "target_persona": target_persona(item),
        "price": item["price"],
        "duration_min": item.get("duration_min"),
        "requires_reservation": item.get("reservation_required", True),
        "inventory_model": to_chinese_value("slot_capacity" if expected_type == "activity" else "table_slot"),
        "package_components": package_components(item, expected_type),
        "health_tags": item.get("health_tags", []),
        "refund_policy": to_chinese_value(item.get("refund_policy")),
        "fulfillment_mode": to_chinese_value(
            "onsite_verify" if expected_type == "activity" else (
                "takeaway_only" if not item.get("dine_in_available", True) else "dine_in_reservation"
            )
        ),
    }


def make_merchant(item: dict[str, Any], expected_type: str) -> dict[str, Any]:
    return {
        "merchant_id": item["merchant_id"],
        "name": item["name"],
        "merchant_type": to_chinese_value("activity_provider" if expected_type == "activity" else "restaurant"),
        "poi_ids": [item["poi_id"]],
        "source_channel": "gaode_poi_search",
        "trust_score": item.get("trust_score", 0.75),
        "review_count": item.get("review_count", 120),
        "recent_order_count": 40,
        "reservation_count": 20 if item.get("reservation_required") else 0,
        "operation_stability_score": item.get("operation_stability_score", 0.82),
        "business_capabilities": to_chinese_tags(merchant_capabilities(item, expected_type)),
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

        calls_left -= 1
        try:
            kwargs = {"city": city} if mode == "transit" else {}
            route = planner.plan(origin, destination, mode=mode, **kwargs)
        except Exception as exc:  # Keep seed generation resilient to route misses.
            route = {"feasible": False, "reason": redact_secret(exc), "mode": mode}

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
    tags = expand_preference_tags(flatten_tags(item.get("tags", {})))
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
    return to_chinese_tags(personas or ["general"])


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


def read_json_or_default(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def normalize_raw_records(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(raw, dict):
        raw = {}
    return {
        "activities": [item for item in raw.get("activities", []) if isinstance(item, dict)],
        "restaurants": [item for item in raw.get("restaurants", []) if isinstance(item, dict)],
    }


def search_record_key(expected_type: str, keyword: str, page: int) -> tuple[str, str, int]:
    return expected_type, str(keyword), int(page)


def existing_success_keys(raw_records: dict[str, list[dict[str, Any]]]) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    for expected_type, group_name in (("activity", "activities"), ("restaurant", "restaurants")):
        for record in raw_records.get(group_name, []):
            if "pois" not in record:
                continue
            try:
                keys.add(search_record_key(expected_type, str(record.get("keyword", "")), int(record.get("page", 1))))
            except (TypeError, ValueError):
                continue
    return keys


def enrich_raw_records(records: list[dict[str, Any]], expected_type: str) -> list[dict[str, Any]]:
    enriched = []
    for record in records:
        keyword = str(record.get("keyword") or "")
        for poi in record.get("pois", []) or []:
            if not isinstance(poi, dict):
                continue
            profile = (
                infer_activity_profile(keyword, poi)
                if expected_type == "activity"
                else infer_restaurant_profile(keyword, poi)
            )
            cleaning_view = {
                **poi,
                **profile,
                "gaode_type": poi.get("type"),
                "raw": poi.get("raw") if isinstance(poi.get("raw"), dict) else poi,
            }
            if should_exclude_poi(cleaning_view, expected_type):
                continue
            enriched.append(
                build_common_fields(
                    poi,
                    expected_type=expected_type,
                    keyword=keyword,
                    profile=profile,
                )
            )
    return dedupe_pois(enriched)


def build_supply_payload(
    activity_raw: list[dict[str, Any]],
    restaurant_raw: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    activities = enrich_raw_records(activity_raw, "activity")
    restaurants = enrich_raw_records(restaurant_raw, "restaurant")
    all_items = activities + restaurants

    for item in all_items:
        item["deal_ids"] = [f"deal_{item['poi_id']}"]
        item["product_ids"] = [f"prod_{item['poi_id']}"]

    deals = [make_deal(item, item["type"]) for item in all_items]
    products = [make_product(item, item["type"]) for item in all_items]
    merchants = [make_merchant(item, item["type"]) for item in all_items]
    availability = {item["poi_id"]: make_availability(item) for item in all_items}
    return activities, restaurants, deals, products, merchants, availability


def write_seed_outputs(
    output_dir: Path,
    *,
    activity_raw: list[dict[str, Any]],
    restaurant_raw: list[dict[str, Any]],
    route_overlays: dict[str, Any],
    report: dict[str, Any],
) -> None:
    activities, restaurants, deals, products, merchants, availability = build_supply_payload(activity_raw, restaurant_raw)
    report = dict(report)
    report.update(
        {
            "activity_count": len(activities),
            "restaurant_count": len(restaurants),
            "deal_count": len(deals),
            "product_count": len(products),
            "merchant_count": len(merchants),
            "route_count": len(route_overlays),
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


def build_search_plan(activity_keywords: list[str], restaurant_keywords: list[str], pages: int) -> list[dict[str, Any]]:
    plan = []
    for expected_type, keywords in (("activity", activity_keywords), ("restaurant", restaurant_keywords)):
        for keyword in keywords:
            for page in range(1, pages + 1):
                plan.append({"expected_type": expected_type, "keyword": keyword, "page": page})
    return plan


def fetch_groups_incrementally(
    searcher: POISearcher,
    *,
    activity_keywords: list[str],
    restaurant_keywords: list[str],
    city: str,
    citylimit: bool,
    pages: int,
    offset: int,
    output_dir: Path,
    report: dict[str, Any],
    resume: bool,
    max_search_calls: int | None,
    stop_on_error: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_records = normalize_raw_records(
        read_json_or_default(output_dir / "gaode_raw_pois.json", {"activities": [], "restaurants": []})
        if resume
        else {"activities": [], "restaurants": []}
    )
    success_keys = existing_success_keys(raw_records)
    search_plan = build_search_plan(activity_keywords, restaurant_keywords, pages)

    previous_report = read_json_or_default(output_dir / "build_report.json", {}) if resume else {}
    errors = (
        redact_json_secrets(list(previous_report.get("search_errors", [])))
        if isinstance(previous_report, dict)
        else []
    )
    attempted_this_run = 0
    successful_this_run = 0
    skipped_existing = 0
    stopped_reason = None
    offset = max(1, min(offset, 25))

    def flush() -> None:
        current_report = dict(report)
        current_report.update(
            {
                "search_plan_count": len(search_plan),
                "max_search_calls": max_search_calls,
                "attempted_search_calls_this_run": attempted_this_run,
                "successful_search_calls_this_run": successful_this_run,
                "skipped_existing_search_calls": skipped_existing,
                "successful_search_calls_total": len(existing_success_keys(raw_records)),
                "failed_search_calls_total": len(errors),
                "search_errors": errors,
                "search_completed": stopped_reason is None and skipped_existing + attempted_this_run >= len(search_plan),
                "stopped_reason": stopped_reason,
            }
        )
        write_seed_outputs(
            output_dir,
            activity_raw=raw_records["activities"],
            restaurant_raw=raw_records["restaurants"],
            route_overlays={},
            report=current_report,
        )

    flush()

    for query in search_plan:
        expected_type = query["expected_type"]
        keyword = query["keyword"]
        page = int(query["page"])
        key = search_record_key(expected_type, keyword, page)

        if resume and key in success_keys:
            skipped_existing += 1
            flush()
            continue

        if max_search_calls is not None and attempted_this_run >= max(0, max_search_calls):
            stopped_reason = "max_search_calls_reached"
            flush()
            break

        attempted_this_run += 1
        try:
            pois = searcher.search(
                keywords=keyword,
                city=city,
                citylimit=citylimit,
                page=page,
                offset=offset,
            )
            pois = pois[:offset]
            group_name = "activities" if expected_type == "activity" else "restaurants"
            raw_records[group_name].append({"keyword": keyword, "page": page, "pois": pois})
            success_keys.add(key)
            successful_this_run += 1
        except Exception as exc:
            error_record = {
                "expected_type": expected_type,
                "keyword": keyword,
                "page": page,
                "error": redact_secret(exc),
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
            }
            errors.append(error_record)
            if stop_on_error:
                stopped_reason = "search_error"
                flush()
                raise
        finally:
            flush()

    return raw_records["activities"], raw_records["restaurants"], {
        "attempted_search_calls_this_run": attempted_this_run,
        "successful_search_calls_this_run": successful_this_run,
        "skipped_existing_search_calls": skipped_existing,
        "search_errors": errors,
        "stopped_reason": stopped_reason,
        "search_plan_count": len(search_plan),
    }


def main() -> int:
    args = parse_args()
    api_key = require_api_key(args)
    activity_keywords = resolve_keywords(
        args.activity_keywords_file,
        args.activity_keywords,
        DEFAULT_ACTIVITY_KEYWORDS,
    )
    restaurant_keywords = resolve_keywords(
        args.restaurant_keywords_file,
        args.restaurant_keywords,
        DEFAULT_RESTAURANT_KEYWORDS,
    )
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
    pages = max(1, args.pages)
    offset = max(1, min(args.offset, 25))

    activity_raw, restaurant_raw, search_summary = fetch_groups_incrementally(
        searcher,
        activity_keywords=activity_keywords,
        restaurant_keywords=restaurant_keywords,
        city=args.city,
        citylimit=citylimit,
        pages=pages,
        offset=offset,
        output_dir=output_dir,
        report=report,
        resume=args.resume,
        max_search_calls=args.max_search_calls,
        stop_on_error=args.stop_on_error,
    )

    activities, restaurants, deals, products, merchants, availability = build_supply_payload(
        activity_raw,
        restaurant_raw,
    )
    all_items = activities + restaurants
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

    report.update(
        {
            **search_summary,
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

    write_seed_outputs(
        output_dir,
        activity_raw=activity_raw,
        restaurant_raw=restaurant_raw,
        route_overlays=route_overlays,
        report=report,
    )

    print(
        f"Wrote {len(activities)} activities, {len(restaurants)} restaurants, "
        f"{len(deals)} deals to {output_dir}. "
        f"Search calls this run: {search_summary['attempted_search_calls_this_run']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
