#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a configurable Chinese-first Gaode POI supply seed for WeekendFlow.

The 40k-call extraction should be treated as an auditable data project, not a
one-off crawler run. This script reads an extraction YAML, builds a balanced
search plan, records a call ledger, supports resume/rebuild, dedupes POIs, and
then generates WeekendFlow mock supply files in the Chinese-first contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is present in our dev env.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.build_supply_from_gaode import (  # noqa: E402
    build_common_fields,
    infer_activity_profile,
    infer_restaurant_profile,
    make_availability,
    make_deal,
    make_merchant,
    make_product,
)
from src.nodes.poi_cleaning import should_exclude_poi  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "gaode_supply_extraction_config.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "mock_data" / "gaode_supply_shanghai_v2_configured"
AMAP_ENDPOINTS = {
    "text": "https://restapi.amap.com/v5/place/text",
    "around": "https://restapi.amap.com/v5/place/around",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configurable Gaode POI supply extraction for WeekendFlow.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", default="pilot", help="Mode under config.modes, e.g. smoke/pilot/main.")
    parser.add_argument("--city", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-calls", type=int, default=None, help="Current run call cap. Overrides mode max_calls.")
    parser.add_argument("--page-size", type=int, default=None)
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--max-pages-per-query", type=int, default=None)
    parser.add_argument("--sleep-sec", type=float, default=None)
    parser.add_argument("--request-timeout-sec", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--retry-backoff-sec", type=float, default=None)
    parser.add_argument("--quota-limit", type=int, default=None, help="Optional external quota ceiling for ledger reporting.")
    parser.add_argument(
        "--dry-run-plan",
        action="store_true",
        help="Build and write the search plan without calling Gaode.",
    )
    parser.add_argument(
        "--rebuild-from-raw",
        action="store_true",
        help="Rebuild normalized outputs from existing raw_pois.jsonl without new Gaode API calls.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip successful plan ids already present in raw_search_requests.jsonl.",
    )
    parser.add_argument(
        "--reset-logs",
        action="store_true",
        help="Delete raw_search_requests.jsonl and raw_pois.jsonl before a fresh API run.",
    )
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def load_local_env() -> None:
    for path in (REPO_ROOT / ".env.local", REPO_ROOT / ".env"):
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lstrip("\ufeff")
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise SystemExit("PyYAML is required to read the extraction config.")
    if not path.exists():
        raise SystemExit(f"Extraction config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"Extraction config must be an object: {path}")
    return config


def merged_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(config.get("defaults") or {})
    mode_settings = dict((config.get("modes") or {}).get(args.mode) or {})
    settings = {**defaults, **mode_settings}

    for key in (
        "city",
        "page_size",
        "radius",
        "max_pages_per_query",
        "sleep_sec",
        "request_timeout_sec",
        "max_calls",
        "max_retries",
        "retry_backoff_sec",
    ):
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value

    settings.setdefault("city", "上海")
    settings.setdefault("page_size", 20)
    settings.setdefault("radius", 3500)
    settings.setdefault("max_pages_per_query", 1)
    settings.setdefault("show_fields", "business")
    settings.setdefault("citylimit", True)
    settings.setdefault("sleep_sec", 0.08)
    settings.setdefault("request_timeout_sec", 8)
    settings.setdefault("max_retries", 2)
    settings.setdefault("retry_backoff_sec", 0.6)
    settings.setdefault("max_calls", 50)
    settings.setdefault("per_category_call_cap", None)
    settings.setdefault("include_area_grids", False)
    return settings


def require_api_key(args: argparse.Namespace) -> str | None:
    api_key = args.api_key or os.getenv("GAODE_API_KEY")
    if api_key:
        return api_key.strip()
    if args.allow_empty or args.dry_run_plan:
        return None
    raise SystemExit("GAODE_API_KEY is missing. Set it in the environment or pass --api-key.")


def normalize_name(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").lower())


def parse_lnglat(value: Any) -> tuple[float | None, float | None]:
    if not value or "," not in str(value):
        return None, None
    lng_raw, lat_raw = str(value).split(",", 1)
    try:
        return float(lng_raw), float(lat_raw)
    except ValueError:
        return None, None


def distance_m(a: str | None, b: str | None) -> float:
    lng1, lat1 = parse_lnglat(a)
    lng2, lat2 = parse_lnglat(b)
    if None in (lng1, lat1, lng2, lat2):
        return math.inf
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lam = math.radians(lng2 - lng1)
    h = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lam / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(h), math.sqrt(1 - h))


def redact_secret(text: str, api_key: str | None = None) -> str:
    redacted = text
    if api_key:
        redacted = redacted.replace(api_key, "<redacted>")
    redacted = re.sub(r"([?&]key=)[^&\s]+", r"\1<redacted>", redacted)
    return redacted


def normalize_poi(raw: dict[str, Any]) -> dict[str, Any]:
    business = raw.get("business") if isinstance(raw.get("business"), dict) else {}
    biz_ext = raw.get("biz_ext") if isinstance(raw.get("biz_ext"), dict) else {}
    if business:
        biz_ext = {
            **biz_ext,
            "rating": business.get("rating") or biz_ext.get("rating"),
            "cost": business.get("cost") or biz_ext.get("cost"),
            "business_area": business.get("business_area"),
            "tag": business.get("tag"),
            "rectag": business.get("rectag"),
            "keytag": business.get("keytag"),
            "alias": business.get("alias"),
            "opentime_today": business.get("opentime_today"),
            "opentime_week": business.get("opentime_week"),
        }
    business_area = biz_ext.get("business_area") or business.get("business_area")
    address = raw.get("address") or business_area or raw.get("adname") or raw.get("cityname") or raw.get("pname") or raw.get("location")
    return {
        "id": raw.get("id"),
        "name": raw.get("name"),
        "type": raw.get("type"),
        "typecode": raw.get("typecode"),
        "address": address,
        "location": raw.get("location"),
        "tel": raw.get("tel") or business.get("tel"),
        "pname": raw.get("pname"),
        "cityname": raw.get("cityname"),
        "adname": raw.get("adname"),
        "pcode": raw.get("pcode"),
        "citycode": raw.get("citycode"),
        "adcode": raw.get("adcode"),
        "distance": raw.get("distance"),
        "biz_ext": biz_ext,
        "raw": raw,
    }


def request_amap(
    api_key: str,
    endpoint: str,
    params: dict[str, Any],
    timeout_sec: float,
    max_retries: int = 2,
    retry_backoff_sec: float = 0.6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = {key: value for key, value in params.items() if value not in (None, "")}
    query.update({"key": api_key, "output": "JSON"})
    url = f"{AMAP_ENDPOINTS[endpoint]}?{urlencode(query)}"
    started = time.time()
    response = None
    last_exc: Exception | None = None
    attempts = max(1, int(max_retries) + 1)
    attempt = 1
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(AMAP_ENDPOINTS[endpoint], params=query, timeout=timeout_sec)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt >= attempts:
                raise
            time.sleep(float(retry_backoff_sec) * attempt)
    if response is None:
        raise last_exc or RuntimeError("Gaode request failed before response was created.")
    elapsed_ms = int((time.time() - started) * 1000)
    payload = response.json()
    meta = {
        "endpoint": endpoint,
        "params": {k: v for k, v in query.items() if k != "key"},
        "url": redact_secret(response.url or url, api_key),
        "status": payload.get("status"),
        "info": payload.get("info"),
        "infocode": payload.get("infocode"),
        "elapsed_ms": elapsed_ms,
        "attempts": attempt,
        "count": len(payload.get("pois") or []),
    }
    if payload.get("status") != "1":
        raise RuntimeError(f"Gaode {endpoint} error: {payload.get('info')} ({payload.get('infocode')})")
    return [normalize_poi(item) for item in payload.get("pois") or []], meta


def stable_plan_id(payload: dict[str, Any]) -> str:
    identity = {
        "endpoint": payload.get("endpoint"),
        "expected_type": payload.get("expected_type"),
        "category_id": payload.get("category_id"),
        "keyword": payload.get("keyword"),
        "center_id": payload.get("center_id"),
        "page_num": payload.get("params", {}).get("page_num"),
        "location": payload.get("params", {}).get("location"),
    }
    digest = hashlib.sha1(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def area_matches(area: dict[str, Any], category: dict[str, Any]) -> bool:
    area_ids = set(category.get("area_ids") or [])
    if area_ids:
        return area.get("id") in area_ids

    area_tags = set(category.get("area_tags") or [])
    if not area_tags:
        return True
    return bool(area_tags.intersection(set(area.get("tags") or [])))


def expand_area_grids(area_grids: list[dict[str, Any]]) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    for grid in area_grids or []:
        if grid.get("enabled") is False:
            continue
        bounds = grid.get("bounds") or {}
        try:
            lng_min = float(bounds["lng_min"])
            lng_max = float(bounds["lng_max"])
            lat_min = float(bounds["lat_min"])
            lat_max = float(bounds["lat_max"])
        except (KeyError, TypeError, ValueError):
            continue

        step_km = max(0.5, float(grid.get("step_km") or 3.0))
        max_centers = int(grid.get("max_centers") or 500)
        lat_mid = (lat_min + lat_max) / 2
        lat_step = step_km / 111.32
        lng_step = step_km / (111.32 * max(0.2, math.cos(math.radians(lat_mid))))

        ix = 0
        grid_count = 0
        lng = lng_min
        while lng <= lng_max + 1e-9:
            iy = 0
            lat = lat_min
            while lat <= lat_max + 1e-9:
                if grid_count >= max_centers:
                    break
                generated.append(
                    {
                        "id": f"{grid.get('id', 'grid')}_{ix:02d}_{iy:02d}",
                        "name": f"{grid.get('name', '网格')} {ix:02d}-{iy:02d}",
                        "district": grid.get("district", grid.get("name", "网格")),
                        "location": f"{lng:.6f},{lat:.6f}",
                        "tags": list(grid.get("tags") or []),
                        "source": "generated_grid",
                    }
                )
                grid_count += 1
                iy += 1
                lat += lat_step
            if grid_count >= max_centers:
                break
            ix += 1
            lng += lng_step
    return generated


def query_pages(category: dict[str, Any], settings: dict[str, Any]) -> int:
    configured = int(category.get("pages") or settings.get("max_pages_per_query") or 1)
    return max(1, min(configured, int(settings.get("max_pages_per_query") or configured)))


def add_plan_record(plan: list[dict[str, Any]], record: dict[str, Any]) -> None:
    record["plan_id"] = stable_plan_id(record)
    record["series_id"] = stable_plan_id({**record, "params": {**record.get("params", {}), "page_num": 0}})
    plan.append(record)


def build_search_plan(config: dict[str, Any], settings: dict[str, Any]) -> list[dict[str, Any]]:
    city = settings["city"]
    page_size = max(1, min(int(settings["page_size"]), 25))
    radius = int(settings["radius"])
    show_fields = settings.get("show_fields", "business")
    citylimit = "true" if settings.get("citylimit", True) else "false"
    plan: list[dict[str, Any]] = []
    areas = list(config.get("areas") or [])
    if settings.get("include_area_grids"):
        areas.extend(expand_area_grids(config.get("area_grids") or []))

    for category in config.get("categories") or []:
        category_id = category["id"]
        include_category_ids = set(settings.get("include_category_ids") or [])
        exclude_category_ids = set(settings.get("exclude_category_ids") or [])
        if include_category_ids and category_id not in include_category_ids:
            continue
        if category_id in exclude_category_ids:
            continue

        expected_type = category["expected_type"]
        category_name = category["category"]
        types = category.get("types")
        pages = query_pages(category, settings)

        for keyword in category.get("text_keywords") or []:
            for page_num in range(1, pages + 1):
                add_plan_record(
                    plan,
                    {
                        "endpoint": "text",
                        "expected_type": expected_type,
                        "category_id": category_id,
                        "category": category_name,
                        "keyword": keyword,
                        "params": {
                            "keywords": keyword,
                            "types": types,
                            "city": city,
                            "citylimit": citylimit,
                            "page_num": page_num,
                            "page_size": page_size,
                            "show_fields": show_fields,
                        },
                    },
                )

        selected_areas = [area for area in areas if area_matches(area, category)]
        for area in selected_areas:
            for keyword in category.get("around_keywords") or []:
                for page_num in range(1, pages + 1):
                    add_plan_record(
                        plan,
                        {
                            "endpoint": "around",
                            "expected_type": expected_type,
                            "category_id": category_id,
                            "category": category_name,
                            "keyword": keyword,
                            "center_id": area.get("id"),
                            "center": area.get("name"),
                            "district_hint": area.get("district"),
                            "params": {
                                "keywords": keyword,
                                "types": types,
                                "location": area.get("location"),
                                "radius": radius,
                                "page_num": page_num,
                                "page_size": page_size,
                                "show_fields": show_fields,
                            },
                        },
                    )

    return apply_plan_caps(plan, settings)


def apply_plan_caps(plan: list[dict[str, Any]], settings: dict[str, Any]) -> list[dict[str, Any]]:
    per_category_cap = settings.get("per_category_call_cap")
    per_category_caps = settings.get("per_category_call_caps")
    if not isinstance(per_category_caps, dict):
        per_category_caps = {}

    if not per_category_cap and not per_category_caps:
        return interleave_by_category(plan)

    kept = []
    counts: dict[str, int] = defaultdict(int)
    for query in plan:
        category_id = query.get("category_id", "unknown")
        raw_cap = per_category_caps.get(category_id, per_category_cap)
        if raw_cap is None:
            kept.append(query)
            continue
        cap = int(raw_cap)
        if cap <= 0 or counts[category_id] >= cap:
            continue
        counts[category_id] += 1
        kept.append(query)
    return interleave_by_category(kept)


def interleave_by_category(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for query in plan:
        category_id = query.get("category_id", "unknown")
        if category_id not in grouped:
            order.append(category_id)
        grouped[category_id].append(query)

    interleaved: list[dict[str, Any]] = []
    index = 0
    while True:
        appended = False
        for category_id in order:
            items = grouped[category_id]
            if index < len(items):
                interleaved.append(items[index])
                appended = True
        if not appended:
            return interleaved
        index += 1


def summarize_plan(plan: list[dict[str, Any]]) -> dict[str, Any]:
    by_endpoint = Counter(item.get("endpoint") for item in plan)
    by_type = Counter(item.get("expected_type") for item in plan)
    by_category = Counter(item.get("category") for item in plan)
    by_district = Counter(item.get("district_hint") or "text_search" for item in plan)
    return {
        "planned_calls": len(plan),
        "by_endpoint": dict(by_endpoint),
        "by_type": dict(by_type),
        "by_category": dict(by_category),
        "top_districts": dict(by_district.most_common(30)),
    }


def planned_run_stats(plan: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    max_calls = max(0, min(int(settings["max_calls"]), len(plan)))
    return {
        "planned_calls_total": len(plan),
        "max_calls_this_run": max_calls,
        "remaining_after_this_run_if_no_resume": max(0, len(plan) - max_calls),
    }


def completed_plan_ids(request_log: Path) -> set[str]:
    completed: set[str] = set()
    for record in read_jsonl(request_log):
        plan_id = record.get("plan_id")
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        if plan_id and meta.get("status") == "1":
            completed.add(plan_id)
    return completed


def merge_source(existing: dict[str, Any], source: dict[str, Any]) -> None:
    sources = existing.setdefault("source_queries", [])
    compact = {
        "plan_id": source.get("plan_id"),
        "endpoint": source.get("endpoint"),
        "expected_type": source.get("expected_type"),
        "category": source.get("category"),
        "category_id": source.get("category_id"),
        "keyword": source.get("keyword"),
        "center": source.get("center"),
        "district_hint": source.get("district_hint"),
        "page_num": (source.get("params") or {}).get("page_num"),
    }
    if compact not in sources:
        sources.append(compact)


def dedupe_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    fuzzy_duplicate_count = 0

    for item in records:
        poi = item["poi"]
        expected_type = item["expected_type"]
        amap_id = poi.get("id")
        key = f"{expected_type}:amap:{amap_id}" if amap_id else ""

        if not key:
            name = normalize_name(poi.get("name"))
            adcode = poi.get("adcode") or "unknown"
            key = f"{expected_type}:name:{name}:{adcode}"

        if key in by_key:
            duplicate_count += 1
            merge_source(by_key[key], item)
            continue

        fuzzy_key = None
        if not amap_id:
            name = normalize_name(poi.get("name"))
            for existing_key, existing in by_key.items():
                if not existing_key.startswith(f"{expected_type}:"):
                    continue
                if normalize_name(existing.get("name")) == name and distance_m(existing.get("location"), poi.get("location")) <= 80:
                    fuzzy_key = existing_key
                    break

        if fuzzy_key:
            fuzzy_duplicate_count += 1
            merge_source(by_key[fuzzy_key], item)
            continue

        stored = {
            **poi,
            "expected_type": expected_type,
            "primary_category": item.get("category"),
            "primary_keyword": item.get("keyword"),
            "source_queries": [],
        }
        merge_source(stored, item)
        by_key[key] = stored

    report = {
        "input_records": len(records),
        "deduped_records": len(by_key),
        "exact_duplicates": duplicate_count,
        "fuzzy_duplicates": fuzzy_duplicate_count,
    }
    return list(by_key.values()), report


def enrich_records(deduped: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    activities = []
    restaurants = []
    for poi in deduped:
        expected_type = poi.get("expected_type")
        if should_exclude_poi(poi, str(expected_type)):
            continue
        keyword = str(poi.get("primary_keyword") or poi.get("primary_category") or "")
        profile = infer_activity_profile(keyword, poi) if expected_type == "activity" else infer_restaurant_profile(keyword, poi)
        item = build_common_fields(poi, expected_type=str(expected_type), keyword=keyword, profile=profile)
        item["source_queries"] = poi.get("source_queries", [])
        item["gaode_typecode"] = poi.get("typecode")
        item["primary_category"] = poi.get("primary_category")
        item["field_sources"] = {
            "identity": "observed_gaode",
            "coordinates": "observed_gaode",
            "category_profile": "rule_imputed_fixture",
            "availability": "execution_mock_state",
            "scores": "b_derived_or_prior",
        }
        if expected_type == "activity":
            activities.append(item)
        else:
            restaurants.append(item)
    return activities, restaurants


def coverage_report(records: list[dict[str, Any]], activities: list[dict[str, Any]], restaurants: list[dict[str, Any]]) -> dict[str, Any]:
    raw_by_type = Counter(item.get("expected_type") for item in records)
    raw_by_category = Counter(item.get("primary_category") for item in records)
    raw_by_district = Counter(item.get("adname") or item.get("district_hint") or "unknown" for item in records)
    restaurant_categories = Counter(item.get("restaurant_category", "unknown") for item in restaurants)
    activity_categories = Counter(item.get("category", "unknown") for item in activities)
    source_query_count = sum(len(item.get("source_queries") or []) for item in activities + restaurants)
    photo_count = sum(1 for item in activities + restaurants if item.get("cover_photo_url") or item.get("photo_urls"))
    phone_count = sum(1 for item in activities + restaurants if item.get("tel"))
    return {
        "raw_by_type": dict(raw_by_type),
        "raw_by_category": dict(raw_by_category),
        "top_raw_districts": dict(raw_by_district.most_common(30)),
        "activity_categories": dict(activity_categories),
        "restaurant_categories": dict(restaurant_categories),
        "normalized_counts": {
            "activities": len(activities),
            "restaurants": len(restaurants),
            "with_phone": phone_count,
            "with_photo_url": photo_count,
            "source_query_links": source_query_count,
        },
    }


def build_payload_from_items(
    activities: list[dict[str, Any]],
    restaurants: list[dict[str, Any]],
) -> dict[str, Any]:
    all_items = activities + restaurants
    for item in all_items:
        item["deal_ids"] = [f"deal_{item['poi_id']}"]
        item["product_ids"] = [f"prod_{item['poi_id']}"]
    return {
        "activities.json": activities,
        "restaurants.json": restaurants,
        "availability.json": {item["poi_id"]: make_availability(item) for item in all_items},
        "deals.json": [make_deal(item, item["type"]) for item in all_items],
        "products.json": [make_product(item, item["type"]) for item in all_items],
        "merchants.json": [make_merchant(item, item["type"]) for item in all_items],
        "routes.json": {},
    }


def normalize_raw_records(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for record in raw_records:
        poi = record.get("poi")
        if not isinstance(poi, dict):
            continue
        if isinstance(poi.get("raw"), dict):
            record = dict(record)
            record["poi"] = normalize_poi(poi["raw"])
        normalized.append(record)
    return normalized


def update_quota_ledger(path: Path, run_record: dict[str, Any], quota_limit: int | None = None) -> None:
    ledger = load_json(path, {"runs": []})
    runs = ledger.setdefault("runs", [])
    runs.append(run_record)
    attempted_total = sum(int(run.get("attempted_calls", 0)) for run in runs)
    quota_attempted_total = sum(int(run.get("quota_attempted_calls", run.get("attempted_calls", 0))) for run in runs)
    successful_total = sum(int(run.get("successful_calls", 0)) for run in runs)
    ledger["totals"] = {
        "attempted_calls": attempted_total,
        "quota_attempted_calls": quota_attempted_total,
        "successful_calls": successful_total,
        "error_calls": sum(int(run.get("error_calls", 0)) for run in runs),
    }
    if quota_limit:
        ledger["quota_limit"] = quota_limit
        ledger["estimated_remaining_calls"] = max(0, quota_limit - quota_attempted_total)
    write_json(path, ledger)


def execute_api_run(
    *,
    api_key: str,
    plan: list[dict[str, Any]],
    args: argparse.Namespace,
    settings: dict[str, Any],
    request_log: Path,
    raw_pois_log: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if args.reset_logs:
        for path in (request_log, raw_pois_log):
            if path.exists():
                path.unlink()

    already_done = completed_plan_ids(request_log) if not args.no_resume else set()
    eligible_plan = [query for query in plan if query["plan_id"] not in already_done]
    max_calls = max(0, min(int(settings["max_calls"]), len(eligible_plan)))
    raw_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    stopped_series: set[str] = set()
    attempted = 0
    successful = 0
    quota_attempted = 0

    for query in eligible_plan:
        if attempted >= max_calls:
            break
        if query["series_id"] in stopped_series:
            continue

        attempted += 1
        page_num = int((query.get("params") or {}).get("page_num") or 1)
        try:
            pois, meta = request_amap(
                api_key,
                query["endpoint"],
                query["params"],
                float(settings["request_timeout_sec"]),
                max_retries=int(settings.get("max_retries", 2)),
                retry_backoff_sec=float(settings.get("retry_backoff_sec", 0.6)),
            )
            quota_attempted += int(meta.get("attempts") or 1)
            successful += 1
            log_record = {
                **{key: value for key, value in query.items() if key != "params"},
                "params": {key: value for key, value in query["params"].items() if key != "key"},
                "meta": meta,
            }
            append_jsonl(request_log, log_record)
            for poi in pois:
                record = {
                    "plan_id": query["plan_id"],
                    "series_id": query["series_id"],
                    "expected_type": query["expected_type"],
                    "category": query["category"],
                    "category_id": query["category_id"],
                    "keyword": query["keyword"],
                    "endpoint": query["endpoint"],
                    "center": query.get("center"),
                    "district_hint": query.get("district_hint"),
                    "params": {key: value for key, value in query["params"].items() if key != "key"},
                    "poi": poi,
                }
                raw_records.append(record)
                append_jsonl(raw_pois_log, record)
            if page_num > 0 and len(pois) < int(settings["page_size"]):
                stopped_series.add(query["series_id"])
        except Exception as exc:  # noqa: BLE001 - keep collection running while logging failures.
            quota_attempted += max(1, int(settings.get("max_retries", 2)) + 1)
            error = {
                "plan_id": query["plan_id"],
                "series_id": query["series_id"],
                "query": {key: value for key, value in query.items() if key != "params"},
                "params": {key: value for key, value in query["params"].items() if key != "key"},
                "error": redact_secret(str(exc), api_key),
            }
            errors.append(error)
            append_jsonl(request_log, {"plan_id": query["plan_id"], "error": error})

        if float(settings.get("sleep_sec") or 0) > 0:
            time.sleep(float(settings["sleep_sec"]))

    stats = {
        "attempted_calls": attempted,
        "quota_attempted_calls": quota_attempted,
        "successful_calls": successful,
        "error_calls": len(errors),
        "skipped_completed_calls": len(already_done),
        "eligible_calls_before_cap": len(eligible_plan),
    }
    return raw_records, errors, stats


def main() -> int:
    load_local_env()
    args = parse_args()
    ensure_dir(args.output_dir)
    config = load_config(args.config)
    settings = merged_settings(args, config)
    api_key = None

    request_log = args.output_dir / "raw_search_requests.jsonl"
    raw_pois_log = args.output_dir / "raw_pois.jsonl"
    quota_ledger_path = args.output_dir / "quota_ledger.json"

    plan = build_search_plan(config, settings)
    plan_summary = summarize_plan(plan)
    run_plan_summary = planned_run_stats(plan, settings)
    write_json(args.output_dir / "search_plan.json", plan)
    write_json(
        args.output_dir / "search_plan_report.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config": str(args.config),
            "mode": args.mode,
            "settings": settings,
            "summary": plan_summary,
            "run_cap": run_plan_summary,
        },
    )

    if args.dry_run_plan:
        print(
            "Dry-run search plan: "
            f"{run_plan_summary['planned_calls_total']} planned calls, "
            f"{run_plan_summary['max_calls_this_run']} max calls this run"
        )
        print(json.dumps(plan_summary["by_category"], ensure_ascii=False, indent=2))
        return 0

    errors: list[dict[str, Any]] = []
    run_stats = {
        "attempted_calls": 0,
        "successful_calls": 0,
        "error_calls": 0,
        "skipped_completed_calls": 0,
        "eligible_calls_before_cap": 0,
    }

    if args.rebuild_from_raw:
        raw_records = normalize_raw_records(read_jsonl(raw_pois_log))
    else:
        api_key = require_api_key(args)
        _, errors, run_stats = execute_api_run(
            api_key=api_key,
            plan=plan,
            args=args,
            settings=settings,
            request_log=request_log,
            raw_pois_log=raw_pois_log,
        )
        raw_records = normalize_raw_records(read_jsonl(raw_pois_log))

    deduped, dedupe = dedupe_records(raw_records)
    activities, restaurants = enrich_records(deduped)
    payload = build_payload_from_items(activities, restaurants)
    for filename, content in payload.items():
        write_json(args.output_dir / filename, content)
    write_json(args.output_dir / "deduped_pois.json", deduped)
    write_json(args.output_dir / "dedupe_report.json", dedupe)

    coverage = coverage_report(deduped, activities, restaurants)
    write_json(args.output_dir / "coverage_report.json", coverage)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config),
        "city": settings["city"],
        "mode": args.mode,
        "rebuilt_from_raw": bool(args.rebuild_from_raw),
        "planned_calls": len(plan),
        "max_calls_this_run": int(settings["max_calls"]),
        **run_stats,
        "raw_records": len(raw_records),
        "deduped_records": len(deduped),
        "activity_count": len(activities),
        "restaurant_count": len(restaurants),
        "errors": errors,
        "dedupe": dedupe,
        "coverage": coverage,
        "photo_policy": config.get("photo_policy", {}),
    }
    write_json(args.output_dir / "build_report.json", report)

    if not args.rebuild_from_raw:
        update_quota_ledger(
            quota_ledger_path,
            {
                "run_started_at": report["generated_at"],
                "mode": args.mode,
                "planned_calls": len(plan),
                "max_calls_this_run": int(settings["max_calls"]),
                **run_stats,
                "raw_records_after_run": len(raw_records),
                "deduped_records_after_run": len(deduped),
            },
            quota_limit=args.quota_limit,
        )

    print(
        f"Wrote {len(activities)} activities and {len(restaurants)} restaurants "
        f"from {len(raw_records)} raw records to {args.output_dir}"
    )
    print(
        f"Run calls attempted={run_stats['attempted_calls']} "
        f"success={run_stats['successful_calls']} errors={run_stats['error_calls']}"
    )
    print(f"Deduped {dedupe['input_records']} raw records -> {dedupe['deduped_records']} POIs")
    if errors:
        print(f"Recorded {len(errors)} errors; inspect build_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
