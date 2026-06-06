"""Route geometry helpers for B candidate generation.

This module keeps coordinate parsing and geo prefiltering separate from the
candidate generation policy. It deliberately uses conservative estimates: these
helpers are planning signals, not a replacement for live route APIs.
"""

from __future__ import annotations

import math

from .b_utils import get_constraint_config_with_profile, to_float


DEFAULT_SHANGHAI_ORIGIN = (121.4737, 31.2304)


def parse_coordinates(value: object) -> tuple[float, float] | None:
    if not value:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        lng, lat = value[0], value[1]
    elif isinstance(value, str) and "," in value:
        lng, lat = value.split(",", 1)
    else:
        return None
    try:
        return float(lng), float(lat)
    except (TypeError, ValueError):
        return None


def haversine_km(coord_a: tuple[float, float], coord_b: tuple[float, float]) -> float:
    lng1, lat1 = coord_a
    lng2, lat2 = coord_b
    radius_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lng / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def item_coordinates(item: dict) -> tuple[float, float] | None:
    coordinates = parse_coordinates(item.get("coordinates"))
    if coordinates:
        return coordinates
    longitude = item.get("longitude") if item.get("longitude") is not None else item.get("lng")
    latitude = item.get("latitude") if item.get("latitude") is not None else item.get("lat")
    if longitude is None or latitude is None:
        return None
    try:
        return float(longitude), float(latitude)
    except (TypeError, ValueError):
        return None


def route_origin_coordinates(constraints: dict | None) -> str | None:
    constraints = constraints or {}
    for key in ("route_origin", "origin_coordinates", "origin", "center"):
        value = constraints.get(key)
        if isinstance(value, str) and "," in value:
            return value.strip()
    location = constraints.get("location")
    if isinstance(location, dict):
        for key in ("route_origin", "origin_coordinates", "coordinates", "center"):
            value = location.get(key)
            if isinstance(value, str) and "," in value:
                return value.strip()
    return None


def looks_like_shanghai_supply(candidates: list[dict]) -> bool:
    inspected = 0
    hits = 0
    for item in candidates[:200]:
        coord = item_coordinates(item)
        if not coord:
            continue
        inspected += 1
        lng, lat = coord
        if 120.8 <= lng <= 122.2 and 30.6 <= lat <= 31.9:
            hits += 1
    return inspected > 0 and hits / inspected >= 0.75


def geo_prefilter_origin(
    constraints: dict | None,
    candidates: list[dict],
) -> tuple[float, float] | None:
    explicit_origin = parse_coordinates(route_origin_coordinates(constraints))
    if explicit_origin:
        return explicit_origin

    constraints = constraints or {}
    city_values = [
        constraints.get("city"),
        constraints.get("district"),
        (constraints.get("location") or {}).get("city") if isinstance(constraints.get("location"), dict) else None,
    ]
    if any("上海" in str(value) for value in city_values if value):
        return DEFAULT_SHANGHAI_ORIGIN
    if looks_like_shanghai_supply(candidates):
        return DEFAULT_SHANGHAI_ORIGIN
    return None


def distance_from_origin_km(
    item: dict,
    origin: tuple[float, float] | None,
) -> float | None:
    coord = item_coordinates(item)
    if origin and coord:
        return haversine_km(origin, coord) * 1.25
    if item.get("distance_km") is not None:
        return to_float(item.get("distance_km"), 999.0)
    return None


def filter_candidates_by_geo_window(
    candidates: list[dict],
    *,
    constraints: dict,
    user_profile: dict | None,
    min_keep: int,
) -> tuple[list[dict], dict]:
    if not candidates:
        return candidates, {"applied": False}

    config = get_constraint_config_with_profile(constraints, user_profile or {})
    max_distance_km = float(config.get("max_distance_km", 8.0))
    origin = geo_prefilter_origin(constraints, candidates)
    scored: list[tuple[float, dict]] = []
    unknown_distance: list[dict] = []

    for item in candidates:
        distance_km = distance_from_origin_km(item, origin)
        if distance_km is None:
            unknown_distance.append(item)
            continue
        copied = dict(item)
        if origin:
            copied["distance_km"] = round(distance_km, 2)
            copied["distance_source"] = "origin_coordinate_estimate"
        scored.append((distance_km, copied))

    if not scored:
        return candidates, {"applied": False, "reason": "missing_distance"}

    scored.sort(key=lambda pair: pair[0])
    selected: list[dict] = []
    selected_radius = None
    for multiplier in (1.15, 1.5, 2.0):
        radius = max_distance_km * multiplier
        within_radius = [item for distance, item in scored if distance <= radius]
        if len(within_radius) >= min_keep or multiplier == 2.0:
            selected = within_radius
            selected_radius = radius
            break

    if len(selected) < min_keep:
        selected_ids = {id(item) for item in selected}
        for _, item in scored:
            if id(item) in selected_ids:
                continue
            selected.append(item)
            if len(selected) >= min_keep:
                break

    if not selected:
        selected = [item for _, item in scored[:min_keep]]

    if unknown_distance and len(selected) < min_keep:
        selected.extend(unknown_distance[: max(0, min_keep - len(selected))])

    return selected, {
        "applied": True,
        "origin": "explicit_or_default",
        "radius_km": round(float(selected_radius or max_distance_km), 2),
        "min_keep": min_keep,
        "unknown_distance_count": len(unknown_distance),
    }
