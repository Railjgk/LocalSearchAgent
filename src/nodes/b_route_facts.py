"""Route fact builders for B-side planning.

Candidate generation needs route facts, but it should not own the mechanics of
loading route overlays, resolving live/estimated legs, and aggregating sequence
traffic. This module keeps that responsibility narrow and testable.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from .b_route_geometry import haversine_km, parse_coordinates, route_origin_coordinates
from .b_utils import destination_city_from_constraints, to_float


@lru_cache(maxsize=16)
def load_route_overlays(mock_data_dir_key: str) -> dict[tuple[str, str], dict]:
    path = Path(mock_data_dir_key) / "routes.json"
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    overlays: dict[tuple[str, str], dict] = {}

    def add_pair(from_id: str | None, to_id: str | None, payload: dict) -> None:
        if not from_id or not to_id:
            return
        overlays[(str(from_id), str(to_id))] = dict(payload)

    if isinstance(raw, dict):
        for item in raw.get("overrides", []) or []:
            if isinstance(item, dict):
                add_pair(item.get("from"), item.get("to"), item)

        for group_name in ("gaode_seed_v1_overlays", "route_overlays"):
            group = raw.get(group_name)
            if isinstance(group, dict):
                for poi_id, route in group.items():
                    if isinstance(route, dict):
                        add_pair("start_point", str(poi_id), route)

        # A gaode seed routes.json is keyed directly by poi_id.
        for poi_id, route in raw.items():
            if isinstance(route, dict) and {"duration_min", "distance_km"}.intersection(route):
                add_pair("start_point", str(poi_id), route)

    return overlays


def normalize_route_leg(
    raw_leg: dict,
    *,
    from_label: str,
    to_label: str,
    from_id: str,
    to_id: str,
) -> dict:
    distance_km = to_float(raw_leg.get("distance_km"), 0.0)
    duration_min = raw_leg.get("travel_time_min")
    if duration_min is None:
        duration_min = raw_leg.get("duration_min")
    if duration_min is None and raw_leg.get("duration") is not None:
        duration_min = to_float(raw_leg.get("duration"), 0.0) / 60.0

    return {
        "from": from_label,
        "to": to_label,
        "from_id": from_id,
        "to_id": to_id,
        "distance_km": round(distance_km, 2),
        "travel_time_min": round(to_float(duration_min, 0.0), 1),
        "mode": raw_leg.get("mode", "driving"),
        "route_source": raw_leg.get("source") or raw_leg.get("route_source", "offline_route_overlay"),
        "walking_time_min": raw_leg.get("walking_time_min"),
        "traffic_status": raw_leg.get("traffic_status") or raw_leg.get("traffic_risk"),
        "feasible": raw_leg.get("feasible", True),
        "raw_route": raw_leg.get("raw_route"),
    }


@lru_cache(maxsize=256)
def fetch_live_route_leg(
    origin: str,
    destination: str,
    mode: str,
    city: str | None,
) -> dict | None:
    if not os.getenv("GAODE_API_KEY"):
        return None

    try:
        from .route_planner import RoutePlanner

        kwargs = {"city": city} if mode == "transit" and city else {}
        route = RoutePlanner().plan(origin, destination, mode=mode, **kwargs)
    except Exception:
        return None

    if not route.get("feasible", False):
        return {
            "feasible": False,
            "reason": route.get("reason"),
            "mode": mode,
            "source": "live_route_api",
        }

    distance_km = to_float(route.get("distance"), 0.0) / 1000.0
    duration_min = to_float(route.get("duration"), 0.0) / 60.0
    return {
        "distance_km": round(distance_km, 2),
        "duration_min": round(duration_min, 1),
        "mode": mode,
        "source": "live_route_api",
        "feasible": True,
        "walking_time_min": round(to_float(route.get("walking_distance"), 0.0) / 80.0, 1)
        if route.get("walking_distance") is not None
        else None,
        "traffic_status": "unknown",
        "raw_route": {
            "provider": "gaode_route_planner",
            "duration_text": route.get("duration_text"),
            "distance_text": route.get("distance_text"),
            "traffic_lights": route.get("traffic_lights"),
            "steps": route.get("steps"),
            "nightflag": route.get("nightflag"),
            "tolls": route.get("tolls"),
        },
    }


def live_route_leg_or_none(
    *,
    origin: str | None,
    destination: str | None,
    from_label: str,
    to_label: str,
    from_id: str,
    to_id: str,
    mode: str,
    city: str | None,
) -> dict | None:
    if not origin or not destination:
        return None
    raw_leg = fetch_live_route_leg(origin, destination, mode, city)
    if not raw_leg:
        return None
    return normalize_route_leg(
        raw_leg,
        from_label=from_label,
        to_label=to_label,
        from_id=from_id,
        to_id=to_id,
    )


def estimate_route_leg(
    from_item: dict | None,
    to_item: dict,
    *,
    from_id: str,
    to_id: str,
    fallback_distance_km: float,
) -> dict:
    from_coord = parse_coordinates((from_item or {}).get("coordinates"))
    to_coord = parse_coordinates(to_item.get("coordinates"))

    if from_coord and to_coord:
        # Road distance is usually longer than straight-line distance. Keep the
        # factor conservative so this remains a planning signal, not a fake API.
        distance_km = haversine_km(from_coord, to_coord) * 1.35
        route_source = "coordinate_estimate"
    else:
        distance_km = fallback_distance_km
        route_source = "poi_distance_fallback"

    mode = "walking" if distance_km <= 2.0 else "driving"
    if mode == "walking":
        travel_time_min = distance_km * 12 + 3
    else:
        travel_time_min = distance_km * 4.5 + 8

    return {
        "from": (from_item or {}).get("name", "start_point"),
        "to": to_item.get("name", to_id),
        "from_id": from_id,
        "to_id": to_id,
        "distance_km": round(distance_km, 2),
        "travel_time_min": round(travel_time_min, 1),
        "mode": mode,
        "route_source": route_source,
        "walking_time_min": round(travel_time_min, 1) if mode == "walking" else None,
        "traffic_status": "low" if distance_km <= 2.0 else ("moderate" if distance_km <= 8.0 else "high"),
        "feasible": True,
        "raw_route": None,
    }


def resolve_route_leg(
    *,
    source_order: list[str],
    overlay: dict | None,
    from_item: dict | None,
    to_item: dict,
    from_id: str,
    to_id: str,
    fallback_distance_km: float,
    live_origin: str | None = None,
    live_mode: str = "driving",
    live_city: str | None = None,
) -> dict:
    for source in source_order:
        if source in {"live_route_api", "c_route_check"}:
            destination = to_item.get("coordinates")
            origin = live_origin
            if origin is None and from_item:
                origin = from_item.get("coordinates")
            leg = live_route_leg_or_none(
                origin=origin,
                destination=destination,
                from_label=(from_item or {}).get("name", "start_point"),
                to_label=to_item.get("name", to_id),
                from_id=from_id,
                to_id=to_id,
                mode=live_mode,
                city=live_city,
            )
            if leg:
                return leg
        elif source in {"offline_routes_json", "offline_route_overlay"} and overlay:
            return normalize_route_leg(
                overlay,
                from_label=(from_item or {}).get("name", "start_point"),
                to_label=to_item.get("name", to_id),
                from_id=from_id,
                to_id=to_id,
            )
        elif source == "coordinate_estimate":
            if parse_coordinates((from_item or {}).get("coordinates")) and parse_coordinates(to_item.get("coordinates")):
                return estimate_route_leg(
                    from_item,
                    to_item,
                    from_id=from_id,
                    to_id=to_id,
                    fallback_distance_km=fallback_distance_km,
                )
        elif source == "poi_distance_fallback":
            return estimate_route_leg(
                None,
                to_item,
                from_id=from_id,
                to_id=to_id,
                fallback_distance_km=fallback_distance_km,
            )

    return estimate_route_leg(
        from_item,
        to_item,
        from_id=from_id,
        to_id=to_id,
        fallback_distance_km=fallback_distance_km,
    )


def route_traffic_status(legs: list[dict]) -> str:
    traffic_values = [str(leg.get("traffic_status") or "").lower() for leg in legs]
    if "high" in traffic_values:
        return "high"
    if "moderate" in traffic_values or "medium" in traffic_values:
        return "moderate"
    return "low"


def build_route_facts(
    activity: dict,
    restaurant: dict,
    constraints: dict | None = None,
    sequence: str = "activity_then_restaurant",
    *,
    mock_data_dir: Path | str,
    source_order: list[str],
) -> dict:
    overlays = load_route_overlays(str(Path(mock_data_dir).resolve()))
    origin_coordinates = route_origin_coordinates(constraints)
    route_mode = str((constraints or {}).get("route_mode") or "driving")
    route_city = destination_city_from_constraints(constraints) or (constraints or {}).get("city") or "上海"
    if sequence == "restaurant_then_activity":
        first_item = restaurant
        second_item = activity
    else:
        first_item = activity
        second_item = restaurant

    first_id = str(first_item.get("poi_id"))
    second_id = str(second_item.get("poi_id"))

    start_overlay = overlays.get(("start_point", first_id))
    start_leg = resolve_route_leg(
        source_order=source_order,
        overlay=start_overlay,
        from_item=None,
        to_item=first_item,
        from_id="start_point",
        to_id=first_id,
        fallback_distance_km=to_float(first_item.get("distance_km"), 0.0),
        live_origin=origin_coordinates,
        live_mode=route_mode,
        live_city=route_city,
    )

    transfer_overlay = overlays.get((first_id, second_id))
    transfer_leg = resolve_route_leg(
        source_order=source_order,
        overlay=transfer_overlay,
        from_item=first_item,
        to_item=second_item,
        from_id=first_id,
        to_id=second_id,
        fallback_distance_km=to_float(second_item.get("distance_km"), 0.0),
        live_mode=route_mode,
        live_city=route_city,
    )

    legs = [start_leg, transfer_leg]
    total_distance = sum(to_float(leg.get("distance_km"), 0.0) for leg in legs)
    total_travel_time = sum(to_float(leg.get("travel_time_min"), 0.0) for leg in legs)

    return {
        "total_distance_km": round(total_distance, 2),
        "total_travel_time_min": round(total_travel_time, 1),
        "traffic_status": route_traffic_status(legs),
        "route_sources": sorted({leg.get("route_source", "unknown") for leg in legs}),
        "legs": legs,
    }


def build_sequence_route_facts(
    nodes: list[dict],
    constraints: dict | None = None,
    *,
    mock_data_dir: Path | str,
    source_order: list[str],
) -> dict:
    overlays = load_route_overlays(str(Path(mock_data_dir).resolve()))
    origin_coordinates = route_origin_coordinates(constraints)
    route_mode = str((constraints or {}).get("route_mode") or "driving")
    route_city = destination_city_from_constraints(constraints) or (constraints or {}).get("city") or "上海"

    legs: list[dict] = []
    previous_item: dict | None = None
    previous_id = "start_point"
    for node in nodes:
        node_id = str(node.get("poi_id") or node.get("id") or "")
        if not node_id:
            continue
        overlay = overlays.get((previous_id, node_id))
        leg = resolve_route_leg(
            source_order=source_order,
            overlay=overlay,
            from_item=previous_item,
            to_item=node,
            from_id=previous_id,
            to_id=node_id,
            fallback_distance_km=to_float(node.get("distance_km"), 0.0),
            live_origin=origin_coordinates if previous_item is None else None,
            live_mode=route_mode,
            live_city=route_city,
        )
        legs.append(leg)
        previous_item = node
        previous_id = node_id

    total_distance = sum(to_float(leg.get("distance_km"), 0.0) for leg in legs)
    total_travel_time = sum(to_float(leg.get("travel_time_min"), 0.0) for leg in legs)
    feasible = all(leg.get("feasible", True) for leg in legs)

    return {
        "total_distance_km": round(total_distance, 2),
        "total_travel_time_min": round(total_travel_time, 1),
        "traffic_status": route_traffic_status(legs),
        "route_sources": sorted({leg.get("route_source", "unknown") for leg in legs}),
        "legs": legs,
        "feasible": feasible,
    }
