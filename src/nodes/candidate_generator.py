try:
    from src.state import PlanState
except ImportError:
    PlanState = dict

from functools import lru_cache
import json
import math
import os
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is optional for smoke demos
    yaml = None

from .mock_api_adapter import (
    fetch_activity_candidates,
    fetch_restaurant_candidates,
)
from .b_utils import (
    collect_preference_sources,
    derive_scenario_activities,
    expand_preference_tags,
    get_constraint_config_with_profile,
    to_float,
    get_scene_template,
    normalize_scene_type,
)



DEFAULT_TOP_K_ACTIVITY = 3
DEFAULT_TOP_K_RESTAURANT = 3
DEFAULT_TRANSITION_BUFFER_MIN = 30
DEFAULT_ROUTE_LOOKAHEAD_MULTIPLIER = 2
DEFAULT_PAIR_POOL_MULTIPLIER = 8
DEFAULT_PLAN_CANDIDATE_LIMIT = 96
DEFAULT_ROUTE_SOURCE_ORDER = ("offline_routes_json", "coordinate_estimate", "poi_distance_fallback")
DEFAULT_MOCK_DATA_DIR = Path(__file__).resolve().parents[2] / "experiments" / "mock_data"
PLAN_TAG_FIELDS = (
    "tags",
    "category",
    "sub_category",
    "experience_type",
    "restaurant_category",
    "service_mode",
    "emotion_tags",
    "atmosphere_tags",
    "local_character_tags",
    "health_tags",
    "menu_health_options",
    "wellness_tags",
    "local_flavor_tags",
)


def _policy_path() -> Path:
    override = os.environ.get("WF_PLANNER_POLICY_PATH", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "experiments" / "planner_policy.yaml"


def _policy_cache_key() -> str:
    return str(_policy_path().resolve())


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


def _get_candidate_generation_config() -> dict:
    policy = _load_policy_config(_policy_cache_key())
    candidate_generation = policy.get("candidate_generation")
    return candidate_generation if isinstance(candidate_generation, dict) else {}


def _get_top_k(name: str, default: int) -> int:
    raw_value = _get_candidate_generation_config().get(name, default)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(1, value)


def _get_route_lookahead_multiplier() -> int:
    raw_value = _get_candidate_generation_config().get(
        "route_lookahead_multiplier",
        DEFAULT_ROUTE_LOOKAHEAD_MULTIPLIER,
    )
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_ROUTE_LOOKAHEAD_MULTIPLIER
    return max(1, min(value, 5))


def _get_pair_pool_multiplier() -> int:
    raw_value = _get_candidate_generation_config().get(
        "pair_pool_multiplier",
        DEFAULT_PAIR_POOL_MULTIPLIER,
    )
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_PAIR_POOL_MULTIPLIER
    return max(1, min(value, 8))


def _get_plan_candidate_limit() -> int:
    raw_value = _get_candidate_generation_config().get(
        "plan_candidate_limit",
        DEFAULT_PLAN_CANDIDATE_LIMIT,
    )
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_PLAN_CANDIDATE_LIMIT
    return max(9, min(value, 300))


def _get_route_source_order() -> list[str]:
    env_order = os.environ.get("WF_ROUTE_SOURCE_ORDER", "").strip()
    if env_order:
        return [item.strip() for item in env_order.split(",") if item.strip()]

    raw_order = _get_candidate_generation_config().get("route_source_order")
    if isinstance(raw_order, list):
        values = [str(item).strip() for item in raw_order if str(item).strip()]
        if values:
            return values

    return list(DEFAULT_ROUTE_SOURCE_ORDER)


def _get_time_slot_policy() -> dict:
    time_slot = _get_candidate_generation_config().get("time_slot")
    return time_slot if isinstance(time_slot, dict) else {}


def _get_transition_buffer_min() -> int:
    raw_value = _get_time_slot_policy().get("default_transition_buffer_min", DEFAULT_TRANSITION_BUFFER_MIN)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_TRANSITION_BUFFER_MIN
    return max(0, value)


def _get_time_slot_bool(name: str, default: bool) -> bool:
    raw_value = _get_time_slot_policy().get(name, default)
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw_value)


def _mock_data_dir() -> Path:
    env_dir = os.environ.get("WF_MOCK_DATA_DIR", "").strip()
    if env_dir:
        path = Path(env_dir)
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[2] / path

    raw_dir = _get_candidate_generation_config().get("local_mock_dir")
    if raw_dir:
        path = Path(str(raw_dir))
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[2] / path

    return DEFAULT_MOCK_DATA_DIR


@lru_cache(maxsize=16)
def _load_route_overlays(mock_data_dir_key: str) -> dict[tuple[str, str], dict]:
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


def _parse_coordinates(value: object) -> tuple[float, float] | None:
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


def _haversine_km(coord_a: tuple[float, float], coord_b: tuple[float, float]) -> float:
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


def _normalize_route_leg(
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


def _route_origin_coordinates(constraints: dict | None) -> str | None:
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


@lru_cache(maxsize=256)
def _fetch_live_route_leg(
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


def _live_route_leg_or_none(
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
    raw_leg = _fetch_live_route_leg(origin, destination, mode, city)
    if not raw_leg:
        return None
    return _normalize_route_leg(
        raw_leg,
        from_label=from_label,
        to_label=to_label,
        from_id=from_id,
        to_id=to_id,
    )


def _estimate_route_leg(
    from_item: dict | None,
    to_item: dict,
    *,
    from_id: str,
    to_id: str,
    fallback_distance_km: float,
) -> dict:
    from_coord = _parse_coordinates((from_item or {}).get("coordinates"))
    to_coord = _parse_coordinates(to_item.get("coordinates"))

    if from_coord and to_coord:
        # Road distance is usually longer than straight-line distance. Keep the
        # factor conservative so this remains a planning signal, not a fake API.
        distance_km = _haversine_km(from_coord, to_coord) * 1.35
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


def _resolve_route_leg(
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
            leg = _live_route_leg_or_none(
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
            return _normalize_route_leg(
                overlay,
                from_label=(from_item or {}).get("name", "start_point"),
                to_label=to_item.get("name", to_id),
                from_id=from_id,
                to_id=to_id,
            )
        elif source == "coordinate_estimate":
            if _parse_coordinates((from_item or {}).get("coordinates")) and _parse_coordinates(to_item.get("coordinates")):
                return _estimate_route_leg(
                    from_item,
                    to_item,
                    from_id=from_id,
                    to_id=to_id,
                    fallback_distance_km=fallback_distance_km,
                )
        elif source == "poi_distance_fallback":
            return _estimate_route_leg(
                None,
                to_item,
                from_id=from_id,
                to_id=to_id,
                fallback_distance_km=fallback_distance_km,
            )

    return _estimate_route_leg(
        from_item,
        to_item,
        from_id=from_id,
        to_id=to_id,
        fallback_distance_km=fallback_distance_km,
    )


def _build_route_facts(activity: dict, restaurant: dict, constraints: dict | None = None) -> dict:
    overlays = _load_route_overlays(str(_mock_data_dir().resolve()))
    source_order = _get_route_source_order()
    origin_coordinates = _route_origin_coordinates(constraints)
    route_mode = str((constraints or {}).get("route_mode") or "driving")
    route_city = (constraints or {}).get("city") or "上海"
    activity_id = str(activity.get("poi_id"))
    restaurant_id = str(restaurant.get("poi_id"))

    start_overlay = overlays.get(("start_point", activity_id))
    start_leg = _resolve_route_leg(
        source_order=source_order,
        overlay=start_overlay,
        from_item=None,
        to_item=activity,
        from_id="start_point",
        to_id=activity_id,
        fallback_distance_km=to_float(activity.get("distance_km"), 0.0),
        live_origin=origin_coordinates,
        live_mode=route_mode,
        live_city=route_city,
    )

    transfer_overlay = overlays.get((activity_id, restaurant_id))
    transfer_leg = _resolve_route_leg(
        source_order=source_order,
        overlay=transfer_overlay,
        from_item=activity,
        to_item=restaurant,
        from_id=activity_id,
        to_id=restaurant_id,
        fallback_distance_km=to_float(restaurant.get("distance_km"), 0.0),
        live_mode=route_mode,
        live_city=route_city,
    )

    legs = [start_leg, transfer_leg]
    total_distance = sum(to_float(leg.get("distance_km"), 0.0) for leg in legs)
    total_travel_time = sum(to_float(leg.get("travel_time_min"), 0.0) for leg in legs)
    feasible = all(leg.get("feasible", True) for leg in legs)
    traffic_values = [str(leg.get("traffic_status") or "").lower() for leg in legs]
    if "high" in traffic_values:
        traffic_status = "high"
    elif "moderate" in traffic_values or "medium" in traffic_values:
        traffic_status = "moderate"
    else:
        traffic_status = "low"

    return {
        "total_distance_km": round(total_distance, 2),
        "total_travel_time_min": round(total_travel_time, 1),
        "traffic_status": traffic_status,
        "route_sources": sorted({leg.get("route_source", "unknown") for leg in legs}),
        "legs": legs,
    }


def _collect_plan_tags(*items: dict) -> list[str]:
    tags: list[str] = []
    for item in items:
        for field in PLAN_TAG_FIELDS:
            raw_values = item.get(field)
            if raw_values is None:
                continue
            if isinstance(raw_values, list):
                tags.extend(str(value).strip() for value in raw_values if str(value).strip())
            elif isinstance(raw_values, dict):
                for value in raw_values.values():
                    if isinstance(value, list):
                        tags.extend(str(part).strip() for part in value if str(part).strip())
                    elif value not in (None, ""):
                        tags.append(str(value).strip())
            elif raw_values not in ("", None):
                tags.append(str(raw_values).strip())

    seen = set()
    deduped = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    return deduped


def _is_supported_plan_template(template: list[str]) -> bool:
    # The current optimizer/action_hints path supports one activity plus one restaurant.
    return template.count("activity") == 1 and template.count("restaurant") == 1


def _get_plan_templates(scene_type: str) -> list[list[str]]:
    fallback = get_scene_template(scene_type)
    policy = _load_policy_config(_policy_cache_key())
    template_policy = policy.get("template_policy")
    if not isinstance(template_policy, dict):
        return [fallback]

    scene_templates = template_policy.get("scene_templates")
    raw_templates = []
    if isinstance(scene_templates, dict):
        raw_templates.append(scene_templates.get(scene_type))
    raw_templates.append(template_policy.get("default_template"))

    templates = []
    seen = set()
    for raw_template in raw_templates:
        if not isinstance(raw_template, list):
            continue
        template = [str(step).strip() for step in raw_template if str(step).strip()]
        key = tuple(template)
        if not _is_supported_plan_template(template) or key in seen:
            continue
        seen.add(key)
        templates.append(template)

    return templates or [fallback]

def _build_activity_candidates() -> list[dict]:
    return [
        {
            "poi_id": "act_001",
            "name": "亲子陶艺体验馆",
            "type": "activity",
            "tags": ["kid_friendly", "indoor", "low_intensity", "family_friendly"],
            "price": 198,
            "distance_km": 3.2,
            "duration_min": 90,
            "rating": 4.7,
            "queue_time_min": 5,
            "available": True,
            "available_slots": [{"time": "14:00"}, {"time": "15:30"}, {"time": "17:00"}],
            "location": "杨浦区大学路",
        },
        {
            "poi_id": "act_002",
            "name": "远郊探险亲子营",
            "type": "activity",
            "tags": ["outdoor", "adventure", "high_intensity"],
            "price": 250,
            "distance_km": 9.5,
            "duration_min": 120,
            "rating": 4.8,
            "queue_time_min": 20,
            "available": True,
            "available_slots": [{"time": "13:00"}, {"time": "15:00"}],
            "location": "浦东新区体育中心",
        },
        {
            "poi_id": "act_003",
            "name": "儿童室内乐园",
            "type": "activity",
            "tags": ["kid_friendly", "indoor", "playful"],
            "price": 160,
            "distance_km": 2.8,
            "duration_min": 100,
            "rating": 4.3,
            "queue_time_min": 80,
            "available": True,
            "available_slots": [{"time": "14:00"}, {"time": "16:00"}],
            "location": "闵行区万达广场",
        },
        {
            "poi_id": "act_004",
            "name": "都市密室逃脱",
            "type": "activity",
            "tags": ["indoor", "strategy", "high_intensity"],
            "price": 180,
            "distance_km": 1.5,
            "duration_min": 90,
            "rating": 4.9,
            "queue_time_min": 10,
            "available": True,
            "available_slots": [{"time": "14:00"}, {"time": "15:30"}],
            "location": "静安区商业街",
        },
    ]


def _build_restaurant_candidates() -> list[dict]:
    return [
        {
            "poi_id": "res_001",
            "name": "轻食日料餐厅",
            "type": "restaurant",
            "tags": ["low_calorie", "light_food", "family_friendly"],
            "price": 220,
            "distance_km": 1.1,
            "duration_min": 90,
            "rating": 4.6,
            "queue_time_min": 10,
            "available": True,
            "available_slots": [{"time": "17:30"}, {"time": "18:30"}, {"time": "19:30"}],
            "location": "五角场商圈",
        },
        {
            "poi_id": "res_002",
            "name": "健身餐盒餐厅",
            "type": "restaurant",
            "tags": ["low_calorie", "light_food"],
            "price": 180,
            "distance_km": 2.2,
            "duration_min": 80,
            "rating": 4.4,
            "queue_time_min": 25,
            "available": True,
            "available_slots": [{"time": "17:00"}, {"time": "18:00"}, {"time": "19:00"}],
            "location": "杨浦区友谊路",
        },
        {
            "poi_id": "res_003",
            "name": "本帮大排档",
            "type": "restaurant",
            "tags": ["high_calorie", "local_cuisine", "crowded_mall"],
            "price": 260,
            "distance_km": 3.8,
            "duration_min": 90,
            "rating": 4.8,
            "queue_time_min": 40,
            "available": True,
            "available_slots": [{"time": "17:00"}, {"time": "18:00"}, {"time": "19:00"}],
            "location": "浦东新区老街",
        },
        {
            "poi_id": "res_005",
            "name": "蔬菜沙拉店",
            "type": "restaurant",
            "tags": ["low_calorie", "light_food", "family_friendly"],
            "price": 150,
            "distance_km": 3.8,
            "duration_min": 60,
            "rating": 4.2,
            "queue_time_min": 15,
            "available": True,
            "available_slots": [{"time": "17:30"}, {"time": "18:30"}],
            "location": "浦东新区世纪大道",
        },
        {
            "poi_id": "res_004",
            "name": "附近快餐小店",
            "type": "restaurant",
            "tags": ["budget", "fast_food"],
            "price": 120,
            "distance_km": 0.8,
            "duration_min": 60,
            "rating": 3.8,
            "queue_time_min": 5,
            "available": True,
            "available_slots": [{"time": "17:00"}, {"time": "18:00"}],
            "location": "杨浦区小北路",
        },
    ]


def _slot_to_minutes(slot: str) -> int:
    if not slot or ":" not in str(slot):
        return -1
    hour, minute = str(slot).split(":", 1)
    return int(hour) * 60 + int(minute)


def _pick_time_slots(activity: dict, restaurant: dict, constraints: dict) -> tuple[str | None, str | None]:
    start_time = str(constraints.get("start_time") or "14:00")
    start_minutes = _slot_to_minutes(start_time)
    transition_buffer_min = _get_transition_buffer_min()
    prefer_earliest_activity = _get_time_slot_bool("prefer_earliest_valid_activity_slot", True)
    prefer_earliest_restaurant = _get_time_slot_bool("prefer_earliest_valid_restaurant_slot", True)
    minimize_transition_gap = _get_time_slot_bool("minimize_transition_gap", True)

    activity_slots = sorted(
        [slot.get("time") for slot in activity.get("available_slots", []) if slot.get("time")],
        key=_slot_to_minutes,
    )
    restaurant_slots = sorted(
        [slot.get("time") for slot in restaurant.get("available_slots", []) if slot.get("time")],
        key=_slot_to_minutes,
    )

    valid_activity_slots = [slot for slot in activity_slots if _slot_to_minutes(slot) >= start_minutes]
    if minimize_transition_gap:
        valid_pairs: list[tuple[int, int, int, str, str]] = []
        activity_pool = valid_activity_slots or activity_slots
        for activity_slot in activity_pool:
            activity_start_minutes = _slot_to_minutes(activity_slot)
            if activity_start_minutes < 0:
                continue
            activity_end_minutes = activity_start_minutes + int(activity.get("duration_min", 0))
            min_restaurant_minutes = activity_end_minutes + transition_buffer_min
            for restaurant_slot in restaurant_slots:
                restaurant_start_minutes = _slot_to_minutes(restaurant_slot)
                if restaurant_start_minutes < min_restaurant_minutes:
                    continue
                transition_gap = restaurant_start_minutes - activity_end_minutes
                valid_pairs.append(
                    (
                        transition_gap,
                        activity_start_minutes,
                        restaurant_start_minutes,
                        activity_slot,
                        restaurant_slot,
                    )
                )

        if valid_pairs:
            if prefer_earliest_activity and prefer_earliest_restaurant:
                valid_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
            elif prefer_earliest_activity:
                valid_pairs.sort(key=lambda item: (item[0], item[1], -item[2]))
            elif prefer_earliest_restaurant:
                valid_pairs.sort(key=lambda item: (item[0], -item[1], item[2]))
            else:
                valid_pairs.sort(key=lambda item: (item[0], -item[1], -item[2]))
            _, _, _, activity_start, restaurant_start = valid_pairs[0]
            return activity_start, restaurant_start

    if prefer_earliest_activity:
        activity_start = valid_activity_slots[0] if valid_activity_slots else (activity_slots[0] if activity_slots else None)
    else:
        activity_start = valid_activity_slots[-1] if valid_activity_slots else (activity_slots[-1] if activity_slots else None)

    if activity_start is None:
        return None, None

    min_restaurant_minutes = _slot_to_minutes(activity_start) + int(activity.get("duration_min", 0)) + transition_buffer_min
    valid_restaurant_slots = [slot for slot in restaurant_slots if _slot_to_minutes(slot) >= min_restaurant_minutes]
    if prefer_earliest_restaurant:
        restaurant_start = valid_restaurant_slots[0] if valid_restaurant_slots else None
    else:
        restaurant_start = valid_restaurant_slots[-1] if valid_restaurant_slots else None

    return activity_start, restaurant_start


def _sort_candidates(
    candidates: list[dict],
    constraints: dict,
    scene_type: str,
    user_profile: dict = None,
    scenario_activities: list = None,
) -> list[dict]:
    user_profile = user_profile or {}
    scenario_activities = derive_scenario_activities(constraints, user_profile, scenario_activities)
    config = get_constraint_config_with_profile(constraints, user_profile)

    child_age = config["child_age"]
    mom_diet = config["mom_diet"]
    preference_tags = list(set(collect_preference_sources(constraints, user_profile, scenario_activities)))

    max_distance_km = config["max_distance_km"]
    max_queue_time = config["max_queue_time"]

    def score(item: dict) -> float:
        base = item.get("rating", 0) * 2

        if not item.get("available", True):
            base -= 12

        distance = to_float(item.get("distance_km"), 0.0)
        if distance > max_distance_km:
            base -= 6 + (distance - max_distance_km)
        else:
            base += max(0.0, (max_distance_km - distance) * 0.1)

        queue_time = to_float(item.get("queue_time_min"), 0.0)
        if queue_time > max_queue_time:
            base -= 4 + (queue_time - max_queue_time) * 0.1
        else:
            base += max(0.0, (max_queue_time - queue_time) * 0.05)

        item_type = item.get("type", "")
        tags = _collect_plan_tags(item)
        normalized_tags = expand_preference_tags(tags) + tags
        normalized_tags = list(set(normalized_tags))

        if child_age is not None and child_age <= 6 and item_type == "activity":
            if "kid_friendly" in normalized_tags or "low_intensity" in normalized_tags:
                base += 6

        if mom_diet == "low_calorie" and item_type == "restaurant":
            if "low_calorie" in normalized_tags or "light_food" in normalized_tags:
                base += 5

        if scene_type == "low_budget" and item_type == "restaurant":
            price = to_float(item.get("price", 0.0), 0.0)
            if price <= 180:
                base += 4
            else:
                base -= 2

        match_count = sum(1 for tag in normalized_tags if tag in preference_tags)
        base += match_count * 2

        if item_type == "restaurant" and "budget" in preference_tags and "budget" in normalized_tags:
            base += 2

        if item_type == "activity" and "nearby" in preference_tags and distance <= max_distance_km:
            base += 1

        return base

    return sorted(candidates, key=score, reverse=True)


def _sort_plan_candidates(
    plan_candidates: list[dict],
    constraints: dict,
    scene_type: str,
    user_profile: dict | None = None,
    scenario_activities: list | None = None,
) -> list[dict]:
    user_profile = user_profile or {}
    scenario_activities = derive_scenario_activities(constraints, user_profile, scenario_activities)
    config = get_constraint_config_with_profile(constraints, user_profile)
    max_distance_km = config["max_distance_km"]
    max_queue_time = config["max_queue_time"]
    duration_range = config["duration_range"]
    budget = config["budget"]
    child_age = config["child_age"]
    mom_diet = config["mom_diet"]
    preference_tags = set(collect_preference_sources(constraints, user_profile, scenario_activities))

    def score(plan: dict) -> float:
        route = plan.get("route", {}) or {}
        budget_info = plan.get("budget", {}) or {}
        availability = plan.get("availability", {}) or {}
        nodes = plan.get("nodes", []) or []
        activity = nodes[0] if nodes else {}
        restaurant = nodes[1] if len(nodes) > 1 else {}
        tags = set(expand_preference_tags(plan.get("tags", []) or []))

        total_distance = to_float(route.get("total_distance_km"), 0.0)
        max_queue = to_float(availability.get("max_queue_time_min"), 0.0)
        total_price = to_float(budget_info.get("total_price"), 0.0)
        estimated_duration = to_float(plan.get("estimated_duration_min"), 0.0)

        value = 0.0
        value += 30.0 if availability.get("all_available", False) else -80.0

        if total_distance <= max_distance_km:
            value += 35.0 * (1.0 - total_distance / max(1.0, max_distance_km))
        else:
            value -= min(60.0, (total_distance - max_distance_km) * 4.0)

        if max_queue <= max_queue_time:
            value += 15.0 * (1.0 - max_queue / max(1.0, max_queue_time))
        else:
            value -= min(35.0, (max_queue - max_queue_time) * 1.5)

        if duration_range[0] <= estimated_duration <= duration_range[1]:
            value += 8.0
        else:
            value -= 12.0

        if total_price <= budget:
            value += 10.0
        elif total_price <= budget * 1.2:
            value += 3.0
        else:
            value -= min(35.0, (total_price - budget * 1.2) / 20.0)

        activity_tags = set(activity.get("tags", []) or [])
        restaurant_tags = set(restaurant.get("tags", []) or [])
        restaurant_health_tags = set(restaurant.get("health_tags", []) or [])
        menu_health_options = set(restaurant.get("menu_health_options", []) or [])

        if child_age is not None and child_age <= 6:
            value += 12.0 if activity_tags.intersection({"kid_friendly", "low_intensity"}) else -20.0
        if mom_diet == "low_calorie":
            health_signals = restaurant_tags | restaurant_health_tags | menu_health_options
            value += (
                12.0
                if health_signals.intersection(
                    {"low_calorie", "light_food", "low_oil", "low_sugar", "high_protein", "vegetable_rich"}
                )
                else -20.0
            )

        if scene_type == "couple" and tags.intersection(
            {"date_friendly", "romantic", "relaxation", "healing", "micro_vacation", "spa"}
        ):
            value += 8.0
        if scene_type == "friends" and tags.intersection(
            {"social", "group_friendly", "escape_room", "board_game", "sports"}
        ):
            value += 8.0
        if scene_type == "low_budget" and total_price <= budget * 1.2:
            value += 8.0

        value += min(12.0, len(preference_tags.intersection(tags)) * 3.0)
        value += to_float(activity.get("rating"), 4.0) + to_float(restaurant.get("rating"), 4.0)
        return value

    return sorted(plan_candidates, key=score, reverse=True)


def _combine_plan_candidates(
    activities: list[dict],
    restaurants: list[dict],
    constraints: dict,
    scene_type: str,
    user_profile: dict | None = None,
) -> list[dict]:
    plan_candidates = []
    plan_index = 1
    plan_template = _get_plan_templates(scene_type)[0]
    config = get_constraint_config_with_profile(constraints, user_profile)

    max_distance_km = config["max_distance_km"]
    max_queue_time = config["max_queue_time"]
    budget = config["budget"]
    child_age_value = config["child_age"]
    mom_diet = config["mom_diet"]
    people_count = config["people_count"]

    for activity in activities:
        for restaurant in restaurants:
            activity_start, restaurant_start = _pick_time_slots(activity, restaurant, constraints)
            if not activity_start or not restaurant_start:
                continue

            route_facts = _build_route_facts(activity, restaurant, constraints)
            total_distance_km = route_facts["total_distance_km"]
            total_travel_time_min = route_facts["total_travel_time_min"]
            total_price = activity["price"] + restaurant["price"]
            max_queue_time_plan = max(activity["queue_time_min"], restaurant["queue_time_min"])
            available = activity["available"] and restaurant["available"]
            activity_end_minutes = _slot_to_minutes(activity_start) + int(activity["duration_min"])
            restaurant_start_minutes = _slot_to_minutes(restaurant_start)
            estimated_duration_min = max(
                240,
                restaurant_start_minutes + int(restaurant["duration_min"]) - _slot_to_minutes(activity_start),
            )
            tags = _collect_plan_tags(activity, restaurant)

            constraint_snapshot = {
                "max_distance_km": max_distance_km,
                "max_queue_time": max_queue_time,
                "budget": budget,
                "child_age": child_age_value,
                "mom_diet": mom_diet,
            }

            availability_detail = {
                "activity_available": activity.get("available", True),
                "restaurant_available": restaurant.get("available", True),
                "activity_queue_min": activity.get("queue_time_min", 0),
                "restaurant_queue_min": restaurant.get("queue_time_min", 0),
                "total_queue_min": max_queue_time_plan,
            }

            if scene_type == "couple":
                min_people = 2
            elif scene_type == "friends":
                min_people = 3
            else:
                min_people = 4

            execution_requirements = {
                "min_people": min_people,
                "people_count": people_count,
                "adult_count": 2 if scene_type == "family" else 1,
                "child_count": 1 if scene_type == "family" else 0,
                "pre_booking_required": (
                    activity.get("queue_time_min", 0) > 20
                    or restaurant.get("queue_time_min", 0) > 30
                ),
                "special_preparation": [],
            }

            if mom_diet == "low_calorie":
                if "low_calorie" not in restaurant.get("tags", []):
                    execution_requirements["special_preparation"].append("提前告知餐厅低卡需求")

            if child_age_value is not None and child_age_value <= 3:
                execution_requirements["special_preparation"].append("携带儿童座椅或垫子")
                execution_requirements["special_preparation"].append("准备纸尿裤和湿巾")

            plan_candidates.append(
                {
                    "plan_id": f"cand_{plan_index:03d}",
                    "scene_type": scene_type,
                    "plan_template": plan_template,
                    "nodes": [activity, restaurant],
                    "route": {
                        "total_distance_km": total_distance_km,
                        "total_travel_time_min": total_travel_time_min,
                        "traffic_status": route_facts.get("traffic_status"),
                        "route_sources": route_facts.get("route_sources", []),
                        "legs": route_facts["legs"],
                    },
                    "schedule": {
                        "activity_start": activity_start,
                        "activity_end": f"{activity_end_minutes // 60:02d}:{activity_end_minutes % 60:02d}",
                        "restaurant_start": restaurant_start,
                    },
                    "budget": {
                        "total_price": total_price,
                    },
                    "availability": {
                        "all_available": available,
                        "max_queue_time_min": max_queue_time_plan,
                        "detail": availability_detail,
                    },
                    "estimated_duration_min": estimated_duration_min,
                    "tags": tags,
                    "constraint_snapshot": constraint_snapshot,
                    "execution_requirements": execution_requirements,
                }
            )
            plan_index += 1

    return plan_candidates


def candidate_generator_node(state: PlanState) -> dict:
    execution_log = state.get("execution_log", [])
    if state.get("need_confirm"):
        execution_log.append(
            "[B] candidate_generator_node waiting for user confirmation; skip candidate generation"
        )
        return {
            "candidates": [],
            "execution_log": execution_log,
        }

    scene_type = normalize_scene_type(state.get("scene_type", "family"))
    constraints = state.get("constraints", {})
    user_profile = state.get("user_profile", {})
    scenario_activities = derive_scenario_activities(
        constraints,
        user_profile,
        state.get("scenario_activities", []),
    )

    top_k_activity = _get_top_k("top_k_activity", DEFAULT_TOP_K_ACTIVITY)
    top_k_restaurant = _get_top_k("top_k_restaurant", DEFAULT_TOP_K_RESTAURANT)
    route_lookahead_multiplier = _get_route_lookahead_multiplier()
    pair_pool_multiplier = _get_pair_pool_multiplier()
    plan_candidate_limit = _get_plan_candidate_limit()

    activity_candidates = fetch_activity_candidates(
        constraints=constraints,
        scene_type=scene_type,
        scenario_activities=scenario_activities,
    ) or _build_activity_candidates()
    restaurant_candidates = fetch_restaurant_candidates(
        constraints=constraints,
        scene_type=scene_type,
        scenario_activities=scenario_activities,
    ) or _build_restaurant_candidates()

    activity_pool_size = top_k_activity * route_lookahead_multiplier * pair_pool_multiplier
    restaurant_pool_size = top_k_restaurant * route_lookahead_multiplier * pair_pool_multiplier

    selected_activities = _sort_candidates(
        activity_candidates,
        constraints,
        scene_type,
        user_profile,
        scenario_activities,
    )[:activity_pool_size]
    selected_restaurants = _sort_candidates(
        restaurant_candidates,
        constraints,
        scene_type,
        user_profile,
        scenario_activities,
    )[:restaurant_pool_size]

    raw_plan_candidates = _combine_plan_candidates(
        selected_activities,
        selected_restaurants,
        constraints,
        scene_type,
        user_profile,
    )
    plan_candidates = _sort_plan_candidates(
        raw_plan_candidates,
        constraints,
        scene_type,
        user_profile,
        scenario_activities,
    )[:plan_candidate_limit]

    execution_log.append(
        f"[B] candidate_generator_node 生成 {len(plan_candidates)} 个 plan_candidates "
        f"(activities={len(activity_candidates)}, restaurants={len(restaurant_candidates)}, "
        f"top_k_activity={top_k_activity}, top_k_restaurant={top_k_restaurant}, "
        f"route_lookahead_multiplier={route_lookahead_multiplier}, "
        f"pair_pool_multiplier={pair_pool_multiplier}, raw_plan_candidates={len(raw_plan_candidates)})"
    )

    return {
        "candidates": plan_candidates,
        "scene_type": scene_type,
        "scenario_activities": scenario_activities,
        "execution_log": execution_log,
    }
