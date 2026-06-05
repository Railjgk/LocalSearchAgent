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
    _normalize_poi as _normalize_mock_poi,
    fetch_activity_candidates,
    fetch_restaurant_candidates,
)
from .b_ai_hints import apply_b_semantic_hints
from .b_itinerary_blueprint import apply_b_itinerary_blueprint
from .b_rag_contract import normalize_rag_node_candidates, rag_candidate_coverage
from .b_requirement_compiler import apply_b_requirement_contract
from .b_semantics import (
    B_ACTIVITY_INTENT_GROUPS,
    B_RESTAURANT_INTENT_GROUPS,
    b_semantic_terms,
    flatten_semantic_values,
    normalize_semantic_text,
    semantic_groups_in_values,
    semantic_groups_for_item,
    semantic_match_score,
    semantic_terms_for_groups,
)
from .weather_client import get_weather_context
from .b_utils import (
    collect_preference_sources,
    derive_scenario_activities,
    destination_city_from_constraints,
    expand_preference_tags,
    get_constraint_config_with_profile,
    item_matches_destination_city,
    parse_duration_range,
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
DEFAULT_MAX_PAIR_COMBINATIONS = 768
DEFAULT_MAX_MULTINODE_CANDIDATES = 24
DEFAULT_ROUTE_SOURCE_ORDER = ("offline_routes_json", "coordinate_estimate", "poi_distance_fallback")
DEFAULT_MOCK_DATA_DIR = Path(__file__).resolve().parents[2] / "experiments" / "mock_data"
DEFAULT_SHANGHAI_ORIGIN = (121.4737, 31.2304)
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
MULTINODE_SUPPORTED_DOMAINS = {"activity", "restaurant"}
GUIDANCE_ONLY_ITINERARY_ROLES = {
    "citywalk_market",
    "park_scenic_walk",
    "convenience_store",
    "souvenir_shopping",
    "parking",
    "nail_salon",
    "pet_grooming",
    "pet_hospital",
    "pet_store",
}
CURRENT_C_EXECUTABLE_NODE_TYPES = {"activity", "restaurant"}
MULTINODE_ROLE_TERMS = {
    "family_activity": ["亲子", "儿童", "孩子", "kid_friendly", "family_friendly", "low_intensity", "indoor"],
    "family_indoor_play": ["室内乐园", "亲子乐园", "儿童乐园", "游乐园", "淘气堡", "蹦床", "kid_friendly", "family_friendly", "indoor_playground"],
    "exhibition": ["展览", "看展", "博物馆", "美术馆", "museum", "art", "exhibition"],
    "citywalk_market": ["citywalk", "城市漫步", "市集", "街区", "历史文化", "历史建筑", "文化街区", "文化景区", "local_market", "local_culture"],
    "park_scenic_walk": ["公园", "游园", "绿地", "滨江", "江边", "河边", "夜景", "散步", "步道", "outdoor", "citywalk"],
    "board_game_escape": ["桌游", "棋牌", "剧本杀", "狼人杀", "密室", "密室逃脱", "board_game", "escape_room"],
    "internet_cafe": ["网吧", "网咖", "电竞", "电玩", "游戏", "通宵", "点播影院", "电影", "internet_cafe", "game"],
    "karaoke": ["KTV", "ktv", "唱歌", "卡拉OK", "karaoke"],
    "bar": ["酒吧", "清吧", "喝一杯", "小酌", "鸡尾酒", "精酿", "bar"],
    "talk_show": ["脱口秀", "喜剧", "剧场", "演出", "livehouse"],
    "cinema": ["电影", "影院", "电影院", "观影", "IMAX"],
    "restaurant_breakfast": ["早餐", "早饭", "包子", "馄饨", "早点", "breakfast"],
    "restaurant_lunch": ["午餐", "中饭", "正餐", "full_meal"],
    "restaurant_dinner": ["晚餐", "晚饭", "正餐", "full_meal"],
    "restaurant_specific": ["餐厅", "吃饭", "正餐", "本帮菜", "上海菜", "小笼包", "小笼", "生煎", "汤包", "full_meal"],
    "cafe": ["咖啡", "下午茶", "甜品", "cafe", "dessert"],
    "cultural_photo": ["汉服", "拍照", "写真", "摄影", "古装", "换装", "文化体验"],
    "tea_house": ["茶艺", "茶馆", "茶室", "品茶", "喝茶", "tea", "teahouse"],
    "nail_salon": ["美甲", "美睫", "甲油胶", "做指甲"],
    "pet_grooming": ["宠物美容", "宠物spa", "宠物洗澡", "宠物洗护", "金毛", "毛发"],
    "pet_cafe": ["宠物友好咖啡", "宠物友好", "可带宠物", "带狗咖啡"],
    "pet_hospital": ["宠物医院", "宠物体检", "兽医", "动物医院"],
    "pet_store": ["宠物店", "宠物用品", "营养品", "狗粮", "猫粮"],
    "dental_clinic": ["牙科", "口腔", "牙医", "洗牙", "补牙", "种植牙", "矫正"],
    "sports_training": ["足球", "足球培训", "足球训练", "青训", "体育培训", "教练", "培训班"],
    "travel_agency": ["旅行社", "出境游", "签证", "办签证", "旅游团", "跟团游", "特价旅游"],
}


def _node_requires_c_execution(node: dict) -> bool:
    role = str(node.get("itinerary_role") or node.get("role") or "")
    node_type = str(node.get("type") or node.get("supply_domain") or "")
    if role in GUIDANCE_ONLY_ITINERARY_ROLES:
        return False
    if role == "lodging" or node_type in {"hotel", "lodging"}:
        return True
    return node_type in CURRENT_C_EXECUTABLE_NODE_TYPES


def _node_is_supported_by_current_c(node: dict) -> bool:
    node_type = str(node.get("type") or node.get("supply_domain") or "")
    return node_type in CURRENT_C_EXECUTABLE_NODE_TYPES
STRICT_MULTINODE_ROLE_TEXT_TERMS = {
    "exhibition": ("美术馆", "博物馆", "展览", "展馆", "艺术馆", "画廊", "文化馆", "艺术", "历史", "museum", "gallery", "exhibition"),
    "board_game_escape": ("剧本杀", "密室", "桌游", "推理", "狼人杀", "血染钟楼", "escape_room", "board_game"),
    "cafe": ("咖啡", "咖啡馆", "咖啡厅", "下午茶", "甜品", "蛋糕", "烘焙", "cafe", "coffee", "dessert"),
    "cultural_photo": ("汉服", "古风", "古装", "拍照", "写真", "摄影", "换装", "文化体验"),
    "park_scenic_walk": ("公园", "游园", "绿地", "滨江", "江边", "河边", "夜景", "散步", "步道", "观景", "外滩"),
    "tea_house": ("茶馆", "茶艺", "茶室", "品茶", "喝茶", "teahouse", "tea"),
    "convenience_store": ("便利店", "超市", "全家", "罗森", "7-eleven", "711", "便利", "零食", "饮料"),
    "parking": ("停车", "停车场", "车库", "车位", "parking"),
    "wellness_massage": ("spa", "按摩", "足疗", "推拿", "养生", "洗脚", "修脚"),
    "fitness": ("健身", "瑜伽", "普拉提", "运动", "fitness", "yoga", "pilates"),
    "lodging": ("酒店", "民宿", "住宿", "宾馆", "hotel", "lodging"),
    "restaurant_breakfast": ("早餐", "早饭", "早点", "包子", "馄饨", "豆浆", "粥", "生煎", "breakfast"),
    "internet_cafe": ("网吧", "网咖", "电竞", "电竞馆", "电玩", "游戏", "通宵", "点播影院", "电影"),
    "bar": ("酒吧", "清吧", "喝一杯", "小酌", "鸡尾酒", "精酿", "夜店"),
    "talk_show": ("脱口秀", "喜剧", "剧场", "演出", "livehouse"),
    "cinema": ("电影", "影院", "电影院", "观影", "IMAX"),
    "nail_salon": ("美甲", "美睫", "甲油胶", "做指甲"),
    "pet_grooming": ("宠物美容", "宠物spa", "宠物洗澡", "宠物洗护", "金毛", "毛发"),
    "pet_cafe": ("宠物友好咖啡", "宠物友好", "可带宠物", "带狗咖啡"),
    "pet_hospital": ("宠物医院", "宠物体检", "兽医", "动物医院"),
    "pet_store": ("宠物店", "宠物用品", "营养品", "狗粮", "猫粮"),
    "dental_clinic": ("牙科", "口腔", "牙医", "洗牙", "补牙", "种植牙", "矫正"),
    "sports_training": (
        "足球培训",
        "足球训练",
        "足球青训",
        "足球教练",
        "少儿足球",
        "青训",
    ),
    "travel_agency": ("旅行社", "出境游", "签证", "办签证", "旅游团", "跟团游", "特价旅游"),
}
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
COMPACT_SEMANTIC_FIELDS = (
    "name",
    "category",
    "sub_category",
    "experience_type",
    "restaurant_category",
    "primary_category",
    "primary_keyword",
    "gaode_keyword",
    "tags",
    "tag_groups",
    "health_tags",
    "menu_health_options",
    "signature_dishes",
    "recommended_dishes",
    "dish_tags",
    "review_keywords",
)
STRICT_MULTINODE_ROLE_FIELDS = (
    "name",
    "category",
    "sub_category",
    "experience_type",
    "restaurant_category",
    "primary_category",
    "primary_keyword",
    "gaode_keyword",
    "gaode_type",
    "service_facilities",
    "parking_fee_policy",
    "source_evidence",
)
RESTAURANT_ROLE_FIELDS = (
    "name",
    "category",
    "restaurant_category",
    "primary_category",
    "primary_keyword",
    "gaode_keyword",
    "gaode_type",
)
_SEMANTIC_TEXT_CACHE_LIMIT = 60000
_SEMANTIC_TEXT_CACHE: dict[tuple[int, tuple[str, ...]], tuple[tuple, str, set[str]]] = {}
_ITEM_TAG_CACHE_LIMIT = 60000
_ITEM_TAG_CACHE: dict[tuple[int, tuple], tuple[tuple, tuple[str, ...]]] = {}
_ITEM_SEMANTIC_SIGNAL_CACHE: dict[tuple[int, tuple], tuple[tuple, set[str]]] = {}
OUTDOOR_ACTIVITY_CATEGORIES = {"citywalk", "local_market", "sports"}
INDOOR_SAFE_TAGS = {"indoor", "museum", "handcraft", "indoor_playground", "escape_room"}
STRICT_ACTIVITY_REQUIREMENT_TAGS = {
    "karaoke",
    "citywalk",
    "local_market",
    "micro_vacation",
    "wellness",
    "museum",
    "handcraft",
    "art_experience",
    "amusement",
    "sports",
    "escape_room",
    "indoor_playground",
}
STRICT_RESTAURANT_REQUIREMENT_TAGS = {
    "hotpot",
    "bbq",
    "japanese",
    "regional_home_cuisine",
}


def _semantic_cache_signature(item: dict) -> tuple:
    return (
        item.get("poi_id") or item.get("id"),
        item.get("name"),
        item.get("category"),
        item.get("sub_category"),
        item.get("experience_type"),
        item.get("restaurant_category"),
        item.get("primary_category"),
        item.get("primary_keyword"),
        item.get("gaode_keyword"),
    )


def _item_cache_key(item: dict) -> tuple[int, tuple]:
    return (id(item), _semantic_cache_signature(item))


def _cache_item_tags(item: dict, tags: list[str]) -> tuple[str, ...]:
    if len(_ITEM_TAG_CACHE) >= _ITEM_TAG_CACHE_LIMIT:
        _ITEM_TAG_CACHE.clear()
        _ITEM_SEMANTIC_SIGNAL_CACHE.clear()
    cached = tuple(tags)
    _ITEM_TAG_CACHE[_item_cache_key(item)] = (_semantic_cache_signature(item), cached)
    return cached


def _normalized_query_terms(values, *, expand_semantics: bool = False) -> list[str]:
    raw_terms = b_semantic_terms(values, include_auxiliary=True) if expand_semantics else flatten_semantic_values(values)
    result: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        normalized = normalize_semantic_text(term)
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _semantic_text_index(
    item: dict,
    *,
    fields: tuple[str, ...] = COMPACT_SEMANTIC_FIELDS,
) -> tuple[str, set[str]]:
    cache_key = (id(item), fields)
    signature = _semantic_cache_signature(item)
    cached = _SEMANTIC_TEXT_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1], cached[2]

    values: list[str] = []
    for field_name in fields:
        values.extend(flatten_semantic_values(item.get(field_name)))

    normalized_values = [
        normalize_semantic_text(value)
        for value in values
        if len(normalize_semantic_text(value)) >= 2
    ]
    value_set = set(normalized_values)
    blob = "\n".join(normalized_values)

    if len(_SEMANTIC_TEXT_CACHE) >= _SEMANTIC_TEXT_CACHE_LIMIT:
        _SEMANTIC_TEXT_CACHE.clear()
    _SEMANTIC_TEXT_CACHE[cache_key] = (signature, blob, value_set)
    return blob, value_set


def _fast_text_match_score(
    normalized_terms: list[str],
    item: dict,
    *,
    fields: tuple[str, ...] = COMPACT_SEMANTIC_FIELDS,
) -> float:
    if not normalized_terms or not item:
        return 0.0
    blob, value_set = _semantic_text_index(item, fields=fields)
    if not blob:
        return 0.0

    best = 0.0
    for term in normalized_terms:
        if term in value_set:
            best = max(best, 3.2)
        elif term in blob:
            best = max(best, 2.3)
    return best
SEQUENCE_ACTIVITY_THEN_RESTAURANT = "activity_then_restaurant"
SEQUENCE_RESTAURANT_THEN_ACTIVITY = "restaurant_then_activity"
RESTAURANT_THEN_ACTIVITY_PHRASES = (
    "吃完",
    "饭后",
    "餐后",
    "用餐后",
    "吃完饭",
    "吃完火锅",
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


def _allow_legacy_fallback_for_multinode() -> bool:
    raw_value = os.environ.get("WF_B_ALLOW_LEGACY_FALLBACK_FOR_MULTINODE", "").strip().lower()
    if raw_value:
        return raw_value in TRUTHY_ENV_VALUES

    config = _get_candidate_generation_config()
    return bool(config.get("allow_legacy_fallback_for_multinode", False))


def _can_plan_multinode_with_current_supply(blueprint: dict | None) -> bool:
    """Return true when the current local activity/restaurant supply can cover the blueprint."""

    blueprint = blueprint or {}
    if blueprint.get("template_mode") != "multi_node":
        return False
    if blueprint.get("unsupported_roles"):
        return False
    if blueprint.get("named_entities"):
        # Exact venue/event names need retrieval evidence rather than generic local pools.
        return False
    node_intents = blueprint.get("node_intents") or []
    if len(node_intents) <= 2:
        return False
    return all(
        str(intent.get("supply_domain") or "") in MULTINODE_SUPPORTED_DOMAINS
        for intent in node_intents
    )


def _can_plan_multinode_with_rag(blueprint: dict | None, coverage: dict | None) -> bool:
    """Return true when retrieval can cover nodes the local pair supply cannot."""

    blueprint = blueprint or {}
    coverage = coverage or {}
    if blueprint.get("template_mode") != "multi_node":
        return False
    node_intents = blueprint.get("node_intents") or []
    if not node_intents:
        return False
    if coverage.get("all_nodes_covered"):
        return True
    if blueprint.get("named_entities"):
        return False
    if not coverage.get("unsupported_roles_covered"):
        return False
    return all(
        str(intent.get("supply_domain") or "") in MULTINODE_SUPPORTED_DOMAINS
        or str(intent.get("node_id") or "") in set(coverage.get("covered_node_ids") or [])
        for intent in node_intents
    )


def _apply_blueprint_duration_defaults(constraints: dict, blueprint: dict | None) -> dict:
    """Widen duration defaults when the request itself asks for a longer itinerary."""

    blueprint = blueprint or {}
    if blueprint.get("template_mode") != "multi_node":
        return constraints
    if constraints.get("duration_range") not in (None, "") or constraints.get("duration") not in (None, ""):
        return constraints

    horizon = blueprint.get("planning_horizon")
    enhanced = dict(constraints)
    if horizon in {"overnight", "two_day"}:
        enhanced["duration_range"] = [480, 1200]
    elif horizon == "full_day":
        enhanced["duration_range"] = [420, 720]
    elif int(blueprint.get("node_count") or len(blueprint.get("node_intents") or [])) >= 3:
        enhanced["duration_range"] = [180, 540]
    return enhanced


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


def _get_max_pair_combinations() -> int:
    raw_value = _get_candidate_generation_config().get(
        "max_pair_combinations",
        DEFAULT_MAX_PAIR_COMBINATIONS,
    )
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_PAIR_COMBINATIONS
    return max(1, min(value, 10000))


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


def _item_coordinates(item: dict) -> tuple[float, float] | None:
    coordinates = _parse_coordinates(item.get("coordinates"))
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


def _looks_like_shanghai_supply(candidates: list[dict]) -> bool:
    inspected = 0
    hits = 0
    for item in candidates[:200]:
        coord = _item_coordinates(item)
        if not coord:
            continue
        inspected += 1
        lng, lat = coord
        if 120.8 <= lng <= 122.2 and 30.6 <= lat <= 31.9:
            hits += 1
    return inspected > 0 and hits / inspected >= 0.75


def _geo_prefilter_origin(
    constraints: dict | None,
    candidates: list[dict],
) -> tuple[float, float] | None:
    explicit_origin = _parse_coordinates(_route_origin_coordinates(constraints))
    if explicit_origin:
        return explicit_origin

    constraints = constraints or {}
    destination_city = destination_city_from_constraints(constraints)
    if destination_city and "上海" not in destination_city:
        return None
    city_values = [
        destination_city or constraints.get("city"),
        constraints.get("district"),
        (constraints.get("location") or {}).get("city") if isinstance(constraints.get("location"), dict) else None,
    ]
    if any("上海" in str(value) for value in city_values if value):
        return DEFAULT_SHANGHAI_ORIGIN
    if _looks_like_shanghai_supply(candidates):
        return DEFAULT_SHANGHAI_ORIGIN
    return None


def _distance_from_origin_km(
    item: dict,
    origin: tuple[float, float] | None,
) -> float | None:
    coord = _item_coordinates(item)
    if origin and coord:
        return _haversine_km(origin, coord) * 1.25
    if item.get("distance_km") is not None:
        return to_float(item.get("distance_km"), 999.0)
    return None


def _filter_candidates_by_geo_window(
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
    origin = _geo_prefilter_origin(constraints, candidates)
    scored: list[tuple[float, dict]] = []
    unknown_distance: list[dict] = []

    for item in candidates:
        distance_km = _distance_from_origin_km(item, origin)
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


def _build_route_facts(
    activity: dict,
    restaurant: dict,
    constraints: dict | None = None,
    sequence: str = SEQUENCE_ACTIVITY_THEN_RESTAURANT,
) -> dict:
    overlays = _load_route_overlays(str(_mock_data_dir().resolve()))
    source_order = _get_route_source_order()
    origin_coordinates = _route_origin_coordinates(constraints)
    route_mode = str((constraints or {}).get("route_mode") or "driving")
    route_city = destination_city_from_constraints(constraints) or "上海"
    if sequence == SEQUENCE_RESTAURANT_THEN_ACTIVITY:
        first_item = restaurant
        second_item = activity
    else:
        first_item = activity
        second_item = restaurant

    first_id = str(first_item.get("poi_id"))
    second_id = str(second_item.get("poi_id"))

    start_overlay = overlays.get(("start_point", first_id))
    start_leg = _resolve_route_leg(
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
    transfer_leg = _resolve_route_leg(
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


def _build_sequence_route_facts(nodes: list[dict], constraints: dict | None = None) -> dict:
    overlays = _load_route_overlays(str(_mock_data_dir().resolve()))
    source_order = _get_route_source_order()
    origin_coordinates = _route_origin_coordinates(constraints)
    route_mode = str((constraints or {}).get("route_mode") or "driving")
    route_city = destination_city_from_constraints(constraints) or "上海"

    legs: list[dict] = []
    previous_item: dict | None = None
    previous_id = "start_point"
    for node in nodes:
        node_id = str(node.get("poi_id") or node.get("id") or "")
        if not node_id:
            continue
        overlay = overlays.get((previous_id, node_id))
        leg = _resolve_route_leg(
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
        "feasible": feasible,
    }


def _collect_plan_tags(*items: dict) -> list[str]:
    tags: list[str] = []
    for item in items:
        if len(items) == 1:
            cache_key = _item_cache_key(item)
            cached = _ITEM_TAG_CACHE.get(cache_key)
            signature = _semantic_cache_signature(item)
            if cached and cached[0] == signature:
                return list(cached[1])

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
    if len(items) == 1:
        _cache_item_tags(items[0], deduped)
    return deduped


def _active_replan_request(state: PlanState | dict | None, constraints: dict | None) -> dict:
    state = state or {}
    constraints = constraints or {}
    request = state.get("b_replan_request") or constraints.get("b_replan_request") or {}
    return request if isinstance(request, dict) else {}


def _replan_avoid_identity_matches(plan: dict, request: dict) -> bool:
    hints = request.get("candidate_generation_hints") or {}
    avoid_plan_ids = {str(item) for item in (hints.get("avoid_plan_ids") or []) if item}
    if str(plan.get("plan_id") or "") in avoid_plan_ids:
        return True

    identity = hints.get("avoid_supply_identity") or {}
    if not isinstance(identity, dict):
        return False
    nodes = plan.get("nodes", []) or []
    activity = next((node for node in nodes if node.get("type") == "activity"), {})
    restaurant = next((node for node in nodes if node.get("type") == "restaurant"), {})
    activity_id = identity.get("activity_id")
    restaurant_id = identity.get("restaurant_id")
    if activity_id and restaurant_id:
        return activity.get("poi_id") == activity_id and restaurant.get("poi_id") == restaurant_id
    if activity_id and activity.get("poi_id") == activity_id:
        return True
    if restaurant_id and restaurant.get("poi_id") == restaurant_id:
        return True
    return False


def _filter_replan_avoided_plans(plan_candidates: list[dict], request: dict) -> tuple[list[dict], dict]:
    if not request:
        return plan_candidates, {"applied": False}
    kept = [plan for plan in plan_candidates if not _replan_avoid_identity_matches(plan, request)]
    removed = len(plan_candidates) - len(kept)
    if removed <= 0 or not kept:
        return plan_candidates, {"applied": False, "removed": removed, "kept": len(kept)}
    return kept, {
        "applied": True,
        "removed": removed,
        "kept": len(kept),
        "source": request.get("source"),
        "status": request.get("status"),
    }


def _semantic_signal_set_for_item(item: dict) -> set[str]:
    cache_key = _item_cache_key(item)
    signature = _semantic_cache_signature(item)
    cached = _ITEM_SEMANTIC_SIGNAL_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return set(cached[1])

    tags = _collect_plan_tags(item)
    signals = set(expand_preference_tags(tags))
    signals.update(b_semantic_terms(tags, include_auxiliary=True))
    signals.update(tags)
    if len(_ITEM_SEMANTIC_SIGNAL_CACHE) >= _ITEM_TAG_CACHE_LIMIT:
        _ITEM_SEMANTIC_SIGNAL_CACHE.clear()
    _ITEM_SEMANTIC_SIGNAL_CACHE[cache_key] = (signature, signals)
    return set(signals)


def _semantic_signal_set_for_values(values: object) -> set[str]:
    return set(expand_preference_tags(values or []) + b_semantic_terms(values or [], include_auxiliary=True))


def _item_matches_tag(item: dict, tag: str) -> bool:
    values = _collect_plan_tags(item)
    values.extend(
        str(item.get(key) or "")
        for key in ("name", "category", "sub_category", "experience_type", "restaurant_category")
    )
    if tag in set(expand_preference_tags(values)):
        return True
    return semantic_match_score([tag], item, fields=COMPACT_SEMANTIC_FIELDS) > 0


def _expanded_required_tokens(required_tags: set[str]) -> set[str]:
    # Hard requirements must stay narrow.  Legacy expand_preference_tags can turn
    # specific categories such as social_hotpot into broad traits like social,
    # which would let generic social restaurants pass a hotpot request.
    return set(
        b_semantic_terms(list(required_tags), include_auxiliary=True)
        + list(required_tags)
    )


def _item_matches_any_tags(item: dict, required_tokens: set[str]) -> bool:
    if not required_tokens:
        return True
    normalized_required_terms = _normalized_query_terms(
        list(required_tokens),
        expand_semantics=True,
    )
    return _fast_text_match_score(normalized_required_terms, item, fields=COMPACT_SEMANTIC_FIELDS) > 0


def _explicit_activity_requirements(constraints: dict | None) -> set[str]:
    constraints = constraints or {}
    planning_preferences = constraints.get("planning_preferences", {}) or {}
    raw_preferences = planning_preferences.get("activity_type")
    expanded = set(expand_preference_tags(raw_preferences))
    semantic_groups = semantic_groups_in_values(raw_preferences).intersection(B_ACTIVITY_INTENT_GROUPS)
    semantic_terms = semantic_terms_for_groups(semantic_groups, include_auxiliary=True)
    return expanded.intersection(STRICT_ACTIVITY_REQUIREMENT_TAGS).union(semantic_terms)


def _list_values(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _excluded_restaurant_requirement_groups(constraints: dict | None) -> set[str]:
    constraints = constraints or {}
    excluded_values: list = []
    excluded_values.extend(_list_values(constraints.get("avoid")))
    contract = constraints.get("b_requirement_contract")
    if isinstance(contract, dict):
        excluded_values.extend(_list_values(contract.get("forbidden_restaurant_groups")))
    explicit_constraints = constraints.get("explicit_constraints")
    if isinstance(explicit_constraints, dict):
        excluded_values.extend(_list_values(explicit_constraints.get("dietary")))
    return semantic_groups_in_values(excluded_values).intersection(B_RESTAURANT_INTENT_GROUPS)


def _explicit_restaurant_requirements(constraints: dict | None) -> set[str]:
    constraints = constraints or {}
    planning_preferences = constraints.get("planning_preferences", {}) or {}
    raw_preferences = []
    for key in ("restaurant_type", "food_type"):
        value = planning_preferences.get(key)
        if isinstance(value, (list, tuple, set)):
            raw_preferences.extend(value)
        elif value:
            raw_preferences.append(value)
    blueprint = constraints.get("b_itinerary_blueprint")
    if isinstance(blueprint, dict):
        for intent in blueprint.get("node_intents", []) or []:
            if not isinstance(intent, dict):
                continue
            if str(intent.get("supply_domain") or "") != "restaurant":
                continue
            raw_preferences.extend(flatten_semantic_values(intent.get("search_terms")))
    expanded = set(expand_preference_tags(raw_preferences))
    excluded_groups = _excluded_restaurant_requirement_groups(constraints)
    semantic_groups = (
        semantic_groups_in_values(raw_preferences)
        .intersection(B_RESTAURANT_INTENT_GROUPS)
        .difference(excluded_groups)
    )
    semantic_terms = semantic_terms_for_groups(semantic_groups, include_auxiliary=True)
    excluded_terms = set(semantic_terms_for_groups(excluded_groups, include_auxiliary=True))
    return (
        expanded.intersection(STRICT_RESTAURANT_REQUIREMENT_TAGS).difference(excluded_terms)
    ).union(semantic_terms)


def _filter_activities_by_requirements(
    activity_candidates: list[dict],
    required_tags: set[str],
) -> list[dict]:
    if not required_tags:
        return activity_candidates
    required_tokens = _normalized_query_terms(
        list(_expanded_required_tokens(required_tags)),
        expand_semantics=True,
    )
    return [
        item
        for item in activity_candidates
        if _fast_text_match_score(required_tokens, item, fields=COMPACT_SEMANTIC_FIELDS) > 0
    ]


def _filter_restaurants_by_requirements(
    restaurant_candidates: list[dict],
    required_tags: set[str],
) -> list[dict]:
    if not required_tags:
        return restaurant_candidates
    required_tokens = _normalized_query_terms(
        list(_expanded_required_tokens(required_tags)),
        expand_semantics=True,
    )
    return [
        item
        for item in restaurant_candidates
        if _fast_text_match_score(required_tokens, item, fields=COMPACT_SEMANTIC_FIELDS) > 0
    ]


def _pretrim_candidates_for_sort(
    candidates: list[dict],
    *,
    constraints: dict,
    user_profile: dict | None,
    scenario_activities: list | None,
    limit: int,
    user_input: str | None = None,
) -> list[dict]:
    if len(candidates) <= limit:
        return candidates

    config = get_constraint_config_with_profile(constraints, user_profile or {})
    max_distance_km = config["max_distance_km"]
    max_queue_time = config["max_queue_time"]
    raw_preference_terms = _raw_preference_sources(constraints, user_profile, scenario_activities)
    direct_query_terms = _normalized_query_terms(raw_preference_terms)
    preferred_restaurant_role = _preferred_restaurant_role(
        constraints,
        user_profile,
        scenario_activities,
        user_input,
    )

    def score(item: dict) -> float:
        distance = to_float(item.get("distance_km"), 999.0)
        queue_time = to_float(item.get("queue_time_min"), 999.0)
        rating = to_float(item.get("rating"), 4.0)
        value = rating * 10.0
        value += 12.0 if item.get("available", True) else -80.0
        if distance <= max_distance_km:
            value += max(0.0, (max_distance_km - distance) * 1.2)
        else:
            value -= min(35.0, (distance - max_distance_km) * 3.0)
        if queue_time <= max_queue_time:
            value += max(0.0, (max_queue_time - queue_time) * 0.12)
        else:
            value -= min(25.0, (queue_time - max_queue_time) * 0.5)
        direct_score = _fast_text_match_score(direct_query_terms, item, fields=COMPACT_SEMANTIC_FIELDS)
        if direct_score:
            value += min(20.0, direct_score * 4.0)
        if item.get("type") == "restaurant":
            value += _restaurant_role_score(item, preferred_restaurant_role)
        return value

    return sorted(candidates, key=score, reverse=True)[:limit]


def _raw_preference_sources(
    constraints: dict | None,
    user_profile: dict | None = None,
    scenario_activities: list | None = None,
) -> list:
    constraints = constraints or {}
    user_profile = user_profile or {}
    planning_preferences = constraints.get("planning_preferences", {}) or {}
    preference_profile = user_profile.get("preference_profile", {}) or {}
    values: list = []
    for key in (
        "activity_type",
        "food_type",
        "emotion_type",
        "atmosphere_type",
        "experience_type",
        "restaurant_type",
    ):
        value = planning_preferences.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(value)
        elif value:
            values.append(value)
    if str(constraints.get("mom_diet") or "").lower() == "low_calorie":
        values.extend(["低卡", "轻食", "健康餐", "少油", "少糖", "蔬菜丰富"])
    for key in ("food_preference", "activity_preference", "emotion_need"):
        value = user_profile.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(value)
        elif value:
            values.append(value)
    for key in ("food", "activity", "emotion"):
        value = preference_profile.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(value)
        elif value:
            values.append(value)
    values.extend(scenario_activities or [])
    return values


def _restaurant_role(item: dict) -> str:
    category_groups = semantic_groups_for_item(item, fields=RESTAURANT_ROLE_FIELDS)
    if "咖啡甜品" in category_groups:
        return "cafe_dessert"
    if "轻食" in category_groups:
        return "light_meal"
    service_mode = str(item.get("service_mode") or "")
    category = str(item.get("restaurant_category") or item.get("category") or "")
    if "饮品" in category or "甜品" in category or "咖啡" in category or "下午茶" in category:
        return "cafe_dessert"
    if service_mode in {"咖啡小坐", "下午茶", "轻食简餐"}:
        return "cafe_dessert" if service_mode in {"咖啡小坐", "下午茶"} else "light_meal"
    return "full_meal"


def _preferred_restaurant_role(
    constraints: dict | None,
    user_profile: dict | None = None,
    scenario_activities: list | None = None,
    user_input: str | None = None,
) -> str | None:
    constraints = constraints or {}
    groups = semantic_groups_in_values(_raw_preference_sources(constraints, user_profile, scenario_activities))
    raw_text = str(user_input or constraints.get("raw_text") or "")
    if "咖啡甜品" in groups or any(
        phrase in raw_text
        for phrase in ("咖啡", "下午茶", "甜品", "小坐", "不想吃正餐", "不吃正餐", "不想正餐")
    ):
        return "cafe_dessert"
    if "轻食" in groups or str(constraints.get("mom_diet") or "").lower() == "low_calorie":
        return "light_meal"
    return None


def _restaurant_role_score(item: dict, preferred_role: str | None) -> float:
    if not preferred_role:
        return 0.0
    actual_role = _restaurant_role(item)
    if preferred_role == "cafe_dessert":
        if actual_role == "cafe_dessert":
            return 10.0
        if actual_role == "light_meal":
            return -2.0
        return -7.0
    if preferred_role == "light_meal":
        if actual_role == "light_meal":
            return 7.0
        if actual_role == "cafe_dessert":
            return 2.0
        return -5.0
    return 0.0


@lru_cache(maxsize=128)
def _strict_role_query_terms(role: str) -> tuple[str, ...]:
    return tuple(_normalized_query_terms(STRICT_MULTINODE_ROLE_TEXT_TERMS.get(role, ())))


PARK_SCENIC_POSITIVE_TERMS = (
    "公园",
    "游园",
    "滨江",
    "江边",
    "河边",
    "夜景",
    "步道",
    "观景",
    "外滩",
    "风景名胜",
    "公园广场",
    "休闲场所",
)
PARK_SCENIC_FALSE_POSITIVE_TERMS = (
    "商务大厦",
    "写字楼",
    "办公楼",
    "公寓",
    "商场",
    "购物中心",
    "店)",
    "店）",
    "绿地缤纷",
    "绿地汇",
    "绿地商务",
    "绿地科创",
)


def _matches_park_scenic_identity(item: dict) -> bool:
    text, values = _semantic_text_index(item, fields=STRICT_MULTINODE_ROLE_FIELDS)
    if any(term in values or term in text for term in PARK_SCENIC_POSITIVE_TERMS):
        return True
    if "绿地" in values or "绿地" in text:
        return not any(term in text for term in PARK_SCENIC_FALSE_POSITIVE_TERMS)
    return False


def _matches_strict_node_role(item: dict, role: str) -> bool:
    """Return whether a POI has its own evidence for a strict itinerary role."""

    item_type = str(item.get("type") or "").lower()
    supply_domain = str(item.get("supply_domain") or "").lower()

    if role == "cafe":
        return _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0
    if role == "exhibition":
        text, values = _semantic_text_index(item, fields=STRICT_MULTINODE_ROLE_FIELDS)
        return _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0 and not any(term in text for term in ("spa", "足疗", "按摩", "推拿", "洗脚", "修脚", "健身", "瑜伽", "普拉提"))
    if role == "board_game_escape":
        return _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0
    if role == "convenience_store":
        return supply_domain in {"shopping", "retail"} and _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0
    if role == "parking":
        return item_type == "transport_service" or supply_domain == "transport_service" or _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0
    if role == "wellness_massage":
        return _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0
    if role == "fitness":
        return _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0
    if role == "lodging":
        return item_type in {"hotel", "lodging"} or supply_domain in {"hotel", "lodging"} or _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0
    if role == "restaurant_breakfast":
        return _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0
    if role == "tea_house":
        return _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0
    if role == "cultural_photo":
        if _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0:
            return True
        signals = _semantic_signal_set_for_item(item)
        text, values = _semantic_text_index(item, fields=STRICT_MULTINODE_ROLE_FIELDS)
        cultural_terms = {
            "museum",
            "gallery",
            "exhibition",
            "art_experience",
            "local_culture",
            "citywalk",
            "博物馆",
            "美术馆",
            "展览",
            "艺术馆",
            "历史",
            "文化",
            "街区",
        }
        false_positive_terms = {
            "spa",
            "足疗",
            "按摩",
            "推拿",
            "洗脚",
            "修脚",
            "健身",
            "瑜伽",
            "普拉提",
        }
        has_cultural_signal = bool(signals.intersection(cultural_terms)) or any(
            term in values or term in text for term in cultural_terms
        )
        return has_cultural_signal and not any(term in text for term in false_positive_terms)
    if role == "park_scenic_walk":
        return _matches_park_scenic_identity(item)
    if role == "bar":
        text, values = _semantic_text_index(item, fields=STRICT_MULTINODE_ROLE_FIELDS)
        return _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0 and not any(term in text for term in ("火锅", "烤肉", "烧烤", "麻辣烫", "寿司", "牛排"))
    if role == "talk_show":
        return _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0
    if role == "cinema":
        return _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0
    return True


def _weather_tags(weather_context: dict | None) -> set[str]:
    weather_context = weather_context or {}
    tags = set(str(tag) for tag in weather_context.get("condition_tags", []) or [])
    tags.update(str(tag) for tag in weather_context.get("risk_tags", []) or [])
    return tags


def _item_weather_sensitivity(item: dict) -> str:
    return str(item.get("weather_sensitivity") or "").strip().lower()


def _is_indoor_safe_item(item: dict, tags: set[str] | None = None) -> bool:
    tags = tags or set(_collect_plan_tags(item))
    category = str(item.get("category") or item.get("experience_type") or "").strip()
    sensitivity = _item_weather_sensitivity(item)
    return (
        sensitivity == "indoor_safe"
        or bool(item.get("indoor_backup"))
        or bool(tags.intersection(INDOOR_SAFE_TAGS))
        or category in {"museum", "handcraft", "indoor_playground", "escape_room", "micro_vacation"}
    )


def _weather_candidate_bonus(item: dict, weather_context: dict | None) -> float:
    if not weather_context or not weather_context.get("available"):
        return 0.0

    if item.get("type") != "activity":
        return 0.0

    tags = set(expand_preference_tags(_collect_plan_tags(item)))
    category = str(item.get("category") or item.get("experience_type") or "").strip()
    sensitivity = _item_weather_sensitivity(item)
    weather_tags = _weather_tags(weather_context)
    prefer_indoor = bool(weather_context.get("prefer_indoor"))
    indoor_safe = _is_indoor_safe_item(item, tags)
    outdoor_like = category in OUTDOOR_ACTIVITY_CATEGORIES or "outdoor" in tags

    bonus = 0.0
    if prefer_indoor:
        if indoor_safe:
            bonus += 5.0
        if sensitivity == "medium":
            bonus -= 2.5
        elif sensitivity == "high":
            bonus -= 6.0
        if outdoor_like and not item.get("indoor_backup"):
            bonus -= 5.0

    if "hot" in weather_tags:
        if indoor_safe:
            bonus += 2.0
        if category in {"sports", "citywalk"} and not item.get("indoor_backup"):
            bonus -= 4.0

    if "comfortable" in weather_tags and category in OUTDOOR_ACTIVITY_CATEGORIES:
        bonus += 2.0

    return bonus


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


def _sequence_preference(constraints: dict | None) -> str:
    constraints = constraints or {}
    explicit_sequence = str(constraints.get("sequence_preference") or "").strip()
    if explicit_sequence in {
        SEQUENCE_ACTIVITY_THEN_RESTAURANT,
        SEQUENCE_RESTAURANT_THEN_ACTIVITY,
    }:
        return explicit_sequence

    raw_text = str(constraints.get("raw_text") or "")
    if raw_text and any(phrase in raw_text for phrase in RESTAURANT_THEN_ACTIVITY_PHRASES):
        return SEQUENCE_RESTAURANT_THEN_ACTIVITY
    return SEQUENCE_ACTIVITY_THEN_RESTAURANT


def _pick_time_slots(activity: dict, restaurant: dict, constraints: dict) -> tuple[str | None, str | None]:
    explicit_start = constraints.get("start_time") not in (None, "")
    explicit_end = constraints.get("end_time") not in (None, "")
    start_time = str(constraints.get("start_time") or "14:00")
    start_minutes = _slot_to_minutes(start_time)
    end_minutes = _slot_to_minutes(str(constraints.get("end_time"))) if explicit_end else -1
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
        activity_pool = valid_activity_slots if explicit_start else (valid_activity_slots or activity_slots)
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
                restaurant_end_minutes = restaurant_start_minutes + int(restaurant.get("duration_min", 0))
                if explicit_end and end_minutes >= start_minutes and restaurant_end_minutes > end_minutes:
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

    if explicit_start and not valid_activity_slots:
        return None, None

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
    if restaurant_start is None:
        return None, None
    if restaurant_start is not None and explicit_end and end_minutes >= start_minutes:
        restaurant_end_minutes = _slot_to_minutes(restaurant_start) + int(restaurant.get("duration_min", 0))
        if restaurant_end_minutes > end_minutes:
            return None, None

    return activity_start, restaurant_start


def _pick_time_slots_restaurant_first(
    activity: dict,
    restaurant: dict,
    constraints: dict,
) -> tuple[str | None, str | None]:
    explicit_start = constraints.get("start_time") not in (None, "")
    explicit_end = constraints.get("end_time") not in (None, "")
    start_time = str(constraints.get("start_time") or "14:00")
    start_minutes = _slot_to_minutes(start_time)
    end_minutes = _slot_to_minutes(str(constraints.get("end_time"))) if explicit_end else -1
    transition_buffer_min = _get_transition_buffer_min()
    prefer_earliest_activity = _get_time_slot_bool("prefer_earliest_valid_activity_slot", True)
    prefer_earliest_restaurant = _get_time_slot_bool("prefer_earliest_valid_restaurant_slot", True)

    activity_slots = sorted(
        [slot.get("time") for slot in activity.get("available_slots", []) if slot.get("time")],
        key=_slot_to_minutes,
    )
    restaurant_slots = sorted(
        [slot.get("time") for slot in restaurant.get("available_slots", []) if slot.get("time")],
        key=_slot_to_minutes,
    )

    valid_restaurant_slots = [
        slot for slot in restaurant_slots if _slot_to_minutes(slot) >= start_minutes
    ]
    restaurant_pool = valid_restaurant_slots if explicit_start else (valid_restaurant_slots or restaurant_slots)
    valid_pairs: list[tuple[int, int, int, str, str]] = []
    for restaurant_slot in restaurant_pool:
        restaurant_start_minutes = _slot_to_minutes(restaurant_slot)
        if restaurant_start_minutes < 0:
            continue
        restaurant_end_minutes = restaurant_start_minutes + int(restaurant.get("duration_min", 0))
        min_activity_minutes = restaurant_end_minutes + transition_buffer_min
        for activity_slot in activity_slots:
            activity_start_minutes = _slot_to_minutes(activity_slot)
            if activity_start_minutes < min_activity_minutes:
                continue
            activity_end_minutes = activity_start_minutes + int(activity.get("duration_min", 0))
            if explicit_end and end_minutes >= start_minutes and activity_end_minutes > end_minutes:
                continue
            transition_gap = activity_start_minutes - restaurant_end_minutes
            valid_pairs.append(
                (
                    transition_gap,
                    restaurant_start_minutes,
                    activity_start_minutes,
                    activity_slot,
                    restaurant_slot,
                )
            )

    if not valid_pairs:
        return None, None

    if prefer_earliest_restaurant and prefer_earliest_activity:
        valid_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    elif prefer_earliest_restaurant:
        valid_pairs.sort(key=lambda item: (item[0], item[1], -item[2]))
    elif prefer_earliest_activity:
        valid_pairs.sort(key=lambda item: (item[0], -item[1], item[2]))
    else:
        valid_pairs.sort(key=lambda item: (item[0], -item[1], -item[2]))
    _, _, _, activity_start, restaurant_start = valid_pairs[0]
    return activity_start, restaurant_start


def _sort_candidates(
    candidates: list[dict],
    constraints: dict,
    scene_type: str,
    user_profile: dict = None,
    scenario_activities: list = None,
    weather_context: dict | None = None,
    user_input: str | None = None,
) -> list[dict]:
    user_profile = user_profile or {}
    scenario_activities = derive_scenario_activities(constraints, user_profile, scenario_activities)
    config = get_constraint_config_with_profile(constraints, user_profile)

    child_age = config["child_age"]
    mom_diet = config["mom_diet"]
    preference_tags = list(set(collect_preference_sources(constraints, user_profile, scenario_activities)))
    raw_preference_terms = _raw_preference_sources(constraints, user_profile, scenario_activities)
    semantic_preference_terms = b_semantic_terms(preference_tags, include_auxiliary=True)
    semantic_query_terms = _normalized_query_terms(semantic_preference_terms, expand_semantics=False)
    direct_query_terms = _normalized_query_terms(raw_preference_terms, expand_semantics=False)
    preferred_restaurant_role = _preferred_restaurant_role(
        constraints,
        user_profile,
        scenario_activities,
        user_input,
    )

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
        normalized_tags = _semantic_signal_set_for_item(item)

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

        semantic_score = _fast_text_match_score(semantic_query_terms, item, fields=COMPACT_SEMANTIC_FIELDS)
        if semantic_score:
            base += min(7.0, semantic_score * 1.35)
        direct_score = _fast_text_match_score(direct_query_terms, item, fields=COMPACT_SEMANTIC_FIELDS)
        if direct_score:
            base += min(8.0, direct_score * 1.8)
        if item_type == "restaurant":
            base += _restaurant_role_score(item, preferred_restaurant_role)

        if item_type == "restaurant" and "budget" in preference_tags and "budget" in normalized_tags:
            base += 2

        if item_type == "activity" and "nearby" in preference_tags and distance <= max_distance_km:
            base += 1

        base += _weather_candidate_bonus(item, weather_context)

        return base

    return sorted(candidates, key=score, reverse=True)


def _sort_plan_candidates(
    plan_candidates: list[dict],
    constraints: dict,
    scene_type: str,
    user_profile: dict | None = None,
    scenario_activities: list | None = None,
    weather_context: dict | None = None,
    user_input: str | None = None,
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
    raw_preference_terms = _raw_preference_sources(constraints, user_profile, scenario_activities)
    semantic_preference_terms = b_semantic_terms(list(preference_tags), include_auxiliary=True)
    semantic_query_terms = _normalized_query_terms(semantic_preference_terms, expand_semantics=False)
    direct_query_terms = _normalized_query_terms(raw_preference_terms, expand_semantics=False)
    preferred_restaurant_role = _preferred_restaurant_role(
        constraints,
        user_profile,
        scenario_activities,
        user_input,
    )

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

        activity_tags = _semantic_signal_set_for_item(activity)
        restaurant_tags = _semantic_signal_set_for_item(restaurant)
        restaurant_health_tags = _semantic_signal_set_for_values(restaurant.get("health_tags", []) or [])
        menu_health_options = _semantic_signal_set_for_values(restaurant.get("menu_health_options", []) or [])

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
        semantic_value = max(
            _fast_text_match_score(semantic_query_terms, activity, fields=COMPACT_SEMANTIC_FIELDS),
            _fast_text_match_score(semantic_query_terms, restaurant, fields=COMPACT_SEMANTIC_FIELDS),
        )
        if semantic_value:
            value += min(14.0, semantic_value * 2.0)
        direct_value = max(
            _fast_text_match_score(direct_query_terms, activity, fields=COMPACT_SEMANTIC_FIELDS),
            _fast_text_match_score(direct_query_terms, restaurant, fields=COMPACT_SEMANTIC_FIELDS),
        )
        if direct_value:
            value += min(16.0, direct_value * 2.2)
        value += _restaurant_role_score(restaurant, preferred_restaurant_role) * 1.25
        value += _weather_candidate_bonus(activity, weather_context)
        value += to_float(activity.get("rating"), 4.0) + to_float(restaurant.get("rating"), 4.0)
        return value

    return sorted(plan_candidates, key=score, reverse=True)


def _ranked_candidate_pairs(
    activities: list[dict],
    restaurants: list[dict],
    max_pairs: int,
) -> list[tuple[dict, dict]]:
    pairs: list[tuple[int, int, dict, dict]] = []
    for activity_index, activity in enumerate(activities):
        for restaurant_index, restaurant in enumerate(restaurants):
            pairs.append((activity_index + restaurant_index, abs(activity_index - restaurant_index), activity, restaurant))
    pairs.sort(key=lambda item: (item[0], item[1]))
    return [(activity, restaurant) for _, _, activity, restaurant in pairs[:max_pairs]]


def _rag_candidates_for_domain(
    rag_candidates_by_node: dict[str, list[dict]] | None,
    domain: str,
) -> list[dict]:
    if not rag_candidates_by_node:
        return []
    expected_type = "restaurant" if domain == "restaurant" else "activity"
    result: list[dict] = []
    seen: set[str] = set()
    for node_candidates in rag_candidates_by_node.values():
        for raw_item in node_candidates or []:
            item_domain = str(raw_item.get("supply_domain") or raw_item.get("type") or "")
            item_type = str(raw_item.get("type") or "")
            if domain == "restaurant":
                matches_domain = item_domain == "restaurant" or item_type == "restaurant"
            else:
                matches_domain = item_domain == "activity" or item_type in {"activity", "play"}
            if not matches_domain:
                continue
            poi_id = str(raw_item.get("poi_id") or raw_item.get("id") or raw_item.get("amap_id") or raw_item.get("name") or "")
            if not poi_id or poi_id in seen:
                continue
            seen.add(poi_id)
            result.append(_normalize_mock_poi(raw_item, expected_type))
    return result


def _intent_query_terms(intent: dict) -> list[str]:
    values: list[str] = []
    values.extend(flatten_semantic_values(intent.get("search_terms")))
    values.extend(flatten_semantic_values(intent.get("label")))
    values.extend(MULTINODE_ROLE_TERMS.get(str(intent.get("role") or ""), []))
    return _normalized_query_terms(values, expand_semantics=True)


def _score_item_for_node_intent(item: dict, intent: dict) -> float:
    role = str(intent.get("role") or "")
    terms = _intent_query_terms(intent)
    signals = _semantic_signal_set_for_item(item)
    score = to_float(item.get("rating"), 4.0) * 2.0
    score += 8.0 if item.get("available", True) else -80.0
    score -= min(12.0, to_float(item.get("queue_time_min"), 0.0) * 0.12)
    score -= min(8.0, to_float(item.get("distance_km"), 0.0) * 0.08)
    score += min(24.0, _fast_text_match_score(terms, item, fields=COMPACT_SEMANTIC_FIELDS) * 5.0)

    if role == "family_activity":
        if signals.intersection({"kid_friendly", "family_friendly", "low_intensity", "indoor"}):
            score += 10.0
    elif role == "family_indoor_play":
        if signals.intersection({"kid_friendly", "family_friendly", "indoor_playground", "amusement", "low_intensity", "indoor"}):
            score += 12.0
    elif role == "exhibition":
        if signals.intersection({"museum", "art_experience", "exhibition"}):
            score += 8.0
    elif role == "citywalk_market":
        if signals.intersection({"citywalk", "local_market", "local_culture", "outdoor"}):
            score += 8.0
    elif role == "park_scenic_walk":
        if signals.intersection({"citywalk", "local_culture", "outdoor", "micro_vacation"}):
            score += 8.0
    elif role == "board_game_escape":
        if signals.intersection({"board_game", "escape_room", "script_murder", "chess_cards", "social"}):
            score += 9.0
    elif role == "karaoke":
        if signals.intersection({"karaoke", "ktv", "social", "group_friendly"}):
            score += 9.0
    elif role == "internet_cafe":
        if signals.intersection({"board_game", "escape_room", "social", "group_friendly"}):
            score += 5.0
    elif role == "cafe":
        score += _restaurant_role_score(item, "cafe_dessert")
    elif role in {"restaurant_breakfast", "restaurant_lunch", "restaurant_dinner", "restaurant_specific"}:
        if _restaurant_role(item) != "cafe_dessert":
            score += 4.0

    return score


def _rank_pool_for_node_intent(
    intent: dict,
    activities: list[dict],
    restaurants: list[dict],
    rag_candidates_by_node: dict[str, list[dict]] | None = None,
) -> list[dict]:
    node_id = str(intent.get("node_id") or "")
    if rag_candidates_by_node and rag_candidates_by_node.get(node_id):
        pool = list(rag_candidates_by_node.get(node_id) or [])
    else:
        pool = []
    domain = str(intent.get("supply_domain") or "")
    if not pool:
        pool = activities if domain == "activity" else restaurants if domain == "restaurant" else []
    if not pool:
        return []
    role = str(intent.get("role") or "")
    if role in STRICT_MULTINODE_ROLE_TEXT_TERMS:
        pool = [item for item in pool if _matches_strict_node_role(item, role)]
        if not pool:
            return []
    if role == "family_activity":
        preferred = [
            item for item in pool
            if _semantic_signal_set_for_item(item).intersection({"kid_friendly", "family_friendly", "low_intensity"})
        ]
        if preferred:
            pool = preferred
    elif role == "family_indoor_play":
        preferred = [
            item for item in pool
            if _semantic_signal_set_for_item(item).intersection({"kid_friendly", "family_friendly", "indoor_playground", "amusement"})
            or _fast_text_match_score(_intent_query_terms(intent), item, fields=COMPACT_SEMANTIC_FIELDS) > 0
        ]
        if preferred:
            pool = preferred
    elif role == "exhibition":
        preferred = [
            item for item in pool
            if _semantic_signal_set_for_item(item).intersection({"museum", "art_experience", "exhibition"})
            or _fast_text_match_score(_intent_query_terms(intent), item, fields=COMPACT_SEMANTIC_FIELDS) > 0
        ]
        if preferred:
            pool = preferred
    elif role in {"restaurant_breakfast", "restaurant_lunch", "restaurant_dinner", "restaurant_specific"}:
        explicit_groups = semantic_groups_in_values(intent.get("search_terms")).intersection(
            B_RESTAURANT_INTENT_GROUPS
        )
        if explicit_groups:
            required_tokens = _normalized_query_terms(
                list(semantic_terms_for_groups(explicit_groups, include_auxiliary=True)),
                expand_semantics=True,
            )
            preferred = [
                item
                for item in pool
                if _fast_text_match_score(required_tokens, item, fields=COMPACT_SEMANTIC_FIELDS) > 0
            ]
            if preferred:
                pool = preferred
        preferred = [item for item in pool if _restaurant_role(item) != "cafe_dessert"]
        if preferred:
            pool = preferred
    return sorted(pool, key=lambda item: _score_item_for_node_intent(item, intent), reverse=True)


def _time_to_minutes(value: object, *, default: int, day: int = 1) -> int:
    text = str(value or "")
    token = ""
    for char in text:
        if char.isdigit() or char == ":":
            token += char
        elif token:
            break
    if ":" not in token:
        return default
    try:
        hour, minute = token.split(":", 1)
        return (max(1, day) - 1) * 1440 + int(hour) * 60 + int(minute)
    except (TypeError, ValueError):
        return default


def _format_itinerary_time(total_minutes: int) -> str:
    minute_of_day = total_minutes % 1440
    hour, minute = divmod(minute_of_day, 60)
    return f"{hour:02d}:{minute:02d}"


def _format_itinerary_time_range(start_minutes: int, end_minutes: int) -> str:
    start_text = _format_itinerary_time(start_minutes)
    end_text = _format_itinerary_time(end_minutes)
    if end_minutes // 1440 > start_minutes // 1440:
        return f"{start_text}-次日{end_text}"
    return f"{start_text}-{end_text}"


def _available_slot_minutes(item: dict, day: int) -> list[int]:
    deal_slots: list[int] = []
    for deal in item.get("deals", []) or []:
        if not isinstance(deal, dict):
            continue
        for raw_time in deal.get("valid_time", []) or []:
            minutes = _time_to_minutes(raw_time, default=-1, day=day)
            if minutes >= 0:
                deal_slots.append(minutes)

    operational_slots: list[int] = []
    for field_name in ("available_slots", "reservation_slots"):
        for slot in item.get(field_name, []) or []:
            if isinstance(slot, dict):
                raw_time = slot.get("time")
            else:
                raw_time = slot
            minutes = _time_to_minutes(raw_time, default=-1, day=day)
            if minutes >= 0:
                operational_slots.append(minutes)

    deal_slot_set = set(deal_slots)
    operational_slot_set = set(operational_slots)
    if deal_slot_set and operational_slot_set:
        intersection = deal_slot_set.intersection(operational_slot_set)
        if intersection:
            return sorted(intersection)
        return sorted(deal_slot_set)
    if deal_slot_set:
        return sorted(deal_slot_set)
    return sorted(operational_slot_set)


def _choose_node_start_time(
    item: dict,
    *,
    desired_start: int,
    earliest_start: int,
    day: int,
) -> int:
    target_start = max(desired_start, earliest_start)
    valid_slots = [slot for slot in _available_slot_minutes(item, day) if slot >= target_start]
    if valid_slots:
        return valid_slots[0]
    return target_start


def _slot_alignment_violation(slot: dict, start: int, *, day: int) -> dict | None:
    if not slot or not slot.get("start_time"):
        return None
    slot_start = _time_to_minutes(slot.get("start_time"), default=-1, day=day)
    if slot_start < 0:
        return None
    role = str(slot.get("role") or "")
    part_of_day = str(slot.get("part_of_day") or "")
    anchored_roles = {
        "restaurant_lunch",
        "restaurant_dinner",
        "talk_show",
        "show",
        "performance",
    }
    anchored_parts = {"morning", "lunch", "dinner", "evening", "overnight"}
    if role not in anchored_roles and part_of_day not in anchored_parts:
        return None
    tolerance_min = 90
    if part_of_day == "morning":
        tolerance_min = 120
    drift_min = start - slot_start
    if drift_min <= tolerance_min:
        return None
    return {
        "node_id": slot.get("node_id"),
        "role": role,
        "part_of_day": part_of_day,
        "slot_start": _format_itinerary_time(slot_start),
        "scheduled_start": _format_itinerary_time(start),
        "drift_min": drift_min,
    }


def _timeline_type_for_node(node: dict) -> str:
    intent = node.get("_itinerary_intent") or {}
    node_type = str(node.get("type") or "")
    domain = str(node.get("supply_domain") or intent.get("supply_domain") or "")
    role = str(node.get("itinerary_role") or intent.get("role") or "")
    if node_type == "restaurant" or domain == "restaurant":
        return "restaurant"
    if node_type == "activity" or domain == "activity":
        return "play"
    if domain in {"hotel", "lodging"} or role == "lodging":
        return "lodging"
    if domain in {"shopping", "retail"}:
        return "shopping"
    if domain == "transport_service" or role == "parking":
        return "transport"
    if domain == "wellness":
        return "wellness"
    return "poi"


def _build_multinode_schedule(
    nodes: list[dict],
    blueprint: dict,
    constraints: dict,
) -> tuple[list[dict], dict]:
    slots_by_node_id = {}
    for day in (blueprint.get("time_skeleton") or {}).get("days", []) or []:
        for slot in day.get("slots", []) or []:
            if slot.get("node_id"):
                slots_by_node_id[slot.get("node_id")] = slot

    timeline: list[dict] = []
    schedule_nodes: list[dict] = []
    transition_buffer = _get_transition_buffer_min()
    slot_start_defaults = [
        _time_to_minutes(slot.get("start_time"), default=0, day=int(slot.get("day") or 1))
        for slot in slots_by_node_id.values()
        if slot.get("start_time")
    ]
    default_start = (
        min(slot_start_defaults)
        if slot_start_defaults
        else 10 * 60 if blueprint.get("planning_horizon") in {"full_day", "two_day"} else 14 * 60
    )
    start_minutes = _time_to_minutes(constraints.get("start_time"), default=default_start, day=1)
    planning_days = int(blueprint.get("planning_days") or 1)
    end_cap = (
        _time_to_minutes(constraints.get("end_time"), default=-1, day=planning_days)
        if constraints.get("end_time") not in (None, "")
        else -1
    )
    if planning_days == 1 and end_cap >= 0 and end_cap <= start_minutes:
        end_cap += 1440
    previous_end = start_minutes
    first_start: int | None = None
    last_end = start_minutes
    active_duration_min = 0
    time_window_feasible = True
    skipped_time_window_nodes: list[dict] = []
    slot_alignment_violations: list[dict] = []

    for index, node in enumerate(nodes):
        intent = node.get("_itinerary_intent") or {}
        slot = slots_by_node_id.get(intent.get("node_id"), {})
        day = int(slot.get("day") or intent.get("day") or intent.get("day_index") or 1)
        if day > 1 and previous_end < (day - 1) * 1440:
            previous_end = (day - 1) * 1440 + 9 * 60
        desired_start = _time_to_minutes(
            slot.get("start_time"),
            default=start_minutes if index == 0 else previous_end + transition_buffer,
            day=day,
        )
        node_role = intent.get("role")
        if node_role in {"convenience_store", "parking"} and index > 0:
            desired_start = min(desired_start, previous_end + transition_buffer)
        earliest_start = start_minutes if index == 0 else previous_end + transition_buffer
        start = _choose_node_start_time(
            node,
            desired_start=desired_start,
            earliest_start=earliest_start,
            day=day,
        )
        duration = int(node.get("duration_min") or intent.get("default_duration_min") or slot.get("duration_min") or 60)
        if node_role == "lodging":
            end = max(start + 60, day * 1440 + 10 * 60)
            duration = end - start
        else:
            end = start + max(15, duration)
        alignment_violation = _slot_alignment_violation(slot, start, day=day)
        if alignment_violation:
            time_window_feasible = False
            skipped_time_window_nodes.append(
                {
                    "poi_id": node.get("poi_id"),
                    "name": node.get("name"),
                    "role": node_role,
                    "day": day,
                    "requested_start": _format_itinerary_time(start),
                    "requested_end": _format_itinerary_time(end),
                    "deadline": slot.get("end_time"),
                    "reason": "slot_alignment_drift",
                }
            )
            slot_alignment_violations.append(
                {
                    **alignment_violation,
                    "poi_id": node.get("poi_id"),
                    "name": node.get("name"),
                }
            )
            continue
        if end_cap >= 0 and day == planning_days and node_role != "lodging" and end > end_cap:
            latest_start = end_cap - max(15, duration)
            valid_fit_slots = [
                slot
                for slot in _available_slot_minutes(node, day)
                if earliest_start <= slot <= latest_start
            ]
            if valid_fit_slots:
                start = max(valid_fit_slots)
                end = start + max(15, duration)
            elif not _available_slot_minutes(node, day) and latest_start >= earliest_start:
                start = latest_start
                end = end_cap
            else:
                time_window_feasible = False
                skipped_time_window_nodes.append(
                    {
                        "poi_id": node.get("poi_id"),
                        "name": node.get("name"),
                        "role": node_role,
                        "day": day,
                        "requested_start": _format_itinerary_time(start),
                        "requested_end": _format_itinerary_time(end),
                        "deadline": _format_itinerary_time(end_cap),
                    }
                )
                continue
        active_duration_min += max(15, duration)
        first_start = start if first_start is None else min(first_start, start)
        last_end = max(last_end, end)
        if node_role == "lodging" and index < len(nodes) - 1:
            # Lodging is an overnight anchor, not an active 12-hour visit that
            # should push dinner or shopping into the early morning.
            previous_end = start + 15
        else:
            previous_end = end
        time_range = _format_itinerary_time_range(start, end)
        schedule_nodes.append(
            {
                "poi_id": node.get("poi_id"),
                "role": node_role,
                "start": _format_itinerary_time(start),
                "end": _format_itinerary_time(end),
                "day": day,
            }
        )
        timeline.append(
            {
                "time": time_range,
                "activity": node.get("name"),
                "poi_id": node.get("poi_id"),
                "type": _timeline_type_for_node(node),
                "role": node_role,
                "day": day,
                "duration_min": duration,
                "price": node.get("price"),
                "notes": [
                    str(intent.get("label") or node_role or node.get("type") or ""),
                    "multi_node_itinerary",
                ],
            }
        )

    return timeline, {
        "nodes": schedule_nodes,
        "first_start_min": first_start or start_minutes,
        "last_end_min": last_end,
        "active_duration_min": active_duration_min,
        "time_window_feasible": time_window_feasible,
        "skipped_time_window_nodes": skipped_time_window_nodes,
        "slot_alignment_violations": slot_alignment_violations,
    }


def _build_multinode_execution_requirements(scene_type: str, people_count: int, nodes: list[dict]) -> dict:
    if scene_type == "couple":
        min_people = 2
    elif scene_type == "friends":
        min_people = 3
    else:
        min_people = people_count
    return {
        "min_people": min_people,
        "people_count": people_count,
        "pre_booking_required": any(to_float(node.get("queue_time_min"), 0.0) > 20 for node in nodes),
        "special_preparation": [],
    }


def _rough_transition_km(previous: dict | None, candidate: dict) -> float:
    if previous:
        previous_coord = _item_coordinates(previous)
        candidate_coord = _item_coordinates(candidate)
        if previous_coord and candidate_coord:
            return _haversine_km(previous_coord, candidate_coord) * 1.35
    return to_float(candidate.get("distance_km"), 8.0)


def _build_multinode_sequences(
    node_intents: list[dict],
    ranked_pools: list[list[dict]],
    *,
    max_candidates: int,
) -> list[list[dict]]:
    """Beam-search compact activity/restaurant sequences before route scoring."""

    beams: list[tuple[float, list[dict], set[str]]] = [(0.0, [], set())]
    branch_limit = 8
    beam_width = max(8, min(max_candidates, 18))
    for intent, pool in zip(node_intents, ranked_pools):
        next_beams: list[tuple[float, list[dict], set[str]]] = []
        for route_cost, sequence, used_ids in beams:
            previous = sequence[-1] if sequence else None
            for candidate in pool[:branch_limit]:
                candidate_id = str(candidate.get("poi_id") or candidate.get("id") or "")
                if not candidate_id or candidate_id in used_ids:
                    continue
                node = dict(candidate)
                node["_itinerary_intent"] = intent
                node["itinerary_role"] = intent.get("role")
                node["itinerary_label"] = intent.get("label")
                role_fit = _score_item_for_node_intent(candidate, intent)
                transition_cost = _rough_transition_km(previous, candidate)
                next_beams.append(
                    (
                        route_cost + transition_cost - min(12.0, role_fit) * 0.08,
                        sequence + [node],
                        used_ids | {candidate_id},
                    )
                )
        next_beams.sort(key=lambda item: item[0])
        beams = next_beams[:beam_width]
        if not beams:
            return []
    return [sequence for _, sequence, _ in beams[:max_candidates]]


def _combine_multinode_plan_candidates(
    activities: list[dict],
    restaurants: list[dict],
    constraints: dict,
    scene_type: str,
    blueprint: dict,
    user_profile: dict | None = None,
    weather_context: dict | None = None,
    rag_candidates_by_node: dict[str, list[dict]] | None = None,
    rag_coverage: dict | None = None,
    max_candidates: int = DEFAULT_MAX_MULTINODE_CANDIDATES,
) -> list[dict]:
    node_intents = blueprint.get("node_intents") or []
    can_plan_full = (
        _can_plan_multinode_with_current_supply(blueprint)
        or _can_plan_multinode_with_rag(blueprint, rag_coverage)
    )
    partial_missing_node_intents: list[dict] = []
    if can_plan_full:
        planning_node_intents = node_intents
    else:
        covered_node_ids = set((rag_coverage or {}).get("covered_node_ids") or [])
        planning_node_intents = [
            intent
            for intent in node_intents
            if str(intent.get("supply_domain") or "") in MULTINODE_SUPPORTED_DOMAINS
            or str(intent.get("node_id") or "") in covered_node_ids
        ]
        partial_missing_node_intents = [
            intent
            for intent in node_intents
            if intent not in planning_node_intents
        ]
        if not planning_node_intents:
            return []

    ranked_pairs = [
        (
            intent,
            _rank_pool_for_node_intent(intent, activities, restaurants, rag_candidates_by_node),
        )
        for intent in planning_node_intents
    ]
    empty_pool_intents = [intent for intent, pool in ranked_pairs if not pool]
    if empty_pool_intents:
        partial_missing_node_intents = partial_missing_node_intents + empty_pool_intents
        ranked_pairs = [(intent, pool) for intent, pool in ranked_pairs if pool]
    if not ranked_pairs:
        return []
    planning_node_intents = [intent for intent, _ in ranked_pairs]
    ranked_pools = [pool for _, pool in ranked_pairs]

    config = get_constraint_config_with_profile(constraints, user_profile)
    people_count = config["people_count"]
    plan_candidates: list[dict] = []
    sequences = _build_multinode_sequences(
        planning_node_intents,
        ranked_pools,
        max_candidates=max_candidates,
    )
    plan_index = 1

    for selected_nodes in sequences:
        timeline, schedule = _build_multinode_schedule(selected_nodes, blueprint, constraints)
        route_facts = _build_sequence_route_facts(selected_nodes, constraints)
        total_price = sum(to_float(node.get("price"), 0.0) for node in selected_nodes)
        max_queue_time_plan = max(to_float(node.get("queue_time_min"), 0.0) for node in selected_nodes)
        available = (
            all(node.get("available", True) for node in selected_nodes)
            and route_facts.get("feasible", True)
            and schedule.get("time_window_feasible", True)
        )
        non_executable_nodes = [
            {
                "poi_id": node.get("poi_id"),
                "name": node.get("name"),
                "role": node.get("itinerary_role"),
                "supply_domain": node.get("supply_domain"),
            }
            for node in selected_nodes
            if _node_requires_c_execution(node) and not _node_is_supported_by_current_c(node)
        ]
        guidance_only_nodes = [
            {
                "poi_id": node.get("poi_id"),
                "name": node.get("name"),
                "role": node.get("itinerary_role"),
                "supply_domain": node.get("supply_domain"),
            }
            for node in selected_nodes
            if not _node_requires_c_execution(node)
        ]
        benchmark_ready = (
            not partial_missing_node_intents
            and
            len(selected_nodes) == len(node_intents)
            and all(node.get("poi_id") for node in selected_nodes)
        )
        if int(blueprint.get("planning_days") or 1) > 1:
            estimated_duration_min = round(
                to_float(schedule.get("active_duration_min"), 0.0)
                + to_float(route_facts.get("total_travel_time_min"), 0.0)
            )
        else:
            estimated_duration_min = max(0, schedule["last_end_min"] - schedule["first_start_min"])
        tags = _collect_plan_tags(*selected_nodes)
        availability_detail = {
            "node_count": len(selected_nodes),
            "unavailable_poi_ids": [
                node.get("poi_id")
                for node in selected_nodes
                if not node.get("available", True)
            ],
            "max_queue_time_min": max_queue_time_plan,
            "time_window_feasible": schedule.get("time_window_feasible", True),
        }
        plan_candidates.append(
            {
                "plan_id": f"cand_multi_{plan_index:03d}",
                "scene_type": scene_type,
                "planner_mode": "multi_node_itinerary",
                "plan_shape": "multi_node",
                "plan_template": [intent.get("supply_domain") for intent in node_intents],
                "nodes": selected_nodes,
                "timeline": timeline,
                "route": {
                    "total_distance_km": route_facts["total_distance_km"],
                    "total_travel_time_min": route_facts["total_travel_time_min"],
                    "traffic_status": route_facts.get("traffic_status"),
                    "route_sources": route_facts.get("route_sources", []),
                    "legs": route_facts["legs"],
                },
                "schedule": schedule,
                "budget": {"total_price": total_price},
                "availability": {
                    "all_available": available,
                    "max_queue_time_min": max_queue_time_plan,
                    "detail": availability_detail,
                },
                "estimated_duration_min": estimated_duration_min,
                "tags": tags,
                "weather_context": weather_context or {},
                "constraint_snapshot": {
                    "planning_horizon": blueprint.get("planning_horizon"),
                    "planning_days": blueprint.get("planning_days"),
                },
                "execution_requirements": _build_multinode_execution_requirements(
                    scene_type,
                    people_count,
                    selected_nodes,
                ),
                "benchmark_ready": benchmark_ready,
                "execution_scope": "partial" if (partial_missing_node_intents or non_executable_nodes) else "full",
                "non_executable_nodes": non_executable_nodes,
                "guidance_only_nodes": guidance_only_nodes,
                "partial_missing_node_intents": partial_missing_node_intents,
                "partial_missing_roles": [
                    str(intent.get("role"))
                    for intent in partial_missing_node_intents
                    if intent.get("role")
                ],
                "rag_candidate_coverage": rag_coverage or {},
                "b_itinerary_blueprint": blueprint,
                "planning_horizon": blueprint.get("planning_horizon"),
                "planning_days": blueprint.get("planning_days"),
            }
        )
        plan_index += 1
        if len(plan_candidates) >= max_candidates:
            break

    return plan_candidates


def _combine_plan_candidates(
    activities: list[dict],
    restaurants: list[dict],
    constraints: dict,
    scene_type: str,
    user_profile: dict | None = None,
    weather_context: dict | None = None,
    max_pair_combinations: int | None = None,
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
    sequence = _sequence_preference(constraints)

    pairs = _ranked_candidate_pairs(
        activities,
        restaurants,
        max_pair_combinations or len(activities) * len(restaurants),
    )

    for activity, restaurant in pairs:
            if sequence == SEQUENCE_RESTAURANT_THEN_ACTIVITY:
                activity_start, restaurant_start = _pick_time_slots_restaurant_first(
                    activity,
                    restaurant,
                    constraints,
                )
            else:
                activity_start, restaurant_start = _pick_time_slots(activity, restaurant, constraints)
            if not activity_start or not restaurant_start:
                continue

            route_facts = _build_route_facts(activity, restaurant, constraints, sequence)
            total_distance_km = route_facts["total_distance_km"]
            total_travel_time_min = route_facts["total_travel_time_min"]
            total_price = activity["price"] + restaurant["price"]
            max_queue_time_plan = max(activity["queue_time_min"], restaurant["queue_time_min"])
            available = activity["available"] and restaurant["available"]
            activity_end_minutes = _slot_to_minutes(activity_start) + int(activity["duration_min"])
            restaurant_start_minutes = _slot_to_minutes(restaurant_start)
            restaurant_end_minutes = restaurant_start_minutes + int(restaurant["duration_min"])
            first_start_minutes = (
                restaurant_start_minutes
                if sequence == SEQUENCE_RESTAURANT_THEN_ACTIVITY
                else _slot_to_minutes(activity_start)
            )
            last_end_minutes = (
                activity_end_minutes
                if sequence == SEQUENCE_RESTAURANT_THEN_ACTIVITY
                else restaurant_end_minutes
            )
            estimated_duration_min = max(
                240,
                last_end_minutes - first_start_minutes,
            )
            tags = _collect_plan_tags(activity, restaurant)
            restaurant_role = _restaurant_role(restaurant)

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
                restaurant_signals = _semantic_signal_set_for_item(restaurant)
                if "low_calorie" not in restaurant_signals:
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
                        "sequence": sequence,
                        "activity_start": activity_start,
                        "activity_end": f"{activity_end_minutes // 60:02d}:{activity_end_minutes % 60:02d}",
                        "restaurant_start": restaurant_start,
                        "restaurant_end": f"{restaurant_end_minutes // 60:02d}:{restaurant_end_minutes % 60:02d}",
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
                    "restaurant_role": restaurant_role,
                    "weather_context": weather_context or {},
                    "constraint_snapshot": constraint_snapshot,
                    "execution_requirements": execution_requirements,
                }
            )
            plan_index += 1

    return plan_candidates


def _text_has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _duration_range_wants_itinerary(constraints: dict, max_single_node_min: int) -> bool:
    raw_duration = constraints.get("duration_range")
    if raw_duration in (None, "") and constraints.get("duration") in (None, ""):
        return False
    duration_range = parse_duration_range(raw_duration or constraints.get("duration"))
    raw_text = str(constraints.get("raw_text") or "")
    explicit_duration_terms = (
        "几个小时",
        "一下午",
        "半天",
        "一整天",
        "全天",
        "一天",
        "两天",
        "2天",
        "周末",
        "过夜",
        "住一晚",
    )
    confidence = constraints.get("confidence") or {}
    try:
        default_duration_pair = (
            isinstance(raw_duration, (list, tuple))
            and len(raw_duration) >= 2
            and float(raw_duration[0]) == 3.0
            and float(raw_duration[1]) == 6.0
        )
    except (TypeError, ValueError):
        default_duration_pair = False
    looks_like_a_default_duration = (
        default_duration_pair
        and str(constraints.get("time_window") or "") in {"", "unspecified"}
        and "time_window" in confidence
        and float(confidence.get("time_window") or 0.0) <= 0.4
        and not any(term in raw_text for term in explicit_duration_terms)
        and constraints.get("duration") in (None, "")
    )
    if looks_like_a_default_duration:
        return False
    return bool(duration_range and duration_range[0] >= max_single_node_min)


def _scenario_has_activity_context(scenario_activities: list | None, text: str) -> bool:
    """Return true when restaurant wording is part of a broader outing request."""

    activity_terms = (
        "好玩",
        "玩",
        "活动",
        "逛",
        "citywalk",
        "市集",
        "本地文化",
        "城市漫步",
        "展览",
        "看展",
        "手作",
        "亲子",
        "运动",
        "拍照",
        "散步",
        "博物馆",
        "美术馆",
        "室内",
    )
    park_activity_terms = ("逛公园", "去公园", "公园散步", "公园玩", "公园夜景", "公园活动")
    dining_terms = (
        "吃",
        "餐",
        "饭",
        "火锅",
        "烤肉",
        "烧烤",
        "咖啡",
        "甜品",
        "轻食",
    )
    if (
        (_text_has_any(text, activity_terms) or _text_has_any(text, park_activity_terms))
        and _text_has_any(text, dining_terms)
    ):
        return True
    return any(str(item) in activity_terms for item in (scenario_activities or []))


def _single_node_plan_shape(
    blueprint: dict | None,
    constraints: dict | None,
    requirement_contract: dict | None,
    user_input: str | None,
    scenario_activities: list | None = None,
) -> str | None:
    """Choose a single-node planner when the request is not an itinerary."""

    blueprint = blueprint or {}
    constraints = constraints or {}
    requirement_contract = requirement_contract or {}
    node_intents = blueprint.get("node_intents") or []
    if blueprint.get("template_mode") != "legacy_pair" or len(node_intents) != 1:
        return None

    intent = node_intents[0]
    role = str(intent.get("role") or "")
    domain = str(intent.get("supply_domain") or "")
    text = " ".join(
        str(value)
        for value in (
            user_input,
            constraints.get("raw_text"),
            constraints.get("scene"),
            constraints.get("activity_type"),
            constraints.get("food_type"),
        )
        if value
    )
    planning_preferences = constraints.get("planning_preferences", {}) or {}
    for key in ("food_type", "restaurant_type", "experience_type"):
        value = planning_preferences.get(key)
        if isinstance(value, (list, tuple, set)):
            text += " " + " ".join(str(item) for item in value)
        elif value:
            text += " " + str(value)

    if domain == "restaurant":
        if _explicit_activity_requirements(constraints):
            return None
        if _duration_range_wants_itinerary(constraints, max_single_node_min=180):
            return None
        if _scenario_has_activity_context(scenario_activities, text):
            return None
        if role == "cafe" or "cafe_non_full_meal" in requirement_contract.get("hard_requirements", []):
            return "cafe_only"
        return "restaurant_only"

    if domain == "activity":
        dining_signals = (
            "吃",
            "餐",
            "饭",
            "轻食",
            "减肥",
            "低卡",
            "咖啡",
            "甜品",
            "火锅",
            "烤肉",
            "烧烤",
        )
        if constraints.get("mom_diet") == "low_calorie":
            return None
        if _text_has_any(text, dining_signals):
            return None
        if _duration_range_wants_itinerary(constraints, max_single_node_min=180):
            return None
        return "activity_only"

    return None


def _apply_single_node_duration_defaults(constraints: dict, shape: str | None) -> dict:
    if not shape:
        return constraints
    if constraints.get("duration_range") not in (None, "") or constraints.get("duration") not in (None, ""):
        return constraints
    enhanced = dict(constraints)
    if shape == "cafe_only":
        enhanced["duration_range"] = [45, 150]
    elif shape == "restaurant_only":
        enhanced["duration_range"] = [60, 180]
    elif shape == "activity_only":
        enhanced["duration_range"] = [60, 240]
    return enhanced


def _default_single_node_start(shape: str, intent: dict, constraints: dict) -> int:
    raw_start = constraints.get("start_time")
    if raw_start:
        parsed = _time_to_minutes(raw_start, default=-1, day=1)
        if parsed >= 0:
            return parsed
    role = str(intent.get("role") or "")
    if shape == "cafe_only":
        return 15 * 60
    if role == "restaurant_breakfast":
        return 9 * 60
    if role == "restaurant_lunch":
        return 12 * 60
    if shape == "restaurant_only":
        return 18 * 60 + 30
    return 14 * 60 + 30


def _build_single_node_timeline(node: dict, shape: str, intent: dict, constraints: dict) -> tuple[list[dict], dict]:
    desired_start = _default_single_node_start(shape, intent, constraints)
    start = _choose_node_start_time(
        node,
        desired_start=desired_start,
        earliest_start=desired_start,
        day=int(intent.get("day") or 1),
    )
    duration = int(node.get("duration_min") or intent.get("default_duration_min") or 60)
    end = start + max(15, duration)
    timeline_type = "restaurant" if shape in {"restaurant_only", "cafe_only"} else "play"
    timeline = [
        {
            "time": f"{_format_itinerary_time(start)}-{_format_itinerary_time(end)}",
            "activity": node.get("name"),
            "poi_id": node.get("poi_id"),
            "type": timeline_type,
            "role": intent.get("role"),
            "day": int(intent.get("day") or 1),
            "duration_min": duration,
            "price": node.get("price"),
            "notes": [
                "single_node_plan",
                str(intent.get("label") or intent.get("role") or node.get("type") or ""),
            ],
        }
    ]
    return timeline, {
        "nodes": [
            {
                "poi_id": node.get("poi_id"),
                "role": intent.get("role"),
                "start": _format_itinerary_time(start),
                "end": _format_itinerary_time(end),
                "day": int(intent.get("day") or 1),
            }
        ],
        "first_start_min": start,
        "last_end_min": end,
        "active_duration_min": duration,
    }


def _combine_single_node_plan_candidates(
    activities: list[dict],
    restaurants: list[dict],
    constraints: dict,
    scene_type: str,
    blueprint: dict,
    shape: str,
    user_profile: dict | None = None,
    weather_context: dict | None = None,
    max_candidates: int = DEFAULT_PLAN_CANDIDATE_LIMIT,
) -> list[dict]:
    node_intents = blueprint.get("node_intents") or []
    if not node_intents:
        return []
    intent = node_intents[0]
    pool = restaurants if shape in {"restaurant_only", "cafe_only"} else activities
    if not pool:
        return []
    if shape == "cafe_only":
        cafe_pool = [item for item in pool if _matches_strict_node_role(item, "cafe")]
        if cafe_pool:
            pool = cafe_pool
        else:
            pool = sorted(
                pool,
                key=lambda item: _restaurant_role_score(item, "cafe_dessert"),
                reverse=True,
            )

    config = get_constraint_config_with_profile(constraints, user_profile)
    people_count = config["people_count"]
    plan_candidates: list[dict] = []
    for index, candidate in enumerate(pool[:max_candidates], start=1):
        node = dict(candidate)
        node["_itinerary_intent"] = intent
        node["itinerary_role"] = intent.get("role")
        timeline, schedule = _build_single_node_timeline(node, shape, intent, constraints)
        route_facts = _build_sequence_route_facts([node], constraints)
        total_price = to_float(node.get("price"), 0.0)
        max_queue_time_plan = to_float(node.get("queue_time_min"), 0.0)
        available = node.get("available", True) and route_facts.get("feasible", True)
        tags = _collect_plan_tags(node)
        plan_candidates.append(
            {
                "plan_id": f"cand_single_{index:03d}",
                "scene_type": scene_type,
                "planner_mode": "single_node",
                "plan_shape": shape,
                "plan_template": [intent.get("supply_domain")],
                "nodes": [node],
                "timeline": timeline,
                "route": {
                    "total_distance_km": route_facts["total_distance_km"],
                    "total_travel_time_min": route_facts["total_travel_time_min"],
                    "traffic_status": route_facts.get("traffic_status"),
                    "route_sources": route_facts.get("route_sources", []),
                    "legs": route_facts["legs"],
                },
                "schedule": schedule,
                "budget": {"total_price": total_price},
                "availability": {
                    "all_available": available,
                    "max_queue_time_min": max_queue_time_plan,
                    "detail": {
                        "node_available": node.get("available", True),
                        "node_queue_min": max_queue_time_plan,
                    },
                },
                "estimated_duration_min": int(schedule["active_duration_min"] + route_facts["total_travel_time_min"]),
                "tags": tags,
                "restaurant_role": _restaurant_role(node) if shape in {"restaurant_only", "cafe_only"} else None,
                "weather_context": weather_context or {},
                "constraint_snapshot": {
                    "planning_horizon": blueprint.get("planning_horizon"),
                    "planning_days": blueprint.get("planning_days"),
                    "plan_shape": shape,
                },
                "execution_requirements": {
                    "min_people": people_count,
                    "people_count": people_count,
                    "pre_booking_required": max_queue_time_plan > 20,
                    "special_preparation": [],
                },
                "benchmark_ready": True,
                "execution_scope": "full",
                "non_executable_nodes": [],
                "b_itinerary_blueprint": blueprint,
                "planning_horizon": blueprint.get("planning_horizon"),
                "planning_days": blueprint.get("planning_days"),
            }
        )
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
    user_input = str(state.get("user_input") or constraints.get("raw_text") or "")
    b_replan_request = _active_replan_request(state, constraints)
    scenario_activities = derive_scenario_activities(
        constraints,
        user_profile,
        state.get("scenario_activities", []),
    )
    constraints, requirement_contract, requirement_metadata = apply_b_requirement_contract(
        state,
        constraints=constraints,
    )
    if requirement_metadata and requirement_metadata.get("success"):
        execution_log.append("[B] candidate_generator_node applied LongCat requirement compiler")
    elif requirement_metadata:
        execution_log.append("[B] candidate_generator_node used deterministic requirement compiler after LongCat fallback")
    elif requirement_contract.get("hard_requirements"):
        execution_log.append("[B] candidate_generator_node applied deterministic requirement compiler")

    constraints, itinerary_blueprint = apply_b_itinerary_blueprint(
        state,
        constraints=constraints,
        scenario_activities=scenario_activities,
    )
    constraints = _apply_blueprint_duration_defaults(constraints, itinerary_blueprint)
    rag_node_candidates, rag_candidate_metadata = normalize_rag_node_candidates(
        state,
        constraints,
        itinerary_blueprint,
    )
    rag_coverage = rag_candidate_coverage(itinerary_blueprint, rag_node_candidates)
    single_node_shape = _single_node_plan_shape(
        itinerary_blueprint,
        constraints,
        requirement_contract,
        user_input,
        scenario_activities,
    )
    constraints = _apply_single_node_duration_defaults(constraints, single_node_shape)
    multinode_issue = None
    can_plan_multinode_with_current_supply = _can_plan_multinode_with_current_supply(itinerary_blueprint)
    can_plan_multinode_with_rag = _can_plan_multinode_with_rag(itinerary_blueprint, rag_coverage)
    can_plan_multinode = can_plan_multinode_with_current_supply or can_plan_multinode_with_rag
    covered_rag_node_ids = set((rag_coverage or {}).get("covered_node_ids") or [])
    can_plan_partial_multinode = bool(
        itinerary_blueprint.get("template_mode") == "multi_node"
        and any(
            str(intent.get("supply_domain") or "") in MULTINODE_SUPPORTED_DOMAINS
            or str(intent.get("node_id") or "") in covered_rag_node_ids
            for intent in itinerary_blueprint.get("node_intents", []) or []
        )
    )
    if single_node_shape:
        execution_log.append(
            f"[B] candidate_generator_node detected single-node plan shape ({single_node_shape})"
        )
    if itinerary_blueprint.get("template_mode") == "multi_node":
        execution_log.append(
            "[B] candidate_generator_node detected multi-node itinerary blueprint "
            f"(nodes={itinerary_blueprint.get('node_count')}, "
            f"unsupported_roles={itinerary_blueprint.get('unsupported_roles', [])})"
        )

        multinode_issue = {
            "type": "multi_node_blueprint_not_yet_planned",
            "template_mode": itinerary_blueprint.get("template_mode"),
            "planning_horizon": itinerary_blueprint.get("planning_horizon"),
            "node_intents": itinerary_blueprint.get("node_intents", []),
            "unsupported_roles": itinerary_blueprint.get("unsupported_roles", []),
            "requires_rag": itinerary_blueprint.get("requires_rag", False),
            "rag_candidate_coverage": rag_coverage,
            "message": "当前主规划器仍是活动+餐厅 pair planner；该需求需要多节点 itinerary planner 才能完整覆盖",
        }
        if can_plan_multinode_with_current_supply:
            execution_log.append(
                "[B] candidate_generator_node will plan supported activity/restaurant multi-node itinerary"
            )
        elif can_plan_multinode_with_rag:
            execution_log.append(
                "[B] candidate_generator_node will plan multi-node itinerary with RAG node candidates "
                f"(covered={rag_coverage.get('covered_node_count')}/{rag_coverage.get('required_node_count')})"
            )
        elif can_plan_partial_multinode:
            execution_log.append(
                "[B] candidate_generator_node will plan partial multi-node itinerary with available RAG/local nodes "
                f"(covered={rag_coverage.get('covered_node_count')}/{rag_coverage.get('required_node_count')})"
            )
        elif not _allow_legacy_fallback_for_multinode():
            execution_log.append(
                "[B] candidate_generator_node stopped before legacy pair fallback "
                "because multi-node itinerary would produce an incomplete plan"
            )
            return {
                "candidates": [],
                "scene_type": scene_type,
                "scenario_activities": scenario_activities,
                "constraints": constraints,
                "user_profile": user_profile,
                "b_requirement_contract": requirement_contract,
                "b_itinerary_blueprint": itinerary_blueprint,
                "b_rag_candidate_metadata": rag_candidate_metadata,
                "b_rag_candidate_coverage": rag_coverage,
                "candidate_generation_issues": [multinode_issue],
                "candidate_recall_diagnostics": {
                    "scene_type": scene_type,
                    "sequence": _sequence_preference(constraints),
                    "blueprint": itinerary_blueprint,
                    "rag_candidate_metadata": rag_candidate_metadata,
                    "rag_candidate_coverage": rag_coverage,
                    "counts": {
                        "activities_initial": 0,
                        "restaurants_initial": 0,
                        "plan_candidates": 0,
                    },
                    "legacy_pair_fallback": "blocked",
                },
                "execution_log": execution_log,
            }

    scenario_activities = derive_scenario_activities(
        constraints,
        user_profile,
        scenario_activities,
    )
    ai_hints_metadata = None
    if rag_node_candidates:
        execution_log.append(
            "[B] candidate_generator_node skipped LongCat semantic hints; "
            "node-level RAG evidence is already available"
        )
    else:
        constraints, user_profile, scenario_activities, ai_hints_metadata = apply_b_semantic_hints(
            state,
            constraints=constraints,
            user_profile=user_profile,
            scenario_activities=scenario_activities,
        )
        if ai_hints_metadata and ai_hints_metadata.get("success"):
            execution_log.append("[B] candidate_generator_node applied LongCat semantic hints")
        elif ai_hints_metadata:
            execution_log.append("[B] candidate_generator_node skipped LongCat semantic hints after fallback")

    weather_context = get_weather_context(
        constraints,
        existing_context=state.get("weather_context"),
    )
    if weather_context.get("available"):
        weather_tags = ",".join(weather_context.get("condition_tags", []) or [])
        execution_log.append(
            f"[B] candidate_generator_node applied WeatherForecaster context ({weather_tags})"
        )
    else:
        execution_log.append(
            f"[B] candidate_generator_node weather fallback ({weather_context.get('source')})"
        )

    top_k_activity = _get_top_k("top_k_activity", DEFAULT_TOP_K_ACTIVITY)
    top_k_restaurant = _get_top_k("top_k_restaurant", DEFAULT_TOP_K_RESTAURANT)
    route_lookahead_multiplier = _get_route_lookahead_multiplier()
    pair_pool_multiplier = _get_pair_pool_multiplier()
    plan_candidate_limit = _get_plan_candidate_limit()
    max_pair_combinations = _get_max_pair_combinations()

    if (
        itinerary_blueprint.get("template_mode") == "multi_node"
        and can_plan_multinode_with_rag
        and rag_coverage.get("all_nodes_covered")
        and not single_node_shape
    ):
        raw_plan_candidates = _combine_multinode_plan_candidates(
            [],
            [],
            constraints,
            scene_type,
            itinerary_blueprint,
            user_profile,
            weather_context,
            rag_node_candidates,
            rag_coverage,
            max_candidates=min(plan_candidate_limit, DEFAULT_MAX_MULTINODE_CANDIDATES),
        )
        raw_plan_candidates, replan_filter_meta = _filter_replan_avoided_plans(
            raw_plan_candidates,
            b_replan_request,
        )
        plan_candidates = _sort_plan_candidates(
            raw_plan_candidates,
            constraints,
            scene_type,
            user_profile,
            scenario_activities,
            weather_context,
            user_input,
        )[:plan_candidate_limit]
        if plan_candidates:
            recall_diagnostics = {
                "scene_type": scene_type,
                "sequence": _sequence_preference(constraints),
                "blueprint": itinerary_blueprint,
                "rag_candidate_metadata": rag_candidate_metadata,
                "rag_candidate_coverage": rag_coverage,
                "single_node_shape": single_node_shape,
                "replan_request_active": bool(b_replan_request),
                "planner_mode": "multi_node_itinerary",
                "fast_path": "rag_only_multinode",
                "replan_filter": replan_filter_meta,
                "counts": {
                    "activities_initial": 0,
                    "restaurants_initial": 0,
                    "selected_activity_pool": 0,
                    "selected_restaurant_pool": 0,
                    "raw_plan_candidates": len(raw_plan_candidates),
                    "plan_candidates": len(plan_candidates),
                },
                "multinode_generation_budget": {
                    "max_multinode_candidates": min(plan_candidate_limit, DEFAULT_MAX_MULTINODE_CANDIDATES),
                    "node_count": len(itinerary_blueprint.get("node_intents", []) or []),
                },
            }
            execution_log.append(
                "[B] candidate_generator_node used RAG-only multi-node fast path "
                f"(raw_plan_candidates={len(raw_plan_candidates)}, plan_candidates={len(plan_candidates)})"
            )
            result = {
                "candidates": plan_candidates,
                "scene_type": scene_type,
                "scenario_activities": scenario_activities,
                "constraints": constraints,
                "user_profile": user_profile,
                "b_requirement_contract": requirement_contract,
                "b_itinerary_blueprint": itinerary_blueprint,
                "b_rag_candidate_metadata": rag_candidate_metadata,
                "b_rag_candidate_coverage": rag_coverage,
                "weather_context": weather_context,
                "candidate_generation_issues": [],
                "candidate_recall_diagnostics": recall_diagnostics,
                "execution_log": execution_log,
            }
            if ai_hints_metadata:
                result["b_ai_semantic_hints"] = ai_hints_metadata
            if requirement_metadata:
                result["b_ai_requirement_compiler"] = requirement_metadata
            return result
        execution_log.append(
            "[B] candidate_generator_node RAG-only multi-node fast path found no valid sequence; "
            "falling back to full local supply"
        )

    rag_activity_candidates = _rag_candidates_for_domain(rag_node_candidates, "activity")
    rag_restaurant_candidates = _rag_candidates_for_domain(rag_node_candidates, "restaurant")
    if rag_activity_candidates:
        activity_candidates = rag_activity_candidates
        execution_log.append(
            "[B] candidate_generator_node used RAG activity candidate pool "
            f"(activities={len(activity_candidates)})"
        )
    else:
        activity_candidates = fetch_activity_candidates(
            constraints=constraints,
            scene_type=scene_type,
            scenario_activities=scenario_activities,
        ) or _build_activity_candidates()
    if rag_restaurant_candidates:
        restaurant_candidates = rag_restaurant_candidates
        execution_log.append(
            "[B] candidate_generator_node used RAG restaurant candidate pool "
            f"(restaurants={len(restaurant_candidates)})"
        )
    else:
        restaurant_candidates = fetch_restaurant_candidates(
            constraints=constraints,
            scene_type=scene_type,
            scenario_activities=scenario_activities,
        ) or _build_restaurant_candidates()
    activity_pool_size = top_k_activity * route_lookahead_multiplier * pair_pool_multiplier
    restaurant_pool_size = top_k_restaurant * route_lookahead_multiplier * pair_pool_multiplier
    destination_city = destination_city_from_constraints(constraints)
    if destination_city:
        activity_count_before_city = len(activity_candidates)
        restaurant_count_before_city = len(restaurant_candidates)
        activity_candidates = [
            item
            for item in activity_candidates
            if item_matches_destination_city(item, destination_city)
        ]
        restaurant_candidates = [
            item
            for item in restaurant_candidates
            if item_matches_destination_city(item, destination_city)
        ]
        if (
            len(activity_candidates) != activity_count_before_city
            or len(restaurant_candidates) != restaurant_count_before_city
        ):
            execution_log.append(
                "[B] candidate_generator_node applied destination-city supply guard "
                f"(city={destination_city}, "
                f"activities={activity_count_before_city}->{len(activity_candidates)}, "
                f"restaurants={restaurant_count_before_city}->{len(restaurant_candidates)})"
            )
    activity_count_before_geo = len(activity_candidates)
    restaurant_count_before_geo = len(restaurant_candidates)
    activity_candidates, activity_geo_meta = _filter_candidates_by_geo_window(
        activity_candidates,
        constraints=constraints,
        user_profile=user_profile,
        min_keep=max(activity_pool_size * 12, 500),
    )
    restaurant_candidates, restaurant_geo_meta = _filter_candidates_by_geo_window(
        restaurant_candidates,
        constraints=constraints,
        user_profile=user_profile,
        min_keep=max(restaurant_pool_size * 12, 500),
    )
    if activity_geo_meta.get("applied") or restaurant_geo_meta.get("applied"):
        execution_log.append(
            "[B] candidate_generator_node applied geo recall window before semantic ranking "
            f"(activities={activity_count_before_geo}->{len(activity_candidates)}, "
            f"restaurants={restaurant_count_before_geo}->{len(restaurant_candidates)}, "
            f"activity_radius={activity_geo_meta.get('radius_km')}, "
            f"restaurant_radius={restaurant_geo_meta.get('radius_km')})"
        )
    candidate_generation_issues = []
    required_activity_tags = _explicit_activity_requirements(constraints)
    required_restaurant_tags = _explicit_restaurant_requirements(constraints)
    sequence = _sequence_preference(constraints)
    recall_semantic_groups = sorted(
        semantic_groups_in_values(
            _raw_preference_sources(constraints, user_profile, scenario_activities)
            + [user_input]
        )
    )
    preferred_restaurant_role = _preferred_restaurant_role(
        constraints,
        user_profile,
        scenario_activities,
        user_input,
    )
    recall_diagnostics = {
        "scene_type": scene_type,
        "sequence": sequence,
        "semantic_groups": recall_semantic_groups,
        "preferred_restaurant_role": preferred_restaurant_role,
        "counts": {
            "activities_initial": activity_count_before_geo,
            "restaurants_initial": restaurant_count_before_geo,
            "activities_after_geo": len(activity_candidates),
            "restaurants_after_geo": len(restaurant_candidates),
        },
        "geo": {
            "activity": activity_geo_meta,
            "restaurant": restaurant_geo_meta,
        },
        "requirements": {
            "activity": sorted(required_activity_tags),
            "restaurant": sorted(required_restaurant_tags),
        },
        "rag_candidate_metadata": rag_candidate_metadata,
        "rag_candidate_coverage": rag_coverage,
        "single_node_shape": single_node_shape,
        "replan_request_active": bool(b_replan_request),
    }
    if required_activity_tags:
        activity_count_before_filter = len(activity_candidates)
        activity_candidates = _filter_activities_by_requirements(
            activity_candidates,
            required_activity_tags,
        )
        execution_log.append(
            "[B] candidate_generator_node applied explicit activity hard filter "
            f"(required={sorted(required_activity_tags)}, "
            f"activities={activity_count_before_filter}->{len(activity_candidates)})"
        )
        if not activity_candidates:
            candidate_generation_issues.append(
                {
                    "type": "missing_activity_supply",
                    "required_activity_tags": sorted(required_activity_tags),
                    "message": "当前 mock 活动供给中没有匹配用户显式活动需求的 POI",
                }
            )
        recall_diagnostics["counts"]["activities_after_requirement_filter"] = len(activity_candidates)
    if required_restaurant_tags:
        restaurant_count_before_filter = len(restaurant_candidates)
        restaurant_candidates = _filter_restaurants_by_requirements(
            restaurant_candidates,
            required_restaurant_tags,
        )
        execution_log.append(
            "[B] candidate_generator_node applied explicit restaurant hard filter "
            f"(required={sorted(required_restaurant_tags)}, "
            f"restaurants={restaurant_count_before_filter}->{len(restaurant_candidates)})"
        )
        if not restaurant_candidates:
            candidate_generation_issues.append(
                {
                    "type": "missing_restaurant_supply",
                    "required_restaurant_tags": sorted(required_restaurant_tags),
                    "message": "当前 mock 餐厅供给中没有匹配用户显式餐饮需求的 POI",
                }
            )
        recall_diagnostics["counts"]["restaurants_after_requirement_filter"] = len(restaurant_candidates)

    if itinerary_blueprint.get("template_mode") == "multi_node" and not can_plan_multinode and multinode_issue:
        candidate_generation_issues.append(multinode_issue)

    activity_pretrim_size = max(activity_pool_size * 8, 300)
    restaurant_pretrim_size = max(restaurant_pool_size * 8, 300)

    activity_candidates_for_sort = _pretrim_candidates_for_sort(
        activity_candidates,
        constraints=constraints,
        user_profile=user_profile,
        scenario_activities=scenario_activities,
        limit=activity_pretrim_size,
        user_input=user_input,
    )
    restaurant_candidates_for_sort = _pretrim_candidates_for_sort(
        restaurant_candidates,
        constraints=constraints,
        user_profile=user_profile,
        scenario_activities=scenario_activities,
        limit=restaurant_pretrim_size,
        user_input=user_input,
    )
    recall_diagnostics["counts"]["activities_after_pretrim"] = len(activity_candidates_for_sort)
    recall_diagnostics["counts"]["restaurants_after_pretrim"] = len(restaurant_candidates_for_sort)
    if len(activity_candidates_for_sort) != len(activity_candidates) or len(restaurant_candidates_for_sort) != len(restaurant_candidates):
        execution_log.append(
            "[B] candidate_generator_node pretrimmed large supply before semantic sort "
            f"(activities={len(activity_candidates)}->{len(activity_candidates_for_sort)}, "
            f"restaurants={len(restaurant_candidates)}->{len(restaurant_candidates_for_sort)})"
        )

    selected_activities = _sort_candidates(
        activity_candidates_for_sort,
        constraints,
        scene_type,
        user_profile,
        scenario_activities,
        weather_context,
        user_input,
    )[:activity_pool_size]
    selected_restaurants = _sort_candidates(
        restaurant_candidates_for_sort,
        constraints,
        scene_type,
        user_profile,
        scenario_activities,
        weather_context,
        user_input,
    )[:restaurant_pool_size]
    recall_diagnostics["counts"]["selected_activity_pool"] = len(selected_activities)
    recall_diagnostics["counts"]["selected_restaurant_pool"] = len(selected_restaurants)
    recall_diagnostics["selected_restaurant_roles"] = {
        role: sum(1 for item in selected_restaurants if _restaurant_role(item) == role)
        for role in ("cafe_dessert", "light_meal", "full_meal")
    }

    if single_node_shape:
        raw_plan_candidates = _combine_single_node_plan_candidates(
            selected_activities,
            selected_restaurants,
            constraints,
            scene_type,
            itinerary_blueprint,
            single_node_shape,
            user_profile,
            weather_context,
            max_candidates=plan_candidate_limit,
        )
        recall_diagnostics["planner_mode"] = "single_node"
        recall_diagnostics["single_node_shape"] = single_node_shape
        if not raw_plan_candidates:
            candidate_generation_issues.append(
                {
                    "type": "no_valid_single_node_candidate",
                    "plan_shape": single_node_shape,
                    "node_intents": itinerary_blueprint.get("node_intents", []),
                    "message": "Single-node planner could not find a valid POI candidate",
                }
            )
    elif can_plan_multinode or can_plan_partial_multinode:
        raw_plan_candidates = _combine_multinode_plan_candidates(
            selected_activities,
            selected_restaurants,
            constraints,
            scene_type,
            itinerary_blueprint,
            user_profile,
            weather_context,
            rag_node_candidates,
            rag_coverage,
            max_candidates=min(plan_candidate_limit, DEFAULT_MAX_MULTINODE_CANDIDATES),
        )
        recall_diagnostics["planner_mode"] = "multi_node_itinerary"
        if not raw_plan_candidates:
            candidate_generation_issues.append(
                {
                    "type": "no_valid_multinode_combination",
                    "node_intents": itinerary_blueprint.get("node_intents", []),
                    "message": "Supported multi-node planner could not assemble a unique activity/restaurant sequence",
                }
            )
    else:
        raw_plan_candidates = _combine_plan_candidates(
            selected_activities,
            selected_restaurants,
            constraints,
            scene_type,
            user_profile,
            weather_context,
            max_pair_combinations=max_pair_combinations,
        )
    if (
        not can_plan_multinode
        and not single_node_shape
        and
        sequence == SEQUENCE_RESTAURANT_THEN_ACTIVITY
        and selected_activities
        and selected_restaurants
        and not raw_plan_candidates
    ):
        candidate_generation_issues.append(
            {
                "type": "no_valid_sequence_schedule",
                "sequence": sequence,
                "message": "当前 mock 时间段无法满足先吃饭再活动的顺序，请补充餐后活动档期或调整行程顺序",
            }
        )
    raw_plan_candidates, replan_filter_meta = _filter_replan_avoided_plans(
        raw_plan_candidates,
        b_replan_request,
    )
    recall_diagnostics["replan_filter"] = replan_filter_meta
    if replan_filter_meta.get("applied"):
        execution_log.append(
            "[B] candidate_generator_node avoided previously criticized plan identity "
            f"(removed={replan_filter_meta.get('removed')}, kept={replan_filter_meta.get('kept')})"
        )
    plan_candidates = _sort_plan_candidates(
        raw_plan_candidates,
        constraints,
        scene_type,
        user_profile,
        scenario_activities,
        weather_context,
        user_input,
    )[:plan_candidate_limit]
    recall_diagnostics["counts"]["raw_plan_candidates"] = len(raw_plan_candidates)
    recall_diagnostics["counts"]["plan_candidates"] = len(plan_candidates)
    if single_node_shape:
        recall_diagnostics["single_node_generation_budget"] = {
            "plan_shape": single_node_shape,
            "candidate_limit": plan_candidate_limit,
        }
    elif can_plan_multinode:
        recall_diagnostics["multinode_generation_budget"] = {
            "max_multinode_candidates": min(plan_candidate_limit, DEFAULT_MAX_MULTINODE_CANDIDATES),
            "node_count": len(itinerary_blueprint.get("node_intents", []) or []),
        }
    else:
        recall_diagnostics["pair_generation_budget"] = {
            "max_pair_combinations": max_pair_combinations,
            "selected_pair_space": len(selected_activities) * len(selected_restaurants),
        }

    execution_log.append(
        f"[B] candidate_generator_node 生成 {len(plan_candidates)} 个 plan_candidates "
        f"(activities={len(activity_candidates)}, restaurants={len(restaurant_candidates)}, "
        f"top_k_activity={top_k_activity}, top_k_restaurant={top_k_restaurant}, "
        f"route_lookahead_multiplier={route_lookahead_multiplier}, "
        f"pair_pool_multiplier={pair_pool_multiplier}, max_pair_combinations={max_pair_combinations}, "
        f"raw_plan_candidates={len(raw_plan_candidates)})"
    )

    result = {
        "candidates": plan_candidates,
        "scene_type": scene_type,
        "scenario_activities": scenario_activities,
        "constraints": constraints,
        "user_profile": user_profile,
        "b_requirement_contract": requirement_contract,
        "b_itinerary_blueprint": itinerary_blueprint,
        "b_rag_candidate_metadata": rag_candidate_metadata,
        "b_rag_candidate_coverage": rag_coverage,
        "weather_context": weather_context,
        "candidate_generation_issues": candidate_generation_issues,
        "candidate_recall_diagnostics": recall_diagnostics,
        "execution_log": execution_log,
    }

    if ai_hints_metadata:
        result["b_ai_semantic_hints"] = ai_hints_metadata
    if requirement_metadata:
        result["b_ai_requirement_compiler"] = requirement_metadata

    return result
