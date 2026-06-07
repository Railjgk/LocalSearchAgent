try:
    from src.state import PlanState
except ImportError:
    PlanState = dict

from functools import lru_cache

from .mock_api_adapter import (
    _normalize_poi as _normalize_mock_poi,
    fetch_activity_candidates,
    fetch_restaurant_candidates,
)
from .b_ai_hints import apply_b_semantic_hints
from .b_candidate_policy import (
    DEFAULT_MAX_MULTINODE_CANDIDATES,
    DEFAULT_PLAN_CANDIDATE_LIMIT,
    DEFAULT_TOP_K_ACTIVITY,
    DEFAULT_TOP_K_RESTAURANT,
    allow_legacy_fallback_for_multinode as _allow_legacy_fallback_for_multinode,
    geo_prefilter_min_keep as _get_geo_prefilter_min_keep,
    max_pair_combinations as _get_max_pair_combinations,
    mock_data_dir as _mock_data_dir,
    pair_pool_multiplier as _get_pair_pool_multiplier,
    plan_candidate_limit as _get_plan_candidate_limit,
    pretrim_min_keep as _get_pretrim_min_keep,
    route_lookahead_multiplier as _get_route_lookahead_multiplier,
    route_source_order as _get_route_source_order,
    top_k as _get_top_k,
    transition_buffer_min as _get_transition_buffer_min,
)
from .b_execution_scope import (
    node_is_supported_by_current_c as _node_is_supported_by_current_c,
    node_requires_c_execution as _node_requires_c_execution,
)
from .b_itinerary_blueprint import apply_b_itinerary_blueprint
from .b_local_food_guardrails import (
    intent_requires_local_shanghai_food as _intent_requires_local_shanghai_food,
    item_conflicts_with_local_shanghai_food as _item_conflicts_with_local_shanghai_food,
    item_has_local_shanghai_food_identity as _item_has_local_shanghai_food_identity,
)
from .b_multinode_policy import (
    MULTINODE_SUPPORTED_DOMAINS,
    apply_blueprint_duration_defaults as _apply_blueprint_duration_defaults,
    can_plan_multinode_with_current_supply as _can_plan_multinode_with_current_supply,
    can_plan_multinode_with_rag as _can_plan_multinode_with_rag,
    can_plan_partial_multinode as _can_plan_partial_multinode,
    mark_rag_resolved_blueprint_roles as _mark_rag_resolved_blueprint_roles,
)
from .b_plan_templates import get_plan_templates as _get_plan_templates
from .b_rag_contract import normalize_rag_node_candidates, rag_candidate_coverage
from .b_requirement_compiler import apply_b_requirement_contract
from .b_replan_filter import (
    active_replan_request as _active_replan_request,
    filter_replan_avoided_plans as _filter_replan_avoided_plans,
    replan_avoid_identity_matches as _replan_avoid_identity_matches,
)
from .b_restaurant_roles import (
    preferred_restaurant_role_from_values as _preferred_restaurant_role_from_values,
    restaurant_role as _restaurant_role,
    restaurant_role_score as _restaurant_role_score,
)
from .b_sequence_policy import (
    SEQUENCE_ACTIVITY_THEN_RESTAURANT,
    SEQUENCE_RESTAURANT_THEN_ACTIVITY,
    sequence_preference as _sequence_preference,
)
from .b_text_match import (
    fast_text_match_score as _fast_text_match_score,
    item_matches_terms as _item_matches_terms,
    normalized_query_terms as _normalized_query_terms,
    semantic_cache_signature as _semantic_cache_signature,
    semantic_text_index as _semantic_text_index,
)
from .b_time_slots import (
    available_slot_minutes as _available_slot_minutes,
    choose_node_start_time as _choose_node_start_time,
    format_itinerary_time as _format_itinerary_time,
    format_itinerary_time_range as _format_itinerary_time_range,
    pick_time_slots as _pick_time_slots,
    pick_time_slots_restaurant_first as _pick_time_slots_restaurant_first,
    slot_to_minutes as _slot_to_minutes,
    time_to_minutes as _time_to_minutes,
)
from .b_route_facts import (
    build_route_facts as _build_route_facts_impl,
    build_sequence_route_facts as _build_sequence_route_facts_impl,
)
from .b_route_geometry import (
    filter_candidates_by_geo_window as _filter_candidates_by_geo_window,
    geo_prefilter_origin as _geo_prefilter_origin,
    haversine_km as _haversine_km,
    item_coordinates as _item_coordinates,
    parse_coordinates as _parse_coordinates,
)
from .b_semantics import (
    B_ACTIVITY_INTENT_GROUPS,
    B_RESTAURANT_INTENT_GROUPS,
    b_semantic_terms,
    flatten_semantic_values,
    normalize_semantic_text,
    semantic_groups_in_values,
    semantic_match_score,
    semantic_terms_for_groups,
)
from .b_weather_scoring import weather_candidate_bonus as _weather_candidate_bonus_impl
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
    normalize_scene_type,
)
MULTINODE_ROLE_TERMS = {
    "family_activity": [
        "亲子",
        "儿童",
        "孩子",
        "亲子手作",
        "亲子手工",
        "手作",
        "手工",
        "陶艺",
        "DIY",
        "diy",
        "kid_friendly",
        "family_friendly",
        "low_intensity",
        "indoor",
    ],
    "family_indoor_play": ["室内乐园", "亲子乐园", "儿童乐园", "游乐园", "淘气堡", "蹦床", "kid_friendly", "family_friendly", "indoor_playground"],
    "exhibition": ["展览", "看展", "博物馆", "美术馆", "museum", "art", "exhibition"],
    "river_cruise": ["游船", "游轮", "邮轮", "黄浦江", "浦江游览", "夜游黄浦江", "包厢", "自助餐", "cruise"],
    "citywalk_market": ["citywalk", "城市漫步", "市集", "街区", "历史文化", "历史建筑", "文化街区", "文化景区", "local_market", "local_culture"],
    "park_scenic_walk": ["公园", "游园", "绿地", "滨江", "江边", "河边", "夜景", "散步", "步道", "outdoor", "citywalk"],
    "board_game_escape": ["桌游", "棋牌", "剧本杀", "狼人杀", "密室", "密室逃脱", "board_game", "escape_room"],
    "internet_cafe": ["网吧", "网咖", "电竞", "电玩", "游戏", "通宵", "点播影院", "电影", "internet_cafe", "game"],
    "karaoke": ["KTV", "ktv", "唱歌", "卡拉OK", "karaoke"],
    "bar": ["酒吧", "清吧", "喝一杯", "小酌", "鸡尾酒", "精酿", "bar"],
    "talk_show": ["脱口秀", "喜剧", "相声", "曲艺", "评弹", "剧场", "演出", "livehouse"],
    "theatre_performance": ["话剧", "戏剧", "舞台剧", "儿童剧", "剧院", "剧场", "演出", "theatre"],
    "cinema": ["电影", "影院", "电影院", "观影", "IMAX"],
    "restaurant_breakfast": ["早餐", "早饭", "包子", "馄饨", "早点", "breakfast"],
    "restaurant_lunch": ["午餐", "中饭", "正餐", "full_meal"],
    "restaurant_dinner": ["晚餐", "晚饭", "正餐", "full_meal"],
    "restaurant_specific": ["餐厅", "吃饭", "正餐", "聚餐", "宴请", "商务宴请", "重要客户", "本帮菜", "上海菜", "小笼包", "小笼", "生煎", "汤包", "full_meal"],
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
STRICT_MULTINODE_ROLE_TEXT_TERMS = {
    "exhibition": ("美术馆", "博物馆", "展览", "展馆", "艺术馆", "画廊", "文化馆", "艺术", "历史", "museum", "gallery", "exhibition"),
    "citywalk_market": (
        "citywalk",
        "城市漫步",
        "老城",
        "街区",
        "市集",
        "历史",
        "文化",
        "景区",
        "栈桥",
        "八大关",
        "中山路",
        "小麦岛",
        "本地文化",
        "local_market",
        "local_culture",
    ),
    "board_game_escape": ("剧本杀", "密室", "桌游", "推理", "狼人杀", "血染钟楼", "escape_room", "board_game"),
    "cafe": ("咖啡", "咖啡馆", "咖啡厅", "下午茶", "甜品", "蛋糕", "烘焙", "cafe", "coffee", "dessert"),
    "cultural_photo": ("汉服", "古风", "古装", "拍照", "写真", "摄影", "换装", "文化体验"),
    "park_scenic_walk": ("公园", "游园", "绿地", "滨江", "江边", "河边", "夜景", "散步", "步道", "观景", "外滩"),
    "river_cruise": ("游船", "游轮", "邮轮", "黄浦江", "浦江游览", "游览船", "观光船", "码头", "包厢", "自助餐", "cruise"),
    "tea_house": ("茶馆", "茶艺", "茶室", "品茶", "喝茶", "teahouse", "tea"),
    "convenience_store": ("便利店", "超市", "全家", "罗森", "7-eleven", "711", "便利", "零食", "饮料"),
    "parking": ("停车", "停车场", "车库", "车位", "parking"),
    "wellness_massage": ("spa", "按摩", "足疗", "推拿", "养生", "洗脚", "修脚"),
    "fitness": ("健身", "瑜伽", "普拉提", "运动", "fitness", "yoga", "pilates"),
    "lodging": ("酒店", "民宿", "住宿", "宾馆", "hotel", "lodging"),
    "restaurant_breakfast": ("早餐", "早饭", "早点", "包子", "馄饨", "豆浆", "粥", "生煎", "breakfast"),
    "internet_cafe": ("网吧", "网咖", "电竞", "电竞馆", "电玩", "游戏", "通宵", "点播影院", "电影"),
    "bar": ("酒吧", "清吧", "喝一杯", "小酌", "鸡尾酒", "精酿", "夜店"),
    "talk_show": ("脱口秀", "喜剧", "相声", "曲艺", "评弹", "剧场", "演出", "livehouse"),
    "theatre_performance": ("话剧", "戏剧", "舞台剧", "儿童剧", "剧院", "剧场", "演出", "theatre"),
    "cinema": ("电影", "影院", "电影院", "观影", "IMAX"),
    "nail_salon": ("美甲", "美睫", "甲油胶", "做指甲"),
    "pet_grooming": ("宠物美容", "宠物spa", "宠物洗澡", "宠物洗护", "金毛", "毛发"),
    "pet_cafe": ("宠物友好咖啡", "宠物友好", "可带宠物", "带狗咖啡"),
    "pet_hospital": ("宠物医院", "宠物体检", "兽医", "动物医院"),
    "pet_store": ("宠物店", "宠物用品", "营养品", "狗粮", "猫粮"),
    "dental_clinic": ("牙科", "口腔", "牙医", "洗牙", "补牙", "种植牙", "矫正"),
    "karaoke": ("ktv", "KTV", "唱歌", "卡拉OK", "卡拉ok", "练歌房", "欢唱", "karaoke"),
    "flower_shop": ("鲜花", "花店", "花束", "花艺", "花坊", "flower"),
    "beauty_cosmetics": ("美妆", "化妆品", "日化", "护肤", "彩妆", "香水", "cosmetics"),
    "souvenir_shopping": ("特产", "土特产", "伴手礼", "纪念品", "文创", "周边", "礼品", "礼物", "老字号", "糕点", "蝴蝶酥", "带回去"),
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
RESTAURANT_ACTUAL_IDENTITY_FIELDS = (
    "name",
    "category",
    "sub_category",
    "gaode_type",
    "tags",
    "signature_dishes",
    "recommended_dishes",
    "dish_tags",
)
_ITEM_TAG_CACHE_LIMIT = 60000
_ITEM_TAG_CACHE: dict[tuple[int, tuple], tuple[tuple, tuple[str, ...]]] = {}
_ITEM_SEMANTIC_SIGNAL_CACHE: dict[tuple[int, tuple], tuple[tuple, set[str]]] = {}
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
CHILD_CONTEXT_TERMS = ("孩子", "小孩", "小朋友", "亲子", "带娃", "宝宝", "儿童", "家庭")
FAMILY_ONLY_SUPPLY_TERMS = (
    "亲子",
    "儿童乐园",
    "室内乐园",
    "淘气堡",
    "宝宝椅",
    "奈尔宝",
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


def _build_route_facts(
    activity: dict,
    restaurant: dict,
    constraints: dict | None = None,
    sequence: str = SEQUENCE_ACTIVITY_THEN_RESTAURANT,
) -> dict:
    return _build_route_facts_impl(
        activity,
        restaurant,
        constraints,
        sequence,
        mock_data_dir=_mock_data_dir(),
        source_order=_get_route_source_order(),
    )


def _build_sequence_route_facts(nodes: list[dict], constraints: dict | None = None) -> dict:
    return _build_sequence_route_facts_impl(
        nodes,
        constraints,
        mock_data_dir=_mock_data_dir(),
        source_order=_get_route_source_order(),
    )


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


def _has_child_context(
    *,
    scene_type: str,
    child_age: int | None,
    constraints: dict | None,
    user_input: str | None,
) -> bool:
    if scene_type == "family" or child_age is not None:
        return True
    text = " ".join(
        str(value)
        for value in (
            user_input,
            (constraints or {}).get("raw_text"),
            (constraints or {}).get("scene"),
        )
        if value not in (None, "")
    )
    return any(term in text for term in CHILD_CONTEXT_TERMS)


def _is_family_only_supply(item: dict) -> bool:
    signals = _semantic_signal_set_for_item(item)
    if signals.intersection({"kid_friendly", "family_friendly", "indoor_playground", "亲子活动", "亲子餐厅"}):
        return True
    text = " ".join(
        str(value)
        for value in (
            item.get("name"),
            item.get("category"),
            item.get("sub_category"),
            item.get("experience_type"),
            item.get("restaurant_category"),
            item.get("primary_category"),
            item.get("gaode_keyword"),
            item.get("tags"),
            item.get("tag_groups"),
        )
        if value not in (None, "")
    )
    return any(term in text for term in FAMILY_ONLY_SUPPLY_TERMS)


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
    raw_preferences = flatten_semantic_values(planning_preferences.get("activity_type"))
    blueprint = constraints.get("b_itinerary_blueprint")
    if isinstance(blueprint, dict):
        for intent in blueprint.get("node_intents", []) or []:
            if not isinstance(intent, dict):
                continue
            if str(intent.get("supply_domain") or "") != "activity":
                continue
            raw_preferences.extend(flatten_semantic_values(intent.get("search_terms")))
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
    blueprint = constraints.get("b_itinerary_blueprint")
    if isinstance(blueprint, dict):
        for intent in blueprint.get("node_intents", []) or []:
            if not isinstance(intent, dict):
                continue
            values.extend(flatten_semantic_values(intent.get("search_terms")))
            values.extend(flatten_semantic_values(intent.get("label")))
    return values


def _preferred_restaurant_role(
    constraints: dict | None,
    user_profile: dict | None = None,
    scenario_activities: list | None = None,
    user_input: str | None = None,
) -> str | None:
    constraints = constraints or {}
    return _preferred_restaurant_role_from_values(
        _raw_preference_sources(constraints, user_profile, scenario_activities),
        mom_diet=constraints.get("mom_diet"),
        raw_text=user_input or constraints.get("raw_text") or "",
    )


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
LODGING_NAME_TERMS = (
    "酒店",
    "宾馆",
    "住宿",
    "民宿",
    "客栈",
    "旅馆",
    "公寓",
    "度假",
    "hotel",
    "lodging",
)
LODGING_NAME_FALSE_POSITIVE_TERMS = (
    "餐厅",
    "餐饮",
    "酒楼",
    "海鲜",
    "鲁菜",
    "家常菜",
    "烧烤",
    "烤肉",
    "火锅",
    "料理",
    "水饺",
    "饺子",
    "渔家宴",
)
KARAOKE_FALSE_POSITIVE_TERMS = (
    "餐厅",
    "餐饮",
    "酒楼",
    "食堂",
    "饭店",
    "小馆",
    "粤菜",
    "本帮菜",
    "上海菜",
    "鲁菜",
    "海鲜",
    "火锅",
    "烧烤",
    "烤肉",
    "日料",
    "日本料理",
    "料理",
    "烧腊",
    "虾饺",
    "叉烧",
    "流沙包",
    "榴莲酥",
    "双人餐",
    "推荐菜",
)
FAMILY_HANDCRAFT_TERMS = (
    "亲子手作",
    "亲子手工",
    "手作",
    "手工",
    "陶艺",
    "DIY",
    "diy",
    "手工坊",
    "木作",
    "银饰",
    "蜡烛",
)


def _matches_park_scenic_identity(item: dict) -> bool:
    text, values = _semantic_text_index(item, fields=STRICT_MULTINODE_ROLE_FIELDS)
    if any(term in values or term in text for term in PARK_SCENIC_POSITIVE_TERMS):
        return True
    if "绿地" in values or "绿地" in text:
        return not any(term in text for term in PARK_SCENIC_FALSE_POSITIVE_TERMS)
    return False


def _matches_lodging_identity(item: dict) -> bool:
    item_type = str(item.get("type") or "").lower()
    supply_domain = str(item.get("supply_domain") or "").lower()
    name_text = normalize_semantic_text(str(item.get("name") or ""))
    identity_text = "\n".join(
        normalize_semantic_text(str(value))
        for field_name in STRICT_MULTINODE_ROLE_FIELDS
        for value in flatten_semantic_values(item.get(field_name))
    )

    if any(term in name_text for term in LODGING_NAME_TERMS):
        return True
    if any(term in name_text for term in LODGING_NAME_FALSE_POSITIVE_TERMS):
        return False
    if _fast_text_match_score(
        list(_strict_role_query_terms("lodging")),
        item,
        fields=STRICT_MULTINODE_ROLE_FIELDS,
    ) > 0:
        return True
    if item_type in {"hotel", "lodging"} or supply_domain in {"hotel", "lodging"}:
        return not any(term in identity_text for term in LODGING_NAME_FALSE_POSITIVE_TERMS)
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
    if role == "citywalk_market":
        text, values = _semantic_text_index(item, fields=STRICT_MULTINODE_ROLE_FIELDS)
        del values
        return _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0 and not any(
            term in text
            for term in (
                "spa",
                "足疗",
                "按摩",
                "推拿",
                "洗脚",
                "修脚",
                "健身",
                "瑜伽",
                "普拉提",
                "陶艺",
                "手作",
                "手工",
                "diy",
                "银饰",
                "玩具城",
                "玩具",
                "购物中心",
                "专卖店",
                "专营店",
                "商场",
                "门店",
                "社区",
                "老年活动室",
                "活动室",
                "党群",
                "服务中心",
                "居委",
                "街道办",
                "酒吧",
                "清吧",
                "bar",
                "club",
            )
        )
    if role == "board_game_escape":
        return _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0
    if role == "river_cruise":
        text, values = _semantic_text_index(item, fields=STRICT_MULTINODE_ROLE_FIELDS)
        del values
        return _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0 and not any(term in text for term in ("公园", "绿地", "步道", "咖啡", "餐厅", "酒吧", "商场", "停车场", "写字楼"))
    if role == "convenience_store":
        return supply_domain in {"shopping", "retail"} and _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0
    if role == "flower_shop":
        text, values = _semantic_text_index(item, fields=STRICT_MULTINODE_ROLE_FIELDS)
        del values
        return supply_domain in {"shopping", "retail"} and _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0 and not any(term in text for term in ("miniso", "名创", "优品", "购物中心", "商场", "超市", "便利店", "美妆", "日化", "餐厅"))
    if role == "beauty_cosmetics":
        text, values = _semantic_text_index(item, fields=STRICT_MULTINODE_ROLE_FIELDS)
        del values
        return supply_domain in {"shopping", "retail"} and _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0 and not any(term in text for term in ("餐厅", "咖啡", "蛋糕", "花店", "鲜花", "便利店", "超市"))
    if role == "souvenir_shopping":
        text, values = _semantic_text_index(item, fields=STRICT_MULTINODE_ROLE_FIELDS)
        del values
        return supply_domain in {"shopping", "retail"} and _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0 and not any(term in text for term in ("咖啡", "咖啡馆", "咖啡厅", "cafe", "bar", "酒吧", "清吧", "餐厅", "便利店", "超市"))
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
        return _matches_lodging_identity(item)
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
    if role == "karaoke":
        text, values = _semantic_text_index(item, fields=STRICT_MULTINODE_ROLE_FIELDS)
        del values
        return _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0 and not any(
            term in text
            for term in (
                "展览",
                "展馆",
                "艺术",
                "画廊",
                "美术馆",
                "博物馆",
                "印象派",
                "酒吧",
                "livehouse",
                "足疗",
                "沐足",
                "按摩",
                "推拿",
                "洗脚",
                "修脚",
                *KARAOKE_FALSE_POSITIVE_TERMS,
            )
        )
    if role == "park_scenic_walk":
        return _matches_park_scenic_identity(item)
    if role == "bar":
        text, values = _semantic_text_index(item, fields=STRICT_MULTINODE_ROLE_FIELDS)
        return _fast_text_match_score(
            list(_strict_role_query_terms(role)),
            item,
            fields=STRICT_MULTINODE_ROLE_FIELDS,
        ) > 0 and not any(term in text for term in ("火锅", "烤肉", "烧烤", "麻辣烫", "寿司", "牛排"))
    if role in {"talk_show", "theatre_performance"}:
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


def _weather_candidate_bonus(item: dict, weather_context: dict | None) -> float:
    return _weather_candidate_bonus_impl(
        item,
        weather_context,
        collect_tags=_collect_plan_tags,
    )


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
    has_child_context = _has_child_context(
        scene_type=scene_type,
        child_age=child_age,
        constraints=constraints,
        user_input=user_input,
    )
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
        if item_type == "activity" and not has_child_context and _is_family_only_supply(item):
            base -= 12.0

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
    has_child_context = _has_child_context(
        scene_type=scene_type,
        child_age=child_age,
        constraints=constraints,
        user_input=user_input,
    )
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
        if not has_child_context and _is_family_only_supply(activity):
            value -= 22.0

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


def _intent_requires_family_handcraft(intent: dict) -> bool:
    intent_text = " ".join(flatten_semantic_values(intent.get("search_terms")))
    intent_text_lower = intent_text.lower()
    return any(term.lower() in intent_text_lower for term in FAMILY_HANDCRAFT_TERMS)


def _item_matches_family_handcraft(item: dict) -> bool:
    return _fast_text_match_score(
        _normalized_query_terms(FAMILY_HANDCRAFT_TERMS, expand_semantics=False),
        item,
        fields=COMPACT_SEMANTIC_FIELDS,
    ) > 0


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
        if _intent_requires_family_handcraft(intent):
            score += 16.0 if _item_matches_family_handcraft(item) else -4.0
        if signals.intersection({"kid_friendly", "family_friendly", "low_intensity", "indoor"}):
            score += 10.0
    elif role == "family_indoor_play":
        if signals.intersection({"kid_friendly", "family_friendly", "indoor_playground", "amusement", "low_intensity", "indoor"}):
            score += 12.0
    elif role == "exhibition":
        if signals.intersection({"museum", "art_experience", "exhibition"}):
            score += 8.0
    elif role == "river_cruise":
        if _fast_text_match_score(_strict_role_query_terms(role), item, fields=COMPACT_SEMANTIC_FIELDS) > 0:
            score += 10.0
    elif role == "theatre_performance":
        if _fast_text_match_score(_strict_role_query_terms(role), item, fields=COMPACT_SEMANTIC_FIELDS) > 0:
            score += 9.0
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
        if _intent_requires_local_shanghai_food(intent):
            if _item_has_local_shanghai_food_identity(item):
                score += 18.0
            elif _item_conflicts_with_local_shanghai_food(item):
                score -= 48.0
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
        requires_handcraft = _intent_requires_family_handcraft(intent)
        preferred = [
            item for item in pool
            if _semantic_signal_set_for_item(item).intersection({"kid_friendly", "family_friendly", "low_intensity"})
            or (requires_handcraft and _item_matches_family_handcraft(item))
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
        if _intent_requires_local_shanghai_food(intent):
            identity_preferred = [
                item
                for item in pool
                if _item_has_local_shanghai_food_identity(item)
                and not _item_conflicts_with_local_shanghai_food(item)
            ]
            if identity_preferred:
                pool = identity_preferred
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
        raw_request_text = " ".join(
            str(value)
            for value in (user_input, constraints.get("raw_text"))
            if value
        )
        if _scenario_has_activity_context([], raw_request_text):
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
    itinerary_blueprint = _mark_rag_resolved_blueprint_roles(
        itinerary_blueprint,
        rag_coverage,
    )
    constraints = {
        **constraints,
        "b_itinerary_blueprint": itinerary_blueprint,
    }
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
    can_plan_partial_multinode = _can_plan_partial_multinode(itinerary_blueprint, rag_coverage)
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
    geo_prefilter_min_keep = _get_geo_prefilter_min_keep()
    pretrim_min_keep = _get_pretrim_min_keep()
    rag_activity_candidates = _rag_candidates_for_domain(rag_node_candidates, "activity")
    rag_restaurant_candidates = _rag_candidates_for_domain(rag_node_candidates, "restaurant")

    if (
        single_node_shape
        and rag_coverage.get("all_nodes_covered")
        and (rag_activity_candidates or rag_restaurant_candidates)
    ):
        raw_plan_candidates = _combine_single_node_plan_candidates(
            rag_activity_candidates,
            rag_restaurant_candidates,
            constraints,
            scene_type,
            itinerary_blueprint,
            single_node_shape,
            user_profile,
            weather_context,
            max_candidates=plan_candidate_limit,
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
                "planner_mode": "single_node",
                "fast_path": "rag_only_single_node",
                "replan_filter": replan_filter_meta,
                "counts": {
                    "activities_initial": 0,
                    "restaurants_initial": 0,
                    "selected_activity_pool": len(rag_activity_candidates),
                    "selected_restaurant_pool": len(rag_restaurant_candidates),
                    "raw_plan_candidates": len(raw_plan_candidates),
                    "plan_candidates": len(plan_candidates),
                },
                "single_node_generation_budget": {
                    "plan_shape": single_node_shape,
                    "candidate_limit": plan_candidate_limit,
                },
            }
            execution_log.append(
                "[B] candidate_generator_node used RAG-only single-node fast path "
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
            "[B] candidate_generator_node RAG-only single-node fast path found no valid candidate; "
            "falling back to full local supply"
        )

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
        min_keep=max(activity_pool_size * 12, geo_prefilter_min_keep),
    )
    restaurant_candidates, restaurant_geo_meta = _filter_candidates_by_geo_window(
        restaurant_candidates,
        constraints=constraints,
        user_profile=user_profile,
        min_keep=max(restaurant_pool_size * 12, geo_prefilter_min_keep),
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

    activity_pretrim_size = max(activity_pool_size * 8, pretrim_min_keep)
    restaurant_pretrim_size = max(restaurant_pool_size * 8, pretrim_min_keep)

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
