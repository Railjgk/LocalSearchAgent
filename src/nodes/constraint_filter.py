try:
    from src.state import PlanState
except ImportError:
    PlanState = dict

import re

from .b_utils import (
    to_float,
    get_constraint_config_with_profile,
    build_filter_summary,
    generate_relaxation_suggestions,
)
from .b_semantics import is_child_compatible_activity
from .b_semantics import (
    B_SEMANTIC_GROUPS,
    CHILD_STRONG_SIGNALS,
    activity_child_signal_set,
    flatten_semantic_values,
    has_item_semantic_group,
    item_semantic_terms,
    normalize_semantic_text,
)

CHILD_CONTEXT_TERMS = ("孩子", "小孩", "小朋友", "儿童", "亲子", "宝宝", "带娃", "家庭")
LOW_CALORIE_CONTEXT_TERMS = (
    "减肥",
    "减脂",
    "低卡",
    "低脂",
    "轻食",
    "少油",
    "少糖",
    "清淡",
    "控糖",
    "健身餐",
    "健康饮食",
    "健康餐",
    "不油腻",
)
WIFE_CONTEXT_TERMS = ("老婆", "妻子", "太太", "爱人")
RELAXED_NON_SPORTS_CONTEXT_TERMS = (
    "轻松活动",
    "轻松一点",
    "放松活动",
    "放松一下",
    "休息一下",
    "坐坐",
    "聊会",
    "聊天",
    "不累",
    "别太累",
    "不要太累",
    "低强度",
)
EXPLICIT_SPORTS_CONTEXT_TERMS = (
    "运动",
    "健身",
    "瑜伽",
    "普拉提",
    "打球",
    "羽毛球",
    "篮球",
    "飞盘",
    "跑步",
    "攀岩",
    "训练",
)
SPORTS_ACTIVITY_TERMS = (
    "健身",
    "瑜伽",
    "普拉提",
    "运动",
    "健身中心",
    "fitness",
    "yoga",
    "pilates",
    "拳击",
    "训练",
    "体能",
    "运动体验",
)
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


def _get_plan_nodes(plan: dict) -> tuple[dict, dict]:
    nodes = plan.get("nodes", []) or []
    activity = next((node for node in nodes if node.get("type") == "activity"), {})
    restaurant = next((node for node in nodes if node.get("type") == "restaurant"), {})
    return activity, restaurant


def _get_plan_nodes_by_type(plan: dict, node_type: str) -> list[dict]:
    return [
        node
        for node in (plan.get("nodes", []) or [])
        if isinstance(node, dict) and node.get("type") == node_type
    ]


def _reject(filter_reasons: dict, plan_id: str, reason: str) -> None:
    filter_reasons[plan_id] = reason


def _item_text(item: dict) -> str:
    values: list[str] = []
    for field_name in (
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
        "service_facilities",
        "decision_profile",
        "business_hours",
        "parking_fee_policy",
        "review_keywords",
        "signature_dishes",
        "recommended_dishes",
    ):
        values.extend(flatten_semantic_values(item.get(field_name)))
    return " ".join(str(value) for value in values)


def _itinerary_annotation_terms(item: dict) -> set[str]:
    values: list[str] = [
        item.get("itinerary_label"),
        item.get("itinerary_role"),
        item.get("role"),
    ]
    intent = item.get("_itinerary_intent") or {}
    if isinstance(intent, dict):
        values.extend([intent.get("label"), intent.get("role")])
    return {
        normalize_semantic_text(value)
        for value in flatten_semantic_values(values)
        if normalize_semantic_text(value)
    }


def _item_identity_text(item: dict, *, exclude_itinerary_annotations: bool = False) -> str:
    values: list[str] = []
    annotation_terms = _itinerary_annotation_terms(item) if exclude_itinerary_annotations else set()
    for field_name in (
        "name",
        "category",
        "sub_category",
        "experience_type",
        "restaurant_category",
        "primary_category",
        "primary_keyword",
        "gaode_keyword",
        "gaode_type",
        "tags",
        "tag_groups",
        "service_facilities",
        "business_hours",
    ):
        for value in flatten_semantic_values(item.get(field_name)):
            if annotation_terms and normalize_semantic_text(value) in annotation_terms:
                continue
            values.append(value)
    return " ".join(str(value) for value in values)


def _text_contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _has_child_context(text: str, constraints: dict) -> bool:
    scene = str(constraints.get("scene") or constraints.get("scene_type") or "")
    return (
        scene == "family"
        or "family" in scene.lower()
        or _text_contains_any(scene, ("家庭", "亲子"))
        or _text_contains_any(text, CHILD_CONTEXT_TERMS)
    )


def _flatten_hard_constraint_context(constraints: dict) -> str:
    values: list[str] = []
    for key in ("hard_tags", "companions"):
        values.extend(flatten_semantic_values(constraints.get(key)))
    return " ".join(str(value) for value in values)


def _wife_profile_has_diet_need(user_profile: dict) -> bool:
    companion_profile = user_profile.get("companion_profile") or {}
    wife_profile = companion_profile.get("wife") or companion_profile.get("spouse") or {}
    if not isinstance(wife_profile, dict):
        return False
    values = flatten_semantic_values(wife_profile.get("state"))
    values.extend(flatten_semantic_values(wife_profile.get("needs")))
    text = " ".join(str(value) for value in values)
    return _text_contains_any(text, ("dieting", "low_calorie", "light_food", "减脂", "减肥", "低卡", "轻食"))


def _has_low_calorie_context(text: str, constraints: dict, user_profile: dict) -> bool:
    if _text_contains_any(text, LOW_CALORIE_CONTEXT_TERMS):
        return True
    hard_context_text = _flatten_hard_constraint_context(constraints)
    if _text_contains_any(hard_context_text, LOW_CALORIE_CONTEXT_TERMS):
        return True
    if _wife_profile_has_diet_need(user_profile) and _text_contains_any(
        " ".join([text, hard_context_text]),
        WIFE_CONTEXT_TERMS,
    ):
        return True
    return False


def _has_explicit_budget_signal(text: str) -> bool:
    if not text:
        return False
    budget_word_pattern = r"(?:人均|每人|单人|预算|总共|控制在|不超过|以内|以下|封顶|左右)"
    money_unit_pattern = r"(?:元|块|rmb|RMB|预算|以内|以下|封顶|左右)"
    return bool(
        re.search(r"[¥￥]\s*\d+", text)
        or re.search(rf"{budget_word_pattern}.{{0,8}}\d+", text)
        or re.search(rf"\d+\s*{money_unit_pattern}", text)
    )


def _is_scoped_small_item_budget(text: str, budget: float) -> bool:
    if not text or budget <= 0 or budget > 50:
        return False
    if any(term in text for term in ("总预算", "总共", "一共", "整体预算", "全程预算")):
        return False
    item_terms = (
        "包子",
        "早餐",
        "早饭",
        "早点",
        "馄饨",
        "豆浆",
        "咖啡",
        "饮料",
        "零食",
        "小吃",
        "吃饱",
        "单品",
    )
    budget_terms = ("以内", "以下", "不超过", "便宜", "实惠")
    return any(term in text for term in item_terms) and any(term in text for term in budget_terms)


def _restaurant_is_cafe_dessert(restaurant: dict) -> bool:
    strict_identity_text = _item_identity_text(restaurant, exclude_itinerary_annotations=True)
    if _text_contains_any(
        strict_identity_text,
        ("咖啡", "甜品", "下午茶", "蛋糕", "面包", "饮品", "茶饮"),
    ):
        return True
    if _itinerary_annotation_terms(restaurant):
        return False
    return has_item_semantic_group(restaurant, "咖啡甜品")


def _restaurant_is_light_meal(restaurant: dict) -> bool:
    strict_identity_text = _item_identity_text(restaurant, exclude_itinerary_annotations=True)
    return has_item_semantic_group(restaurant, "轻食") or _text_contains_any(
        strict_identity_text,
        ("轻食", "沙拉", "简餐", "健康餐", "低卡", "清淡"),
    )


def _restaurant_has_primary_group_evidence(restaurant: dict, group: str) -> bool:
    text = normalize_semantic_text(_item_text(restaurant))
    if not text:
        return False
    for term in B_SEMANTIC_GROUPS.get(group, {}).get("primary", []):
        normalized = normalize_semantic_text(term)
        if normalized and normalized in text:
            return True
    return False


def _restaurant_has_group(restaurant: dict, group: str) -> bool:
    if group == "正餐":
        return not (_restaurant_is_cafe_dessert(restaurant) or _restaurant_is_light_meal(restaurant))
    if group in {"烤肉", "火锅"}:
        return _restaurant_has_primary_group_evidence(restaurant, group)
    return has_item_semantic_group(restaurant, group)


def _restaurant_conflicts_low_calorie(restaurant: dict) -> bool:
    """Return whether a restaurant is a poor fit for an explicit low-calorie need."""

    if any(_restaurant_has_group(restaurant, group) for group in ("烤肉", "火锅", "炸鸡小吃")):
        return True
    text = normalize_semantic_text(_item_text(restaurant))
    return _text_contains_any(
        text,
        (
            "牛排",
            "巴西牛排",
            "巴西烤肉",
            "自助烤肉",
            "烧肉",
            "烤肉",
            "炸鸡",
            "油炸",
            "重油",
            "肥牛",
            "五花肉",
        ),
    )


def _has_relaxed_non_sports_context(text: str, constraints: dict) -> bool:
    context = " ".join(
        str(value)
        for value in (
            text,
            constraints.get("planning_preferences"),
            constraints.get("soft_tags"),
            constraints.get("hard_tags"),
        )
        if value not in (None, "")
    )
    if _text_contains_any(context, EXPLICIT_SPORTS_CONTEXT_TERMS):
        return False
    return _text_contains_any(context, RELAXED_NON_SPORTS_CONTEXT_TERMS)


def _activity_is_sports_like(activity: dict) -> bool:
    text = normalize_semantic_text(_item_text(activity))
    return _text_contains_any(text, SPORTS_ACTIVITY_TERMS)


def _node_role(item: dict) -> str:
    return str(item.get("itinerary_role") or item.get("role") or "")


def _node_matches_itinerary_role(item: dict, *, allow_cafe_light_meal_fallback: bool = False) -> bool:
    """Guard multi-node plans from filling a role with the wrong POI category."""

    role = _node_role(item)
    if not role:
        return True
    text = normalize_semantic_text(_item_text(item))
    identity_text = normalize_semantic_text(
        _item_identity_text(item, exclude_itinerary_annotations=True)
    )
    item_type = str(item.get("type") or "").lower()
    if not text and not identity_text:
        return True

    if role == "cafe":
        if _text_contains_any(identity_text, ("咖啡", "咖啡馆", "咖啡厅", "下午茶", "甜品", "蛋糕", "烘焙")):
            return True
        return allow_cafe_light_meal_fallback and _text_contains_any(
            identity_text,
            ("轻食", "沙拉", "简餐", "饮品", "茶饮"),
        )
    if role == "tea_house":
        return _text_contains_any(identity_text, ("茶", "茶馆", "茶艺", "茶室", "品茶"))
    if role == "convenience_store":
        return _text_contains_any(
            identity_text,
            ("便利店", "超市", "全家", "罗森", "7-eleven", "711", "便利", "零食"),
        )
    if role == "parking":
        return item_type == "transport_service" or _text_contains_any(identity_text, ("停车", "车库", "车位"))
    if role == "wellness_massage":
        return _text_contains_any(identity_text, ("spa", "按摩", "足疗", "推拿", "养生", "洗脚", "修脚"))
    if role == "fitness":
        return _activity_is_sports_like(item)
    if role == "lodging":
        return item_type in {"hotel", "lodging"} or _text_contains_any(identity_text, ("酒店", "民宿", "住宿", "宾馆"))
    if role == "exhibition":
        return _text_contains_any(
            identity_text,
            ("美术馆", "博物馆", "展览", "展馆", "艺术馆", "画廊", "文化馆", "文化", "艺术", "历史", "cultural"),
        )
    if role == "restaurant_breakfast":
        return _text_contains_any(identity_text, ("早餐", "早饭", "早点", "包子", "馄饨", "豆浆", "粥", "生煎"))
    if role in {"restaurant_lunch", "restaurant_dinner", "restaurant_specific"}:
        return item_type == "restaurant"
    if role == "cultural_photo":
        return _text_contains_any(identity_text, ("汉服", "写真", "摄影", "古风", "拍照", "换装", "豫园", "文化街区", "珠宝"))
    if role == "park_scenic_walk":
        if _text_contains_any(identity_text, PARK_SCENIC_POSITIVE_TERMS):
            return True
        if "绿地" in identity_text:
            return not _text_contains_any(identity_text, PARK_SCENIC_FALSE_POSITIVE_TERMS)
        return False
    if role in {"family_activity", "family_indoor_play"}:
        return item_type == "activity"
    if role == "board_game_escape":
        return _text_contains_any(identity_text, ("剧本杀", "密室", "桌游", "推理", "狼人杀", "血染钟楼"))
    if role == "internet_cafe":
        return _text_contains_any(identity_text, ("网吧", "网咖", "电竞", "电玩", "游戏机", "ps5", "ns"))
    if role == "bar":
        return item_type in {"activity", "restaurant"} and _text_contains_any(
            identity_text,
            ("酒吧", "清吧", "精酿", "鸡尾酒", "live", "音乐"),
        )
    if role == "talk_show":
        return _text_contains_any(identity_text, ("脱口秀", "喜剧", "剧场", "演出", "livehouse", "live house"))
    if role in {"karaoke", "citywalk_market"}:
        return item_type == "activity"
    return True


def _has_halal_evidence(restaurant: dict) -> bool:
    return _text_contains_any(_item_text(restaurant).lower(), ("清真", "halal", "穆斯林"))


def _has_pet_friendly_evidence(item: dict) -> bool:
    return _text_contains_any(_item_text(item), ("宠物友好", "可带宠物", "带狗", "狗狗友好", "宠物"))


def _is_pet_specific_node(item: dict) -> bool:
    role = _node_role(item)
    item_type = str(item.get("type") or "").lower()
    supply_domain = str(item.get("supply_domain") or item.get("domain") or "").lower()
    text = _item_text(item)
    return (
        role in {"pet_grooming", "pet_cafe", "pet_hospital", "pet_store"}
        or item_type == "pet_service"
        or supply_domain == "pet_service"
        or _text_contains_any(
            text,
            ("宠物美容", "宠物洗护", "宠物医院", "动物医院", "宠物店", "宠物用品", "宠物友好咖啡"),
        )
    )


def _has_explicit_total_duration_signal(raw_text: str) -> bool:
    """Detect user-level itinerary duration without mistaking per-POI time hints for it."""

    if not raw_text:
        return False
    if any(term in raw_text for term in ("几个小时", "半天", "一整天", "全天", "两天", "2天", "一天", "过夜", "住一晚")):
        return True
    if re.search(
        r"(?:玩|逛|安排|行程|计划|活动|出去|约会|周末|上午|下午|晚上|白天).{0,8}\d+\s*(?:个)?小时",
        raw_text,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\d+\s*(?:个)?小时.{0,8}(?:左右|以内|内|上下|行程|计划|安排|路线|总共|整体)",
        raw_text,
        flags=re.IGNORECASE,
    ):
        return True
    return False


def _business_open_after(item: dict, start_time: str, *, min_open_minutes: int = 90) -> bool:
    if not start_time or ":" not in start_time:
        return True
    try:
        start_hour, start_minute = [int(part) for part in start_time.split(":", 1)]
    except ValueError:
        return True
    start_minutes = start_hour * 60 + start_minute
    if start_minutes < 21 * 60:
        return True

    hours = item.get("business_hours") or {}
    values = []
    if isinstance(hours, dict):
        values.extend(str(value) for value in hours.values())
    else:
        values.append(str(hours))

    for value in values:
        if "-" not in value:
            continue
        end_text = value.split("-")[-1].strip()
        if ":" not in end_text:
            continue
        try:
            end_hour, end_minute = [int(part) for part in end_text.split(":", 1)]
        except ValueError:
            continue
        end_minutes = end_hour * 60 + end_minute
        if end_minutes <= 6 * 60:
            end_minutes += 24 * 60
        if end_minutes - start_minutes >= min_open_minutes:
            return True
    return False


def _has_concrete_route_origin(constraints: dict, route: dict) -> bool:
    """Whether route distance can be treated as a hard user-origin constraint."""

    if any(
        constraints.get(key)
        for key in ("origin_coordinates", "route_origin_coordinates", "user_coordinates")
    ):
        return True
    if route.get("origin_coordinates"):
        return True
    # Textual origins such as "外滩附近" or "复旦附近" are useful retrieval
    # hints, but without geocoded coordinates candidate_generator uses
    # synthetic fallback legs. Keep these distance checks soft until A/C gives
    # B concrete coordinates.
    return False


def _contract_reject_reason(
    contract: dict,
    *,
    activity: dict,
    restaurant: dict,
    activities: list[dict] | None = None,
    restaurants: list[dict] | None = None,
    plan_nodes: list[dict] | None = None,
    constraints: dict,
) -> str | None:
    hard_requirements = set(contract.get("hard_requirements") or [])
    forbidden_groups = set(contract.get("forbidden_restaurant_groups") or [])
    activities = activities or ([activity] if activity else [])
    restaurants = restaurants or ([restaurant] if restaurant else [])
    plan_nodes = plan_nodes or activities + restaurants

    for group in forbidden_groups:
        if any(_restaurant_has_group(item, str(group)) for item in restaurants):
            return f"餐厅命中用户明确规避的{group}需求"

    if "child_friendly_activity" in hard_requirements and activities:
        has_child_activity = any(
            activity_child_signal_set(item).intersection(CHILD_STRONG_SIGNALS)
            for item in activities
        )
        if not has_child_activity:
            return "缺少明确儿童友好/亲子活动证据"

    if "cafe_non_full_meal" in hard_requirements and not any(
        _restaurant_is_cafe_dessert(item) or _restaurant_is_light_meal(item)
        for item in restaurants
    ):
        return "用户想咖啡小坐且不吃正餐，当前餐厅不匹配"

    if "halal_restaurant" in hard_requirements and not any(_has_halal_evidence(item) for item in restaurants):
        return "缺少清真餐厅证据"

    if "pet_friendly" in hard_requirements:
        pet_check_nodes = activities + restaurants
        pet_specific_nodes = [item for item in plan_nodes if _is_pet_specific_node(item)]
        generic_nodes_need_evidence = [
            item for item in pet_check_nodes if not _is_pet_specific_node(item)
        ]
        if not pet_check_nodes and not pet_specific_nodes:
            return "缺少活动和餐厅均宠物友好的证据"
        if any(not _has_pet_friendly_evidence(item) for item in generic_nodes_need_evidence):
            return "缺少活动和餐厅均宠物友好的证据"

    if "parking_needed" in hard_requirements:
        if not any(item.get("parking_available") for item in activities + restaurants):
            return "缺少可停车证据"

    if "late_night_open" in hard_requirements:
        start_time = str(constraints.get("start_time") or "")
        if not all(_business_open_after(item, start_time, min_open_minutes=60) for item in activities):
            return "营业时间不满足深夜/夜宵需求"
        if not all(_business_open_after(item, start_time, min_open_minutes=90) for item in restaurants):
            return "营业时间不满足深夜/夜宵需求"

    return None


def constraint_filter_node(state: PlanState) -> dict:
    """
    Plan-level hard constraint filter.

    输入：
    - candidates: plan_candidates
    - constraints: 用户约束
    - user_profile: 用户画像

    输出：
    - filtered_candidates
    - filter_reasons
    - execution_log

    注意：
    - crowded_mall 在 B 中作为 soft risk，不在这里硬过滤；
    - 本节点只过滤明显不可执行或严重违背需求的方案。
    """
    execution_log = state.get("execution_log", [])
    candidates = state.get("candidates", []) or []
    constraints = state.get("constraints", {}) or {}
    user_profile = state.get("user_profile", {}) or {}
    candidate_generation_issues = state.get("candidate_generation_issues", []) or []
    contract = state.get("b_requirement_contract") or constraints.get("b_requirement_contract") or {}

    config = get_constraint_config_with_profile(constraints, user_profile)
    max_distance_km = config["max_distance_km"]
    max_queue_time = config["max_queue_time"]
    duration_range = config["duration_range"]
    budget = config["budget"]
    child_age = config["child_age"]
    mom_diet = config["mom_diet"]

    filtered_candidates = []
    filter_reasons = {}

    for plan in candidates:
        plan_id = plan.get("plan_id", "unknown")

        route = plan.get("route", {}) or {}
        budget_info = plan.get("budget", {}) or {}
        availability = plan.get("availability", {}) or {}
        activity, restaurant = _get_plan_nodes(plan)
        plan_nodes = plan.get("nodes", []) or []
        activities = _get_plan_nodes_by_type(plan, "activity")
        restaurants = _get_plan_nodes_by_type(plan, "restaurant")
        is_partial_itinerary = plan.get("execution_scope") == "partial"
        is_multiday_itinerary = (
            int(to_float(plan.get("planning_days"), 1.0)) > 1
            or str(plan.get("planning_horizon") or "").lower() in {"overnight", "two_day"}
        )
        partial_missing_roles = set(plan.get("partial_missing_roles") or [])
        has_lodging_node = any(
            node.get("itinerary_role") == "lodging"
            or node.get("role") == "lodging"
            or node.get("type") in {"hotel", "lodging"}
            for node in plan_nodes
        )
        raw_constraint_text = " ".join(
            str(value)
            for value in (
                constraints.get("raw_text"),
                constraints.get("user_input"),
            )
            if value not in (None, "")
        )
        has_numeric_budget_signal = _has_explicit_budget_signal(raw_constraint_text)
        has_explicit_budget = constraints.get("budget") not in (None, "") and has_numeric_budget_signal
        is_per_person_budget = (
            str(constraints.get("budget_type") or "").lower() in {"per_person", "per-person", "pp"}
            or "人均" in raw_constraint_text
        )
        has_explicit_duration_signal = _has_explicit_total_duration_signal(raw_constraint_text)
        duration_upper_limit = duration_range[1]

        total_distance = to_float(route.get("total_distance_km"), 0.0)
        route_legs = route.get("legs") if isinstance(route.get("legs"), list) else []
        leg_distances = [
            to_float(leg.get("distance_km"), 0.0)
            for leg in route_legs
            if isinstance(leg, dict)
        ]
        max_single_leg_distance = max(leg_distances or [total_distance])
        max_queue = to_float(availability.get("max_queue_time_min"), 0.0)
        estimated_duration = to_float(plan.get("estimated_duration_min"), 0.0)
        total_price = to_float(budget_info.get("total_price"), 0.0)
        is_multi_node_itinerary = len(plan_nodes) > 2 or plan.get("planner_mode") == "multi_node_itinerary"

        # 0. B requirement contract guardrails.
        contract_for_plan = contract
        if is_partial_itinerary:
            hard_requirements = set(contract.get("hard_requirements") or [])
            if "parking" in partial_missing_roles:
                hard_requirements.discard("parking_needed")
            if any(
                node.get("itinerary_role") == "parking"
                or node.get("role") == "parking"
                or node.get("type") == "transport_service"
                for node in plan_nodes
            ):
                hard_requirements.discard("parking_needed")
            contract_for_plan = {
                **contract,
                "hard_requirements": list(hard_requirements),
            }
        contract_reason = _contract_reject_reason(
            contract_for_plan,
            activity=activity,
            restaurant=restaurant,
            activities=activities,
            restaurants=restaurants,
            plan_nodes=plan_nodes,
            constraints=constraints,
        )
        if contract_reason:
            _reject(filter_reasons, plan_id, contract_reason)
            continue

        if is_multi_node_itinerary:
            allow_cafe_light_meal_fallback = (
                plan.get("planner_mode") == "single_node"
                and plan.get("plan_shape") == "cafe_only"
            )
            mismatched_role = next(
                (
                    _node_role(item)
                    for item in plan_nodes
                    if not _node_matches_itinerary_role(
                        item,
                        allow_cafe_light_meal_fallback=allow_cafe_light_meal_fallback,
                    )
                ),
                "",
            )
            if mismatched_role:
                _reject(filter_reasons, plan_id, f"节点角色与POI类型不匹配：{mismatched_role}")
                continue

        # 1. 库存 / 可用性
        if not availability.get("all_available", False):
            _reject(filter_reasons, plan_id, "活动或餐厅当前不可用")
            continue

        # 2. 距离
        if (
            is_multi_node_itinerary
            and (
                any(term in raw_constraint_text for term in ("一整天", "全天", "一天"))
                or constraints.get("time_window") == "full_day"
            )
        ):
            duration_upper_limit = max(duration_upper_limit, 720)
            if any(term in raw_constraint_text for term in ("晚上", "夜宵", "酒吧", "live", "Live", "演出")):
                duration_upper_limit = max(duration_upper_limit, 960)
        if is_multi_node_itinerary:
            route_has_concrete_origin = _has_concrete_route_origin(constraints, route)
            leg_distances_for_filter = leg_distances
            aggregate_distance = total_distance
            if not route_has_concrete_origin and len(leg_distances) > 1:
                # The first leg is a synthetic start-to-first-POI estimate when no
                # real user origin is available; do not let it kill clustered POI plans.
                leg_distances_for_filter = leg_distances[1:]
                aggregate_distance = sum(leg_distances_for_filter)
            max_single_leg_for_filter = max(leg_distances_for_filter or [aggregate_distance])
            effective_max_leg_distance = max_distance_km
            has_explicit_distance_number = bool(
                re.search(r"\d+(?:\.\d+)?\s*(?:公里|千米|km|KM|米|m|M)", raw_constraint_text)
            )
            if (
                str(constraints.get("route_mode") or constraints.get("transport_mode") or "").lower()
                in {"walking", "walk", "步行"}
                and not has_explicit_distance_number
            ):
                effective_max_leg_distance = max(effective_max_leg_distance, 2.5)
            if not route_has_concrete_origin and not has_explicit_distance_number:
                # A text-only anchor such as "外滩附近" or "复旦附近" is useful for
                # retrieval, but synthetic route legs can still be noisy on sparse POI
                # coverage. Keep explicit numeric distance requests strict; otherwise
                # allow a small buffer so B can return a truthful partial itinerary
                # instead of an empty skeleton.
                route_mode = str(
                    constraints.get("route_mode") or constraints.get("transport_mode") or ""
                ).lower()
                text_anchor_floor = 4.0 if route_mode in {"walking", "walk", "步行"} else 8.0
                effective_max_leg_distance = max(
                    effective_max_leg_distance,
                    max_distance_km * 2,
                    text_anchor_floor,
                )
            aggregate_distance_limit = effective_max_leg_distance * max(1, min(4, len(plan_nodes) - 1))
            if max_single_leg_for_filter > effective_max_leg_distance:
                _reject(filter_reasons, plan_id, "单段距离超过用户可接受范围")
                continue
            if aggregate_distance > aggregate_distance_limit:
                _reject(filter_reasons, plan_id, "总路线距离超过多节点行程可接受范围")
                continue
        else:
            distance_limit = max_distance_km
            if not _has_concrete_route_origin(constraints, route):
                distance_limit = max(distance_limit, max_distance_km * 2)
        if not is_multi_node_itinerary and total_distance > distance_limit:
            _reject(filter_reasons, plan_id, "距离超过用户可接受范围")
            continue

        # 3. 排队
        if max_queue > max_queue_time:
            _reject(filter_reasons, plan_id, "排队时间过长")
            continue

        # 4. 时长
        if not is_multiday_itinerary:
            if is_multi_node_itinerary:
                if has_explicit_duration_signal and estimated_duration > duration_upper_limit:
                    _reject(filter_reasons, plan_id, "时长不满足用户的时间范围")
                    continue
            elif (
                (
                    has_explicit_duration_signal
                    and not is_partial_itinerary
                    and estimated_duration < duration_range[0]
                )
                or (has_explicit_duration_signal and estimated_duration > duration_range[1])
            ):
                _reject(filter_reasons, plan_id, "时长不满足用户的时间范围")
                continue

        # 5. 预算：允许 1.2 倍软浮动，但超过则硬过滤
        effective_budget = budget
        if has_explicit_budget and is_per_person_budget:
            effective_budget = budget * max(1, int(to_float(config.get("people_count"), 1)))
        budget_limit = effective_budget * 1.2
        if not has_explicit_budget and has_lodging_node:
            budget_limit = max(budget_limit, 1200.0)
        elif not has_explicit_budget and is_multi_node_itinerary:
            budget_limit = max(budget_limit, 300.0 * max(2, len(plan_nodes)))
        scoped_small_item_budget = (
            is_multi_node_itinerary
            and has_explicit_budget
            and _is_scoped_small_item_budget(raw_constraint_text, budget)
        )
        if scoped_small_item_budget:
            priced_food_nodes = [
                item
                for item in restaurants
                if to_float(item.get("price"), 0.0) > 0
            ]
            if priced_food_nodes and min(to_float(item.get("price"), 0.0) for item in priced_food_nodes) > budget_limit:
                _reject(filter_reasons, plan_id, "单项餐饮预算超出可接受上限")
                continue
        elif total_price > budget_limit:
            _reject(filter_reasons, plan_id, "预算超出可接受上限")
            continue

        # 6. 低龄儿童约束
        if child_age is not None and child_age <= 6 and _has_child_context(raw_constraint_text, constraints):
            if not any(is_child_compatible_activity(item, child_age) for item in activities):
                _reject(filter_reasons, plan_id, "不满足低龄儿童友好要求")
                continue

        # 7. “轻松/放松/不累” is a non-sports social/leisure intent unless
        # the user explicitly asks for sports, fitness, yoga, etc.
        if _has_relaxed_non_sports_context(raw_constraint_text, constraints):
            if any(_activity_is_sports_like(item) for item in activities):
                _reject(filter_reasons, plan_id, "轻松放松需求不匹配运动健身类活动")
                continue

        # 8. 减脂 / 低卡饮食约束
        if mom_diet == "low_calorie" and _has_low_calorie_context(raw_constraint_text, constraints, user_profile):
            if restaurants:
                restaurant_health_ok = True
                for item in restaurants:
                    if _restaurant_conflicts_low_calorie(item):
                        restaurant_health_ok = False
                        break
                    tags = item.get("tags", []) or []
                    health_tags = item.get("health_tags", []) or []
                    menu_health_options = item.get("menu_health_options", []) or []
                    health_signals = set(tags) | set(health_tags) | set(menu_health_options)
                    if not health_signals.intersection(
                        {
                            "low_calorie",
                            "light_food",
                            "low_oil",
                            "low_sugar",
                            "high_protein",
                            "vegetable_rich",
                            "低卡",
                            "轻食",
                            "少油",
                            "低糖",
                            "高蛋白",
                            "蔬菜丰富",
                            "健康",
                            "健康餐",
                            "清淡",
                        }
                    ):
                        restaurant_health_ok = False
                        break
            else:
                restaurant_health_ok = False
            if not restaurant_health_ok:
                _reject(filter_reasons, plan_id, "不符合低卡或轻食需求")
                continue

        filtered_candidates.append(plan)

    summary_detail = build_filter_summary(
        total_candidates=len(candidates),
        valid_candidates=len(filtered_candidates),
        filter_reasons=filter_reasons,
    )

    relaxation_suggestions = generate_relaxation_suggestions(filter_reasons, constraints)

    if candidate_generation_issues and not candidates:
        issue_text = "；".join(
            str(issue.get("message") or issue.get("type"))
            for issue in candidate_generation_issues
            if isinstance(issue, dict)
        )
        filter_reasons["_summary"] = (
            "候选生成阶段未找到满足显式活动或餐饮类型的供给。"
            f"{issue_text if issue_text else ''}"
        )
        filter_reasons["_relaxation_suggestions"] = [
            "补充对应活动或餐饮类型的 mock POI、商品和库存",
            "或改用当前 mock 数据中已有的活动类型",
        ]
        filter_reasons["_candidate_generation_issues"] = candidate_generation_issues
    elif summary_detail["invalid_candidates"] > 0:
        reason_text = "；".join(
            f"{reason}（{count} 个）"
            for reason, count in summary_detail["reason_counts"].items()
        )
        filter_reasons["_summary"] = (
            f"共生成 {summary_detail['total_candidates']} 个候选方案，"
            f"其中 {summary_detail['valid_candidates']} 个满足硬约束，"
            f"{summary_detail['invalid_candidates']} 个被过滤。"
            f"{reason_text if reason_text else ''}"
        )
    else:
        filter_reasons["_summary"] = (
            f"共生成 {summary_detail['total_candidates']} 个候选方案，全部满足硬约束。"
        )

    filter_reasons["_summary_detail"] = summary_detail
    filter_reasons.setdefault("_relaxation_suggestions", relaxation_suggestions)
    if candidate_generation_issues:
        filter_reasons.setdefault("_candidate_generation_issues", candidate_generation_issues)

    if not filtered_candidates:
        execution_log.append(
            "[B] constraint_filter_node 未找到满足硬约束的方案；"
            f"过滤原因统计={summary_detail['reason_counts']}"
        )
    else:
        execution_log.append(
            f"[B] constraint_filter_node 过滤后剩余 {len(filtered_candidates)} 个方案；"
            f"过滤 {summary_detail['invalid_candidates']} 个"
        )

    return {
        "filtered_candidates": filtered_candidates,
        "filter_reasons": filter_reasons,
        "execution_log": execution_log,
    }
