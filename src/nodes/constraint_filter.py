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
)
WIFE_CONTEXT_TERMS = ("老婆", "妻子", "太太", "爱人")


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


def _restaurant_is_cafe_dessert(restaurant: dict) -> bool:
    text = _item_text(restaurant)
    return has_item_semantic_group(restaurant, "咖啡甜品") or _text_contains_any(
        text,
        ("咖啡", "甜品", "下午茶", "蛋糕", "面包", "饮品", "茶饮"),
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
        return not _restaurant_is_cafe_dessert(restaurant)
    if group in {"烤肉", "火锅"}:
        return _restaurant_has_primary_group_evidence(restaurant, group)
    return has_item_semantic_group(restaurant, group)


def _has_halal_evidence(restaurant: dict) -> bool:
    return _text_contains_any(_item_text(restaurant).lower(), ("清真", "halal", "穆斯林"))


def _has_pet_friendly_evidence(item: dict) -> bool:
    return _text_contains_any(_item_text(item), ("宠物友好", "可带宠物", "带狗", "狗狗友好", "宠物"))


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


def _contract_reject_reason(
    contract: dict,
    *,
    activity: dict,
    restaurant: dict,
    activities: list[dict] | None = None,
    restaurants: list[dict] | None = None,
    constraints: dict,
) -> str | None:
    hard_requirements = set(contract.get("hard_requirements") or [])
    forbidden_groups = set(contract.get("forbidden_restaurant_groups") or [])
    activities = activities or ([activity] if activity else [])
    restaurants = restaurants or ([restaurant] if restaurant else [])

    for group in forbidden_groups:
        if any(_restaurant_has_group(item, str(group)) for item in restaurants):
            return f"餐厅命中用户明确规避的{group}需求"

    if "child_friendly_activity" in hard_requirements:
        has_child_activity = any(
            activity_child_signal_set(item).intersection(CHILD_STRONG_SIGNALS)
            for item in activities
        )
        if not has_child_activity:
            return "缺少明确儿童友好/亲子活动证据"

    if "cafe_non_full_meal" in hard_requirements and not any(_restaurant_is_cafe_dessert(item) for item in restaurants):
        return "用户想咖啡小坐且不吃正餐，当前餐厅不匹配"

    if "halal_restaurant" in hard_requirements and not any(_has_halal_evidence(item) for item in restaurants):
        return "缺少清真餐厅证据"

    if "pet_friendly" in hard_requirements:
        if not (
            activities
            and restaurants
            and all(_has_pet_friendly_evidence(item) for item in activities + restaurants)
        ):
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
        has_explicit_duration_signal = bool(
            re.search(r"\d+\s*(?:个)?小时|\d+\s*h", raw_constraint_text, flags=re.IGNORECASE)
            or any(term in raw_constraint_text for term in ("几个小时", "半天", "一整天", "全天", "两天", "2天"))
        )
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
            constraints=constraints,
        )
        if contract_reason:
            _reject(filter_reasons, plan_id, contract_reason)
            continue

        # 1. 库存 / 可用性
        if not availability.get("all_available", False):
            _reject(filter_reasons, plan_id, "活动或餐厅当前不可用")
            continue

        # 2. 距离
        is_multi_node_itinerary = len(plan_nodes) > 2 or plan.get("planner_mode") == "multi_node_itinerary"
        if is_multi_node_itinerary and any(term in raw_constraint_text for term in ("一整天", "全天")):
            duration_upper_limit = max(duration_upper_limit, 720)
        if is_multi_node_itinerary:
            aggregate_distance_limit = max_distance_km * max(1, min(4, len(plan_nodes) - 1))
            if max_single_leg_distance > max_distance_km:
                _reject(filter_reasons, plan_id, "单段距离超过用户可接受范围")
                continue
            if total_distance > aggregate_distance_limit:
                _reject(filter_reasons, plan_id, "总路线距离超过多节点行程可接受范围")
                continue
        elif total_distance > max_distance_km:
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
                (not is_partial_itinerary and estimated_duration < duration_range[0])
                or estimated_duration > duration_range[1]
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
        if total_price > budget_limit:
            _reject(filter_reasons, plan_id, "预算超出可接受上限")
            continue

        # 6. 低龄儿童约束
        if child_age is not None and child_age <= 6 and _has_child_context(raw_constraint_text, constraints):
            if not any(is_child_compatible_activity(item, child_age) for item in activities):
                _reject(filter_reasons, plan_id, "不满足低龄儿童友好要求")
                continue

        # 7. 减脂 / 低卡饮食约束
        if mom_diet == "low_calorie" and _has_low_calorie_context(raw_constraint_text, constraints, user_profile):
            if restaurants:
                restaurant_health_ok = True
                for item in restaurants:
                    tags = item.get("tags", []) or []
                    health_tags = item.get("health_tags", []) or []
                    menu_health_options = item.get("menu_health_options", []) or []
                    health_signals = set(tags) | set(health_tags) | set(menu_health_options)
                    if not health_signals.intersection({"low_calorie", "light_food", "low_oil", "low_sugar", "high_protein", "vegetable_rich"}):
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
