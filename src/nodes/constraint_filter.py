try:
    from src.state import PlanState
except ImportError:
    PlanState = dict

from .b_utils import (
    to_float,
    get_constraint_config_with_profile,
    build_filter_summary,
    generate_relaxation_suggestions,
)


def _get_plan_nodes(plan: dict) -> tuple[dict, dict]:
    nodes = plan.get("nodes", []) or []
    activity = next((node for node in nodes if node.get("type") == "activity"), {})
    restaurant = next((node for node in nodes if node.get("type") == "restaurant"), {})
    return activity, restaurant


def _reject(filter_reasons: dict, plan_id: str, reason: str) -> None:
    filter_reasons[plan_id] = reason


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

        total_distance = to_float(route.get("total_distance_km"), 0.0)
        max_queue = to_float(availability.get("max_queue_time_min"), 0.0)
        estimated_duration = to_float(plan.get("estimated_duration_min"), 0.0)
        total_price = to_float(budget_info.get("total_price"), 0.0)

        activity_tags = activity.get("tags", []) or []
        restaurant_tags = restaurant.get("tags", []) or []
        restaurant_health_tags = restaurant.get("health_tags", []) or []
        menu_health_options = restaurant.get("menu_health_options", []) or []

        # 1. 库存 / 可用性
        if not availability.get("all_available", False):
            _reject(filter_reasons, plan_id, "活动或餐厅当前不可用")
            continue

        # 2. 距离
        if total_distance > max_distance_km:
            _reject(filter_reasons, plan_id, "距离超过用户可接受范围")
            continue

        # 3. 排队
        if max_queue > max_queue_time:
            _reject(filter_reasons, plan_id, "排队时间过长")
            continue

        # 4. 时长
        if estimated_duration < duration_range[0] or estimated_duration > duration_range[1]:
            _reject(filter_reasons, plan_id, "时长不满足用户的时间范围")
            continue

        # 5. 预算：允许 1.2 倍软浮动，但超过则硬过滤
        if total_price > budget * 1.2:
            _reject(filter_reasons, plan_id, "预算超出可接受上限")
            continue

        # 6. 低龄儿童约束
        if child_age is not None and child_age <= 6:
            if "kid_friendly" not in activity_tags and "low_intensity" not in activity_tags:
                _reject(filter_reasons, plan_id, "不满足低龄儿童友好要求")
                continue

        # 7. 减脂 / 低卡饮食约束
        if mom_diet == "low_calorie":
            health_signals = set(restaurant_tags) | set(restaurant_health_tags) | set(menu_health_options)
            if not health_signals.intersection({"low_calorie", "light_food", "low_oil", "low_sugar", "high_protein", "vegetable_rich"}):
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
