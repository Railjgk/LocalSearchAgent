try:
    from src.state import PlanState
except ImportError:
    PlanState = dict


def _extract_plan_items(selected_plan: dict) -> tuple[dict, dict]:
    timeline = selected_plan.get("timeline") or []
    activity_types = {"activity", "play", "amusement", "museum", "art"}
    restaurant_types = {"restaurant", "eat"}

    activity_item = next(
        (item for item in timeline if item.get("type") in activity_types),
        {},
    )
    restaurant_item = next(
        (item for item in timeline if item.get("type") in restaurant_types),
        {},
    )

    return activity_item, restaurant_item


def explainability_node(state: PlanState) -> dict:
    """
    Generate comprehensive explanation for plan selection.
    Incorporates objective analysis, risk factors, constraint status, and alternatives.
    """
    execution_log = state.get("execution_log", [])
    selected_plan = state.get("selected_plan", {})
    alternative_plans = state.get("alternative_plans", [])
    optimization_score = state.get("optimization_score", 0.0)
    filter_reasons = state.get("filter_reasons", {})

    if not selected_plan:
        summary = filter_reasons.get("_summary", "当前约束过于严格或候选不足")
        relaxation_suggestions = filter_reasons.get("_relaxation_suggestions", [])

        suggestion_text = ""
        if relaxation_suggestions:
            suggestion_text = "建议放宽约束：" + "；".join(relaxation_suggestions) + "。"

        explanation_text = (
            f"当前没有找到满足所有硬约束的可执行方案。{summary}。"
            f"{suggestion_text}"
            "或考虑更多儿童友好/低卡活动和餐厅。"
        )

        execution_log.append("[B] explainability_node 生成无解提示并建议放宽约束")

        return {
            "explanation_text": explanation_text,
            "execution_log": execution_log,
        }

    activity_item, restaurant_item = _extract_plan_items(selected_plan)
    activity_name = activity_item.get("activity", "活动")
    restaurant_name = restaurant_item.get("activity", "餐厅")

    objective = selected_plan.get("objective_vector", {})
    risk_factors = selected_plan.get("risk_factors", [])
    constraint_summary = selected_plan.get("constraint_summary", {})
    execution_ready = selected_plan.get("execution_ready", False)

    highlights = []

    if objective.get("group_fit", 0) >= 0.6:
        highlights.append("儿童友好与家庭适配性高")
    if objective.get("availability", 0) >= 0.6:
        highlights.append("可预约性好，排队风险低")
    if objective.get("route", 0) >= 0.6:
        highlights.append("路线控制合理，通勤距离适中")
    if objective.get("budget", 0) >= 0.6:
        highlights.append("预算使用合理")
    if objective.get("experience", 0) >= 0.6:
        highlights.append("体验评价较高")
    if objective.get("preference", 0) >= 0.6:
        highlights.append("偏好匹配度高")

    best_factors = highlights or ["综合平衡较好"]

    scene_type = state.get("scene_type", "family")
    scene_description = {
        "family": "它兼顾了儿童友好、低强度与轻松用餐的家庭场景需求。",
        "friends": "它兼顾了体验感与社交氛围，适合朋友聚会。",
        "couple": "它兼顾了舒适节奏与轻松体验，适合情侣周末约会。",
        "low_budget": "它在预算控制上表现更优，同时兼顾出行距离和可执行性。",
    }.get(scene_type, "它体现了当前候选集合中的综合协调性和执行可行性。")

    risk_explanation = ""
    if risk_factors:
        risk_explanation = f"潜在风险因素：{'; '.join(risk_factors)}。建议提前预留时间或联系商家。"

    constraint_status_parts = []

    for key, value in constraint_summary.items():
        if value != "✓":
            if key == "distance_status":
                constraint_status_parts.append("距离可能超出预期")
            elif key == "queue_status":
                constraint_status_parts.append("排队时间可能较长")
            elif key == "budget_status":
                constraint_status_parts.append("预算可能超出")
            elif key == "child_friendly_status":
                constraint_status_parts.append("儿童友好程度可能不足")
            elif key == "diet_status":
                constraint_status_parts.append("饮食需求可能不完全满足")

    constraint_caveat = ""
    if constraint_status_parts:
        constraint_caveat = f"注意：{'; '.join(constraint_status_parts)}。"
    elif execution_ready:
        constraint_caveat = "✓ 所有约束条件已满足，方案完全可执行。"

    alt_parts = []
    for alt in alternative_plans[:2]:
        title = alt.get("title", "备选方案")
        dominant = alt.get("dominant_dimension", "")
        tradeoff = alt.get("tradeoff", "提供了不同取舍")

        if dominant:
            alt_parts.append(f"{title}（优势：{dominant}）：{tradeoff}")
        else:
            alt_parts.append(f"{title}：{tradeoff}")

    if alternative_plans:
        alternatives_explanation = f"备选方案中，{'; '.join(alt_parts)}。"
    else:
        alternatives_explanation = "当前候选集合中没有其他显著差异的备选方案。"

    explanation_text = (
        f"推荐这个方案，是因为它在关键目标上表现优秀：{'; '.join(best_factors)}。"
        f"活动选择【{activity_name}】，餐厅选择【{restaurant_name}】，{scene_description}"
        f"该方案预计总预算为 {selected_plan.get('total_price')} 元、"
        f"总时长 {selected_plan.get('total_duration_min')} 分钟、"
        f"总距离 {selected_plan.get('total_distance_km')} 公里，"
        f"综合得分 {optimization_score}。"
    )

    if constraint_caveat:
        explanation_text += constraint_caveat

    if risk_explanation:
        explanation_text += risk_explanation

    explanation_text += alternatives_explanation

    execution_log.append(
        f"[B] explainability_node 生成多维度解释 "
        f"(execution_ready={execution_ready}, risk_factors={len(risk_factors)})"
    )

    return {
        "explanation_text": explanation_text,
        "execution_log": execution_log,
    }
