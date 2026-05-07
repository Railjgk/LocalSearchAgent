try:
    from src.state import PlanState
except ImportError:
    PlanState = dict

from .b_utils import (
    to_float,
    parse_child_age,
    normalize,
    safe_match_count,
)


ABSOLUTE_MAX_DISTANCE_KM = 15.0
ABSOLUTE_MAX_QUEUE_TIME_MIN = 60.0
ABSOLUTE_MIN_RATING = 3.0
ABSOLUTE_MAX_RATING = 5.0


def _derive_weights(scene_type: str) -> dict[str, float]:
    """Get scene-type specific weights for 7-dimensional objective vector."""
    weights = {
        "family": {
            "preference": 0.05,
            "group_fit": 0.30,
            "route": 0.20,
            "budget": 0.15,
            "availability": 0.20,
            "experience": 0.10,
            "risk": -0.20,
        },
        "friends": {
            "preference": 0.20,
            "group_fit": 0.10,
            "route": 0.15,
            "budget": 0.15,
            "availability": 0.15,
            "experience": 0.25,
            "risk": -0.15,
        },
        "couple": {
            "preference": 0.20,
            "group_fit": 0.05,
            "route": 0.20,
            "budget": 0.10,
            "availability": 0.15,
            "experience": 0.30,
            "risk": -0.15,
        },
        "low_budget": {
            "preference": 0.05,
            "group_fit": 0.15,
            "route": 0.20,
            "budget": 0.35,
            "availability": 0.15,
            "experience": 0.10,
            "risk": -0.15,
        },
    }
    return weights.get(scene_type, weights["family"])


def _score_preference(preference_sources: list[str], tags: list[str]) -> float:
    """Score preference match (0-1) independently of other plans."""
    if not preference_sources or not tags:
        return 0.5

    matches = safe_match_count(preference_sources, tags)
    return min(1.0, matches / max(1, len(preference_sources)) * 0.8 + 0.2)


def _score_group_fit(
    activity_tags: list,
    restaurant_tags: list,
    child_age: int | None,
    mom_diet: str | None,
    scene_type: str,
) -> float:
    """
    Score group suitability (0-1) based on absolute criteria.
    Not dependent on other plans in the set.
    """
    score = 0.0
    max_points = 0.0

    if scene_type == "family":
        max_points += 0.5
        if child_age is not None and child_age <= 6:
            if "kid_friendly" in activity_tags or "low_intensity" in activity_tags:
                score += 0.5
        else:
            score += 0.3

    max_points += 0.5
    if mom_diet == "low_calorie":
        if "low_calorie" in restaurant_tags or "light_food" in restaurant_tags:
            score += 0.5
    else:
        score += 0.3

    return min(1.0, score / max(1.0, max_points)) if max_points > 0 else 0.5


def _score_route(distance_km: float, travel_time_min: float) -> float:
    """
    Score route quality (0-1) using absolute boundaries.
    1.0 = nearby and short; 0.0 = too far or too long.
    """
    distance_score = normalize(distance_km, 0, ABSOLUTE_MAX_DISTANCE_KM)
    time_score = normalize(travel_time_min, 0, ABSOLUTE_MAX_DISTANCE_KM * 6)
    return max(0.0, 1.0 - 0.5 * distance_score - 0.5 * time_score)


def _score_budget(total_price: float, user_budget: float, scene_type: str) -> float:
    """
    Score budget appropriateness (0-1) using absolute boundaries.
    Considers scene-type specific preferences.
    """
    total_price = to_float(total_price, 0.0)
    user_budget = to_float(user_budget, 500.0)

    if total_price > user_budget * 1.2:
        return 0.0

    if scene_type == "low_budget":
        return max(0.0, 1.0 - total_price / max(1.0, user_budget * 1.2))

    if total_price <= user_budget:
        utilization = total_price / max(1.0, user_budget)
        return min(1.0, 0.6 + 0.4 * utilization)

    return max(0.0, 1.0 - (total_price - user_budget) / max(1.0, user_budget) * 2)


def _score_availability(all_available: bool, queue_time_min: float) -> float:
    """Score availability (0-1) using absolute criteria."""
    if not all_available:
        return 0.1

    queue_score = normalize(queue_time_min, 0, ABSOLUTE_MAX_QUEUE_TIME_MIN)
    return 0.5 + 0.5 * (1.0 - queue_score)


def _score_experience(activity_rating: float, restaurant_rating: float, tags: list) -> float:
    """Score experience quality (0-1) independently."""
    rating = to_float((activity_rating + restaurant_rating) / 2.0, 4.0)
    rating_score = normalize(rating, ABSOLUTE_MIN_RATING, ABSOLUTE_MAX_RATING)
    tag_diversity = min(1.0, len(set(tags)) / 8.0)
    return 0.6 * rating_score + 0.4 * tag_diversity


def _calc_risk_factors(
    distance_km: float,
    queue_time_min: float,
    activity_rating: float,
    restaurant_rating: float,
    tags: list,
) -> tuple[float, list[str]]:
    """
    Calculate risk score (0-1, lower is better) and identify specific risk factors.
    Uses absolute criteria, not dependent on plan collection statistics.
    """
    risk_factors = []
    risk_score = 0.0

    if distance_km > ABSOLUTE_MAX_DISTANCE_KM * 0.8:
        risk_score += 0.3
        risk_factors.append(f"距离较远 ({distance_km:.1f} 公里)")

    if queue_time_min > ABSOLUTE_MAX_QUEUE_TIME_MIN * 0.7:
        risk_score += 0.25
        risk_factors.append(f"可能排队较长 ({queue_time_min:.0f} 分钟)")

    avg_rating = (activity_rating + restaurant_rating) / 2.0
    if avg_rating < 4.2:
        risk_score += 0.2
        risk_factors.append(f"评分不够高 ({avg_rating:.1f})")

    if "crowded_mall" in tags:
        risk_score += 0.15
        risk_factors.append("可能人流较多")

    return min(1.0, risk_score), risk_factors


def _build_timeline(activity: dict, restaurant: dict, start_hour: int = 14, start_minute: int = 30) -> list[dict]:
    """Build detailed timeline with activity, transition, restaurant."""
    schedule = activity.get("_selected_schedule", {}) or {}
    activity_start_str = schedule.get("activity_start")
    activity_end_str = schedule.get("activity_end")
    restaurant_start_str = schedule.get("restaurant_start")

    def format_time(hour: int, minute: int) -> str:
        return f"{hour:02d}:{minute:02d}"

    if activity_start_str and ":" in activity_start_str:
        start_hour, start_minute = [int(x) for x in activity_start_str.split(":", 1)]

    activity_start_total = start_hour * 60 + start_minute
    activity_end_total = activity_start_total + activity.get("duration_min", 0)
    if activity_end_str and ":" in activity_end_str:
        activity_end_hour, activity_end_minute = [int(x) for x in activity_end_str.split(":", 1)]
        activity_end_total = activity_end_hour * 60 + activity_end_minute

    activity_end_hour, activity_end_minute = divmod(activity_end_total, 60)

    transition_start_hour, transition_start_minute = activity_end_hour, activity_end_minute
    if restaurant_start_str and ":" in restaurant_start_str:
        restaurant_start_hour, restaurant_start_minute = [int(x) for x in restaurant_start_str.split(":", 1)]
        transition_end_total = restaurant_start_hour * 60 + restaurant_start_minute
    else:
        transition_end_total = activity_end_total + 30
    transition_buffer_min = max(0, transition_end_total - activity_end_total)
    transition_end_hour, transition_end_minute = divmod(transition_end_total, 60)

    restaurant_start_hour, restaurant_start_minute = divmod(transition_end_total, 60)
    restaurant_end_total = (
        restaurant_start_hour * 60
        + restaurant_start_minute
        + restaurant.get("duration_min", 0)
    )
    restaurant_end_hour, restaurant_end_minute = divmod(restaurant_end_total, 60)

    return [
        {
            "time": f"{format_time(start_hour, start_minute)}-{format_time(activity_end_hour, activity_end_minute)}",
            "activity": activity.get("name"),
            "poi_id": activity.get("poi_id"),
            "type": "play",
            "duration_min": activity.get("duration_min"),
            "price": activity.get("price"),
            "notes": [
                "适合儿童" if "kid_friendly" in activity.get("tags", []) else "体验型活动",
                "低强度" if "low_intensity" in activity.get("tags", []) else "强度适中",
            ],
        },
        {
            "time": f"{format_time(transition_start_hour, transition_start_minute)}-{format_time(transition_end_hour, transition_end_minute)}",
            "activity": "附近休息与转场",
            "poi_id": None,
            "type": "transition",
            "duration_min": transition_buffer_min,
            "price": 0,
            "notes": ["避免行程过满"],
        },
        {
            "time": f"{format_time(restaurant_start_hour, restaurant_start_minute)}-{format_time(restaurant_end_hour, restaurant_end_minute)}",
            "activity": restaurant.get("name"),
            "poi_id": restaurant.get("poi_id"),
            "type": "restaurant",
            "duration_min": restaurant.get("duration_min"),
            "price": restaurant.get("price"),
            "notes": [
                "低卡" if "low_calorie" in restaurant.get("tags", []) else "普通餐饮",
                "轻食" if "light_food" in restaurant.get("tags", []) else "口味偏重",
            ],
        },
    ]


def plan_optimizer_node(state: PlanState) -> dict:
    """
    Multi-objective plan optimization with absolute scoring and enhanced metadata.
    """
    execution_log = state.get("execution_log", [])
    filtered_candidates = state.get("filtered_candidates", [])
    scene_type = state.get("scene_type", "family")
    constraints = state.get("constraints", {})
    user_profile = state.get("user_profile", {})

    if not filtered_candidates:
        execution_log.append("[B] plan_optimizer_node 未找到可行方案")
        return {
            "selected_plan": {},
            "optimization_score": 0.0,
            "alternative_plans": [],
            "execution_log": execution_log,
        }

    weights = _derive_weights(scene_type)
    budget = to_float(constraints.get("budget"), 500.0)
    child_age = parse_child_age(constraints.get("child_age"))
    max_distance = to_float(constraints.get("max_distance_km"), 8.0)
    max_queue_time = to_float(constraints.get("max_queue_time"), 30.0)

    preference_sources = []
    food_pref = user_profile.get("food_preference")

    if isinstance(food_pref, list):
        preference_sources.extend(food_pref)
    elif isinstance(food_pref, str) and food_pref:
        preference_sources.append(food_pref)

    scenario_activities = state.get("scenario_activities", []) or []
    preference_sources.extend(scenario_activities)

    scored_candidates = []

    for plan in filtered_candidates:
        tags = plan.get("tags", []) or []
        route = plan.get("route", {}) or {}
        budget_info = plan.get("budget", {}) or {}
        availability = plan.get("availability", {}) or {}

        activity = next((node for node in plan.get("nodes", []) if node.get("type") == "activity"), {})
        restaurant = next((node for node in plan.get("nodes", []) if node.get("type") == "restaurant"), {})

        activity_tags = activity.get("tags", []) or []
        restaurant_tags = restaurant.get("tags", []) or []

        preference = _score_preference(preference_sources, tags)
        group_fit = _score_group_fit(
            activity_tags,
            restaurant_tags,
            child_age,
            constraints.get("mom_diet"),
            scene_type,
        )
        route_value = _score_route(
            route.get("total_distance_km", 0),
            route.get("total_travel_time_min", 0),
        )
        budget_value = _score_budget(budget_info.get("total_price", 0), budget, scene_type)
        availability_value = _score_availability(
            availability.get("all_available", False),
            availability.get("max_queue_time_min", 0),
        )
        experience_value = _score_experience(
            activity.get("rating", 0),
            restaurant.get("rating", 0),
            tags,
        )
        risk_score, risk_factors = _calc_risk_factors(
            route.get("total_distance_km", 0),
            availability.get("max_queue_time_min", 0),
            activity.get("rating", 0),
            restaurant.get("rating", 0),
            tags,
        )

        objective_vector = {
            "preference": round(preference, 3),
            "group_fit": round(group_fit, 3),
            "route": round(route_value, 3),
            "budget": round(budget_value, 3),
            "availability": round(availability_value, 3),
            "experience": round(experience_value, 3),
            "risk": round(risk_score, 3),
        }

        weighted_score = (
            objective_vector["preference"] * weights["preference"]
            + objective_vector["group_fit"] * weights["group_fit"]
            + objective_vector["route"] * weights["route"]
            + objective_vector["budget"] * weights["budget"]
            + objective_vector["availability"] * weights["availability"]
            + objective_vector["experience"] * weights["experience"]
            - objective_vector["risk"] * abs(weights["risk"])
        )

        avoid = user_profile.get("avoid", []) or []
        if "crowded_mall" in avoid and "crowded_mall" in tags:
            weighted_score *= 0.85

        score_breakdown = {
            "preference": round(objective_vector["preference"] * weights["preference"], 3),
            "group_fit": round(objective_vector["group_fit"] * weights["group_fit"], 3),
            "route": round(objective_vector["route"] * weights["route"], 3),
            "budget": round(objective_vector["budget"] * weights["budget"], 3),
            "availability": round(objective_vector["availability"] * weights["availability"], 3),
            "experience": round(objective_vector["experience"] * weights["experience"], 3),
            "risk": round(objective_vector["risk"] * abs(weights["risk"]), 3),
        }

        scored_candidates.append(
            {
                "plan": plan,
                "objective_vector": objective_vector,
                "weighted_score": round(weighted_score, 4),
                "score_breakdown": score_breakdown,
                "risk_factors": risk_factors,
            }
        )

    scored_candidates.sort(key=lambda x: x["weighted_score"], reverse=True)

    selected = scored_candidates[0]
    selected_plan_base = selected["plan"]

    activity = next((node for node in selected_plan_base.get("nodes", []) if node.get("type") == "activity"), {})
    restaurant = next((node for node in selected_plan_base.get("nodes", []) if node.get("type") == "restaurant"), {})

    activity_tags = activity.get("tags", []) or []
    restaurant_tags = restaurant.get("tags", []) or []

    people_count = constraints.get("people_count", user_profile.get("people_count", 3))
    activity["_selected_schedule"] = selected_plan_base.get("schedule", {})
    timeline = _build_timeline(activity, restaurant)

    activity_notes = []
    if "kid_friendly" in activity_tags:
        activity_notes.append("kid_friendly")
    if "low_intensity" in activity_tags:
        activity_notes.append("low_intensity")

    restaurant_notes = []
    if "family_friendly" in restaurant_tags:
        restaurant_notes.append("child_seat")
    if "low_calorie" in restaurant_tags or "light_food" in restaurant_tags:
        restaurant_notes.append("low_oil_low_salt")

    child_fit_ok = (
        child_age is None
        or "kid_friendly" in activity_tags
        or "low_intensity" in activity_tags
    )

    diet_ok = (
        constraints.get("mom_diet") != "low_calorie"
        or "low_calorie" in restaurant_tags
        or "light_food" in restaurant_tags
    )

    constraint_summary = {
        "distance_status": (
            "✓"
            if selected_plan_base.get("route", {}).get("total_distance_km", 0) <= max_distance
            else "⚠"
        ),
        "queue_status": (
            "✓"
            if selected_plan_base.get("availability", {}).get("max_queue_time_min", 0) <= max_queue_time
            else "⚠"
        ),
        "budget_status": (
            "✓"
            if selected_plan_base.get("budget", {}).get("total_price", 0) <= budget * 1.2
            else "⚠"
        ),
        "child_friendly_status": "✓" if child_fit_ok else "⚠",
        "diet_status": "✓" if diet_ok else "⚠",
    }

    selected_plan = {
        "plan_id": selected_plan_base.get("plan_id", "plan_001").replace("cand_", "plan_"),
        "title": (
            "轻松亲子下午计划"
            if ("kid_friendly" in selected_plan_base.get("tags", []) or "low_intensity" in activity_tags)
            else "周末休闲计划"
        ),
        "scene_type": scene_type,
        "timeline": timeline,
        "total_price": selected_plan_base.get("budget", {}).get("total_price", 0),
        "total_duration_min": selected_plan_base.get("estimated_duration_min", 0),
        "total_distance_km": selected_plan_base.get("route", {}).get("total_distance_km", 0),
        "people_count": people_count,
        "route": selected_plan_base.get("route", {}),
        "budget": selected_plan_base.get("budget", {}),
        "availability": selected_plan_base.get("availability", {}),
        "objective_vector": selected["objective_vector"],
        "score_breakdown": selected["score_breakdown"],
        "weighted_score": selected["weighted_score"],
        "weights": weights,
        "risk_factors": selected["risk_factors"],
        "constraint_summary": constraint_summary,
        "execution_ready": all(v == "✓" for v in constraint_summary.values()),
        "action_hints": [
            {
                "action_type": "order_activity_ticket",
                "poi_id": activity.get("poi_id"),
                "time": timeline[0].get("time", "14:30").split("-")[0],
                "quantity": people_count,
                "notes": activity_notes,
            },
            {
                "action_type": "reserve_restaurant",
                "poi_id": restaurant.get("poi_id"),
                "time": timeline[2].get("time", "17:00").split("-")[0],
                "people": people_count,
                "notes": restaurant_notes,
            },
        ],
    }

    alternative_plans = []
    seen_plan_ids = {selected["plan"].get("plan_id")}

    best_by_metric = [
        ("cheapest", min(scored_candidates, key=lambda x: x["plan"].get("budget", {}).get("total_price", float("inf")))),
        ("nearest", min(scored_candidates, key=lambda x: x["plan"].get("route", {}).get("total_distance_km", float("inf")))),
        ("best_experience", max(scored_candidates, key=lambda x: x["objective_vector"].get("experience", 0))),
    ]

    for metric_name, candidate in best_by_metric:
        if candidate is None or candidate["plan"].get("plan_id") in seen_plan_ids:
            continue

        plan_base = candidate["plan"]

        if len(alternative_plans) >= 2:
            break

        if metric_name == "cheapest":
            title = "低预算备选方案"
            dominant_dimension = "budget"
            tradeoff = "价格更低，但可能牺牲路线或体验匹配。"
        elif metric_name == "nearest":
            title = "路线更短备选方案"
            dominant_dimension = "route"
            tradeoff = "通勤更短，但可能预算更高或体验略弱。"
        else:
            title = "体验更强备选方案"
            dominant_dimension = "experience"
            tradeoff = "体验评分更高，但可能预算压力更大或排队更久。"

        alternative_plans.append(
            {
                "plan_id": plan_base.get("plan_id", "unknown").replace("cand_", "plan_"),
                "title": title,
                "dominant_dimension": dominant_dimension,
                "weighted_score": candidate["weighted_score"],
                "total_price": plan_base.get("budget", {}).get("total_price", 0),
                "total_distance_km": plan_base.get("route", {}).get("total_distance_km", 0),
                "objective_vector": candidate["objective_vector"],
                "tradeoff": tradeoff,
            }
        )
        seen_plan_ids.add(plan_base.get("plan_id"))

    if len(alternative_plans) < 2:
        for candidate in scored_candidates:
            if candidate["plan"].get("plan_id") in seen_plan_ids:
                continue

            if len(alternative_plans) >= 2:
                break

            plan_base = candidate["plan"]
            alternative_plans.append(
                {
                    "plan_id": plan_base.get("plan_id", "unknown").replace("cand_", "plan_"),
                    "title": "综合备选方案",
                    "dominant_dimension": "balance",
                    "weighted_score": candidate["weighted_score"],
                    "total_price": plan_base.get("budget", {}).get("total_price", 0),
                    "total_distance_km": plan_base.get("route", {}).get("total_distance_km", 0),
                    "objective_vector": candidate["objective_vector"],
                    "tradeoff": "整体分数接近，但优势维度不同。",
                }
            )
            seen_plan_ids.add(plan_base.get("plan_id"))

    optimization_score = round(selected_plan["weighted_score"] * 100, 2)

    execution_log.append(
        f"[B] plan_optimizer_node 选出 weighted_score={selected_plan['weighted_score']}, "
        f"execution_ready={selected_plan['execution_ready']}"
    )

    return {
        "selected_plan": selected_plan,
        "optimization_score": optimization_score,
        "alternative_plans": alternative_plans,
        "execution_log": execution_log,
    }
