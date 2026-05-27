# -*- coding: utf-8 -*-
"""Plan-quality helpers for WeekendFlow B optimizer.

This module turns rich local-life supply fields into explicit scoring evidence.
It keeps optimization auditable: rules can adjust scores and weights, while the
selected plan can explain which supply facts supported the decision.
"""

from __future__ import annotations

from typing import Any

from .b_semantics import b_semantic_terms
from .b_utils import expand_preference_tags, normalize_scene_type, to_float


QUALITY_DIMENSION_LABELS = {
    "preference": "偏好匹配",
    "group_fit": "同行人适配",
    "route": "路线顺畅",
    "budget": "预算适配",
    "availability": "履约可用",
    "experience": "体验质量",
    "time": "时间节奏",
    "atmosphere": "氛围匹配",
    "novelty": "新鲜感",
    "weather_fit": "天气适配",
    "commercial_addon": "优惠与套餐",
    "risk": "风险扣分",
}

_WEIGHT_INTENT_TERMS = {
    "health": {"低卡", "轻食", "健康", "减脂", "少油", "少盐", "low_calorie", "light_food", "healthy"},
    "nearby": {"近", "附近", "别太远", "不要太远", "低心智负担", "省心", "nearby", "short_distance"},
    "quiet_date": {"安静", "约会", "氛围", "舒服", "romantic", "quiet", "date_friendly", "atmosphere"},
    "budget": {"预算", "便宜", "平价", "省钱", "budget", "value_for_money"},
    "family": {"亲子", "孩子", "儿童", "带娃", "kid_friendly", "family_friendly"},
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        values: list[Any] = []
        for nested in value.values():
            values.extend(_flatten(nested))
        return values
    if isinstance(value, (list, tuple, set)):
        values: list[Any] = []
        for item in value:
            values.extend(_flatten(item))
        return values
    return [value]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _semantic_set(values: Any) -> set[str]:
    flattened = _flatten(values)
    return set(b_semantic_terms(flattened, include_auxiliary=True) + expand_preference_tags(flattened))


def _direct_term_set(values: Any) -> set[str]:
    flattened = [str(item).strip() for item in _flatten(values) if str(item).strip()]
    return set(flattened + expand_preference_tags(flattened))


def _bool_score(value: Any, *, unknown: float = 0.5) -> float:
    if value is True:
        return 1.0
    if value is False:
        return 0.0
    return unknown


def _average(values: list[float], default: float) -> float:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return default
    return sum(cleaned) / len(cleaned)


def _review_quality(item: dict[str, Any]) -> float:
    review = item.get("review_breakdown")
    values: list[float] = []
    if isinstance(review, dict):
        for value in review.values():
            numeric = to_float(value, -1.0)
            if numeric >= 0:
                values.append(min(1.0, numeric / 5.0))
    if values:
        return _clamp01(_average(values, 0.75))
    rating = to_float(item.get("rating"), 4.0)
    return _clamp01(rating / 5.0)


def _package_score(item: dict[str, Any]) -> float:
    packages = [package for package in _as_list(item.get("package_options")) if isinstance(package, dict)]
    deals = [deal for deal in _as_list(item.get("deals")) if isinstance(deal, dict)]
    product_count = len(_as_list(item.get("products")))
    if not packages and not deals and product_count == 0:
        return 0.35

    discount_scores = []
    for package in packages + deals:
        sale_price = to_float(package.get("sale_price") or package.get("amount"), 0.0)
        original_price = to_float(package.get("original_price"), 0.0)
        if sale_price > 0 and original_price > sale_price:
            discount_scores.append((original_price - sale_price) / original_price)

    score = 0.50 + min(0.20, (len(packages) + len(deals) + product_count) * 0.04)
    if any(bool(package.get("coupon_available")) for package in packages):
        score += 0.10
    if discount_scores:
        score += min(0.20, max(discount_scores) * 1.2)
    return _clamp01(score)


def _slot_confidence(item: dict[str, Any]) -> float:
    slots = [slot for slot in _as_list(item.get("reservation_slots") or item.get("available_slots")) if isinstance(slot, dict)]
    if not slots:
        return 0.55 if not item.get("reservation_required") else 0.35

    positive = 0
    for slot in slots:
        if slot.get("reservable") is False:
            continue
        remaining = slot.get("inventory_left")
        if remaining is None:
            remaining = slot.get("remaining")
        if remaining is None or to_float(remaining, 0.0) > 0:
            positive += 1
    return _clamp01(0.45 + 0.55 * positive / max(1, len(slots)))


def _fulfillment_confidence(activity: dict[str, Any], restaurant: dict[str, Any]) -> float:
    activity_score = _slot_confidence(activity)
    restaurant_score = _slot_confidence(restaurant)
    business_hours_score = 0.85 if activity.get("business_hours") and restaurant.get("business_hours") else 0.65
    reservation_signal = 0.85 if restaurant.get("dine_in_available") is not False else 0.25
    return _clamp01(
        0.34 * activity_score
        + 0.34 * restaurant_score
        + 0.18 * business_hours_score
        + 0.14 * reservation_signal
    )


def _family_facility_score(activity: dict[str, Any], restaurant: dict[str, Any], child_age: int | None) -> float:
    activity_signals = _semantic_set(activity.get("tags"))
    restaurant_signals = _semantic_set(restaurant.get("tags"))
    score = 0.45
    if activity_signals.intersection({"kid_friendly", "family_friendly", "亲子", "儿童友好", "低强度"}):
        score += 0.22
    if restaurant_signals.intersection({"family_friendly", "kid_friendly", "child_seat", "儿童椅", "亲子餐厅"}):
        score += 0.16
    if restaurant.get("baby_chair_available") is True or restaurant.get("child_menu") is True:
        score += 0.12
    if activity.get("parking_available") is True or restaurant.get("parking_available") is True:
        score += 0.05
    age_range = activity.get("age_range") if isinstance(activity.get("age_range"), dict) else {}
    if child_age is not None and age_range:
        min_age = to_float(age_range.get("min"), 0.0)
        max_age = to_float(age_range.get("max"), 99.0)
        score += 0.08 if min_age <= child_age <= max_age else -0.12
    return _clamp01(score)


def _diet_flexibility_score(restaurant: dict[str, Any]) -> float:
    options = restaurant.get("dietary_options") if isinstance(restaurant.get("dietary_options"), dict) else {}
    tags = _semantic_set(
        [
            restaurant.get("tags"),
            restaurant.get("health_tags"),
            restaurant.get("menu_health_options"),
            restaurant.get("recommended_dishes"),
            restaurant.get("dish_tags"),
        ]
    )
    score = 0.38
    positive_keys = {"可少油", "可少盐", "有低糖饮品", "有素食/蔬菜选项", "有高蛋白选项", "减脂期友好"}
    if options:
        score += min(0.34, sum(1 for key in positive_keys if options.get(key) is True) * 0.07)
    if tags.intersection({"low_calorie", "light_food", "low_oil", "low_sugar", "high_protein", "vegetable_rich", "低卡", "轻食"}):
        score += 0.22
    if tags.intersection({"high_calorie", "high_oil", "高热量"}):
        score -= 0.12
    return _clamp01(score)


def _peak_risk_score(activity: dict[str, Any], restaurant: dict[str, Any]) -> float:
    risk = 0.0
    for item in (activity, restaurant):
        queue_profile = item.get("queue_time_by_period") if isinstance(item.get("queue_time_by_period"), dict) else {}
        if queue_profile:
            max_queue = max(to_float(value, 0.0) for value in queue_profile.values())
            if max_queue >= 45:
                risk += 0.18
            elif max_queue >= 30:
                risk += 0.12
            elif max_queue >= 20:
                risk += 0.06
        peak_profile = item.get("peak_risk_profile") if isinstance(item.get("peak_risk_profile"), dict) else {}
        risk_text = " ".join(str(value) for value in _flatten(peak_profile.get("主要风险")))
        if any(term in risk_text for term in ("排队久", "座位紧张", "库存变化", "天气不适合")):
            risk += 0.06
    if restaurant.get("noise_level") in {"偏热闹", "吵"}:
        risk += 0.04
    return _clamp01(risk)


def build_plan_quality_profile(
    *,
    activity: dict[str, Any],
    restaurant: dict[str, Any],
    scene_type: str,
    constraints: dict[str, Any] | None,
    preference_sources: list[str],
    child_age: int | None,
    mom_diet: str | None,
) -> dict[str, Any]:
    """Summarize rich mock supply facts for scoring and explanation."""

    scene_type = normalize_scene_type(scene_type)
    activity_review = _review_quality(activity)
    restaurant_review = _review_quality(restaurant)
    review_quality = _average([activity_review, restaurant_review], 0.75)
    fulfillment = _fulfillment_confidence(activity, restaurant)
    package_score = _average([_package_score(activity), _package_score(restaurant)], 0.40)
    family_fit = _family_facility_score(activity, restaurant, child_age)
    diet_fit = _diet_flexibility_score(restaurant)
    peak_risk = _peak_risk_score(activity, restaurant)

    positive_evidence: list[str] = []
    risk_evidence: list[str] = []

    if restaurant.get("signature_dishes"):
        positive_evidence.append(f"餐厅有招牌菜：{'、'.join(str(item) for item in _as_list(restaurant.get('signature_dishes'))[:3])}")
    if restaurant.get("recommended_dishes"):
        positive_evidence.append(f"网友推荐：{'、'.join(str(item) for item in _as_list(restaurant.get('recommended_dishes'))[:3])}")
    if restaurant.get("baby_chair_available") is True:
        positive_evidence.append("餐厅支持宝宝椅")
    if restaurant.get("dietary_options"):
        positive_evidence.append("餐厅支持少油/低糖等饮食备注")
    if activity.get("package_options") or restaurant.get("package_options"):
        positive_evidence.append("有套餐或团购券可执行")
    if activity.get("business_hours") and restaurant.get("business_hours"):
        positive_evidence.append("活动和餐厅都有营业时间信息")

    if peak_risk >= 0.12:
        risk_evidence.append("高峰期存在排队、座位或库存波动风险")
    if restaurant.get("parking_available") is False:
        risk_evidence.append("餐厅停车不稳定，建议地铁/打车")
    if mom_diet == "low_calorie" and diet_fit < 0.62:
        risk_evidence.append("减脂/低卡适配不足，需要备注少油少盐")
    if scene_type == "couple" and restaurant.get("noise_level") == "偏热闹":
        risk_evidence.append("餐厅偏热闹，约会安静感一般")

    return {
        "review_quality": round(review_quality, 3),
        "fulfillment_confidence": round(fulfillment, 3),
        "package_value": round(package_score, 3),
        "family_facility_fit": round(family_fit, 3),
        "diet_flexibility": round(diet_fit, 3),
        "peak_risk": round(peak_risk, 3),
        "positive_evidence": positive_evidence[:6],
        "risk_evidence": risk_evidence[:6],
        "field_sources": {
            "activity": activity.get("mock_detail_sources", {}),
            "restaurant": restaurant.get("mock_detail_sources", {}),
        },
    }


def apply_quality_profile_to_objectives(
    objective_vector: dict[str, float],
    *,
    risk_score: float,
    risk_factors: list[str],
    quality_profile: dict[str, Any],
    scene_type: str,
    child_age: int | None,
    mom_diet: str | None,
) -> tuple[dict[str, float], float, list[str], list[dict[str, Any]]]:
    """Blend rich-supply evidence into existing optimizer objectives."""

    scene_type = normalize_scene_type(scene_type)
    adjusted = dict(objective_vector)
    adjustments: list[dict[str, Any]] = []

    def blend(key: str, evidence_key: str, ratio: float, reason: str) -> None:
        old = to_float(adjusted.get(key), 0.0)
        evidence_score = to_float(quality_profile.get(evidence_key), old)
        new = _clamp01(old * (1.0 - ratio) + evidence_score * ratio)
        adjusted[key] = round(new, 3)
        if abs(new - old) >= 0.015:
            adjustments.append(
                {
                    "dimension": key,
                    "label": QUALITY_DIMENSION_LABELS.get(key, key),
                    "from": round(old, 3),
                    "to": round(new, 3),
                    "reason": reason,
                }
            )

    blend("experience", "review_quality", 0.22, "结合口味/服务/环境/体验等评价细项")
    blend("availability", "fulfillment_confidence", 0.18, "结合营业时间、库存/订座槽位和堂食可用性")
    blend("commercial_addon", "package_value", 0.55, "结合套餐、券、产品丰富度和折扣力度")

    if scene_type == "family" or child_age is not None:
        blend("group_fit", "family_facility_fit", 0.24, "结合宝宝椅、适龄、亲子/低强度设施")
    if mom_diet == "low_calorie":
        blend("group_fit", "diet_flexibility", 0.20, "结合低卡、少油少盐和菜品可备注性")

    risk_delta = to_float(quality_profile.get("peak_risk"), 0.0) * 0.35
    new_risk = _clamp01(risk_score + risk_delta)
    new_factors = list(risk_factors)
    if risk_delta >= 0.025:
        new_factors.extend(quality_profile.get("risk_evidence", []) or [])
        adjustments.append(
            {
                "dimension": "risk",
                "label": QUALITY_DIMENSION_LABELS["risk"],
                "from": round(risk_score, 3),
                "to": round(new_risk, 3),
                "reason": "结合高峰排队、座位、库存和停车等履约风险",
            }
        )

    return adjusted, round(new_risk, 3), new_factors, adjustments


def adjust_weights_for_context(
    weights: dict[str, float],
    *,
    scene_type: str,
    constraints: dict[str, Any] | None,
    preference_sources: list[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Apply small intent-level weight shifts on top of scene weights."""

    scene_type = normalize_scene_type(scene_type)
    adjusted = dict(weights)
    reasons: list[dict[str, Any]] = []
    terms = _semantic_set([preference_sources, constraints or {}])
    direct_terms = _direct_term_set([preference_sources, constraints or {}])

    def add(key: str, delta: float, reason: str) -> None:
        if key not in adjusted:
            return
        before = adjusted[key]
        adjusted[key] = round(before + delta, 3)
        reasons.append({"dimension": key, "delta": round(delta, 3), "reason": reason})

    if scene_type == "family" or terms.intersection(_WEIGHT_INTENT_TERMS["family"]):
        add("group_fit", 0.03, "亲子/家庭需求提高同行人适配权重")
        add("availability", 0.02, "带娃场景提高履约确定性权重")
    if terms.intersection(_WEIGHT_INTENT_TERMS["health"]):
        add("group_fit", 0.03, "健康/减脂需求提高饮食适配权重")
        add("risk", -0.02, "健康需求提高风险扣分敏感度")
    if terms.intersection(_WEIGHT_INTENT_TERMS["nearby"]):
        add("route", 0.03, "近场/省心需求提高路线权重")
        add("time", 0.01, "近场/省心需求提高节奏权重")
    if scene_type == "couple" or direct_terms.intersection(_WEIGHT_INTENT_TERMS["quiet_date"]):
        add("atmosphere", 0.03, "约会/安静需求提高氛围权重")
    if scene_type == "low_budget" or terms.intersection(_WEIGHT_INTENT_TERMS["budget"]):
        add("budget", 0.04, "预算敏感需求提高价格权重")
        add("commercial_addon", 0.02, "预算敏感需求提高套餐/券权重")

    positive_keys = [key for key, value in adjusted.items() if key != "risk" and value > 0]
    original_sum = sum(value for key, value in weights.items() if key != "risk" and value > 0)
    adjusted_sum = sum(adjusted[key] for key in positive_keys)
    if adjusted_sum > 0 and original_sum > 0:
        scale = original_sum / adjusted_sum
        for key in positive_keys:
            adjusted[key] = round(adjusted[key] * scale, 3)

    return adjusted, reasons


def build_score_breakdown_details(
    objective_vector: dict[str, float],
    weights: dict[str, float],
    score_breakdown: dict[str, float],
    quality_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    details = []
    for key, score in objective_vector.items():
        weight = abs(to_float(weights.get(key), 0.0))
        contribution = to_float(score_breakdown.get(key), 0.0)
        details.append(
            {
                "key": key,
                "label": QUALITY_DIMENSION_LABELS.get(key, key),
                "score": round(to_float(score), 3),
                "weight": round(weight, 3),
                "contribution": round(contribution, 3),
                "direction": "penalty" if key == "risk" else "positive",
            }
        )
    details.sort(key=lambda item: item["contribution"], reverse=True)
    if quality_profile.get("positive_evidence"):
        details[0]["evidence"] = quality_profile.get("positive_evidence", [])[:3]
    return details


def build_why_selected(
    objective_vector: dict[str, float],
    risk_factors: list[str],
    quality_profile: dict[str, Any],
) -> dict[str, Any]:
    ranked = sorted(
        (
            {
                "key": key,
                "label": QUALITY_DIMENSION_LABELS.get(key, key),
                "score": round(to_float(value), 3),
            }
            for key, value in objective_vector.items()
            if key != "risk"
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return {
        "top_reasons": ranked[:4],
        "evidence": quality_profile.get("positive_evidence", [])[:5],
        "tradeoffs": list(dict.fromkeys((risk_factors or []) + (quality_profile.get("risk_evidence") or [])))[:5],
    }
