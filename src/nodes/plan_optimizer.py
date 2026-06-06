try:
    from src.state import PlanState
except ImportError:
    PlanState = dict

from functools import lru_cache
import os
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is optional for smoke demos
    yaml = None

from .b_utils import (
    collect_tag_fields,
    collect_preference_sources,
    expand_preference_tags,
    get_constraint_config_with_profile,
    normalize_scene_type,
    to_float,
    normalize,
    safe_match_count,
)
from .b_semantics import (
    CHILD_STRONG_SIGNALS,
    activity_child_signal_set,
    b_semantic_terms,
    has_item_semantic_group,
    is_child_compatible_activity,
)
from .b_plan_quality import (
    adjust_weights_for_context,
    apply_quality_profile_to_objectives,
    build_plan_quality_profile,
    build_score_breakdown_details,
    build_why_selected,
)
from .b_ai_plan_critic import apply_b_plan_critic
from .b_execution_scope import (
    node_is_supported_by_current_c as _node_is_supported_by_current_c,
    node_requires_c_execution as _node_requires_c_execution,
)


ABSOLUTE_MAX_DISTANCE_KM = 15.0
ABSOLUTE_MAX_QUEUE_TIME_MIN = 60.0
ABSOLUTE_MIN_RATING = 3.0
ABSOLUTE_MAX_RATING = 5.0
_SEMANTIC_TAG_SET_CACHE_LIMIT = 20000
_SEMANTIC_TAG_SET_CACHE: dict[tuple[str, ...], set[str]] = {}

DEFAULT_SCORE_THRESHOLDS = {
    "route": {
        "absolute_max_distance_km": 15.0,
        "distance_warning_ratio": 0.8,
        "travel_minutes_per_km": 6.0,
    },
    "availability": {
        "absolute_max_queue_time_min": 60.0,
        "queue_warning_ratio": 0.7,
    },
    "experience": {
        "min_rating": 3.0,
        "max_rating": 5.0,
        "low_rating_warning": 4.2,
        "tag_diversity_cap": 8.0,
    },
    "budget": {
        "over_budget_hard_ratio": 1.2,
        "low_budget_restaurant_price_target": 180.0,
    },
}
DEFAULT_PENALTIES = {
    "unavailable_plan": 0.90,
    "crowded_mall": 0.15,
    "far_distance": 0.30,
    "long_queue": 0.25,
    "low_rating": 0.20,
    "avoid_tag_hit_multiplier": 0.85,
}

HEAVY_MEAL_TERMS = {
    "烧烤",
    "烤肉",
    "火锅",
    "大排档",
    "啤酒屋",
    "串",
    "油烟",
    "高热量",
    "bbq",
    "barbecue",
    "hotpot",
}
LOCAL_SEAFOOD_TERMS = {
    "海鲜",
    "青岛菜",
    "胶东菜",
    "鲁菜",
    "海鲜水饺",
    "啤酒屋",
}
LOW_EFFORT_TERMS = {
    "轻松",
    "别太累",
    "不累",
    "低强度",
    "少折腾",
    "省心",
    "放松",
    "少排队",
    "排队久",
    "低心智负担",
}
HIGH_FRICTION_TERMS = {
    "排队久",
    "商场拥挤",
    "拥挤",
    "油烟味",
    "高热量",
    "天气敏感",
    "not_low_intensity",
    "long_queue",
    "crowded_mall",
}


def _plan_quality_text(item: dict | None) -> str:
    if not isinstance(item, dict):
        return ""
    values: list[str] = []
    for key in (
        "name",
        "category",
        "restaurant_category",
        "sub_category",
        "primary_category",
        "itinerary_role",
        "itinerary_label",
        "gaode_keyword",
        "gaode_type",
        "address",
    ):
        value = item.get(key)
        if value:
            values.append(str(value))
    for key in (
        "tags",
        "risk_tags",
        "review_keywords",
        "signature_dishes",
        "recommended_dishes",
        "dish_tags",
        "health_tags",
        "menu_health_options",
    ):
        value = item.get(key)
        if isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, (list, tuple, set)):
                    values.extend(str(item) for item in nested if str(item).strip())
                elif nested:
                    values.append(str(nested))
        elif isinstance(value, (list, tuple, set)):
            values.extend(str(item) for item in value if str(item).strip())
        elif value:
            values.append(str(value))
    semantic_tags = item.get("semantic_tags")
    if isinstance(semantic_tags, dict):
        for nested in semantic_tags.values():
            if isinstance(nested, (list, tuple, set)):
                values.extend(str(item) for item in nested if str(item).strip())
            elif nested:
                values.append(str(nested))
    return " ".join(values)


def _matches_any_text_term(text: str, terms: set[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


def _apply_multinode_itinerary_quality_guards(
    *,
    plan: dict,
    constraints: dict,
    preference_sources: list[str],
    preference: float,
    risk_score: float,
    risk_factors: list[str],
) -> tuple[float, float, list[str]]:
    """Score the whole itinerary, not only the first activity/restaurant pair."""

    if plan.get("planner_mode") != "multi_node_itinerary":
        return preference, risk_score, risk_factors

    nodes = plan.get("nodes", []) or []
    restaurants = [node for node in nodes if str(node.get("type") or "") == "restaurant"]
    if not restaurants:
        return preference, risk_score, risk_factors

    intent_text = " ".join(
        str(value)
        for value in (
            [constraints.get("raw_text"), constraints.get("time_window")]
            + list(preference_sources or [])
            + collect_tag_fields(constraints, "hard_tags", "soft_tags", "avoid", "hard", "soft")
        )
        if value
    )
    low_effort_requested = _matches_any_text_term(intent_text, LOW_EFFORT_TERMS)

    heavy_count = 0
    seafood_count = 0
    high_friction_count = 0
    category_signatures: list[str] = []
    for restaurant in restaurants:
        text = _plan_quality_text(restaurant)
        if _matches_any_text_term(text, HEAVY_MEAL_TERMS):
            heavy_count += 1
        if _matches_any_text_term(text, LOCAL_SEAFOOD_TERMS):
            seafood_count += 1
        if _matches_any_text_term(text, HIGH_FRICTION_TERMS):
            high_friction_count += 1
        signature_terms = []
        for term in sorted(LOCAL_SEAFOOD_TERMS | HEAVY_MEAL_TERMS):
            if term and term in text:
                signature_terms.append(term)
        if signature_terms:
            category_signatures.append("|".join(signature_terms[:3]))

    if len(restaurants) >= 2:
        unique_signatures = len(set(category_signatures))
        if unique_signatures <= 1 and category_signatures:
            preference = max(0.0, preference - 0.10)
            risk_score = min(1.0, risk_score + 0.07)
            risk_factors.append("全天餐饮类型重复度偏高")
        if seafood_count >= 2 and heavy_count >= 1:
            risk_score = min(1.0, risk_score + 0.06)
            risk_factors.append("全天海鲜主题较集中，晚餐偏重口")
        if low_effort_requested and heavy_count >= 2:
            preference = max(0.0, preference - 0.15)
            risk_score = min(1.0, risk_score + 0.15)
            risk_factors.append("轻松需求下连续重口餐饮负担偏高")

    if low_effort_requested and high_friction_count:
        risk_score = min(1.0, risk_score + min(0.18, 0.08 * high_friction_count))
        risk_factors.append("轻松需求下存在排队、拥挤或油烟风险")

    return preference, risk_score, risk_factors


AI_REPLAN_TRIGGER_TERMS = (
    "structural",
    "re-search",
    "research",
    "mismatch",
    "incoherent",
    "not coherent",
    "clash",
    "replace",
    "重新",
    "重排",
    "不匹配",
    "不连贯",
    "冲突",
    "替换",
)
HEALTH_MATCH_TAGS = {
    "low_calorie",
    "light_food",
    "low_oil",
    "low_sugar",
    "high_protein",
    "vegetable_rich",
}
FRIENDS_FIT_TAGS = {
    "social",
    "group_friendly",
    "chat_friendly",
    "local_market",
    "local_culture",
    "citywalk",
    "escape_room",
    "board_game",
    "sports",
}
COUPLE_FIT_TAGS = {
    "romantic",
    "date_friendly",
    "atmosphere",
    "quiet",
    "photogenic",
    "relaxation",
    "healing",
    "micro_vacation",
    "spa",
}
BUDGET_FIT_TAGS = {
    "budget",
    "budget_activity",
    "budget_restaurant",
    "value_for_money",
    "coupon_available",
}
ATMOSPHERE_TAGS = {
    "atmosphere",
    "romantic",
    "date_friendly",
    "warm",
    "quiet",
    "photogenic",
    "ritual",
    "healing",
    "relaxation",
    "creative",
}
LOCAL_CULTURE_TAGS = {
    "citywalk",
    "local_culture",
    "city_limited",
    "local_market",
    "local_experience",
    "cultural",
    "regional_home_cuisine",
}
NOVELTY_TAGS = {
    "city_limited",
    "local_experience",
    "local_market",
    "micro_vacation",
    "wellness",
    "immersive",
    "story_driven",
    "hands_on",
    "creative",
    "ritual",
    "photogenic",
}
RELATED_PREFERENCE_TAGS = {
    "romantic": {"date_friendly", "atmosphere", "warm", "ritual", "quiet", "photogenic"},
    "atmosphere": {"romantic", "date_friendly", "warm", "ritual", "quiet", "photogenic", "cultural", "healing"},
    "local_culture": {"local_experience", "city_limited", "citywalk", "local_market", "cultural"},
    "citywalk": {"local_experience", "city_limited", "local_culture", "local_market", "cultural"},
    "light_food": {"low_calorie", "healthy", "low_oil", "low_sugar", "vegetable_rich", "japanese_light_food"},
    "low_calorie": {"light_food", "healthy", "low_oil", "low_sugar", "vegetable_rich"},
    "social": {"group_friendly", "chat_friendly", "escape_room", "board_game"},
    "烤肉": {"烧烤", "烤串", "羊肉串", "炭火", "炭烤", "日式烧肉", "韩式烤肉", "bbq", "barbecue"},
    "烧烤": {"烤肉", "烤串", "羊肉串", "炭火", "炭烤", "bbq", "barbecue"},
    "bbq": {"烤肉", "烧烤", "barbecue"},
    "barbecue": {"烤肉", "烧烤", "bbq"},
    "火锅": {"hotpot", "涮锅", "牛油锅"},
    "hotpot": {"火锅", "涮锅"},
}
WEIGHT_KEYS = (
    "preference",
    "group_fit",
    "route",
    "budget",
    "availability",
    "experience",
    "time",
    "atmosphere",
    "novelty",
    "weather_fit",
    "commercial_addon",
    "risk",
)
DEFAULT_SCENE_WEIGHTS = {
    "family": {
        "preference": 0.07,
        "group_fit": 0.26,
        "route": 0.18,
        "budget": 0.13,
        "availability": 0.18,
        "experience": 0.08,
        "time": 0.05,
        "atmosphere": 0.03,
        "novelty": 0.02,
        "weather_fit": 0.03,
        "commercial_addon": 0.00,
        "risk": -0.20,
    },
    "friends": {
        "preference": 0.22,
        "group_fit": 0.10,
        "route": 0.13,
        "budget": 0.12,
        "availability": 0.13,
        "experience": 0.18,
        "time": 0.04,
        "atmosphere": 0.05,
        "novelty": 0.03,
        "weather_fit": 0.02,
        "commercial_addon": 0.00,
        "risk": -0.15,
    },
    "couple": {
        "preference": 0.18,
        "group_fit": 0.04,
        "route": 0.16,
        "budget": 0.08,
        "availability": 0.13,
        "experience": 0.20,
        "time": 0.04,
        "atmosphere": 0.17,
        "novelty": 0.00,
        "weather_fit": 0.02,
        "commercial_addon": 0.00,
        "risk": -0.15,
    },
    "low_budget": {
        "preference": 0.07,
        "group_fit": 0.12,
        "route": 0.18,
        "budget": 0.34,
        "availability": 0.13,
        "experience": 0.07,
        "time": 0.04,
        "atmosphere": 0.00,
        "novelty": 0.02,
        "weather_fit": 0.03,
        "commercial_addon": 0.03,
        "risk": -0.15,
    },
    "solo": {
        "preference": 0.17,
        "group_fit": 0.04,
        "route": 0.20,
        "budget": 0.16,
        "availability": 0.16,
        "experience": 0.17,
        "time": 0.04,
        "atmosphere": 0.03,
        "novelty": 0.03,
        "weather_fit": 0.03,
        "commercial_addon": 0.00,
        "risk": -0.15,
    },
}


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


def _default_weights(scene_type: str) -> dict[str, float]:
    return dict(DEFAULT_SCENE_WEIGHTS.get(scene_type, DEFAULT_SCENE_WEIGHTS["family"]))


@lru_cache(maxsize=8)
def _load_scene_weights_from_policy(policy_path_key: str) -> dict[str, dict[str, float]]:
    policy = _load_policy_config(policy_path_key)
    scene_weights = policy.get("scene_weights")
    if not isinstance(scene_weights, dict):
        return {}

    normalized_weights = {}
    for scene_name, raw_weights in scene_weights.items():
        if not isinstance(raw_weights, dict):
            continue
        scene_key = normalize_scene_type(str(scene_name).strip())
        merged = _default_weights(scene_key)
        for key in WEIGHT_KEYS:
            if key in raw_weights:
                merged[key] = to_float(raw_weights.get(key), merged[key])
        normalized_weights[scene_key] = merged
    return normalized_weights


@lru_cache(maxsize=8)
def _load_score_thresholds_from_policy(policy_path_key: str) -> dict[str, dict[str, float]]:
    policy = _load_policy_config(policy_path_key)
    raw_thresholds = policy.get("score_thresholds")
    if not isinstance(raw_thresholds, dict):
        raw_thresholds = {}

    normalized = {}
    for section_name, defaults in DEFAULT_SCORE_THRESHOLDS.items():
        merged = dict(defaults)
        raw_section = raw_thresholds.get(section_name)
        if isinstance(raw_section, dict):
            for key, default_value in defaults.items():
                if key in raw_section:
                    merged[key] = to_float(raw_section.get(key), default_value)
        normalized[section_name] = merged
    return normalized


@lru_cache(maxsize=8)
def _load_penalties_from_policy(policy_path_key: str) -> dict[str, float]:
    policy = _load_policy_config(policy_path_key)
    raw_penalties = policy.get("penalties")
    if not isinstance(raw_penalties, dict):
        raw_penalties = {}

    merged = dict(DEFAULT_PENALTIES)
    for key, default_value in DEFAULT_PENALTIES.items():
        if key in raw_penalties:
            merged[key] = to_float(raw_penalties.get(key), default_value)
    return merged


def _get_threshold(section: str, key: str, default: float) -> float:
    thresholds = _load_score_thresholds_from_policy(_policy_cache_key())
    section_values = thresholds.get(section, {})
    return to_float(section_values.get(key), default)


def _get_penalty(name: str, default: float) -> float:
    penalties = _load_penalties_from_policy(_policy_cache_key())
    return to_float(penalties.get(name), default)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        token = str(value).strip()
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def _semantic_values_cache_key(values: list | set | tuple | str | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, set):
        raw_values = sorted(values)
    elif isinstance(values, (list, tuple)):
        raw_values = values
    else:
        raw_values = [values]
    return tuple(str(value).strip() for value in raw_values if str(value).strip())


def _semantic_tag_set(values: list | set | tuple | str | None) -> set[str]:
    """Chinese-first terms plus legacy canonical indexes for B scoring."""

    cache_key = _semantic_values_cache_key(values)
    cached = _SEMANTIC_TAG_SET_CACHE.get(cache_key)
    if cached is not None:
        return set(cached)

    result = set(
        _dedupe_keep_order(
            b_semantic_terms(list(cache_key), include_auxiliary=True)
            + expand_preference_tags(list(cache_key))
        )
    )
    if len(_SEMANTIC_TAG_SET_CACHE) >= _SEMANTIC_TAG_SET_CACHE_LIMIT:
        _SEMANTIC_TAG_SET_CACHE.clear()
    _SEMANTIC_TAG_SET_CACHE[cache_key] = result
    return set(result)


def _build_ai_planning_review(selected: dict, plan_critic_metadata: dict | None) -> dict | None:
    """Compact LongCat critic output into a planner-facing review object."""

    if not plan_critic_metadata or not plan_critic_metadata.get("enabled"):
        return None

    plan = selected.get("plan", {}) or {}
    selected_plan_id = str(plan.get("plan_id") or "").strip()
    applied_adjustments = plan_critic_metadata.get("applied_adjustments") or []
    selected_adjustment = next(
        (
            adjustment
            for adjustment in applied_adjustments
            if str(adjustment.get("plan_id") or "").strip() == selected_plan_id
        ),
        None,
    )

    global_notes = [
        str(note).strip()
        for note in (plan_critic_metadata.get("global_notes") or [])
        if str(note).strip()
    ][:4]
    selected_reasons = []
    selected_evidence = []
    if selected_adjustment:
        selected_reasons = [
            str(reason).strip()
            for reason in (selected_adjustment.get("reasons") or [])
            if str(reason).strip()
        ][:5]
        selected_evidence = [
            str(item).strip()
            for item in (selected_adjustment.get("evidence") or [])
            if str(item).strip()
        ][:5]

    review_text = " ".join(global_notes + selected_reasons + selected_evidence).lower()
    risk_delta = to_float((selected_adjustment or {}).get("risk_delta"), 0.0)
    score_delta = to_float((selected_adjustment or {}).get("score_delta"), 0.0)
    confidence = to_float((selected_adjustment or {}).get("confidence"), 0.0)
    term_trigger = any(term.lower() in review_text for term in AI_REPLAN_TRIGGER_TERMS)
    strong_negative_adjustment = confidence >= 0.6 and (risk_delta >= 0.03 or score_delta <= -0.02)
    needs_replan = bool(plan_critic_metadata.get("success") and (term_trigger or strong_negative_adjustment))
    replan_guidance = (selected_reasons or global_notes)[:3] if needs_replan else []

    return {
        "enabled": bool(plan_critic_metadata.get("enabled")),
        "success": bool(plan_critic_metadata.get("success")),
        "provider": plan_critic_metadata.get("provider"),
        "model": plan_critic_metadata.get("model"),
        "selected_after_critic": plan_critic_metadata.get("selected_after_critic"),
        "selected_adjustment": selected_adjustment or {},
        "global_notes": global_notes,
        "needs_replan": needs_replan,
        "replan_trigger": {
            "term_trigger": term_trigger,
            "strong_negative_adjustment": strong_negative_adjustment,
            "risk_delta": round(risk_delta, 4),
            "score_delta": round(score_delta, 4),
            "confidence": round(confidence, 3),
        },
        "replan_guidance": replan_guidance,
        "guardrails": plan_critic_metadata.get("guardrails", {}),
    }


def _build_b_replan_request(
    selected_plan: dict,
    plan_base: dict,
    ai_planning_review: dict | None,
    constraints: dict,
) -> dict | None:
    """Create a bounded request for a future B replan loop."""

    if not ai_planning_review or not ai_planning_review.get("needs_replan"):
        return None

    planning_preferences = constraints.get("planning_preferences") or {}
    guidance = ai_planning_review.get("replan_guidance", [])
    global_notes = ai_planning_review.get("global_notes", [])
    guidance_text = " ".join(str(item) for item in guidance + global_notes).lower()
    supply_identity = selected_plan.get("supply_identity", {})
    target_domains = []
    if any(term in guidance_text for term in ("activity", "museum", "poi", "活动", "博物馆", "场所")):
        target_domains.append("activity")
    if any(term in guidance_text for term in ("restaurant", "hotpot", "餐厅", "火锅", "吃")):
        target_domains.append("restaurant")
    if not target_domains:
        target_domains = ["activity", "restaurant"]

    rag_target_nodes = []
    for domain in target_domains:
        avoid_poi_ids = []
        if domain == "activity" and supply_identity.get("activity_id"):
            avoid_poi_ids.append(supply_identity.get("activity_id"))
        if domain == "restaurant" and supply_identity.get("restaurant_id"):
            avoid_poi_ids.append(supply_identity.get("restaurant_id"))
        rag_target_nodes.append(
            {
                "supply_domain": domain,
                "query_terms": guidance[:3] or global_notes[:3],
                "avoid_poi_ids": avoid_poi_ids,
                "max_candidates": 12,
            }
        )

    return {
        "source": "longcat_plan_critic",
        "status": "needs_replan",
        "next_step": "rerun_candidate_generation",
        "reason": "LongCat critic found plan-level experience or coherence risks before execution.",
        "rejected_plan_id": selected_plan.get("plan_id"),
        "rejected_candidate_id": plan_base.get("plan_id"),
        "preserve_constraints": {
            "scene_type": selected_plan.get("scene_type"),
            "people_count": selected_plan.get("people_count"),
            "budget": constraints.get("budget"),
            "max_distance_km": constraints.get("max_distance_km"),
            "max_queue_time_min": constraints.get("max_queue_time_min")
            or constraints.get("max_queue_time"),
            "duration_range": constraints.get("duration_range"),
            "planning_preferences": planning_preferences,
            "b_requirement_contract": constraints.get("b_requirement_contract", {}),
        },
        "guidance": guidance,
        "global_notes": global_notes,
        "critic_trigger": ai_planning_review.get("replan_trigger", {}),
        "rag_request": {
            "request_type": "replacement_poi_candidates",
            "target_nodes": rag_target_nodes,
            "expected_output_key": "b_rag_candidate_evidence",
            "compatible_output_keys": ["b_rag_candidate_evidence", "b_rag_node_candidates"],
            "contract": "Return b_rag_candidate_evidence.node_evidence[].candidates[] or node-keyed b_rag_node_candidates compatible with src.nodes.b_rag_contract.normalize_rag_node_candidates",
        },
        "candidate_generation_hints": {
            "avoid_plan_ids": [plan_base.get("plan_id")] if plan_base.get("plan_id") else [],
            "avoid_supply_identity": supply_identity,
            "prefer_terms_from_guidance": guidance,
        },
    }


def _activity_signal_set(activity: dict | None, activity_tags: list | None = None) -> set[str]:
    signals = activity_child_signal_set(activity)
    if activity_tags is not None:
        signals.update(_semantic_tag_set(activity_tags))
    return signals


def _canonical_preference_tokens(values: list[str]) -> list[str]:
    tokens: list[str] = []
    for raw_value in values or []:
        raw_token = str(raw_value).strip()
        tokens.extend(b_semantic_terms(raw_token, include_auxiliary=True))
        expanded = expand_preference_tags(raw_token)
        mapped_tokens = [token for token in expanded if token != raw_token]
        tokens.extend(mapped_tokens or expanded)
    return _dedupe_keep_order(tokens)


def _derive_weights(scene_type: str, constraints: dict | None = None) -> dict[str, float]:
    """Get scene-type specific weights for 7-dimensional objective vector."""
    scene_type = normalize_scene_type(scene_type)
    policy_weights = _load_scene_weights_from_policy(_policy_cache_key())
    resolved = dict(policy_weights.get(scene_type, _default_weights(scene_type)))
    overrides = (constraints or {}).get("score_weights", {}) or {}

    if isinstance(overrides, dict):
        alias_map = {"health": "group_fit"}
        for key, value in overrides.items():
            mapped_key = alias_map.get(key, key)
            if mapped_key in resolved:
                resolved[mapped_key] = round(to_float(value, resolved[mapped_key]), 3)

    return resolved


def _score_preference(preference_sources: list[str], tags: list[str]) -> float:
    """Score preference match (0-1) independently of other plans."""
    preference_tokens = _canonical_preference_tokens(preference_sources or [])
    tag_tokens = list(_semantic_tag_set(tags or []))
    if not preference_tokens or not tag_tokens:
        return 0.5

    tag_set = set(tag_tokens)
    matches = 0
    for token in preference_tokens:
        related = RELATED_PREFERENCE_TAGS.get(token, set())
        if (
            token in tag_set
            or tag_set.intersection(related)
            or safe_match_count([token], tag_tokens) > 0
        ):
            matches += 1
    return min(1.0, matches / max(1, len(preference_tokens)) * 0.8 + 0.2)


def _score_group_fit(
    activity_tags: list,
    restaurant_tags: list,
    child_age: int | None,
    mom_diet: str | None,
    scene_type: str,
    activity: dict | None = None,
    restaurant: dict | None = None,
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
            if is_child_compatible_activity(activity, child_age):
                score += 0.5
        else:
            score += 0.3
    elif scene_type == "friends":
        max_points += 0.5
        social_signals = _semantic_tag_set(activity_tags) | _semantic_tag_set(restaurant_tags)
        if social_signals.intersection(FRIENDS_FIT_TAGS):
            score += 0.5
        elif "kid_friendly" in social_signals and not social_signals.intersection({"social", "group_friendly"}):
            score += 0.1
        else:
            score += 0.25
    elif scene_type == "couple":
        max_points += 0.5
        couple_signals = _semantic_tag_set(activity_tags) | _semantic_tag_set(restaurant_tags)
        if couple_signals.intersection(COUPLE_FIT_TAGS):
            score += 0.5
        else:
            score += 0.2
    elif scene_type == "low_budget":
        max_points += 0.5
        budget_signals = _semantic_tag_set(activity_tags) | _semantic_tag_set(restaurant_tags)
        if budget_signals.intersection(BUDGET_FIT_TAGS):
            score += 0.5
        else:
            score += 0.2

    max_points += 0.5
    if mom_diet == "low_calorie":
        health_signals = _restaurant_health_signals(restaurant, restaurant_tags)
        if health_signals.intersection(HEALTH_MATCH_TAGS):
            score += 0.5
    else:
        score += 0.3

    return min(1.0, score / max(1.0, max_points)) if max_points > 0 else 0.5


def _score_route(distance_km: float, travel_time_min: float) -> float:
    """
    Score route quality (0-1) using absolute boundaries.
    1.0 = nearby and short; 0.0 = too far or too long.
    """
    max_distance = _get_threshold("route", "absolute_max_distance_km", ABSOLUTE_MAX_DISTANCE_KM)
    minutes_per_km = _get_threshold("route", "travel_minutes_per_km", 6.0)
    distance_score = normalize(distance_km, 0, max_distance)
    time_score = normalize(travel_time_min, 0, max_distance * minutes_per_km)
    return max(0.0, 1.0 - 0.5 * distance_score - 0.5 * time_score)


def _score_budget(total_price: float, user_budget: float, scene_type: str) -> float:
    """
    Score budget appropriateness (0-1) using absolute boundaries.
    Considers scene-type specific preferences.
    """
    total_price = to_float(total_price, 0.0)
    user_budget = to_float(user_budget, 500.0)

    over_budget_ratio = _get_threshold("budget", "over_budget_hard_ratio", 1.2)

    if total_price > user_budget * over_budget_ratio:
        return 0.0

    if scene_type == "low_budget":
        return max(0.0, 1.0 - total_price / max(1.0, user_budget * over_budget_ratio))

    if total_price <= user_budget:
        utilization = total_price / max(1.0, user_budget)
        return min(1.0, 0.6 + 0.4 * utilization)

    return max(0.0, 1.0 - (total_price - user_budget) / max(1.0, user_budget) * 2)


def _score_time_fit(estimated_duration_min: float, duration_range: list[int], route: dict | None = None) -> float:
    """Score pace and time fit against the requested duration window."""

    route = route or {}
    duration = to_float(estimated_duration_min, 0.0)
    if not duration_range or len(duration_range) < 2:
        return 0.6

    lower = to_float(duration_range[0], 240.0)
    upper = to_float(duration_range[1], 360.0)
    if upper < lower:
        lower, upper = upper, lower

    if lower <= duration <= upper:
        window_midpoint = (lower + upper) / 2.0
        half_window = max(1.0, (upper - lower) / 2.0)
        duration_score = 1.0 - min(1.0, abs(duration - window_midpoint) / half_window) * 0.25
    else:
        nearest = lower if duration < lower else upper
        duration_score = max(0.0, 1.0 - abs(duration - nearest) / max(60.0, upper - lower))

    total_travel_time = to_float(route.get("total_travel_time_min"), 0.0)
    travel_share = total_travel_time / max(1.0, duration)
    travel_score = max(0.0, 1.0 - min(1.0, travel_share / 0.30))

    return min(1.0, 0.70 * duration_score + 0.30 * travel_score)


def _score_availability(all_available: bool, queue_time_min: float) -> float:
    """Score availability (0-1) using absolute criteria."""
    if not all_available:
        return 0.1

    max_queue_time = _get_threshold("availability", "absolute_max_queue_time_min", ABSOLUTE_MAX_QUEUE_TIME_MIN)
    queue_score = normalize(queue_time_min, 0, max_queue_time)
    return 0.5 + 0.5 * (1.0 - queue_score)


def _score_experience(
    activity_rating: float,
    restaurant_rating: float,
    tags: list,
    activity: dict | None = None,
    restaurant: dict | None = None,
) -> float:
    """Score experience quality (0-1) independently."""
    activity = activity or {}
    restaurant = restaurant or {}
    rating = to_float((activity_rating + restaurant_rating) / 2.0, 4.0)
    min_rating = _get_threshold("experience", "min_rating", ABSOLUTE_MIN_RATING)
    max_rating = _get_threshold("experience", "max_rating", ABSOLUTE_MAX_RATING)
    tag_diversity_cap = _get_threshold("experience", "tag_diversity_cap", 8.0)
    rating_score = normalize(rating, min_rating, max_rating)
    tag_diversity = min(1.0, len(set(tags)) / max(1.0, tag_diversity_cap))
    trust_score = (to_float(activity.get("trust_score"), 0.7) + to_float(restaurant.get("trust_score"), 0.7)) / 2.0
    ritual_score = (to_float(activity.get("ritual_score"), 0.5) + to_float(restaurant.get("ritual_score"), 0.5)) / 2.0
    stability_score = to_float(restaurant.get("operation_stability_score"), 0.8)
    return min(
        1.0,
        0.45 * rating_score
        + 0.20 * tag_diversity
        + 0.20 * trust_score
        + 0.10 * ritual_score
        + 0.05 * stability_score,
    )


def _score_atmosphere(
    scene_type: str,
    tags: list,
    activity: dict | None,
    restaurant: dict | None,
    preference_sources: list[str],
) -> float:
    """Score scene-specific soft fit such as date atmosphere or local culture."""

    activity = activity or {}
    restaurant = restaurant or {}
    scene_type = normalize_scene_type(scene_type)
    tag_set = _semantic_tag_set(tags or [])
    preference_set = _semantic_tag_set(preference_sources or [])
    activity_category = str(activity.get("category") or activity.get("experience_type") or "")
    restaurant_category = str(restaurant.get("restaurant_category") or restaurant.get("category") or "")

    if preference_set.intersection(LOCAL_CULTURE_TAGS):
        score = 0.25
        if tag_set.intersection(LOCAL_CULTURE_TAGS):
            score += 0.40
        if activity_category in {"citywalk", "museum", "local_market"}:
            score += 0.25
        if activity_category == "escape_room" and "escape_room" not in preference_set:
            score -= 0.30
        if restaurant_category in {"regional_home_cuisine", "local_cuisine"}:
            score += 0.10
        return max(0.0, min(1.0, score))

    if scene_type == "couple":
        date_signals = {"romantic", "date_friendly", "atmosphere"}
        ambience_signals = {"quiet", "photogenic", "cultural", "healing", "relaxation", "ritual"}
        soft_signals = {"warm", "creative"}
        score = 0.22
        score += min(0.24, len(tag_set.intersection(date_signals)) * 0.12)
        score += min(0.42, len(tag_set.intersection(ambience_signals)) * 0.14)
        score += min(0.08, len(tag_set.intersection(soft_signals)) * 0.04)
        if preference_set.intersection({"atmosphere", "romantic", "date_friendly", "relaxation", "healing"}):
            score += 0.10 if tag_set.intersection(date_signals | ambience_signals) else -0.10
        if activity_category in {"micro_vacation", "museum"}:
            score += 0.18
        elif activity_category == "handcraft":
            score += 0.06
        if activity_category == "escape_room" and "escape_room" not in preference_set:
            score -= 0.10
        return max(0.0, min(1.0, score))

    if scene_type == "friends":
        social_signals = FRIENDS_FIT_TAGS | {"immersive", "story_driven", "chat_friendly"}
        score = 0.35 + min(0.45, len(tag_set.intersection(social_signals)) * 0.12)
        if preference_set.intersection({"social", "group_friendly"}) and not tag_set.intersection(social_signals):
            score -= 0.15
        return max(0.0, min(1.0, score))

    if scene_type == "family":
        family_signals = {"kid_friendly", "family_friendly", "low_intensity", "indoor", "warm", "educational"}
        return min(1.0, 0.35 + len(tag_set.intersection(family_signals)) * 0.10)

    if scene_type == "low_budget":
        return min(1.0, 0.35 + len(tag_set.intersection(BUDGET_FIT_TAGS)) * 0.15)

    return min(1.0, 0.40 + len(tag_set.intersection(ATMOSPHERE_TAGS | NOVELTY_TAGS)) * 0.08)


def _score_novelty(tags: list, activity: dict | None, restaurant: dict | None) -> float:
    """Score freshness using supply-side signals available in the current mock data."""

    activity = activity or {}
    restaurant = restaurant or {}
    tag_set = _semantic_tag_set(tags or [])
    score = 0.35 + min(0.40, len(tag_set.intersection(NOVELTY_TAGS)) * 0.08)
    activity_category = str(activity.get("category") or "")
    if activity_category in {"citywalk", "museum", "micro_vacation", "local_market", "escape_room", "handcraft"}:
        score += 0.12
    if str(restaurant.get("restaurant_category") or "") in {"regional_home_cuisine", "japanese_light_food", "hotpot"}:
        score += 0.06
    return max(0.0, min(1.0, score))


def _score_commercial_addon(activity: dict | None, restaurant: dict | None) -> float:
    """Score optional product/deal richness without letting commerce dominate."""

    activity = activity or {}
    restaurant = restaurant or {}
    items = [activity, restaurant]
    deal_count = sum(len(item.get("deals", []) or []) for item in items)
    product_count = sum(len(item.get("products", []) or []) for item in items)
    has_coupon = any("coupon_available" in set(expand_preference_tags(item.get("tags", []) or [])) for item in items)
    score = 0.40
    score += min(0.30, product_count * 0.08)
    score += min(0.20, deal_count * 0.08)
    if has_coupon:
        score += 0.10
    return max(0.0, min(1.0, score))


def _has_health_food_intent(preference_sources: list[str]) -> bool:
    preference_tokens = set(_canonical_preference_tokens(preference_sources or []))
    return bool(preference_tokens.intersection(HEALTH_MATCH_TAGS | {"healthy"}))


def _has_light_food_intent(preference_sources: list[str]) -> bool:
    preference_tokens = set(_canonical_preference_tokens(preference_sources or []))
    return bool(preference_tokens.intersection({"light_food", "low_calorie", "healthy"}))


def _restaurant_health_signals(restaurant: dict | None, restaurant_tags: list | None = None) -> set[str]:
    restaurant = restaurant or {}
    health_signals = _semantic_tag_set(restaurant_tags or restaurant.get("tags", []) or [])
    health_signals.update(_semantic_tag_set(restaurant.get("health_tags", []) or []))
    health_signals.update(_semantic_tag_set(restaurant.get("menu_health_options", []) or []))
    health_signals.update(_semantic_tag_set(restaurant.get("restaurant_category") or ""))
    return health_signals


def _is_light_food_restaurant(restaurant: dict | None, restaurant_tags: list | None = None) -> bool:
    restaurant = restaurant or {}
    category = str(restaurant.get("restaurant_category") or restaurant.get("category") or "")
    category_signals = {"轻食", "沙拉轻食", "日料轻食", "素食轻食", "light_food", "salad_light_food", "japanese_light_food", "vegetarian_light_food"}
    if category in category_signals:
        return True
    if has_item_semantic_group(restaurant, "轻食"):
        return True
    direct_tags = _semantic_tag_set(restaurant_tags or restaurant.get("tags", []) or [])
    direct_tags.update(_semantic_tag_set(restaurant.get("health_tags", []) or []))
    return bool(direct_tags.intersection({"light_food", "low_calorie", "salad_light_food", "japanese_light_food"}))


OUTDOOR_ACTIVITY_CATEGORIES = {"citywalk", "local_market", "sports"}
INDOOR_SAFE_TAGS = {"indoor", "museum", "handcraft", "indoor_playground", "escape_room"}


def _weather_tags(weather_context: dict | None) -> set[str]:
    weather_context = weather_context or {}
    tags = set(str(tag) for tag in weather_context.get("condition_tags", []) or [])
    tags.update(str(tag) for tag in weather_context.get("risk_tags", []) or [])
    return tags


def _activity_weather_profile(activity: dict | None, activity_tags: list | None = None) -> dict:
    activity = activity or {}
    tags = set(expand_preference_tags(activity_tags or activity.get("tags", []) or []))
    tags.update(str(tag).strip() for tag in activity.get("tags", []) or [])
    category = str(activity.get("category") or activity.get("experience_type") or "")
    sensitivity = str(activity.get("weather_sensitivity") or "").strip().lower()
    indoor_safe = (
        sensitivity == "indoor_safe"
        or bool(activity.get("indoor_backup"))
        or bool(tags.intersection(INDOOR_SAFE_TAGS))
        or category in {"museum", "handcraft", "indoor_playground", "escape_room", "micro_vacation"}
    )
    outdoor_like = category in OUTDOOR_ACTIVITY_CATEGORIES or "outdoor" in tags
    return {
        "category": category,
        "sensitivity": sensitivity,
        "indoor_safe": indoor_safe,
        "outdoor_like": outdoor_like,
        "has_indoor_backup": bool(activity.get("indoor_backup")),
    }


def _score_weather_fit(activity: dict | None, activity_tags: list | None, weather_context: dict | None) -> float:
    if not weather_context or not weather_context.get("available"):
        return 0.6

    profile = _activity_weather_profile(activity, activity_tags)
    weather_tags = _weather_tags(weather_context)
    prefer_indoor = bool(weather_context.get("prefer_indoor"))

    score = 0.72
    if prefer_indoor:
        if profile["indoor_safe"]:
            score = 0.95
        elif profile["sensitivity"] == "medium":
            score = 0.55
        elif profile["sensitivity"] == "high":
            score = 0.28
        if profile["outdoor_like"] and not profile["has_indoor_backup"]:
            score -= 0.20

    if "hot" in weather_tags:
        if profile["indoor_safe"]:
            score += 0.05
        if profile["category"] in {"sports", "citywalk"} and not profile["has_indoor_backup"]:
            score -= 0.25

    if "comfortable" in weather_tags and profile["category"] in OUTDOOR_ACTIVITY_CATEGORIES:
        score = max(score, 0.88)

    return max(0.0, min(1.0, score))


def _weather_risk_factors(
    activity: dict | None,
    activity_tags: list | None,
    weather_context: dict | None,
) -> tuple[float, list[str]]:
    if not weather_context or not weather_context.get("available"):
        return 0.0, []

    weather_fit = _score_weather_fit(activity, activity_tags, weather_context)
    if weather_fit >= 0.7:
        return 0.0, []

    weather = weather_context.get("weather") or ",".join(weather_context.get("condition_tags", []) or [])
    profile = _activity_weather_profile(activity, activity_tags)
    factors = [
        f"Weather risk: {weather} may affect {profile['category'] or 'activity'}"
    ]
    return round((0.7 - weather_fit) * 0.35, 3), factors


def _calc_risk_factors(
    distance_km: float,
    travel_time_min: float,
    queue_time_min: float,
    activity_rating: float,
    restaurant_rating: float,
    tags: list,
    route: dict | None = None,
) -> tuple[float, list[str]]:
    """
    Calculate risk score (0-1, lower is better) and identify specific risk factors.
    Uses absolute criteria, not dependent on plan collection statistics.
    """
    risk_factors = []
    risk_score = 0.0
    route = route or {}
    max_distance = _get_threshold("route", "absolute_max_distance_km", ABSOLUTE_MAX_DISTANCE_KM)
    distance_warning_ratio = _get_threshold("route", "distance_warning_ratio", 0.8)
    minutes_per_km = _get_threshold("route", "travel_minutes_per_km", 6.0)
    max_queue_time = _get_threshold("availability", "absolute_max_queue_time_min", ABSOLUTE_MAX_QUEUE_TIME_MIN)
    queue_warning_ratio = _get_threshold("availability", "queue_warning_ratio", 0.7)
    low_rating_warning = _get_threshold("experience", "low_rating_warning", 4.2)

    if distance_km > max_distance * distance_warning_ratio:
        risk_score += _get_penalty("far_distance", 0.3)
        risk_factors.append(f"距离较远 ({distance_km:.1f} 公里)")

    travel_warning_min = max_distance * minutes_per_km * distance_warning_ratio
    if travel_time_min > travel_warning_min:
        risk_score += _get_penalty("far_distance", 0.3) * 0.6
        risk_factors.append(f"路上时间较长 ({travel_time_min:.0f} 分钟)")

    traffic_status = str(route.get("traffic_status") or "").lower()
    if traffic_status in {"high", "heavy", "severe"}:
        risk_score += 0.18
        risk_factors.append("交通状态偏紧张")

    if queue_time_min > max_queue_time * queue_warning_ratio:
        risk_score += _get_penalty("long_queue", 0.25)
        risk_factors.append(f"可能排队较长 ({queue_time_min:.0f} 分钟)")

    avg_rating = (activity_rating + restaurant_rating) / 2.0
    if avg_rating < low_rating_warning:
        risk_score += _get_penalty("low_rating", 0.2)
        risk_factors.append(f"评分不够高 ({avg_rating:.1f})")

    if "crowded_mall" in tags:
        risk_score += _get_penalty("crowded_mall", 0.15)
        risk_factors.append("可能人流较多")

    return min(1.0, risk_score), risk_factors


def _build_timeline(
    activity: dict,
    restaurant: dict,
    start_hour: int = 14,
    start_minute: int = 30,
    *,
    scene_type: str = "family",
    child_age: int | None = None,
) -> list[dict]:
    """Build detailed timeline with activity, transition, restaurant."""
    schedule = activity.get("_selected_schedule", {}) or {}
    restaurant_health_signals = _restaurant_health_signals(restaurant, restaurant.get("tags", []) or [])
    activity_signals = _activity_signal_set(activity, activity.get("tags", []) or [])
    activity_child_compatible = scene_type == "family" and is_child_compatible_activity(
        activity,
        child_age,
    )
    activity_start_str = schedule.get("activity_start")
    activity_end_str = schedule.get("activity_end")
    restaurant_start_str = schedule.get("restaurant_start")

    def format_time(hour: int, minute: int) -> str:
        return f"{hour:02d}:{minute:02d}"

    if schedule.get("sequence") == "restaurant_then_activity":
        restaurant_start_str = schedule.get("restaurant_start")
        restaurant_end_str = schedule.get("restaurant_end")
        if restaurant_start_str and ":" in restaurant_start_str:
            restaurant_start_hour, restaurant_start_minute = [
                int(x) for x in restaurant_start_str.split(":", 1)
            ]
        else:
            restaurant_start_hour, restaurant_start_minute = start_hour, start_minute

        restaurant_start_total = restaurant_start_hour * 60 + restaurant_start_minute
        restaurant_end_total = restaurant_start_total + restaurant.get("duration_min", 0)
        if restaurant_end_str and ":" in restaurant_end_str:
            restaurant_end_hour, restaurant_end_minute = [
                int(x) for x in restaurant_end_str.split(":", 1)
            ]
            restaurant_end_total = restaurant_end_hour * 60 + restaurant_end_minute
        restaurant_end_hour, restaurant_end_minute = divmod(restaurant_end_total, 60)

        if activity_start_str and ":" in activity_start_str:
            activity_start_hour, activity_start_minute = [
                int(x) for x in activity_start_str.split(":", 1)
            ]
            transition_end_total = activity_start_hour * 60 + activity_start_minute
        else:
            transition_end_total = restaurant_end_total + 30
        transition_buffer_min = max(0, transition_end_total - restaurant_end_total)
        transition_end_hour, transition_end_minute = divmod(transition_end_total, 60)

        activity_start_hour, activity_start_minute = divmod(transition_end_total, 60)
        activity_end_total = (
            activity_start_hour * 60
            + activity_start_minute
            + activity.get("duration_min", 0)
        )
        if activity_end_str and ":" in activity_end_str:
            activity_end_hour, activity_end_minute = [
                int(x) for x in activity_end_str.split(":", 1)
            ]
            activity_end_total = activity_end_hour * 60 + activity_end_minute
        activity_end_hour, activity_end_minute = divmod(activity_end_total, 60)

        return [
            {
                "time": f"{format_time(restaurant_start_hour, restaurant_start_minute)}-{format_time(restaurant_end_hour, restaurant_end_minute)}",
                "activity": restaurant.get("name"),
                "poi_id": restaurant.get("poi_id"),
                "type": "restaurant",
                "duration_min": restaurant.get("duration_min"),
                "price": restaurant.get("price"),
                "notes": [
                    "低卡/少油选项" if restaurant_health_signals.intersection(HEALTH_MATCH_TAGS) else "普通餐饮",
                    "轻食" if "light_food" in restaurant_health_signals else "口味清淡可备注" if restaurant_health_signals.intersection({"low_oil", "low_sugar", "vegetable_rich"}) else "口味偏重",
                ],
            },
            {
                "time": f"{format_time(restaurant_end_hour, restaurant_end_minute)}-{format_time(transition_end_hour, transition_end_minute)}",
                "activity": "附近休息与转场",
                "poi_id": None,
                "type": "transition",
                "duration_min": transition_buffer_min,
                "price": 0,
                "notes": ["避免行程过满"],
            },
            {
                "time": f"{format_time(activity_start_hour, activity_start_minute)}-{format_time(activity_end_hour, activity_end_minute)}",
                "activity": activity.get("name"),
                "poi_id": activity.get("poi_id"),
                "type": "play",
                "duration_min": activity.get("duration_min"),
                "price": activity.get("price"),
                "notes": [
                    "kid_friendly" if activity_child_compatible else "体验型活动",
                    "低强度" if "low_intensity" in activity_signals else "强度适中",
                ],
            },
        ]

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
                "kid_friendly" if activity_child_compatible else "体验型活动",
                "低强度" if "low_intensity" in activity_signals else "强度适中",
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
                "低卡/少油选项" if restaurant_health_signals.intersection(HEALTH_MATCH_TAGS) else "普通餐饮",
                "轻食" if "light_food" in restaurant_health_signals else "口味清淡可备注" if restaurant_health_signals.intersection({"low_oil", "low_sugar", "vegetable_rich"}) else "口味偏重",
            ],
        },
    ]


def _first_id(value) -> str | None:
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    if value not in (None, ""):
        return str(value)
    return None


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _id_values(value) -> set[str]:
    return {str(item) for item in _as_list(value) if item not in (None, "")}


def _ordered_ids(value) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in _as_list(value):
        if item in (None, ""):
            continue
        item_id = str(item)
        if item_id not in seen:
            seen.add(item_id)
            result.append(item_id)
    return result


def _find_record(records, id_field: str, item_id: str | None) -> dict:
    if not item_id:
        return {}
    for record in _as_list(records):
        if isinstance(record, dict) and str(record.get(id_field, "")) == str(item_id):
            return record
    return {}


def _slot_values(slots) -> set[str]:
    result: set[str] = set()
    for slot in _as_list(slots):
        if isinstance(slot, dict):
            value = slot.get("time")
        else:
            value = slot
        if value not in (None, ""):
            result.add(str(value))
    return result


def _record_requires_reservation(record: dict) -> bool:
    return bool(record.get("requires_reservation") or record.get("reservation_required"))


def _select_supply_ids(node: dict, time: str | None) -> tuple[str | None, str | None]:
    product_ids = _ordered_ids(node.get("product_ids"))
    deal_records = [
        deal
        for deal in _as_list(node.get("deals"))
        if isinstance(deal, dict) and deal.get("deal_id")
    ]
    for deal in deal_records:
        valid_times = _slot_values(deal.get("valid_time"))
        if valid_times and time and str(time) not in valid_times:
            continue
        deal_product_id = str(deal.get("product_id")) if deal.get("product_id") else None
        if product_ids and deal_product_id and deal_product_id not in product_ids:
            continue
        return deal_product_id or (product_ids[0] if product_ids else None), str(deal.get("deal_id"))
    fallback_deal_id = None if deal_records else _first_id(node.get("deal_ids"))
    return (product_ids[0] if product_ids else None), fallback_deal_id


def _build_action_hint(
    action_type: str,
    node: dict,
    time: str | None,
    people_count: int,
    notes: list[str],
    count_field: str,
) -> dict:
    product_id, deal_id = _select_supply_ids(node, time)
    product = _find_record(node.get("products"), "product_id", product_id)
    deal = _find_record(node.get("deals"), "deal_id", deal_id)

    hint = {
        "action_type": action_type,
        "poi_id": node.get("poi_id"),
        "merchant_id": node.get("merchant_id"),
        "product_id": product_id,
        "deal_id": deal_id,
        "time": time,
        count_field: people_count,
        "notes": notes,
        "requires_reservation": bool(
            node.get("reservation_required")
            or _record_requires_reservation(product)
            or _record_requires_reservation(deal)
        ),
    }
    if node.get("restaurant_role"):
        hint["restaurant_role"] = node.get("restaurant_role")
    for field in ("product_type", "inventory_model", "fulfillment_mode"):
        if product.get(field):
            hint[field] = product.get(field)
    if deal.get("deal_type"):
        hint["deal_type"] = deal.get("deal_type")
    if deal.get("coupon_type"):
        hint["coupon_type"] = deal.get("coupon_type")
    return hint


def _build_lodging_action_hint(node: dict, time: str | None, people_count: int, notes: list[str]) -> dict:
    return {
        "action_type": "reserve_lodging",
        "poi_id": node.get("poi_id"),
        "merchant_id": node.get("merchant_id"),
        "time": time or "15:00",
        "check_in_date": node.get("check_in_date") or "2026-06-01",
        "check_out_date": node.get("check_out_date") or "2026-06-02",
        "room_count": int(to_float(node.get("room_count"), 1.0)) or 1,
        "people_count": people_count,
        "room_type": node.get("room_type"),
        "notes": notes,
        "requires_reservation": True,
    }


def _build_single_node_action_hints(plan_base: dict, timeline: list[dict], people_count: int) -> list[dict]:
    nodes = plan_base.get("nodes", []) or []
    if not nodes:
        return []
    node = nodes[0]
    timeline_item = next(
        (item for item in timeline if item.get("poi_id") == node.get("poi_id")),
        timeline[0] if timeline else {},
    )
    action_time = str(timeline_item.get("time") or "17:00").split("-")[0]
    if node.get("type") == "restaurant" or plan_base.get("plan_shape") in {"restaurant_only", "cafe_only"}:
        return [
            _build_action_hint(
                "reserve_restaurant",
                node,
                action_time,
                people_count,
                ["single_node_plan"],
                "people",
            )
        ]
    if node.get("type") == "activity" or plan_base.get("plan_shape") == "activity_only":
        return [
            _build_action_hint(
                "order_activity_ticket",
                node,
                action_time,
                people_count,
                ["single_node_plan"],
                "quantity",
            )
        ]
    if node.get("type") in {"hotel", "lodging"} or node.get("itinerary_role") == "lodging":
        return [
            _build_lodging_action_hint(
                node,
                action_time,
                people_count,
                ["single_node_plan"],
            )
        ]
    return []


def _add_contract_check(checks: list[dict], blocking_reasons: list[str], name: str, ok: bool, message: str) -> None:
    checks.append({"name": name, "status": "pass" if ok else "fail", "message": message})
    if not ok:
        blocking_reasons.append(message)


def _validate_action_hint(
    *,
    checks: list[dict],
    blocking_reasons: list[str],
    role: str,
    hint: dict,
    node: dict,
    people_count: int,
    count_field: str,
) -> None:
    poi_id = hint.get("poi_id")
    merchant_id = hint.get("merchant_id")
    product_id = hint.get("product_id")
    deal_id = hint.get("deal_id")
    time = hint.get("time")
    action_type = hint.get("action_type")

    _add_contract_check(
        checks,
        blocking_reasons,
        f"{role}_poi_id",
        bool(poi_id) and (not node.get("poi_id") or poi_id == node.get("poi_id")),
        f"{role} action must target the selected poi_id",
    )
    _add_contract_check(
        checks,
        blocking_reasons,
        f"{role}_merchant_id",
        action_type == "reserve_lodging"
        or (bool(merchant_id) and (not node.get("merchant_id") or merchant_id == node.get("merchant_id"))),
        f"{role} action must target the selected merchant_id",
    )
    _add_contract_check(
        checks,
        blocking_reasons,
        f"{role}_product_or_deal",
        action_type == "reserve_lodging" or bool(product_id or deal_id),
        f"{role} action must include product_id or deal_id",
    )

    product_ids = _id_values(node.get("product_ids"))
    deal_ids = _id_values(node.get("deal_ids"))
    product = _find_record(node.get("products"), "product_id", product_id)
    deal = _find_record(node.get("deals"), "deal_id", deal_id)

    if product_id:
        _add_contract_check(
            checks,
            blocking_reasons,
            f"{role}_product_id_known",
            (not product_ids or str(product_id) in product_ids) and (not node.get("products") or bool(product)),
            f"{role} product_id must exist in selected supply",
        )
    if deal_id:
        _add_contract_check(
            checks,
            blocking_reasons,
            f"{role}_deal_id_known",
            (not deal_ids or str(deal_id) in deal_ids) and (not node.get("deals") or bool(deal)),
            f"{role} deal_id must exist in selected supply",
        )
    if product_id and deal.get("product_id"):
        _add_contract_check(
            checks,
            blocking_reasons,
            f"{role}_deal_product_match",
            str(deal.get("product_id")) == str(product_id),
            f"{role} deal_id must map to selected product_id",
        )

    slot_values = set()
    slot_values.update(_slot_values(node.get("available_slots")))
    slot_values.update(_slot_values(node.get("reservation_slots")))
    _add_contract_check(
        checks,
        blocking_reasons,
        f"{role}_time_in_slot",
        bool(time) and (not slot_values or str(time) in slot_values),
        f"{role} action time must be in available/reservation slots",
    )
    deal_times = _slot_values(deal.get("valid_time")) if deal else set()
    if deal_times:
        _add_contract_check(
            checks,
            blocking_reasons,
            f"{role}_time_in_deal",
            str(time) in deal_times,
            f"{role} action time must be valid for selected deal",
        )

    count_value = hint.get(count_field)
    _add_contract_check(
        checks,
        blocking_reasons,
        f"{role}_{count_field}",
        count_value == people_count,
        f"{role} action {count_field} must match people_count",
    )

    requires_reservation = bool(
        hint.get("requires_reservation")
        or node.get("reservation_required")
        or _record_requires_reservation(product)
        or _record_requires_reservation(deal)
    )
    _add_contract_check(
        checks,
        blocking_reasons,
        f"{role}_reservation_time",
        not requires_reservation or bool(time),
        f"{role} reservation-required supply must include action time",
    )

    if role == "restaurant":
        _add_contract_check(
            checks,
            blocking_reasons,
            "restaurant_dine_in",
            node.get("dine_in_available") is not False,
            "restaurant reservation requires dine_in_available supply",
        )


def _validate_execution_contract(selected_plan: dict, activity: dict, restaurant: dict, people_count: int) -> dict:
    checks: list[dict] = []
    blocking_reasons: list[str] = []
    hints = selected_plan.get("action_hints", []) or []

    expected_actions = [
        ("activity", "order_activity_ticket", activity, "quantity"),
        ("restaurant", "reserve_restaurant", restaurant, "people"),
    ]
    for role, action_type, node, count_field in expected_actions:
        hint = next((item for item in hints if item.get("action_type") == action_type), {})
        _add_contract_check(
            checks,
            blocking_reasons,
            f"{role}_action_present",
            bool(hint),
            f"{role} action_hints must include {action_type}",
        )
        if hint:
            _validate_action_hint(
                checks=checks,
                blocking_reasons=blocking_reasons,
                role=role,
                hint=hint,
                node=node,
                people_count=people_count,
                count_field=count_field,
            )

    return {
        "ready": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "checks": checks,
    }


def _validate_multinode_execution_contract(selected_plan: dict, nodes: list[dict], people_count: int) -> dict:
    checks: list[dict] = []
    blocking_reasons: list[str] = []
    node_ids = {str(node.get("poi_id")) for node in nodes if node.get("poi_id")}
    hints = selected_plan.get("action_hints", []) or []

    for index, hint in enumerate(hints, start=1):
        action_type = hint.get("action_type")
        poi_id = str(hint.get("poi_id") or "")
        time = hint.get("time")
        if action_type == "reserve_restaurant":
            count_value = hint.get("people")
            count_ok = count_value == people_count
        elif action_type == "order_activity_ticket":
            count_value = hint.get("quantity")
            count_ok = count_value == people_count
        elif action_type == "reserve_lodging":
            count_value = hint.get("room_count")
            count_ok = int(to_float(count_value, 0.0)) >= 1 and hint.get("people_count") == people_count
        else:
            count_value = None
            count_ok = False
        role = f"node_{index}"
        _add_contract_check(
            checks,
            blocking_reasons,
            f"{role}_poi_id",
            bool(poi_id) and poi_id in node_ids,
            f"{role} action must target a selected poi_id",
        )
        _add_contract_check(
            checks,
            blocking_reasons,
            f"{role}_action_type",
            action_type in {"order_activity_ticket", "reserve_restaurant", "reserve_lodging"},
            f"{role} action type must be executable by C",
        )
        _add_contract_check(
            checks,
            blocking_reasons,
            f"{role}_time",
            bool(time),
            f"{role} action must include time",
        )
        _add_contract_check(
            checks,
            blocking_reasons,
            f"{role}_count",
            count_ok,
            f"{role} action count must match people_count",
        )

    _add_contract_check(
        checks,
        blocking_reasons,
        "all_nodes_have_actions",
        len(hints) >= len(
            [
                node
                for node in nodes
                if _node_requires_c_execution(node)
                and _node_is_supported_by_current_c(node)
            ]
        ),
        "multi-node itinerary must expose one execution hint per selected activity/restaurant node",
    )
    non_executable_nodes = [
        {
            "poi_id": node.get("poi_id"),
            "name": node.get("name"),
            "role": node.get("itinerary_role"),
            "supply_domain": node.get("supply_domain"),
        }
        for node in nodes
        if _node_requires_c_execution(node) and not _node_is_supported_by_current_c(node)
    ]
    guidance_only_nodes = [
        {
            "poi_id": node.get("poi_id"),
            "name": node.get("name"),
            "role": node.get("itinerary_role"),
            "supply_domain": node.get("supply_domain"),
        }
        for node in nodes
        if not _node_requires_c_execution(node)
    ]
    _add_contract_check(
        checks,
        blocking_reasons,
        "all_nodes_supported_by_c_execution",
        not non_executable_nodes,
        "multi-node itinerary contains POI nodes that C cannot execute yet",
    )
    return {
        "ready": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "checks": checks,
        "mode": "multi_node_best_effort",
        "execution_scope": "partial" if non_executable_nodes else "full",
        "non_executable_nodes": non_executable_nodes,
        "guidance_only_nodes": guidance_only_nodes,
    }


def _build_multinode_action_hints(plan_base: dict, timeline: list[dict], people_count: int) -> list[dict]:
    nodes_by_id = {
        str(node.get("poi_id")): node
        for node in plan_base.get("nodes", []) or []
        if node.get("poi_id")
    }
    action_hints: list[dict] = []
    for item in timeline:
        poi_id = item.get("poi_id")
        if not poi_id:
            continue
        node = nodes_by_id.get(str(poi_id), {})
        action_time = str(item.get("time") or "").split("-")[0]
        if not _node_requires_c_execution(node):
            continue
        if item.get("type") == "lodging" or node.get("type") in {"hotel", "lodging"} or node.get("itinerary_role") == "lodging":
            action_hints.append(
                _build_lodging_action_hint(
                    node,
                    action_time,
                    people_count,
                    ["multi_node_itinerary"],
                )
            )
        elif item.get("type") == "restaurant" or node.get("type") == "restaurant":
            action_hints.append(
                _build_action_hint(
                    "reserve_restaurant",
                    node,
                    action_time,
                    people_count,
                    ["multi_node_itinerary"],
                    "people",
                )
            )
        elif item.get("type") in {"play", "activity", "amusement", "museum", "art"} or node.get("type") == "activity":
            action_hints.append(
                _build_action_hint(
                    "order_activity_ticket",
                    node,
                    action_time,
                    people_count,
                    ["multi_node_itinerary"],
                    "quantity",
                )
            )
    return action_hints


def _build_multinode_plan_title(plan_base: dict) -> str:
    days = int(plan_base.get("planning_days") or 1)
    if days >= 2:
        return "两天本地生活行程"
    if plan_base.get("planning_horizon") == "full_day":
        return "一日多节点行程"
    return "多节点本地生活行程"


def _plan_identity(plan_base: dict) -> dict:
    nodes = plan_base.get("nodes", []) or []
    activity = next((node for node in nodes if node.get("type") == "activity"), {})
    restaurant = next((node for node in nodes if node.get("type") == "restaurant"), {})
    return {
        "activity_id": activity.get("poi_id"),
        "activity_name": activity.get("name"),
        "activity_category": activity.get("category"),
        "restaurant_id": restaurant.get("poi_id"),
        "restaurant_name": restaurant.get("name"),
        "restaurant_category": restaurant.get("restaurant_category") or restaurant.get("category"),
        "restaurant_role": plan_base.get("restaurant_role") or restaurant.get("restaurant_role"),
    }


def _build_plan_title(
    scene_type: str,
    activity: dict,
    restaurant: dict,
    child_age: int | None,
) -> str:
    """Build a user-facing title from the actual selected supply."""

    if scene_type == "family" or child_age is not None:
        return "轻松亲子下午计划"

    if has_item_semantic_group(activity, "密室桌游") and has_item_semantic_group(restaurant, "火锅"):
        return "桌游火锅朋友聚会计划"
    if has_item_semantic_group(activity, "密室桌游"):
        return "朋友社交游戏计划"
    if has_item_semantic_group(restaurant, "烤肉"):
        return "烤肉聚会轻松计划"
    if has_item_semantic_group(restaurant, "火锅"):
        return "火锅聚会轻松计划"
    if has_item_semantic_group(activity, "博物馆展览") and has_item_semantic_group(restaurant, "咖啡甜品"):
        return "看展咖啡放松计划"
    if has_item_semantic_group(activity, "博物馆展览"):
        return "城市看展放松计划"
    if has_item_semantic_group(restaurant, "咖啡甜品"):
        return "咖啡小坐放松计划"

    if scene_type == "couple":
        return "轻松约会计划"
    if scene_type == "friends":
        return "朋友聚会计划"
    if scene_type == "solo":
        return "一个人轻松探索计划"
    if scene_type == "low_budget":
        return "高性价比周末计划"
    return "周末休闲计划"


def _build_multinode_skeleton_plan(
    *,
    blueprint: dict,
    filter_reasons: dict,
    scene_type: str,
) -> dict:
    """Build a non-executable itinerary skeleton for multi-node requests."""

    timeline: list[dict] = []
    total_duration = 0
    for day in (blueprint.get("time_skeleton") or {}).get("days", []) or []:
        day_index = day.get("day", 1)
        for slot in day.get("slots", []) or []:
            duration = int(slot.get("duration_min") or 0)
            if slot.get("part_of_day") != "overnight":
                total_duration += duration
            timeline.append(
                {
                    "time": f"{slot.get('start_time')}-{slot.get('end_time')}",
                    "activity": slot.get("label") or slot.get("role"),
                    "poi_id": None,
                    "type": "planning_intent",
                    "role": slot.get("role"),
                    "supply_domain": slot.get("supply_domain"),
                    "day": day_index,
                    "duration_min": duration,
                    "price": None,
                    "notes": [
                        "等待 RAG/多城市供给返回候选",
                        "当前不是可执行商家节点",
                    ],
                }
            )

    missing_roles = blueprint.get("unsupported_roles") or []
    issue_summary = filter_reasons.get("_summary", "")
    return {
        "plan_id": "plan_itinerary_skeleton",
        "title": "多节点行程骨架",
        "scene_type": scene_type,
        "plan_status": "needs_rag_candidate_evidence",
        "planning_horizon": blueprint.get("planning_horizon"),
        "planning_days": blueprint.get("planning_days"),
        "timeline": timeline,
        "total_price": None,
        "total_duration_min": total_duration,
        "total_distance_km": None,
        "people_count": None,
        "route": {},
        "budget": {},
        "availability": {"all_available": False, "reason": "missing_rag_candidate_evidence"},
        "objective_vector": {},
        "score_breakdown": {},
        "score_breakdown_details": {},
        "weighted_score": 0.0,
        "weights": {},
        "base_weights": {},
        "weight_adjustments": [],
        "plan_quality": {},
        "quality_adjustments": [],
        "why_selected": [
            "已识别为全日/两天多节点需求",
            "需要 RAG 返回每个节点的候选商家和证据后才能优化",
        ],
        "risk_factors": [
            "当前不是可执行方案",
            "缺少多节点候选池" if not missing_roles else f"缺少供给域: {', '.join(missing_roles)}",
        ],
        "constraint_summary": {
            "candidate_evidence_status": "⚠",
            "execution_status": "⚠",
        },
        "execution_ready": False,
        "action_hints": [],
        "b_itinerary_blueprint": blueprint,
        "candidate_generation_summary": issue_summary,
    }


BUSINESS_HOURS_REJECT_REASON = "营业时间不满足深夜/夜宵需求"


def _candidate_poi_node_count(plan: dict) -> int:
    return sum(
        1
        for item in plan.get("timeline", []) or []
        if isinstance(item, dict) and item.get("poi_id")
    )


def _best_rejected_candidate_for_reason(
    candidates: list[dict],
    filter_reasons: dict,
    reason: str,
) -> dict:
    rejected = [
        plan
        for plan in candidates
        if filter_reasons.get(plan.get("plan_id", "unknown")) == reason
    ]
    if not rejected:
        return {}

    def sort_key(plan: dict) -> tuple[float, float, float]:
        route = plan.get("route", {}) or {}
        budget = plan.get("budget", {}) or {}
        return (
            float(_candidate_poi_node_count(plan)),
            -to_float(route.get("total_distance_km"), 999.0),
            -to_float(budget.get("total_price"), 999999.0),
        )

    return max(rejected, key=sort_key)


def _build_time_adjustment_fallback_plan(
    *,
    candidate: dict,
    blueprint: dict,
    filter_reasons: dict,
    scene_type: str,
    people_count: int,
) -> dict:
    """Return a truthful non-executable plan when all POI candidates are closed.

    This is intentionally not marked execution-ready. The point is to preserve
    useful POI evidence and tell the user which constraint needs negotiation.
    """

    timeline = []
    for item in candidate.get("timeline", []) or []:
        entry = dict(item)
        notes = list(entry.get("notes", []) or [])
        notes.append("当前请求时间可能无法履约，需要调整时间或替换为深夜营业点")
        entry["notes"] = notes
        timeline.append(entry)

    nodes = candidate.get("nodes", []) or []
    non_executable_nodes = [
        {
            "poi_id": node.get("poi_id"),
            "name": node.get("name"),
            "role": node.get("itinerary_role") or node.get("role"),
            "supply_domain": node.get("supply_domain") or node.get("type"),
            "blocker": BUSINESS_HOURS_REJECT_REASON,
        }
        for node in nodes
        if isinstance(node, dict)
    ]
    issue_summary = filter_reasons.get("_summary", "")
    return {
        "plan_id": str(candidate.get("plan_id") or "plan_time_adjustment").replace("cand_", "plan_"),
        "title": "需要调整时间的深夜候选方案",
        "scene_type": scene_type,
        "planner_mode": candidate.get("planner_mode"),
        "plan_shape": candidate.get("plan_shape"),
        "planning_horizon": candidate.get("planning_horizon") or blueprint.get("planning_horizon"),
        "planning_days": candidate.get("planning_days") or blueprint.get("planning_days"),
        "benchmark_ready": bool(candidate.get("benchmark_ready", False)),
        "execution_scope": "partial",
        "plan_status": "time_adjustment_required",
        "timeline": timeline,
        "total_price": (candidate.get("budget") or {}).get("total_price"),
        "total_duration_min": candidate.get("estimated_duration_min"),
        "total_distance_km": (candidate.get("route") or {}).get("total_distance_km"),
        "people_count": people_count,
        "route": candidate.get("route", {}),
        "budget": candidate.get("budget", {}),
        "availability": {
            **(candidate.get("availability", {}) or {}),
            "all_available": False,
            "reason": "requested_time_closed",
        },
        "objective_vector": {},
        "score_breakdown": {},
        "score_breakdown_details": {},
        "weighted_score": 0.0,
        "weights": {},
        "base_weights": {},
        "weight_adjustments": [],
        "plan_quality": {},
        "quality_adjustments": [],
        "why_selected": [
            "已找到满足地点/类型的真实 POI 候选",
            "但当前请求时间过晚，营业时间不满足履约要求",
            "建议提前开始或改为深夜营业供给后再执行",
        ],
        "risk_factors": [BUSINESS_HOURS_REJECT_REASON],
        "constraint_summary": {
            "candidate_evidence_status": "✅",
            "business_hours_status": "⚠️",
            "execution_status": "⚠️",
        },
        "execution_ready": False,
        "action_hints": [],
        "execution_blockers": [BUSINESS_HOURS_REJECT_REASON],
        "non_executable_nodes": non_executable_nodes,
        "partial_missing_roles": [],
        "partial_missing_node_intents": [],
        "time_adjustment": {
            "requested_time_infeasible": True,
            "suggested_time_windows": ["20:00 前开始", "改为次日下午/傍晚", "改搜深夜营业咖啡/酒吧/夜宵"],
            "reason": BUSINESS_HOURS_REJECT_REASON,
        },
        "b_itinerary_blueprint": blueprint,
        "candidate_generation_summary": issue_summary,
    }


def plan_optimizer_node(state: PlanState) -> dict:
    """
    Multi-objective plan optimization with absolute scoring and enhanced metadata.
    """
    execution_log = state.get("execution_log", [])
    filtered_candidates = state.get("filtered_candidates", [])
    scene_type = normalize_scene_type(state.get("scene_type", "family"))
    constraints = state.get("constraints", {})
    user_profile = state.get("user_profile", {})
    state_weather_context = state.get("weather_context", {}) or {}

    if not filtered_candidates:
        blueprint = state.get("b_itinerary_blueprint") or constraints.get("b_itinerary_blueprint") or {}
        filter_reasons = state.get("filter_reasons", {}) or {}
        candidates = state.get("candidates", []) or []
        business_hours_rejected = _best_rejected_candidate_for_reason(
            candidates,
            filter_reasons,
            BUSINESS_HOURS_REJECT_REASON,
        )
        if business_hours_rejected:
            config = get_constraint_config_with_profile(constraints, user_profile)
            fallback_plan = _build_time_adjustment_fallback_plan(
                candidate=business_hours_rejected,
                blueprint=blueprint,
                filter_reasons=filter_reasons,
                scene_type=scene_type,
                people_count=config["people_count"],
            )
            execution_log.append(
                "[B] plan_optimizer_node returned time-adjustment fallback; "
                "POI evidence exists but requested time is not executable"
            )
            return {
                "selected_plan": fallback_plan,
                "optimization_score": 0.0,
                "alternative_plans": [],
                "execution_log": execution_log,
            }
        if blueprint.get("template_mode") == "multi_node":
            skeleton_plan = _build_multinode_skeleton_plan(
                blueprint=blueprint,
                filter_reasons=filter_reasons,
                scene_type=scene_type,
            )
            execution_log.append(
                "[B] plan_optimizer_node returned multi-node itinerary skeleton; "
                "waiting for RAG candidate evidence"
            )
            return {
                "selected_plan": skeleton_plan,
                "optimization_score": 0.0,
                "alternative_plans": [],
                "execution_log": execution_log,
            }
        execution_log.append("[B] plan_optimizer_node 未找到可行方案")
        return {
            "selected_plan": {},
            "optimization_score": 0.0,
            "alternative_plans": [],
            "execution_log": execution_log,
        }

    config = get_constraint_config_with_profile(constraints, user_profile)
    budget = config["budget"]
    child_age = config["child_age"]
    max_distance = config["max_distance_km"]
    max_queue_time = config["max_queue_time"]
    duration_range = config["duration_range"]
    mom_diet = config["mom_diet"]

    scenario_activities = state.get("scenario_activities", []) or []
    preference_sources = collect_preference_sources(constraints, user_profile, scenario_activities)
    current_preference_sources = collect_preference_sources(constraints, {}, scenario_activities)
    base_weights = _derive_weights(scene_type, constraints)
    weights, weight_adjustments = adjust_weights_for_context(
        base_weights,
        scene_type=scene_type,
        constraints=constraints,
        preference_sources=preference_sources,
    )

    scored_candidates = []

    for plan in filtered_candidates:
        tags = plan.get("tags", []) or []
        route = plan.get("route", {}) or {}
        budget_info = plan.get("budget", {}) or {}
        availability = plan.get("availability", {}) or {}
        weather_context = plan.get("weather_context") or state_weather_context

        plan_nodes = plan.get("nodes", []) or []
        activity = next((node for node in plan_nodes if node.get("type") == "activity"), {})
        restaurant = next((node for node in plan_nodes if node.get("type") == "restaurant"), {})
        primary_node = plan_nodes[0] if plan_nodes else {}
        activity_for_score = activity or primary_node
        restaurant_for_score = restaurant or primary_node

        activity_tags = activity_for_score.get("tags", []) or []
        restaurant_tags = restaurant_for_score.get("tags", []) or []

        preference = _score_preference(preference_sources, tags)
        group_fit = _score_group_fit(
            activity_tags,
            restaurant_tags,
            child_age,
            mom_diet,
            scene_type,
            activity_for_score,
            restaurant_for_score,
        )
        route_value = _score_route(
            route.get("total_distance_km", 0),
            route.get("total_travel_time_min", 0),
        )
        budget_value = _score_budget(budget_info.get("total_price", 0), budget, scene_type)
        time_value = _score_time_fit(
            plan.get("estimated_duration_min", 0),
            duration_range,
            route,
        )
        availability_value = _score_availability(
            availability.get("all_available", False),
            availability.get("max_queue_time_min", 0),
        )
        experience_value = _score_experience(
            activity_for_score.get("rating", 0),
            restaurant_for_score.get("rating", 0),
            tags,
            activity_for_score,
            restaurant_for_score,
        )
        atmosphere_value = _score_atmosphere(
            scene_type,
            tags,
            activity_for_score,
            restaurant_for_score,
            preference_sources,
        )
        novelty_value = _score_novelty(tags, activity_for_score, restaurant_for_score)
        weather_fit_value = _score_weather_fit(activity_for_score, activity_tags, weather_context)
        commercial_addon_value = _score_commercial_addon(activity_for_score, restaurant_for_score)
        risk_score, risk_factors = _calc_risk_factors(
            route.get("total_distance_km", 0),
            route.get("total_travel_time_min", 0),
            availability.get("max_queue_time_min", 0),
            activity_for_score.get("rating", 0),
            restaurant_for_score.get("rating", 0),
            tags,
            route,
        )
        weather_risk, weather_risk_factors = _weather_risk_factors(
            activity_for_score,
            activity_tags,
            weather_context,
        )
        risk_score = min(1.0, risk_score + weather_risk)
        risk_factors.extend(weather_risk_factors)

        restaurant_category = restaurant_for_score.get("restaurant_category") or restaurant_for_score.get("category")
        restaurant_diet_conflict = (
            str(restaurant_category) in {"火锅", "烤肉", "炸鸡小吃", "hotpot", "bbq", "barbecue", "fried_chicken"}
            or has_item_semantic_group(restaurant_for_score, "烤肉")
            or has_item_semantic_group(restaurant_for_score, "火锅")
            or has_item_semantic_group(restaurant_for_score, "炸鸡小吃")
        )
        if mom_diet == "low_calorie" and restaurant_diet_conflict:
            risk_score = min(1.0, risk_score + 0.20)
            risk_factors.append(f"{restaurant_category} 与低卡需求存在冲突")
        if restaurant and restaurant.get("dine_in_available") is False:
            risk_score = min(1.0, risk_score + 0.35)
            risk_factors.append("该餐厅不支持堂食订座")
        if _has_health_food_intent(current_preference_sources):
            health_signals = _restaurant_health_signals(restaurant_for_score, restaurant_tags)
            if not health_signals.intersection(HEALTH_MATCH_TAGS | {"healthy", "japanese_light_food"}):
                preference = max(0.0, preference - 0.20)
                risk_score = min(1.0, risk_score + 0.10)
                risk_factors.append("餐厅与轻食/健康偏好匹配不足")
        if _has_light_food_intent(current_preference_sources) and not _is_light_food_restaurant(restaurant_for_score, restaurant_tags):
            preference = max(0.0, preference - 0.15)
            risk_score = min(1.0, risk_score + 0.08)
            risk_factors.append("餐厅不是明确轻食供给")

        preference, risk_score, risk_factors = _apply_multinode_itinerary_quality_guards(
            plan=plan,
            constraints=constraints,
            preference_sources=current_preference_sources,
            preference=preference,
            risk_score=risk_score,
            risk_factors=risk_factors,
        )

        objective_vector = {
            "preference": round(preference, 3),
            "group_fit": round(group_fit, 3),
            "route": round(route_value, 3),
            "budget": round(budget_value, 3),
            "availability": round(availability_value, 3),
            "experience": round(experience_value, 3),
            "time": round(time_value, 3),
            "atmosphere": round(atmosphere_value, 3),
            "novelty": round(novelty_value, 3),
            "weather_fit": round(weather_fit_value, 3),
            "commercial_addon": round(commercial_addon_value, 3),
            "risk": round(risk_score, 3),
        }

        quality_profile = build_plan_quality_profile(
            activity=activity_for_score,
            restaurant=restaurant_for_score,
            scene_type=scene_type,
            constraints=constraints,
            preference_sources=preference_sources,
            child_age=child_age,
            mom_diet=mom_diet,
        )
        objective_vector, risk_score, risk_factors, quality_adjustments = apply_quality_profile_to_objectives(
            objective_vector,
            risk_score=risk_score,
            risk_factors=risk_factors,
            quality_profile=quality_profile,
            scene_type=scene_type,
            child_age=child_age,
            mom_diet=mom_diet,
        )
        objective_vector["risk"] = round(risk_score, 3)

        weighted_score = sum(
            objective_vector[key] * weights.get(key, 0.0)
            for key in WEIGHT_KEYS
            if key != "risk"
        ) - objective_vector["risk"] * abs(weights.get("risk", 0.0))

        avoid = expand_preference_tags(
            (user_profile.get("avoid", []) or [])
            + (user_profile.get("preference_profile", {}).get("avoid", []) or [])
            + collect_tag_fields(constraints, "avoid")
        )
        if "crowded_mall" in avoid and "crowded_mall" in tags:
            weighted_score *= _get_penalty("avoid_tag_hit_multiplier", 0.85)

        score_breakdown = {
            "preference": round(objective_vector["preference"] * weights["preference"], 3),
            "group_fit": round(objective_vector["group_fit"] * weights["group_fit"], 3),
            "route": round(objective_vector["route"] * weights["route"], 3),
            "budget": round(objective_vector["budget"] * weights["budget"], 3),
            "availability": round(objective_vector["availability"] * weights["availability"], 3),
            "experience": round(objective_vector["experience"] * weights["experience"], 3),
            "time": round(objective_vector["time"] * weights.get("time", 0.0), 3),
            "atmosphere": round(objective_vector["atmosphere"] * weights.get("atmosphere", 0.0), 3),
            "novelty": round(objective_vector["novelty"] * weights.get("novelty", 0.0), 3),
            "weather_fit": round(objective_vector["weather_fit"] * weights.get("weather_fit", 0.0), 3),
            "commercial_addon": round(
                objective_vector["commercial_addon"] * weights.get("commercial_addon", 0.0),
                3,
            ),
            "risk": round(objective_vector["risk"] * abs(weights.get("risk", 0.0)), 3),
        }
        score_breakdown_details = build_score_breakdown_details(
            objective_vector,
            weights,
            score_breakdown,
            quality_profile,
        )
        why_selected = build_why_selected(
            objective_vector,
            risk_factors,
            quality_profile,
        )

        scored_candidates.append(
            {
                "plan": plan,
                "objective_vector": objective_vector,
                "weighted_score": round(weighted_score, 4),
                "score_breakdown": score_breakdown,
                "score_breakdown_details": score_breakdown_details,
                "risk_factors": risk_factors,
                "plan_quality": quality_profile,
                "quality_adjustments": quality_adjustments,
                "why_selected": why_selected,
            }
        )

    scored_candidates.sort(key=lambda x: x["weighted_score"], reverse=True)
    plan_critic_metadata = None
    if scored_candidates and scored_candidates[0]["plan"].get("execution_scope") == "partial":
        plan_critic_metadata = {
            "enabled": True,
            "provider": "longcat",
            "success": False,
            "skipped": True,
            "reason": "partial_plan_not_fully_executable",
        }
    else:
        scored_candidates, plan_critic_metadata = apply_b_plan_critic(
            state,
            scored_candidates,
            weights=weights,
        )
    if plan_critic_metadata:
        if plan_critic_metadata.get("success"):
            execution_log.append(
                "[B] LongCat plan critic reviewed top candidates "
                f"and selected={plan_critic_metadata.get('selected_after_critic')}"
            )
        elif plan_critic_metadata.get("skipped"):
            execution_log.append(
                "[B] LongCat plan critic skipped; deterministic ranking was clear"
            )
        else:
            execution_log.append("[B] LongCat plan critic unavailable; kept deterministic ranking")

    selected = scored_candidates[0]
    selected_plan_base = selected["plan"]
    is_multinode_plan = selected_plan_base.get("planner_mode") == "multi_node_itinerary"
    is_single_node_plan = selected_plan_base.get("planner_mode") == "single_node"

    activity = next((node for node in selected_plan_base.get("nodes", []) if node.get("type") == "activity"), {})
    restaurant = next((node for node in selected_plan_base.get("nodes", []) if node.get("type") == "restaurant"), {})
    activity_nodes = [
        node for node in selected_plan_base.get("nodes", []) or []
        if node.get("type") == "activity"
    ]
    restaurant_nodes = [
        node for node in selected_plan_base.get("nodes", []) or []
        if node.get("type") == "restaurant"
    ]
    if selected_plan_base.get("restaurant_role"):
        restaurant = dict(restaurant)
        restaurant["restaurant_role"] = selected_plan_base.get("restaurant_role")

    activity_tags = activity.get("tags", []) or []
    restaurant_tags = restaurant.get("tags", []) or []
    activity_signal_set = _activity_signal_set(activity, activity_tags)
    restaurant_signal_set = _semantic_tag_set(restaurant_tags)
    restaurant_health_signals = _restaurant_health_signals(restaurant, restaurant_tags)

    people_count = config["people_count"]
    activity["_selected_schedule"] = selected_plan_base.get("schedule", {})
    if is_multinode_plan or is_single_node_plan:
        timeline = selected_plan_base.get("timeline", []) or []
    else:
        timeline = _build_timeline(
            activity,
            restaurant,
            scene_type=scene_type,
            child_age=child_age,
        )

    activity_notes = []
    if is_child_compatible_activity(activity, child_age):
        activity_notes.append("kid_friendly")
    if "low_intensity" in activity_signal_set:
        activity_notes.append("low_intensity")

    restaurant_notes = []
    if "family_friendly" in restaurant_signal_set:
        restaurant_notes.append("child_seat")
    if restaurant_health_signals.intersection(HEALTH_MATCH_TAGS):
        restaurant_notes.append("low_oil_low_salt")

    if child_age is None:
        child_fit_ok = True
    elif activity_nodes:
        child_fit_ok = any(is_child_compatible_activity(item, child_age) for item in activity_nodes)
    elif restaurant_nodes:
        child_fit_ok = bool(
            restaurant_signal_set.intersection({"family_friendly", "kid_friendly", "child_seat", "亲子餐厅"})
        )
    else:
        child_fit_ok = True

    diet_ok = (
        mom_diet != "low_calorie"
        or not restaurant_nodes
        or all(
            _restaurant_health_signals(item, item.get("tags", []) or []).intersection(HEALTH_MATCH_TAGS)
            for item in restaurant_nodes
        )
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

    activity_timeline_item = next(
        (item for item in timeline if item.get("type") in {"activity", "play", "amusement", "museum", "art"}),
        timeline[0] if timeline else {},
    )
    restaurant_timeline_item = next(
        (item for item in timeline if item.get("type") in {"restaurant", "eat"}),
        timeline[-1] if timeline else {},
    )
    activity_action_time = activity_timeline_item.get("time", "14:30").split("-")[0]
    restaurant_action_time = restaurant_timeline_item.get("time", "17:00").split("-")[0]
    constraint_warnings = [
        key
        for key, value in constraint_summary.items()
        if value != "✓"
    ]
    # The constraint filter has already removed hard-invalid candidates.  The
    # summary here is a product explanation layer, so warning badges should not
    # block C execution when the selected POIs still have valid action hints.
    constraint_ready = True
    if is_multinode_plan:
        action_hints = _build_multinode_action_hints(selected_plan_base, timeline, people_count)
    elif is_single_node_plan:
        action_hints = _build_single_node_action_hints(selected_plan_base, timeline, people_count)
    else:
        activity_action_hint = _build_action_hint(
            "order_activity_ticket",
            activity,
            activity_action_time,
            people_count,
            activity_notes,
            "quantity",
        )
        restaurant_action_hint = _build_action_hint(
            "reserve_restaurant",
            restaurant,
            restaurant_action_time,
            people_count,
            restaurant_notes,
            "people",
        )
        if selected_plan_base.get("schedule", {}).get("sequence") == "restaurant_then_activity":
            action_hints = [restaurant_action_hint, activity_action_hint]
        else:
            action_hints = [activity_action_hint, restaurant_action_hint]

    selected_plan = {
        "plan_id": selected_plan_base.get("plan_id", "plan_001").replace("cand_", "plan_"),
        "supply_identity": _plan_identity(selected_plan_base),
        "title": _build_multinode_plan_title(selected_plan_base) if is_multinode_plan else _build_plan_title(scene_type, activity, restaurant, child_age),
        "scene_type": scene_type,
        "planner_mode": selected_plan_base.get("planner_mode"),
        "plan_shape": selected_plan_base.get("plan_shape"),
        "planning_horizon": selected_plan_base.get("planning_horizon"),
        "planning_days": selected_plan_base.get("planning_days"),
        "benchmark_ready": bool(selected_plan_base.get("benchmark_ready", False)),
        "execution_scope": selected_plan_base.get("execution_scope", "full"),
        "non_executable_nodes": selected_plan_base.get("non_executable_nodes", []),
        "guidance_only_nodes": selected_plan_base.get("guidance_only_nodes", []),
        "rag_candidate_coverage": selected_plan_base.get("rag_candidate_coverage", {}),
        "timeline": timeline,
        "total_price": selected_plan_base.get("budget", {}).get("total_price", 0),
        "total_duration_min": selected_plan_base.get("estimated_duration_min", 0),
        "total_distance_km": selected_plan_base.get("route", {}).get("total_distance_km", 0),
        "people_count": people_count,
        "route": selected_plan_base.get("route", {}),
        "restaurant_role": selected_plan_base.get("restaurant_role"),
        "weather_context": selected_plan_base.get("weather_context") or state_weather_context,
        "budget": selected_plan_base.get("budget", {}),
        "availability": selected_plan_base.get("availability", {}),
        "objective_vector": selected["objective_vector"],
        "score_breakdown": selected["score_breakdown"],
        "score_breakdown_details": selected["score_breakdown_details"],
        "weighted_score": selected["weighted_score"],
        "weights": weights,
        "base_weights": base_weights,
        "weight_adjustments": weight_adjustments,
        "plan_quality": selected["plan_quality"],
        "quality_adjustments": selected["quality_adjustments"],
        "why_selected": selected["why_selected"],
        "risk_factors": selected["risk_factors"],
        "constraint_summary": constraint_summary,
        "constraint_warnings": constraint_warnings,
        "execution_ready": constraint_ready,
        "action_hints": action_hints,
    }
    if selected_plan_base.get("b_itinerary_blueprint"):
        selected_plan["b_itinerary_blueprint"] = selected_plan_base.get("b_itinerary_blueprint")
    ai_planning_review = _build_ai_planning_review(selected, plan_critic_metadata)
    if plan_critic_metadata:
        selected_plan["b_ai_plan_critic"] = plan_critic_metadata
    if ai_planning_review:
        selected_plan["b_ai_planning_review"] = ai_planning_review
    if is_multinode_plan or is_single_node_plan:
        execution_contract = _validate_multinode_execution_contract(
            selected_plan,
            selected_plan_base.get("nodes", []) or [],
            people_count,
        )
    else:
        execution_contract = _validate_execution_contract(selected_plan, activity, restaurant, people_count)
    selected_plan["execution_contract"] = execution_contract
    if is_multinode_plan or is_single_node_plan:
        selected_plan["execution_scope"] = execution_contract.get(
            "execution_scope",
            selected_plan.get("execution_scope"),
        )
        selected_plan["non_executable_nodes"] = execution_contract.get(
            "non_executable_nodes",
            selected_plan.get("non_executable_nodes", []),
        )
        selected_plan["guidance_only_nodes"] = execution_contract.get(
            "guidance_only_nodes",
            selected_plan.get("guidance_only_nodes", []),
        )
    selected_plan["execution_ready"] = constraint_ready and execution_contract["ready"]
    partial_missing_roles = selected_plan_base.get("partial_missing_roles", [])
    if selected_plan.get("execution_scope") == "partial" or partial_missing_roles:
        selected_plan["execution_scope"] = "partial"
        selected_plan["plan_status"] = "partial_executable"
        selected_plan["execution_ready"] = False
        selected_plan["partial_missing_roles"] = partial_missing_roles
        selected_plan["partial_missing_node_intents"] = selected_plan_base.get("partial_missing_node_intents", [])
        selected_plan["execution_blockers"] = _dedupe_keep_order(
            list(selected_plan.get("execution_blockers", []) or [])
            + [
                "Some itinerary nodes still need RAG/domain supply before full execution",
            ]
        )
    b_replan_request = _build_b_replan_request(
        selected_plan,
        selected_plan_base,
        ai_planning_review,
        constraints,
    )
    if b_replan_request:
        selected_plan["plan_status"] = "needs_ai_replan"
        selected_plan["execution_ready"] = False
        selected_plan["execution_blockers"] = _dedupe_keep_order(
            list(selected_plan.get("execution_blockers", []) or [])
            + ["LongCat critic requested B replan before C execution"]
        )
        selected_plan["b_replan_request"] = b_replan_request

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
                "supply_identity": _plan_identity(plan_base),
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
                    "supply_identity": _plan_identity(plan_base),
                    "tradeoff": "整体分数接近，但优势维度不同。",
                }
            )
            seen_plan_ids.add(plan_base.get("plan_id"))

    optimization_score = round(selected_plan["weighted_score"] * 100, 2)

    execution_log.append(
        f"[B] plan_optimizer_node 选出 weighted_score={selected_plan['weighted_score']}, "
        f"execution_ready={selected_plan['execution_ready']}"
    )

    result = {
        "selected_plan": selected_plan,
        "optimization_score": optimization_score,
        "alternative_plans": alternative_plans,
        "execution_log": execution_log,
    }
    if plan_critic_metadata:
        result["b_ai_plan_critic"] = plan_critic_metadata
    if ai_planning_review:
        result["b_ai_planning_review"] = ai_planning_review
    if b_replan_request:
        result["b_replan_request"] = b_replan_request
    return result
