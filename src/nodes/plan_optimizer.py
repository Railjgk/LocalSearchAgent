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


ABSOLUTE_MAX_DISTANCE_KM = 15.0
ABSOLUTE_MAX_QUEUE_TIME_MIN = 60.0
ABSOLUTE_MIN_RATING = 3.0
ABSOLUTE_MAX_RATING = 5.0

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


def _canonical_preference_tokens(values: list[str]) -> list[str]:
    tokens: list[str] = []
    for raw_value in values or []:
        raw_token = str(raw_value).strip()
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
    tag_tokens = _dedupe_keep_order(expand_preference_tags(tags or []))
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
            if "kid_friendly" in activity_tags or "low_intensity" in activity_tags:
                score += 0.5
        else:
            score += 0.3
    elif scene_type == "friends":
        max_points += 0.5
        social_signals = set(activity_tags) | set(restaurant_tags)
        if social_signals.intersection(FRIENDS_FIT_TAGS):
            score += 0.5
        elif "kid_friendly" in social_signals and not social_signals.intersection({"social", "group_friendly"}):
            score += 0.1
        else:
            score += 0.25
    elif scene_type == "couple":
        max_points += 0.5
        couple_signals = set(activity_tags) | set(restaurant_tags)
        if couple_signals.intersection(COUPLE_FIT_TAGS):
            score += 0.5
        else:
            score += 0.2
    elif scene_type == "low_budget":
        max_points += 0.5
        budget_signals = set(activity_tags) | set(restaurant_tags)
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
    tag_set = set(expand_preference_tags(tags or []))
    preference_set = set(expand_preference_tags(preference_sources or []))
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
    tag_set = set(expand_preference_tags(tags or []))
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
    health_signals = set(expand_preference_tags(restaurant_tags or restaurant.get("tags", []) or []))
    health_signals.update(expand_preference_tags(restaurant.get("health_tags", []) or []))
    health_signals.update(expand_preference_tags(restaurant.get("menu_health_options", []) or []))
    health_signals.update(expand_preference_tags(restaurant.get("restaurant_category") or ""))
    return health_signals


def _is_light_food_restaurant(restaurant: dict | None, restaurant_tags: list | None = None) -> bool:
    restaurant = restaurant or {}
    category = str(restaurant.get("restaurant_category") or restaurant.get("category") or "")
    category_signals = {"light_food", "salad_light_food", "japanese_light_food", "vegetarian_light_food"}
    if category in category_signals:
        return True
    direct_tags = set()
    for raw_tag in restaurant_tags or restaurant.get("tags", []) or []:
        direct_tags.add(str(raw_tag).strip())
    direct_tags.update(str(tag).strip() for tag in restaurant.get("health_tags", []) or [])
    return bool(direct_tags.intersection({"light_food", "low_calorie", "salad_light_food", "japanese_light_food"}))


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
        risk_factors.append(f"???? ({distance_km:.1f} ??)")

    travel_warning_min = max_distance * minutes_per_km * distance_warning_ratio
    if travel_time_min > travel_warning_min:
        risk_score += _get_penalty("far_distance", 0.3) * 0.6
        risk_factors.append(f"?????? ({travel_time_min:.0f} ??)")

    traffic_status = str(route.get("traffic_status") or "").lower()
    if traffic_status in {"high", "heavy", "severe"}:
        risk_score += 0.18
        risk_factors.append("???????")

    if queue_time_min > max_queue_time * queue_warning_ratio:
        risk_score += _get_penalty("long_queue", 0.25)
        risk_factors.append(f"?????? ({queue_time_min:.0f} ??)")

    avg_rating = (activity_rating + restaurant_rating) / 2.0
    if avg_rating < low_rating_warning:
        risk_score += _get_penalty("low_rating", 0.2)
        risk_factors.append(f"????? ({avg_rating:.1f})")

    if "crowded_mall" in tags:
        risk_score += _get_penalty("crowded_mall", 0.15)
        risk_factors.append("??????")

    return min(1.0, risk_score), risk_factors


def _build_timeline(activity: dict, restaurant: dict, start_hour: int = 14, start_minute: int = 30) -> list[dict]:
    """Build detailed timeline with activity, transition, restaurant."""
    schedule = activity.get("_selected_schedule", {}) or {}
    restaurant_health_signals = set(restaurant.get("tags", []) or [])
    restaurant_health_signals.update(restaurant.get("health_tags", []) or [])
    restaurant_health_signals.update(restaurant.get("menu_health_options", []) or [])
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
                "????" if "kid_friendly" in activity.get("tags", []) else "?????",
                "???" if "low_intensity" in activity.get("tags", []) else "????",
            ],
        },
        {
            "time": f"{format_time(transition_start_hour, transition_start_minute)}-{format_time(transition_end_hour, transition_end_minute)}",
            "activity": "???????",
            "poi_id": None,
            "type": "transition",
            "duration_min": transition_buffer_min,
            "price": 0,
            "notes": ["??????"],
        },
        {
            "time": f"{format_time(restaurant_start_hour, restaurant_start_minute)}-{format_time(restaurant_end_hour, restaurant_end_minute)}",
            "activity": restaurant.get("name"),
            "poi_id": restaurant.get("poi_id"),
            "type": "restaurant",
            "duration_min": restaurant.get("duration_min"),
            "price": restaurant.get("price"),
            "notes": [
                "??/????" if restaurant_health_signals.intersection(HEALTH_MATCH_TAGS) else "????",
                "??" if "light_food" in restaurant_health_signals else "???????" if restaurant_health_signals.intersection({"low_oil", "low_sugar", "vegetable_rich"}) else "????",
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
    for field in ("product_type", "inventory_model", "fulfillment_mode"):
        if product.get(field):
            hint[field] = product.get(field)
    if deal.get("deal_type"):
        hint["deal_type"] = deal.get("deal_type")
    if deal.get("coupon_type"):
        hint["coupon_type"] = deal.get("coupon_type")
    return hint


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
        bool(merchant_id) and (not node.get("merchant_id") or merchant_id == node.get("merchant_id")),
        f"{role} action must target the selected merchant_id",
    )
    _add_contract_check(
        checks,
        blocking_reasons,
        f"{role}_product_or_deal",
        bool(product_id or deal_id),
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

    if not filtered_candidates:
        execution_log.append("[B] plan_optimizer_node ???????")
        return {
            "selected_plan": {},
            "optimization_score": 0.0,
            "alternative_plans": [],
            "execution_log": execution_log,
        }

    config = get_constraint_config_with_profile(constraints, user_profile)
    weights = _derive_weights(scene_type, constraints)
    budget = config["budget"]
    child_age = config["child_age"]
    max_distance = config["max_distance_km"]
    max_queue_time = config["max_queue_time"]
    duration_range = config["duration_range"]
    mom_diet = config["mom_diet"]

    scenario_activities = state.get("scenario_activities", []) or []
    preference_sources = collect_preference_sources(constraints, user_profile, scenario_activities)

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
            mom_diet,
            scene_type,
            restaurant,
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
            activity.get("rating", 0),
            restaurant.get("rating", 0),
            tags,
            activity,
            restaurant,
        )
        atmosphere_value = _score_atmosphere(
            scene_type,
            tags,
            activity,
            restaurant,
            preference_sources,
        )
        novelty_value = _score_novelty(tags, activity, restaurant)
        commercial_addon_value = _score_commercial_addon(activity, restaurant)
        risk_score, risk_factors = _calc_risk_factors(
            route.get("total_distance_km", 0),
            route.get("total_travel_time_min", 0),
            availability.get("max_queue_time_min", 0),
            activity.get("rating", 0),
            restaurant.get("rating", 0),
            tags,
            route,
        )

        restaurant_category = restaurant.get("restaurant_category") or restaurant.get("category")
        if mom_diet == "low_calorie" and restaurant_category in {"hotpot", "bbq", "fried_chicken"}:
            risk_score = min(1.0, risk_score + 0.20)
            risk_factors.append(f"{restaurant_category} ?????????")
        if restaurant.get("dine_in_available") is False:
            risk_score = min(1.0, risk_score + 0.35)
            risk_factors.append("??????????")
        if _has_health_food_intent(preference_sources):
            health_signals = _restaurant_health_signals(restaurant, restaurant_tags)
            if not health_signals.intersection(HEALTH_MATCH_TAGS | {"healthy", "japanese_light_food"}):
                preference = max(0.0, preference - 0.20)
                risk_score = min(1.0, risk_score + 0.10)
                risk_factors.append("?????/????????")
        if _has_light_food_intent(preference_sources) and not _is_light_food_restaurant(restaurant, restaurant_tags):
            preference = max(0.0, preference - 0.15)
            risk_score = min(1.0, risk_score + 0.08)
            risk_factors.append("??????????")

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
            "commercial_addon": round(commercial_addon_value, 3),
            "risk": round(risk_score, 3),
        }

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
            "commercial_addon": round(
                objective_vector["commercial_addon"] * weights.get("commercial_addon", 0.0),
                3,
            ),
            "risk": round(objective_vector["risk"] * abs(weights.get("risk", 0.0)), 3),
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
    restaurant_health_signals = set(restaurant_tags)
    restaurant_health_signals.update(restaurant.get("health_tags", []) or [])
    restaurant_health_signals.update(restaurant.get("menu_health_options", []) or [])

    people_count = config["people_count"]
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
    if restaurant_health_signals.intersection(HEALTH_MATCH_TAGS):
        restaurant_notes.append("low_oil_low_salt")

    child_fit_ok = (
        child_age is None
        or "kid_friendly" in activity_tags
        or "low_intensity" in activity_tags
    )

    diet_ok = (
        mom_diet != "low_calorie"
        or bool(restaurant_health_signals.intersection(HEALTH_MATCH_TAGS))
    )

    constraint_summary = {
        "distance_status": (
            "?"
            if selected_plan_base.get("route", {}).get("total_distance_km", 0) <= max_distance
            else "?"
        ),
        "queue_status": (
            "?"
            if selected_plan_base.get("availability", {}).get("max_queue_time_min", 0) <= max_queue_time
            else "?"
        ),
        "budget_status": (
            "?"
            if selected_plan_base.get("budget", {}).get("total_price", 0) <= budget * 1.2
            else "?"
        ),
        "child_friendly_status": "?" if child_fit_ok else "?",
        "diet_status": "?" if diet_ok else "?",
    }

    activity_action_time = timeline[0].get("time", "14:30").split("-")[0]
    restaurant_action_time = timeline[2].get("time", "17:00").split("-")[0]
    constraint_ready = all(v == "?" for v in constraint_summary.values())

    selected_plan = {
        "plan_id": selected_plan_base.get("plan_id", "plan_001").replace("cand_", "plan_"),
        "supply_identity": _plan_identity(selected_plan_base),
        "title": (
            "????????"
            if ("kid_friendly" in selected_plan_base.get("tags", []) or "low_intensity" in activity_tags)
            else "??????"
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
        "execution_ready": constraint_ready,
        "action_hints": [
            _build_action_hint(
                "order_activity_ticket",
                activity,
                activity_action_time,
                people_count,
                activity_notes,
                "quantity",
            ),
            _build_action_hint(
                "reserve_restaurant",
                restaurant,
                restaurant_action_time,
                people_count,
                restaurant_notes,
                "people",
            ),
        ],
    }
    execution_contract = _validate_execution_contract(selected_plan, activity, restaurant, people_count)
    selected_plan["execution_contract"] = execution_contract
    selected_plan["execution_ready"] = constraint_ready and execution_contract["ready"]

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
            title = "???????"
            dominant_dimension = "budget"
            tradeoff = "??????????????????"
        elif metric_name == "nearest":
            title = "????????"
            dominant_dimension = "route"
            tradeoff = "??????????????????"
        else:
            title = "????????"
            dominant_dimension = "experience"
            tradeoff = "??????????????????????"

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
                    "title": "??????",
                    "dominant_dimension": "balance",
                    "weighted_score": candidate["weighted_score"],
                    "total_price": plan_base.get("budget", {}).get("total_price", 0),
                    "total_distance_km": plan_base.get("route", {}).get("total_distance_km", 0),
                    "objective_vector": candidate["objective_vector"],
                    "supply_identity": _plan_identity(plan_base),
                    "tradeoff": "???????????????",
                }
            )
            seen_plan_ids.add(plan_base.get("plan_id"))

    optimization_score = round(selected_plan["weighted_score"] * 100, 2)

    execution_log.append(
        f"[B] plan_optimizer_node ?? weighted_score={selected_plan['weighted_score']}, "
        f"execution_ready={selected_plan['execution_ready']}"
    )

    return {
        "selected_plan": selected_plan,
        "optimization_score": optimization_score,
        "alternative_plans": alternative_plans,
        "execution_log": execution_log,
    }
