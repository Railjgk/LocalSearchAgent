"""B scoring policy defaults and policy-file overrides."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is optional for smoke demos
    yaml = None

from .b_utils import normalize_scene_type, to_float


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


def policy_path() -> Path:
    override = os.environ.get("WF_PLANNER_POLICY_PATH", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "experiments" / "planner_policy.yaml"


def policy_cache_key() -> str:
    return str(policy_path().resolve())


@lru_cache(maxsize=8)
def load_policy_config(policy_path_key: str) -> dict:
    if yaml is None:
        return {}

    path = Path(policy_path_key)
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def default_weights(scene_type: str) -> dict[str, float]:
    return dict(DEFAULT_SCENE_WEIGHTS.get(scene_type, DEFAULT_SCENE_WEIGHTS["family"]))


@lru_cache(maxsize=8)
def load_scene_weights_from_policy(policy_path_key: str) -> dict[str, dict[str, float]]:
    policy = load_policy_config(policy_path_key)
    scene_weights = policy.get("scene_weights")
    if not isinstance(scene_weights, dict):
        return {}

    normalized_weights = {}
    for scene_name, raw_weights in scene_weights.items():
        if not isinstance(raw_weights, dict):
            continue
        scene_key = normalize_scene_type(str(scene_name).strip())
        merged = default_weights(scene_key)
        for key in WEIGHT_KEYS:
            if key in raw_weights:
                merged[key] = to_float(raw_weights.get(key), merged[key])
        normalized_weights[scene_key] = merged
    return normalized_weights


@lru_cache(maxsize=8)
def load_score_thresholds_from_policy(policy_path_key: str) -> dict[str, dict[str, float]]:
    policy = load_policy_config(policy_path_key)
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
def load_penalties_from_policy(policy_path_key: str) -> dict[str, float]:
    policy = load_policy_config(policy_path_key)
    raw_penalties = policy.get("penalties")
    if not isinstance(raw_penalties, dict):
        raw_penalties = {}

    merged = dict(DEFAULT_PENALTIES)
    for key, default_value in DEFAULT_PENALTIES.items():
        if key in raw_penalties:
            merged[key] = to_float(raw_penalties.get(key), default_value)
    return merged


def get_threshold(section: str, key: str, default: float) -> float:
    thresholds = load_score_thresholds_from_policy(policy_cache_key())
    section_values = thresholds.get(section, {})
    return to_float(section_values.get(key), default)


def get_penalty(name: str, default: float) -> float:
    penalties = load_penalties_from_policy(policy_cache_key())
    return to_float(penalties.get(name), default)
