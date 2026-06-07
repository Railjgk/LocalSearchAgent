"""Runtime policy helpers for B candidate generation.

This module keeps environment variables, planner policy YAML, and default
candidate budgets out of the main candidate generation flow.
"""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is optional for smoke demos
    yaml = None


DEFAULT_TOP_K_ACTIVITY = 3
DEFAULT_TOP_K_RESTAURANT = 3
DEFAULT_TRANSITION_BUFFER_MIN = 30
DEFAULT_ROUTE_LOOKAHEAD_MULTIPLIER = 2
DEFAULT_PAIR_POOL_MULTIPLIER = 8
DEFAULT_PLAN_CANDIDATE_LIMIT = 96
DEFAULT_MAX_PAIR_COMBINATIONS = 768
DEFAULT_MAX_MULTINODE_CANDIDATES = 24
DEFAULT_ROUTE_SOURCE_ORDER = ("offline_routes_json", "coordinate_estimate", "poi_distance_fallback")
DEFAULT_MOCK_DATA_DIR = Path(__file__).resolve().parents[2] / "experiments" / "mock_data"
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


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

    resolved_policy_path = Path(policy_path_key)
    if not resolved_policy_path.exists():
        return {}

    try:
        with resolved_policy_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def candidate_generation_config() -> dict:
    policy = load_policy_config(policy_cache_key())
    candidate_generation = policy.get("candidate_generation")
    return candidate_generation if isinstance(candidate_generation, dict) else {}


def allow_legacy_fallback_for_multinode() -> bool:
    raw_value = os.environ.get("WF_B_ALLOW_LEGACY_FALLBACK_FOR_MULTINODE", "").strip().lower()
    if raw_value:
        return raw_value in TRUTHY_ENV_VALUES

    config = candidate_generation_config()
    return bool(config.get("allow_legacy_fallback_for_multinode", False))


def top_k(name: str, default: int) -> int:
    env_name = f"WF_B_{name.upper()}"
    raw_value = os.environ.get(env_name)
    if raw_value is None:
        raw_value = candidate_generation_config().get(name, default)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(1, value)


def route_lookahead_multiplier() -> int:
    raw_value = os.environ.get("WF_B_ROUTE_LOOKAHEAD_MULTIPLIER")
    if raw_value is None:
        raw_value = candidate_generation_config().get(
            "route_lookahead_multiplier",
            DEFAULT_ROUTE_LOOKAHEAD_MULTIPLIER,
        )
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_ROUTE_LOOKAHEAD_MULTIPLIER
    return max(1, min(value, 5))


def pair_pool_multiplier() -> int:
    raw_value = os.environ.get("WF_B_PAIR_POOL_MULTIPLIER")
    if raw_value is None:
        raw_value = candidate_generation_config().get(
            "pair_pool_multiplier",
            DEFAULT_PAIR_POOL_MULTIPLIER,
        )
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_PAIR_POOL_MULTIPLIER
    return max(1, min(value, 8))


def plan_candidate_limit() -> int:
    raw_value = os.environ.get("WF_B_PLAN_CANDIDATE_LIMIT")
    if raw_value is None:
        raw_value = candidate_generation_config().get(
            "plan_candidate_limit",
            DEFAULT_PLAN_CANDIDATE_LIMIT,
        )
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_PLAN_CANDIDATE_LIMIT
    return max(9, min(value, 300))


def max_pair_combinations() -> int:
    raw_value = os.environ.get("WF_B_MAX_PAIR_COMBINATIONS")
    if raw_value is None:
        raw_value = candidate_generation_config().get(
            "max_pair_combinations",
            DEFAULT_MAX_PAIR_COMBINATIONS,
        )
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_PAIR_COMBINATIONS
    return max(1, min(value, 10000))


def geo_prefilter_min_keep() -> int:
    raw_value = os.environ.get("WF_B_GEO_PREFILTER_MIN_KEEP")
    if raw_value is None:
        raw_value = candidate_generation_config().get("geo_prefilter_min_keep", 500)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return 500
    return max(40, min(value, 5000))


def pretrim_min_keep() -> int:
    raw_value = os.environ.get("WF_B_PRETRIM_MIN_KEEP")
    if raw_value is None:
        raw_value = candidate_generation_config().get("pretrim_min_keep", 300)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return 300
    return max(40, min(value, 2000))


def route_source_order() -> list[str]:
    env_order = os.environ.get("WF_ROUTE_SOURCE_ORDER", "").strip()
    if env_order:
        return [item.strip() for item in env_order.split(",") if item.strip()]

    raw_order = candidate_generation_config().get("route_source_order")
    if isinstance(raw_order, list):
        values = [str(item).strip() for item in raw_order if str(item).strip()]
        if values:
            return values

    return list(DEFAULT_ROUTE_SOURCE_ORDER)


def time_slot_policy() -> dict:
    time_slot = candidate_generation_config().get("time_slot")
    return time_slot if isinstance(time_slot, dict) else {}


def transition_buffer_min() -> int:
    raw_value = time_slot_policy().get("default_transition_buffer_min", DEFAULT_TRANSITION_BUFFER_MIN)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_TRANSITION_BUFFER_MIN
    return max(0, value)


def time_slot_bool(name: str, default: bool) -> bool:
    raw_value = time_slot_policy().get(name, default)
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw_value)


def mock_data_dir() -> Path:
    env_dir = os.environ.get("WF_MOCK_DATA_DIR", "").strip()
    if env_dir:
        path = Path(env_dir)
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[2] / path

    raw_dir = candidate_generation_config().get("local_mock_dir")
    if raw_dir:
        path = Path(str(raw_dir))
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[2] / path

    return DEFAULT_MOCK_DATA_DIR
