from __future__ import annotations

from pathlib import Path

from src.nodes import b_candidate_policy as policy


def _clear_policy_env(monkeypatch) -> None:
    for key in (
        "WF_PLANNER_POLICY_PATH",
        "WF_B_TOP_K_ACTIVITY",
        "WF_B_PLAN_CANDIDATE_LIMIT",
        "WF_B_PAIR_POOL_MULTIPLIER",
        "WF_B_MAX_PAIR_COMBINATIONS",
        "WF_ROUTE_SOURCE_ORDER",
        "WF_MOCK_DATA_DIR",
        "WF_B_ALLOW_LEGACY_FALLBACK_FOR_MULTINODE",
    ):
        monkeypatch.delenv(key, raising=False)
    policy.load_policy_config.cache_clear()


def test_policy_env_overrides_candidate_budget(monkeypatch) -> None:
    _clear_policy_env(monkeypatch)
    monkeypatch.setenv("WF_B_TOP_K_ACTIVITY", "7")
    monkeypatch.setenv("WF_B_PLAN_CANDIDATE_LIMIT", "999")
    monkeypatch.setenv("WF_B_PAIR_POOL_MULTIPLIER", "99")

    assert policy.top_k("top_k_activity", policy.DEFAULT_TOP_K_ACTIVITY) == 7
    assert policy.plan_candidate_limit() == 300
    assert policy.pair_pool_multiplier() == 8


def test_policy_yaml_supplies_defaults(tmp_path, monkeypatch) -> None:
    _clear_policy_env(monkeypatch)
    mock_dir = tmp_path / "mock_data"
    config_path = tmp_path / "planner_policy.yaml"
    config_path.write_text(
        "\n".join(
            [
                "candidate_generation:",
                "  top_k_activity: 5",
                "  plan_candidate_limit: 42",
                "  route_source_order:",
                "    - live_route_api",
                "    - coordinate_estimate",
                "  local_mock_dir: " + mock_dir.as_posix(),
                "  time_slot:",
                "    default_transition_buffer_min: 18",
                "    prefer_earliest_valid_activity_slot: false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WF_PLANNER_POLICY_PATH", str(config_path))

    assert policy.top_k("top_k_activity", policy.DEFAULT_TOP_K_ACTIVITY) == 5
    assert policy.plan_candidate_limit() == 42
    assert policy.route_source_order() == ["live_route_api", "coordinate_estimate"]
    assert policy.mock_data_dir() == mock_dir
    assert policy.transition_buffer_min() == 18
    assert policy.time_slot_bool("prefer_earliest_valid_activity_slot", True) is False


def test_relative_mock_data_dir_resolves_from_repo_root(tmp_path, monkeypatch) -> None:
    _clear_policy_env(monkeypatch)
    monkeypatch.setenv("WF_MOCK_DATA_DIR", "experiments/mock_data")

    expected = Path(policy.__file__).resolve().parents[2] / "experiments" / "mock_data"
    assert policy.mock_data_dir() == expected


def test_legacy_multinode_fallback_env_overrides_policy(tmp_path, monkeypatch) -> None:
    _clear_policy_env(monkeypatch)
    config_path = tmp_path / "planner_policy.yaml"
    config_path.write_text(
        "\n".join(
            [
                "candidate_generation:",
                "  allow_legacy_fallback_for_multinode: false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WF_PLANNER_POLICY_PATH", str(config_path))
    monkeypatch.setenv("WF_B_ALLOW_LEGACY_FALLBACK_FOR_MULTINODE", "true")

    assert policy.allow_legacy_fallback_for_multinode() is True
