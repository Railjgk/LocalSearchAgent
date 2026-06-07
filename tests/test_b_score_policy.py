from __future__ import annotations

from src.nodes import b_score_policy as policy


def _clear_policy_caches() -> None:
    policy.load_policy_config.cache_clear()
    policy.load_scene_weights_from_policy.cache_clear()
    policy.load_score_thresholds_from_policy.cache_clear()
    policy.load_penalties_from_policy.cache_clear()


def test_default_weights_returns_scene_specific_copy() -> None:
    couple = policy.default_weights("couple")
    couple["route"] = 9.9

    assert policy.default_weights("couple")["atmosphere"] == 0.17
    assert policy.default_weights("unknown") == policy.default_weights("family")


def test_policy_file_overrides_scene_weights_thresholds_and_penalties(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "planner_policy.yaml"
    config_path.write_text(
        "\n".join(
            [
                "scene_weights:",
                "  couple:",
                "    route: 0.31",
                "    budget: 0.04",
                "    unsupported: 9.9",
                "score_thresholds:",
                "  route:",
                "    absolute_max_distance_km: 9.5",
                "penalties:",
                "  long_queue: 0.42",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WF_PLANNER_POLICY_PATH", str(config_path))
    _clear_policy_caches()

    weights = policy.load_scene_weights_from_policy(policy.policy_cache_key())

    assert weights["couple"]["route"] == 0.31
    assert weights["couple"]["budget"] == 0.04
    assert "unsupported" not in weights["couple"]
    assert policy.get_threshold("route", "absolute_max_distance_km", 15.0) == 9.5
    assert policy.get_penalty("long_queue", 0.25) == 0.42


def test_missing_policy_file_keeps_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WF_PLANNER_POLICY_PATH", str(tmp_path / "missing.yaml"))
    _clear_policy_caches()

    assert policy.load_policy_config(policy.policy_cache_key()) == {}
    assert policy.get_threshold("route", "absolute_max_distance_km", 15.0) == 15.0
    assert policy.get_penalty("long_queue", 0.25) == 0.25
