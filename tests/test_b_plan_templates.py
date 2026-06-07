from __future__ import annotations

from src.nodes import b_candidate_policy
from src.nodes.b_plan_templates import get_plan_templates, is_supported_plan_template


def _clear_policy(monkeypatch) -> None:
    monkeypatch.delenv("WF_PLANNER_POLICY_PATH", raising=False)
    b_candidate_policy.load_policy_config.cache_clear()


def test_supported_template_requires_one_activity_and_one_restaurant() -> None:
    assert is_supported_plan_template(["activity", "restaurant"]) is True
    assert is_supported_plan_template(["activity", "transition", "restaurant"]) is True
    assert is_supported_plan_template(["activity", "activity", "restaurant"]) is False
    assert is_supported_plan_template(["restaurant"]) is False


def test_plan_templates_fall_back_without_policy(monkeypatch) -> None:
    _clear_policy(monkeypatch)

    assert get_plan_templates("family") == [["activity", "transition", "restaurant"]]


def test_plan_templates_filter_unsupported_and_dedupe_policy_entries(tmp_path, monkeypatch) -> None:
    _clear_policy(monkeypatch)
    config_path = tmp_path / "planner_policy.yaml"
    config_path.write_text(
        "\n".join(
            [
                "template_policy:",
                "  scene_templates:",
                "    family:",
                "      - activity",
                "      - activity",
                "      - restaurant",
                "  default_template:",
                "    - restaurant",
                "    - activity",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WF_PLANNER_POLICY_PATH", str(config_path))
    b_candidate_policy.load_policy_config.cache_clear()

    assert get_plan_templates("family") == [["restaurant", "activity"]]
