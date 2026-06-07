from __future__ import annotations

from src.nodes.b_multinode_policy import (
    apply_blueprint_duration_defaults,
    can_plan_multinode_with_current_supply,
    can_plan_multinode_with_rag,
    can_plan_partial_multinode,
    mark_rag_resolved_blueprint_roles,
)


def test_current_supply_multinode_requires_supported_domains_without_named_entities() -> None:
    blueprint = {
        "template_mode": "multi_node",
        "node_intents": [
            {"node_id": "n1", "supply_domain": "activity"},
            {"node_id": "n2", "supply_domain": "restaurant"},
            {"node_id": "n3", "supply_domain": "activity"},
        ],
    }

    assert can_plan_multinode_with_current_supply(blueprint) is True

    assert can_plan_multinode_with_current_supply({**blueprint, "named_entities": ["some venue"]}) is False
    assert can_plan_multinode_with_current_supply({**blueprint, "unsupported_roles": ["lodging"]}) is False
    assert can_plan_multinode_with_current_supply(
        {**blueprint, "node_intents": blueprint["node_intents"][:2]}
    ) is False


def test_rag_can_cover_unsupported_node_ids_without_claiming_named_entities() -> None:
    blueprint = {
        "template_mode": "multi_node",
        "node_intents": [
            {"node_id": "n1", "role": "lodging", "supply_domain": "lodging"},
            {"node_id": "n2", "role": "dinner", "supply_domain": "restaurant"},
        ],
    }
    coverage = {
        "covered_node_ids": ["n1"],
        "unsupported_roles_covered": True,
    }

    assert can_plan_multinode_with_rag(blueprint, coverage) is True
    assert can_plan_partial_multinode(blueprint, coverage) is True
    assert can_plan_multinode_with_rag({**blueprint, "named_entities": ["exact hotel"]}, coverage) is False


def test_mark_rag_resolved_blueprint_roles_preserves_unresolved_roles() -> None:
    blueprint = {
        "unsupported_roles": ["lodging", "parking"],
        "node_intents": [
            {"node_id": "hotel_1", "role": "lodging"},
            {"node_id": "parking_1", "role": "parking"},
        ],
    }
    coverage = {"covered_node_ids": ["hotel_1"]}

    marked = mark_rag_resolved_blueprint_roles(blueprint, coverage)

    assert marked["unsupported_roles"] == ["parking"]
    assert marked["rag_resolved_roles"] == ["lodging"]
    assert blueprint["unsupported_roles"] == ["lodging", "parking"]


def test_blueprint_duration_defaults_expand_only_missing_duration() -> None:
    assert apply_blueprint_duration_defaults(
        {},
        {"template_mode": "multi_node", "planning_horizon": "full_day"},
    ) == {"duration_range": [420, 720]}

    assert apply_blueprint_duration_defaults(
        {"duration_range": [60, 120]},
        {"template_mode": "multi_node", "planning_horizon": "two_day"},
    ) == {"duration_range": [60, 120]}

    assert apply_blueprint_duration_defaults(
        {},
        {"template_mode": "multi_node", "node_count": 3},
    ) == {"duration_range": [180, 540]}
