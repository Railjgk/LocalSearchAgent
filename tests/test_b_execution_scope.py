from __future__ import annotations

from src.nodes.b_execution_scope import (
    node_is_supported_by_current_c,
    node_requires_c_execution,
)


def test_guidance_only_roles_do_not_require_c_execution() -> None:
    node = {"itinerary_role": "parking", "type": "activity"}

    assert node_requires_c_execution(node) is False
    assert node_is_supported_by_current_c(node) is True


def test_lodging_requires_and_is_supported_by_c_execution() -> None:
    node = {"itinerary_role": "lodging", "type": "poi"}

    assert node_requires_c_execution(node) is True
    assert node_is_supported_by_current_c(node) is True


def test_non_executable_supply_domain_stays_guidance_only_unless_supported() -> None:
    node = {"role": "citywalk_market", "supply_domain": "poi"}
    unknown = {"role": "dental_clinic", "supply_domain": "dental"}

    assert node_requires_c_execution(node) is False
    assert node_requires_c_execution(unknown) is False
    assert node_is_supported_by_current_c(unknown) is False
