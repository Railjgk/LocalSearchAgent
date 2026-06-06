"""B-side execution scope rules for C handoff readiness.

B may plan guidance-only nodes such as city walks or parking hints, but only
some itinerary nodes should be sent into C's executable mock API flow.
"""
from __future__ import annotations


GUIDANCE_ONLY_ITINERARY_ROLES = {
    "citywalk_market",
    "park_scenic_walk",
    "convenience_store",
    "souvenir_shopping",
    "parking",
    "nail_salon",
    "pet_grooming",
    "pet_hospital",
    "pet_store",
}

CURRENT_C_EXECUTABLE_NODE_TYPES = {"activity", "restaurant", "hotel", "lodging"}


def node_requires_c_execution(node: dict) -> bool:
    role = str(node.get("itinerary_role") or node.get("role") or "")
    node_type = str(node.get("type") or node.get("supply_domain") or "")
    if role in GUIDANCE_ONLY_ITINERARY_ROLES:
        return False
    if role == "lodging" or node_type in {"hotel", "lodging"}:
        return True
    return node_type in CURRENT_C_EXECUTABLE_NODE_TYPES


def node_is_supported_by_current_c(node: dict) -> bool:
    role = str(node.get("itinerary_role") or node.get("role") or "")
    node_type = str(node.get("type") or node.get("supply_domain") or "")
    if role == "lodging":
        return True
    return node_type in CURRENT_C_EXECUTABLE_NODE_TYPES
