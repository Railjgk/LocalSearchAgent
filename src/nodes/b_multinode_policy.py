"""Planning policy helpers for B multi-node itineraries."""
from __future__ import annotations


MULTINODE_SUPPORTED_DOMAINS = {"activity", "restaurant"}


def can_plan_multinode_with_current_supply(blueprint: dict | None) -> bool:
    """Return true when the current local activity/restaurant supply can cover the blueprint."""

    blueprint = blueprint or {}
    if blueprint.get("template_mode") != "multi_node":
        return False
    if blueprint.get("unsupported_roles"):
        return False
    if blueprint.get("named_entities"):
        # Exact venue/event names need retrieval evidence rather than generic local pools.
        return False
    node_intents = blueprint.get("node_intents") or []
    if len(node_intents) <= 2:
        return False
    return all(
        str(intent.get("supply_domain") or "") in MULTINODE_SUPPORTED_DOMAINS
        for intent in node_intents
    )


def can_plan_multinode_with_rag(blueprint: dict | None, coverage: dict | None) -> bool:
    """Return true when retrieval can cover nodes the local pair supply cannot."""

    blueprint = blueprint or {}
    coverage = coverage or {}
    if blueprint.get("template_mode") != "multi_node":
        return False
    node_intents = blueprint.get("node_intents") or []
    if not node_intents:
        return False
    if coverage.get("all_nodes_covered"):
        return True
    if blueprint.get("named_entities"):
        return False
    if not coverage.get("unsupported_roles_covered"):
        return False

    covered_node_ids = set(coverage.get("covered_node_ids") or [])
    return all(
        str(intent.get("supply_domain") or "") in MULTINODE_SUPPORTED_DOMAINS
        or str(intent.get("node_id") or "") in covered_node_ids
        for intent in node_intents
    )


def can_plan_partial_multinode(blueprint: dict | None, coverage: dict | None) -> bool:
    """Return true when at least one itinerary node can be planned by local supply or RAG."""

    blueprint = blueprint or {}
    if blueprint.get("template_mode") != "multi_node":
        return False

    covered_node_ids = set((coverage or {}).get("covered_node_ids") or [])
    return any(
        str(intent.get("supply_domain") or "") in MULTINODE_SUPPORTED_DOMAINS
        or str(intent.get("node_id") or "") in covered_node_ids
        for intent in blueprint.get("node_intents", []) or []
    )


def mark_rag_resolved_blueprint_roles(blueprint: dict | None, coverage: dict | None) -> dict:
    """Clear unsupported role flags once local POI RAG has concrete candidates."""

    blueprint = dict(blueprint or {})
    coverage = coverage or {}
    unsupported_roles = list(blueprint.get("unsupported_roles") or [])
    if not unsupported_roles:
        return blueprint

    covered_node_ids = set(coverage.get("covered_node_ids") or [])
    resolved_roles: list[str] = []
    remaining_roles: list[str] = []
    for intent in blueprint.get("node_intents") or []:
        role = str(intent.get("role") or "")
        if role not in unsupported_roles:
            continue
        if str(intent.get("node_id") or "") in covered_node_ids:
            resolved_roles.append(role)
        else:
            remaining_roles.append(role)

    if not resolved_roles:
        return blueprint

    blueprint["unsupported_roles"] = remaining_roles
    blueprint["rag_resolved_roles"] = sorted(set(resolved_roles))
    return blueprint


def apply_blueprint_duration_defaults(constraints: dict, blueprint: dict | None) -> dict:
    """Widen duration defaults when the request itself asks for a longer itinerary."""

    blueprint = blueprint or {}
    if blueprint.get("template_mode") != "multi_node":
        return constraints
    if constraints.get("duration_range") not in (None, "") or constraints.get("duration") not in (None, ""):
        return constraints

    horizon = blueprint.get("planning_horizon")
    enhanced = dict(constraints)
    if horizon in {"overnight", "two_day"}:
        enhanced["duration_range"] = [480, 1200]
    elif horizon == "full_day":
        enhanced["duration_range"] = [420, 720]
    elif int(blueprint.get("node_count") or len(blueprint.get("node_intents") or [])) >= 3:
        enhanced["duration_range"] = [180, 540]
    return enhanced
