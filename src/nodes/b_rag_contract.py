"""Normalize RAG/POI retrieval results into B-stage itinerary candidates.

The retrieval layer is intentionally kept outside B.  B only needs a stable
contract: for each itinerary node, provide a short list of POI candidates with
enough evidence to schedule, score, and explain a plan.
"""

from __future__ import annotations

from typing import Any

try:
    from src.state import PlanState
except ImportError:  # pragma: no cover
    PlanState = dict

from .b_semantics import flatten_semantic_values
from .b_utils import to_float


RAG_CANDIDATE_KEYS = (
    "b_rag_node_candidates",
    "rag_node_candidates",
)
RAG_CANDIDATE_EVIDENCE_KEYS = (
    "b_rag_candidate_evidence",
    "rag_candidate_evidence",
)

SUPPORTED_EXECUTION_DOMAINS = {"activity", "restaurant"}
DEFAULT_ROLE_DURATIONS = {
    "lodging": 720,
    "restaurant_breakfast": 45,
    "restaurant_lunch": 70,
    "restaurant_dinner": 80,
    "restaurant_specific": 75,
    "cafe": 60,
    "family_activity": 120,
    "family_indoor_play": 150,
    "exhibition": 120,
    "citywalk_market": 120,
    "board_game_escape": 120,
    "karaoke": 120,
    "bar": 75,
    "talk_show": 100,
    "cinema": 120,
    "souvenir_shopping": 45,
    "beauty_cosmetics": 35,
    "nail_salon": 70,
    "flower_shop": 25,
    "convenience_store": 20,
    "wellness_massage": 60,
    "pet_grooming": 90,
    "pet_cafe": 60,
    "pet_hospital": 60,
    "pet_store": 35,
    "parking": 15,
}
DOMAIN_TO_NODE_TYPE = {
    "activity": "activity",
    "restaurant": "restaurant",
    "hotel": "hotel",
    "lodging": "hotel",
    "shopping": "shopping",
    "retail": "retail",
    "transport_service": "transport_service",
    "wellness": "wellness",
    "pet_service": "pet_service",
    "beauty_service": "beauty_service",
}


def _first_present(record: dict, keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default


def _coordinate_pair(record: dict) -> tuple[float | None, float | None]:
    coordinates = record.get("coordinates") or record.get("location")
    if isinstance(coordinates, str) and "," in coordinates:
        lng, lat = coordinates.split(",", 1)
    elif isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
        lng, lat = coordinates[0], coordinates[1]
    else:
        lng = _first_present(record, ("longitude", "lng", "lon"))
        lat = _first_present(record, ("latitude", "lat"))
    try:
        return float(lng), float(lat)
    except (TypeError, ValueError):
        return None, None


def _candidate_container(value: Any) -> bool:
    return isinstance(value, dict) and "candidates" in value and isinstance(value.get("candidates"), list)


def _looks_like_candidate(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return any(key in value for key in ("poi_id", "amap_id", "gaode_id", "id", "name"))


def _iter_candidate_blocks(raw: Any) -> list[tuple[str | None, list[dict]]]:
    blocks: list[tuple[str | None, list[dict]]] = []
    if raw is None:
        return blocks

    if isinstance(raw, dict) and isinstance(raw.get("node_evidence"), list):
        for node_block in raw.get("node_evidence", []):
            if not isinstance(node_block, dict):
                continue
            selector = str(
                _first_present(node_block, ("node_id", "role", "supply_domain"), "") or ""
            ) or None
            candidates = []
            for candidate in node_block.get("candidates", []) or []:
                if not isinstance(candidate, dict):
                    continue
                enriched = dict(candidate)
                enriched.setdefault("node_id", node_block.get("node_id"))
                enriched.setdefault("role", node_block.get("role"))
                enriched.setdefault("supply_domain", node_block.get("supply_domain"))
                enriched.setdefault("category", node_block.get("label"))
                enriched.setdefault("source", "b_rag_candidate_evidence")
                enriched.setdefault("coverage_status", node_block.get("coverage_status"))
                enriched.setdefault("search_queries", node_block.get("search_queries"))
                if node_block.get("missing_reason") and not enriched.get("evidence_text"):
                    enriched["evidence_text"] = node_block.get("missing_reason")
                candidates.append(enriched)
            blocks.append((selector, candidates))
        return blocks

    if isinstance(raw, list):
        for item in raw:
            if _candidate_container(item):
                selector = str(_first_present(item, ("node_id", "role", "supply_domain"), "") or "") or None
                blocks.append((selector, [c for c in item.get("candidates", []) if isinstance(c, dict)]))
            elif _looks_like_candidate(item):
                blocks.append((None, [item]))
        return blocks

    if _candidate_container(raw):
        selector = str(_first_present(raw, ("node_id", "role", "supply_domain"), "") or "") or None
        blocks.append((selector, [c for c in raw.get("candidates", []) if isinstance(c, dict)]))
        return blocks

    if _looks_like_candidate(raw):
        blocks.append((None, [raw]))
        return blocks

    if isinstance(raw, dict):
        for key, payload in raw.items():
            selector = str(key)
            if isinstance(payload, list):
                blocks.append((selector, [c for c in payload if isinstance(c, dict)]))
            elif _candidate_container(payload):
                payload_selector = str(
                    _first_present(payload, ("node_id", "role", "supply_domain"), selector) or selector
                )
                blocks.append(
                    (
                        payload_selector,
                        [c for c in payload.get("candidates", []) if isinstance(c, dict)],
                    )
                )
            elif _looks_like_candidate(payload):
                blocks.append((selector, [payload]))
    return blocks


def _intent_indexes(blueprint: dict | None) -> tuple[dict[str, dict], dict[str, list[dict]], dict[str, list[dict]]]:
    node_by_id: dict[str, dict] = {}
    nodes_by_role: dict[str, list[dict]] = {}
    nodes_by_domain: dict[str, list[dict]] = {}
    for intent in (blueprint or {}).get("node_intents", []) or []:
        node_id = str(intent.get("node_id") or "")
        if node_id:
            node_by_id[node_id] = intent
        role = str(intent.get("role") or "")
        if role:
            nodes_by_role.setdefault(role, []).append(intent)
        domain = str(intent.get("supply_domain") or "")
        if domain:
            nodes_by_domain.setdefault(domain, []).append(intent)
    return node_by_id, nodes_by_role, nodes_by_domain


def _candidate_role_domain(candidate: dict) -> tuple[str, str]:
    role = str(_first_present(candidate, ("role", "itinerary_role"), "") or "")
    domain = str(_first_present(candidate, ("supply_domain", "domain"), "") or "")
    return role, domain


def _compatible_with_intent(candidate: dict, intent: dict) -> bool:
    """Return whether a RAG candidate can safely attach to a B itinerary node.

    RAG providers may send node ids from their own blueprint.  We treat node_id
    as a shortcut only when role/domain do not contradict B's current blueprint.
    """

    candidate_role, candidate_domain = _candidate_role_domain(candidate)
    intent_role = str(intent.get("role") or "")
    intent_domain = str(intent.get("supply_domain") or "")
    if candidate_role and intent_role and candidate_role != intent_role:
        return False
    if candidate_domain and intent_domain and candidate_domain != intent_domain:
        return False
    return True


def _target_intents(
    *,
    selector: str | None,
    candidate: dict,
    blueprint: dict | None,
    node_by_id: dict[str, dict],
    nodes_by_role: dict[str, list[dict]],
    nodes_by_domain: dict[str, list[dict]],
) -> list[dict]:
    node_id = str(_first_present(candidate, ("node_id", "itinerary_node_id"), selector or "") or "")
    if node_id in node_by_id and _compatible_with_intent(candidate, node_by_id[node_id]):
        return [node_by_id[node_id]]

    selector_text = str(selector or "")
    if selector_text in node_by_id and _compatible_with_intent(candidate, node_by_id[selector_text]):
        return [node_by_id[selector_text]]
    if selector_text in nodes_by_role:
        return nodes_by_role[selector_text]
    if selector_text in nodes_by_domain:
        return nodes_by_domain[selector_text]

    role, domain = _candidate_role_domain(candidate)
    if role in nodes_by_role:
        return nodes_by_role[role]

    if domain in nodes_by_domain:
        return nodes_by_domain[domain]

    node_intents = (blueprint or {}).get("node_intents", []) or []
    return node_intents if len(node_intents) == 1 else []


def _dedupe_tags(values: list[Any]) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for value in flatten_semantic_values(values):
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        tags.append(text)
    return tags


def _normalize_candidate(record: dict, intent: dict) -> dict | None:
    poi_id = str(_first_present(record, ("poi_id", "amap_id", "gaode_id", "id"), "") or "").strip()
    name = str(record.get("name") or "").strip()
    if not poi_id or not name:
        return None

    role = str(intent.get("role") or record.get("role") or record.get("itinerary_role") or "")
    domain = str(
        intent.get("supply_domain")
        or record.get("supply_domain")
        or record.get("domain")
        or ""
    )
    node_type = str(record.get("type") or DOMAIN_TO_NODE_TYPE.get(domain) or "poi")
    if domain == "activity":
        node_type = "activity"
    elif domain == "restaurant":
        node_type = "restaurant"

    longitude, latitude = _coordinate_pair(record)
    coordinates = record.get("coordinates") or record.get("location")
    if not coordinates and longitude is not None and latitude is not None:
        coordinates = [longitude, latitude]

    tags = _dedupe_tags(
        [
            record.get("tags"),
            record.get("tag_groups"),
            record.get("category"),
            record.get("sub_category"),
            record.get("primary_category"),
            record.get("restaurant_category"),
            record.get("business_area"),
            role,
            domain,
            intent.get("label"),
        ]
    )
    evidence = _dedupe_tags(
        [
            record.get("evidence"),
            record.get("evidence_text"),
            record.get("review_keywords"),
            record.get("recommended_dishes"),
            record.get("signature_dishes"),
            record.get("business_hours"),
            record.get("address"),
        ]
    )
    duration = int(
        to_float(
            _first_present(
                record,
                ("duration_min", "visit_duration_min", "stay_duration_min"),
                intent.get("default_duration_min") or DEFAULT_ROLE_DURATIONS.get(role, 60),
            ),
            DEFAULT_ROLE_DURATIONS.get(role, 60),
        )
    )
    normalized = dict(record)
    normalized.update(
        {
            "poi_id": poi_id,
            "name": name,
            "type": node_type,
            "supply_domain": domain,
            "itinerary_role": role,
            "category": (
                record.get("category")
                or record.get("primary_category")
                or record.get("restaurant_category")
                or record.get("gaode_keyword")
                or ""
            ),
            "coordinates": coordinates,
            "longitude": longitude if longitude is not None else record.get("longitude"),
            "latitude": latitude if latitude is not None else record.get("latitude"),
            "price": to_float(_first_present(record, ("price", "avg_price", "price_per_person", "cost"), 0.0), 0.0),
            "duration_min": max(15, duration),
            "rating": to_float(_first_present(record, ("rating", "score"), 4.0), 4.0),
            "queue_time_min": to_float(
                _first_present(record, ("queue_time_min", "wait_time_min", "estimated_wait_min"), 0.0),
                0.0,
            ),
            "available": bool(record.get("available", True)),
            "distance_km": to_float(_first_present(record, ("distance_km", "distance"), 0.0), 0.0),
            "tags": tags,
            "evidence": evidence,
            "evidence_status": "present" if evidence else "missing",
            "source": record.get("source") or "rag_node_candidates",
        }
    )
    return normalized


def _collect_raw_sources(state: PlanState, constraints: dict | None) -> list[Any]:
    sources: list[Any] = []
    constraints = constraints or {}
    for key in RAG_CANDIDATE_KEYS + RAG_CANDIDATE_EVIDENCE_KEYS:
        if state.get(key) is not None:
            sources.append(state.get(key))
        if constraints.get(key) is not None:
            sources.append(constraints.get(key))
    return sources


def normalize_rag_node_candidates(
    state: PlanState,
    constraints: dict | None,
    blueprint: dict | None,
) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    """Return normalized candidates keyed by B itinerary node_id."""

    node_by_id, nodes_by_role, nodes_by_domain = _intent_indexes(blueprint)
    candidates_by_node: dict[str, list[dict]] = {node_id: [] for node_id in node_by_id}
    seen_by_node: dict[str, set[str]] = {node_id: set() for node_id in node_by_id}
    unmatched_count = 0
    raw_count = 0

    for raw_source in _collect_raw_sources(state, constraints):
        for selector, records in _iter_candidate_blocks(raw_source):
            for record in records:
                raw_count += 1
                targets = _target_intents(
                    selector=selector,
                    candidate=record,
                    blueprint=blueprint,
                    node_by_id=node_by_id,
                    nodes_by_role=nodes_by_role,
                    nodes_by_domain=nodes_by_domain,
                )
                if not targets:
                    unmatched_count += 1
                    continue
                for intent in targets:
                    node_id = str(intent.get("node_id") or "")
                    if not node_id:
                        continue
                    normalized = _normalize_candidate(record, intent)
                    if not normalized:
                        unmatched_count += 1
                        continue
                    poi_id = str(normalized.get("poi_id"))
                    if poi_id in seen_by_node.setdefault(node_id, set()):
                        continue
                    seen_by_node[node_id].add(poi_id)
                    candidates_by_node.setdefault(node_id, []).append(normalized)

    for node_id, items in list(candidates_by_node.items()):
        items.sort(
            key=lambda item: (
                item.get("evidence_status") == "present",
                to_float(item.get("rating"), 0.0),
                -to_float(item.get("distance_km"), 0.0),
            ),
            reverse=True,
        )

    covered_node_ids = [node_id for node_id, items in candidates_by_node.items() if items]
    metadata = {
        "raw_candidate_count": raw_count,
        "normalized_candidate_count": sum(len(items) for items in candidates_by_node.values()),
        "covered_node_ids": covered_node_ids,
        "unmatched_candidate_count": unmatched_count,
        "source": "rag_candidate_inputs",
        "supported_input_keys": list(RAG_CANDIDATE_KEYS + RAG_CANDIDATE_EVIDENCE_KEYS),
    }
    return candidates_by_node, metadata


def rag_candidate_coverage(blueprint: dict | None, candidates_by_node: dict[str, list[dict]]) -> dict[str, Any]:
    node_intents = (blueprint or {}).get("node_intents", []) or []
    required_node_ids = [str(intent.get("node_id")) for intent in node_intents if intent.get("node_id")]
    covered_node_ids = [node_id for node_id in required_node_ids if candidates_by_node.get(node_id)]
    missing_node_ids = [node_id for node_id in required_node_ids if node_id not in covered_node_ids]
    unsupported_missing_roles = [
        str(intent.get("role"))
        for intent in node_intents
        if str(intent.get("supply_domain") or "") not in SUPPORTED_EXECUTION_DOMAINS
        and not candidates_by_node.get(str(intent.get("node_id") or ""))
    ]
    return {
        "required_node_count": len(required_node_ids),
        "covered_node_count": len(covered_node_ids),
        "covered_node_ids": covered_node_ids,
        "missing_node_ids": missing_node_ids,
        "all_nodes_covered": bool(required_node_ids) and not missing_node_ids,
        "unsupported_roles_covered": not unsupported_missing_roles,
        "unsupported_missing_roles": unsupported_missing_roles,
    }
