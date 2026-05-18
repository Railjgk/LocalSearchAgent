"""Current-turn memory ingestion and profile updates."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

from src.memory.policy import has_child_request, has_spouse_request
from src.memory.schema import (
    MAX_EPISODIC_ITEMS,
    MAX_SHORT_TERM_ITEMS,
    MEMORY_POLICY,
    build_memory_atom,
)
from src.memory.storage import build_memory_graph
from src.memory.utils import as_list, now_iso, unique
from src.state import PlanState, ValueMemoryItem


def current_turn_memory_atoms(
    state: PlanState,
    constraints: dict[str, Any],
    active_value_memory: list[ValueMemoryItem],
) -> list[dict[str, Any]]:
    user_input = str(state.get("user_input") or "").strip()
    if not user_input:
        return []

    user_id = str(state.get("user_id") or "u001")
    timestamp = now_iso()
    tags = [
        constraints.get("scene"),
        *as_list(constraints.get("hard_tags")),
        *as_list(constraints.get("soft_tags")),
        *as_list(constraints.get("avoid")),
        *[item.get("value_id") for item in active_value_memory],
    ]

    atoms = [
        build_memory_atom(
            f"episode_{uuid4().hex[:10]}",
            "episode",
            "short_term",
            user_input,
            tags,
            {
                "user_input": user_input,
                "constraints": deepcopy(constraints),
                "created_at": timestamp,
            },
            user_id=user_id,
            source="current_turn",
            confidence=0.9,
            ttl_turns=6,
            raw_ref={"type": "inline", "text": user_input},
            created_at=timestamp,
            updated_at=timestamp,
        )
    ]

    child_age = constraints.get("child_age")
    if has_child_request(constraints) and child_age not in (None, ""):
        atoms.append(
            build_memory_atom(
                "current_child_profile",
                "companion",
                "long_term",
                f"孩子年龄 {child_age}",
                ["child", "kid_friendly", "low_intensity"],
                {
                    "role": "child",
                    "age": child_age,
                    "needs": ["kid_friendly", "low_intensity"],
                },
                user_id=user_id,
                source="current_turn",
                confidence=0.92,
                entities=["child"],
                relations=[
                    {
                        "source": "user",
                        "predicate": "travels_with",
                        "target": "child",
                        "confidence": 0.92,
                    }
                ],
                created_at=timestamp,
                updated_at=timestamp,
            )
        )

    mom_diet = constraints.get("mom_diet")
    if has_spouse_request(constraints) and mom_diet == "low_calorie":
        atoms.append(
            build_memory_atom(
                "current_spouse_diet",
                "companion",
                "short_term",
                "伴侣近期偏低卡轻食",
                ["wife", "dieting", "low_calorie", "light_food"],
                {
                    "role": "wife",
                    "state": "dieting",
                    "needs": ["low_calorie", "light_food"],
                },
                user_id=user_id,
                source="current_turn",
                confidence=0.9,
                ttl_turns=4,
                entities=["wife"],
                relations=[
                    {
                        "source": "wife",
                        "predicate": "prefers",
                        "target": "low_calorie",
                        "confidence": 0.9,
                    }
                ],
                created_at=timestamp,
                updated_at=timestamp,
            )
        )

    return atoms


def _upsert_memory_item(
    items: list[dict[str, Any]],
    new_item: dict[str, Any],
    max_items: int,
) -> list[dict[str, Any]]:
    memory_id = new_item.get("memory_id")
    retained = [
        item
        for item in items
        if isinstance(item, dict) and item.get("memory_id") != memory_id
    ]
    retained.append(new_item)
    return retained[-max_items:]


def _duplicate_signature(item: dict[str, Any]) -> tuple[Any, ...]:
    payload = item.get("payload", {}) or {}
    return (
        item.get("kind"),
        item.get("scope"),
        item.get("summary") or item.get("text"),
        tuple(sorted(str(tag) for tag in item.get("tags", []) or [])),
        payload.get("role"),
        payload.get("state"),
        payload.get("age"),
    )


def _find_duplicate_atom(
    atoms: list[dict[str, Any]],
    new_atom: dict[str, Any],
) -> dict[str, Any] | None:
    new_id = new_atom.get("memory_id")
    new_signature = _duplicate_signature(new_atom)
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        if new_id and atom.get("memory_id") == new_id:
            return atom
        if _duplicate_signature(atom) == new_signature:
            return atom
    return None


def _refresh_duplicate_atom(
    existing_atom: dict[str, Any],
    new_atom: dict[str, Any],
) -> dict[str, Any]:
    refreshed = dict(new_atom)
    refreshed["memory_id"] = existing_atom.get("memory_id")
    refreshed["created_at"] = existing_atom.get("created_at") or new_atom.get(
        "created_at"
    )
    refreshed["updated_at"] = now_iso()
    return refreshed


def _upsert_atom(memory: dict[str, Any], atom: dict[str, Any]) -> None:
    memory["atoms"] = _upsert_memory_item(memory.get("atoms", []) or [], atom, 200)
    memory["graph"] = build_memory_graph(memory["atoms"])


def apply_memory_updates(
    memory: dict[str, Any],
    turn_atoms: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Write current-turn facts into the compatibility memory projection."""

    updated = deepcopy(memory)
    operations: list[dict[str, Any]] = []

    for atom in turn_atoms:
        existing_atom = _find_duplicate_atom(updated.get("atoms", []) or [], atom)
        op = "refresh" if existing_atom else "upsert"
        if existing_atom:
            atom = _refresh_duplicate_atom(existing_atom, atom)
        kind = atom.get("kind")
        payload = atom.get("payload", {}) or {}
        _upsert_atom(updated, atom)

        if kind == "episode":
            updated["episodic_memory"] = _upsert_memory_item(
                updated.get("episodic_memory", []) or [],
                atom,
                MAX_EPISODIC_ITEMS,
            )
            updated["short_term_items"] = _upsert_memory_item(
                updated.get("short_term_items", []) or [],
                atom,
                MAX_SHORT_TERM_ITEMS,
            )
            operations.append(
                {
                    "op": op,
                    "target": "episodic_memory",
                    "memory_id": atom.get("memory_id"),
                    "ttl_turns": atom.get("ttl_turns"),
                    "reason": "duplicate_fact" if op == "refresh" else "new_fact",
                }
            )

        elif kind == "companion" and payload.get("role") == "child":
            child_profile = updated.setdefault("companion_profile", {}).setdefault(
                "child",
                {},
            )
            child_profile.update(
                {
                    "age": payload.get("age"),
                    "needs": unique(payload.get("needs")),
                    "confidence": atom.get("confidence", 0.85),
                    "source": atom.get("source", "current_turn"),
                    "ttl": atom.get("scope", "long_term"),
                    "updated_at": now_iso(),
                }
            )
            operations.append(
                {
                    "op": op,
                    "target": "companion_profile.child",
                    "memory_id": atom.get("memory_id"),
                    "reason": "duplicate_fact" if op == "refresh" else "new_fact",
                }
            )

        elif kind == "companion" and payload.get("role") == "wife":
            wife_profile = updated.setdefault("companion_profile", {}).setdefault(
                "wife",
                {},
            )
            wife_profile.update(
                {
                    "state": payload.get("state"),
                    "needs": unique(payload.get("needs")),
                    "confidence": atom.get("confidence", 0.8),
                    "source": atom.get("source", "current_turn"),
                    "ttl": atom.get("scope", "short_term"),
                    "ttl_turns": atom.get("ttl_turns"),
                    "updated_at": now_iso(),
                }
            )
            updated["short_term_items"] = _upsert_memory_item(
                updated.get("short_term_items", []) or [],
                atom,
                MAX_SHORT_TERM_ITEMS,
            )
            operations.append(
                {
                    "op": op,
                    "target": "companion_profile.wife",
                    "memory_id": atom.get("memory_id"),
                    "ttl_turns": atom.get("ttl_turns"),
                    "reason": "duplicate_fact" if op == "refresh" else "new_fact",
                }
            )

    return updated, operations


def build_memory_trace(
    retrieved_memories: list[dict[str, Any]],
    memory_updates: list[dict[str, Any]],
    active_value_memory: list[ValueMemoryItem],
    retrieval_trace: dict[str, Any] | None = None,
    semantic_sync: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a compact audit payload for downstream explanations and tests."""

    return {
        "policy": MEMORY_POLICY,
        "retrieved": [
            {
                "memory_id": item.get("memory_id"),
                "kind": item.get("kind"),
                "scope": item.get("scope"),
                "source": item.get("source"),
                "retrieval_sources": item.get("retrieval_sources", []),
                "confidence": item.get("confidence"),
                "relevance": item.get("relevance"),
                "graph_hop": item.get("graph_hop"),
                "expansion_level": item.get("expansion_level"),
            }
            for item in retrieved_memories
        ],
        "skipped": (retrieval_trace or {}).get("skipped", []),
        "retrieval_sources": (retrieval_trace or {}).get("sources", {}),
        "active_value_ids": [item.get("value_id") for item in active_value_memory],
        "updates": memory_updates,
        "semantic_sync": semantic_sync
        or {
            "backend": "base_store",
            "status": "skipped",
            "reason": "no_base_store",
        },
    }
