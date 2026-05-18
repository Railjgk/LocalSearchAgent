"""Hybrid memory retrieval used by the A-stage memory node."""

from __future__ import annotations

from typing import Any

from src.memory.graph_index import expand_graph_memory_ids
from src.memory.policy import (
    has_child_request,
    has_spouse_request,
    select_active_value_memory,
)
from src.memory.schema import build_memory_atom
from src.memory.semantic import semantic_search_memories
from src.memory.utils import as_list, unique


def _score_memory_item(item: dict[str, Any], constraints: dict[str, Any]) -> float:
    query_tags = unique(
        [
            constraints.get("scene"),
            *as_list(constraints.get("hard_tags")),
            *as_list(constraints.get("soft_tags")),
            *as_list(constraints.get("avoid")),
            *as_list(constraints.get("active_value_ids")),
        ]
    )
    item_tags = set(str(tag) for tag in item.get("tags", []) or [])
    query_tag_set = set(str(tag) for tag in query_tags if tag not in (None, ""))
    overlap = len(item_tags.intersection(query_tag_set))

    base_by_kind = {
        "value": 0.55,
        "companion": 0.50,
        "preference": 0.48,
        "stable_profile": 0.38,
        "feedback": 0.34,
        "episode": 0.24,
    }
    base = base_by_kind.get(item.get("kind"), 0.25)
    confidence = float(item.get("confidence", 0.5) or 0.5)
    return base + min(0.25, overlap * 0.08) + min(0.2, confidence * 0.2)


def _fallback_profile_items(
    constraints: dict[str, Any],
    memory: dict[str, Any],
) -> list[dict[str, Any]]:
    user_id = str(memory.get("user_id") or "u001")
    items: list[dict[str, Any]] = []
    stable_profile = memory.get("stable_profile", {}) or {}
    preference_profile = memory.get("preference_profile", {}) or {}

    items.append(
        build_memory_atom(
            "stable_profile",
            "stable_profile",
            "long_term",
            "用户长期稳定画像",
            [
                stable_profile.get("consumption_level"),
                stable_profile.get("default_transport"),
            ],
            stable_profile,
            user_id=user_id,
            source="stable_profile",
            confidence=0.82,
            entities=["user"],
        )
    )

    items.append(
        build_memory_atom(
            "preference_profile",
            "preference",
            "long_term",
            "用户偏好和规避项",
            [
                *as_list(preference_profile.get("food")),
                *as_list(preference_profile.get("activity")),
                *as_list(preference_profile.get("avoid")),
            ],
            preference_profile,
            user_id=user_id,
            source="preference_profile",
            confidence=0.78,
            entities=["user"],
        )
    )

    companion_profile = memory.get("companion_profile", {}) or {}
    if has_child_request(constraints) or constraints.get("scene") == "family":
        child_profile = companion_profile.get("child")
        if isinstance(child_profile, dict):
            items.append(
                build_memory_atom(
                    "companion_child",
                    "companion",
                    child_profile.get("ttl", "long_term"),
                    "常见同行孩子画像",
                    ["child", *as_list(child_profile.get("needs"))],
                    child_profile,
                    user_id=user_id,
                    source=child_profile.get("source", "companion_profile"),
                    confidence=child_profile.get("confidence", 0.75),
                    entities=["child"],
                )
            )

    if has_spouse_request(constraints):
        spouse_profile = companion_profile.get("wife")
        if isinstance(spouse_profile, dict) and spouse_profile.get("state"):
            items.append(
                build_memory_atom(
                    "companion_wife",
                    "companion",
                    spouse_profile.get("ttl", "short_term"),
                    "伴侣近期饮食状态",
                    [
                        "wife",
                        spouse_profile.get("state"),
                        *as_list(spouse_profile.get("needs")),
                    ],
                    spouse_profile,
                    user_id=user_id,
                    source=spouse_profile.get("source", "companion_profile"),
                    confidence=spouse_profile.get("confidence", 0.7),
                    ttl_turns=spouse_profile.get("ttl_turns"),
                    entities=["wife"],
                )
            )

    for item in select_active_value_memory(constraints, memory):
        items.append(
            build_memory_atom(
                f"value_{item.get('value_id')}",
                "value",
                item.get("ttl", "long_term"),
                item.get("label", item.get("value_id", "value")),
                [item.get("value_id"), *as_list(item.get("evidence"))],
                dict(item),
                user_id=user_id,
                source=item.get("source", "value_profile"),
                confidence=item.get("confidence", 0.7),
                entities=["user"],
            )
        )

    for idx, feedback in enumerate(memory.get("history_feedback", []) or []):
        if not isinstance(feedback, dict):
            continue
        items.append(
            build_memory_atom(
                f"feedback_{feedback.get('plan_id', idx)}",
                "feedback",
                "long_term",
                "历史方案反馈",
                [
                    *as_list(feedback.get("positive")),
                    *as_list(feedback.get("negative")),
                ],
                feedback,
                user_id=user_id,
                source=feedback.get("source", "history_feedback"),
                confidence=feedback.get("confidence", 0.65),
                entities=["user"],
            )
        )

    return items


def _scene_safety_reason(
    item: dict[str, Any],
    constraints: dict[str, Any],
) -> str | None:
    memory_id = item.get("memory_id")
    if memory_id == "companion_child":
        if not (has_child_request(constraints) or constraints.get("scene") == "family"):
            return "child_memory_requires_family_or_child_request"
    if memory_id == "companion_wife":
        if not has_spouse_request(constraints):
            return "spouse_memory_requires_spouse_request"
    if memory_id == "value_family_care":
        if not (has_child_request(constraints) or constraints.get("scene") == "family"):
            return "family_value_requires_family_scene"
    if memory_id == "value_health":
        if "health" not in (constraints.get("active_value_ids") or []):
            return "health_value_not_active"
    tags = {str(tag) for tag in item.get("tags", []) or []}
    entities = {str(entity) for entity in item.get("entities", []) or []}
    if "child" in tags.union(entities) and not (
        has_child_request(constraints) or constraints.get("scene") == "family"
    ):
        return "child_tag_requires_family_or_child_request"
    if "wife" in tags.union(entities) and not has_spouse_request(constraints):
        return "spouse_tag_requires_spouse_request"
    return None


def _scope_score(item: dict[str, Any]) -> float:
    if item.get("scope") == "short_term":
        return 0.04
    if item.get("scope") == "long_term":
        return 0.02
    return 0.0


def _semantic_score(item: dict[str, Any]) -> float:
    score = item.get("semantic_score")
    if score is None:
        return 0.0
    try:
        return min(0.12, max(0.0, float(score)) * 0.12)
    except (TypeError, ValueError):
        return 0.0


def _candidate_memories(
    constraints: dict[str, Any],
    memory: dict[str, Any],
    semantic_store: Any = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    user_id = str(memory.get("user_id") or "u001")
    graph_hits, graph_paths = expand_graph_memory_ids(memory, constraints)
    semantic_items, semantic_trace = semantic_search_memories(
        semantic_store,
        user_id,
        constraints,
    )

    grouped_candidates = [
        ("sparse", _fallback_profile_items(constraints, memory)),
        (
            "atom",
            [item for item in memory.get("atoms", []) or [] if isinstance(item, dict)],
        ),
        (
            "short_term",
            [
                item
                for item in memory.get("short_term_items", []) or []
                if isinstance(item, dict)
            ],
        ),
        (
            "episodic",
            [
                item
                for item in memory.get("episodic_memory", []) or []
                if isinstance(item, dict)
            ],
        ),
        ("semantic", semantic_items),
    ]

    candidates: dict[str, dict[str, Any]] = {}
    for source, items in grouped_candidates:
        for item in items:
            memory_id = item.get("memory_id")
            if not memory_id:
                continue
            candidate = candidates.setdefault(str(memory_id), dict(item))
            candidate_sources = set(candidate.get("retrieval_sources", []))
            candidate_sources.add(source)
            candidate["retrieval_sources"] = sorted(candidate_sources)
            if item.get("semantic_score") is not None:
                candidate["semantic_score"] = item.get("semantic_score")

    for memory_id, graph_hit in graph_hits.items():
        candidate = candidates.get(memory_id)
        if not candidate:
            continue
        candidate_sources = set(candidate.get("retrieval_sources", []))
        candidate_sources.add("graph")
        candidate["retrieval_sources"] = sorted(candidate_sources)
        candidate["graph_hop"] = graph_hit.get("hop")
        candidate["graph_path"] = graph_hit.get("path")

    return candidates, {
        "graph": {
            "status": "ok",
            "hits": graph_hits,
            "paths": graph_paths,
        },
        "semantic": semantic_trace,
    }


def retrieve_relevant_memories_with_trace(
    constraints: dict[str, Any],
    memory: dict[str, Any],
    limit: int = 12,
    *,
    semantic_store: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve safe memories and return retrieval audit details."""

    candidates, source_trace = _candidate_memories(constraints, memory, semantic_store)
    skipped: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []

    for item in candidates.values():
        reason = _scene_safety_reason(item, constraints)
        if reason:
            skipped.append(
                {
                    "memory_id": item.get("memory_id"),
                    "reason": reason,
                    "sources": item.get("retrieval_sources", []),
                }
            )
            continue

        scored_item = dict(item)
        graph_bonus = (
            0.08 if "graph" in scored_item.get("retrieval_sources", []) else 0.0
        )
        scored_item["relevance"] = round(
            _score_memory_item(scored_item, constraints)
            + graph_bonus
            + _semantic_score(scored_item)
            + _scope_score(scored_item),
            3,
        )
        scored_item["expansion_level"] = (
            "payload" if scored_item["relevance"] >= 0.78 else "summary"
        )
        scored.append(scored_item)

    scored.sort(
        key=lambda item: (
            item.get("relevance", 0.0),
            item.get("confidence", 0.0),
            1 if item.get("scope") == "short_term" else 0,
        ),
        reverse=True,
    )
    retrieved = scored[:limit]
    return retrieved, {
        "sources": source_trace,
        "skipped": skipped,
        "retrieved": [
            {
                "memory_id": item.get("memory_id"),
                "sources": item.get("retrieval_sources", []),
                "relevance": item.get("relevance"),
                "graph_hop": item.get("graph_hop"),
                "expansion_level": item.get("expansion_level"),
            }
            for item in retrieved
        ],
    }


def retrieve_relevant_memories(
    constraints: dict[str, Any],
    memory: dict[str, Any],
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Retrieve the memory items that are safe and relevant for this turn."""

    retrieved, _trace = retrieve_relevant_memories_with_trace(
        constraints,
        memory,
        limit,
    )
    return retrieved
