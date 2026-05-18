"""Graph expansion helpers for Memory Atom retrieval."""

from __future__ import annotations

from collections import deque
from typing import Any

from src.memory.policy import companion_roles, has_child_request, has_spouse_request
from src.memory.utils import as_list, unique


def query_entities_from_constraints(constraints: dict[str, Any]) -> list[str]:
    """Infer graph starting entities from current planner constraints."""

    entities: list[str] = []
    roles = companion_roles(constraints)
    if has_child_request(constraints) or constraints.get("scene") == "family":
        entities.append("child")
    if has_spouse_request(constraints):
        entities.append("wife")
    if constraints.get("scene") in {"family", "friends", "couple", "solo"}:
        entities.append("user")

    entities.extend(
        role for role in roles if role in {"child", "wife", "partner", "spouse"}
    )
    entities.extend(
        tag
        for tag in [
            constraints.get("mom_diet"),
            *as_list(constraints.get("soft_tags")),
            *as_list(constraints.get("hard_tags")),
            *as_list(constraints.get("active_value_ids")),
        ]
        if tag in {"low_calorie", "light_food", "health", "family_care"}
    )
    return [str(entity) for entity in unique(entities)]


def expand_graph_memory_ids(
    memory: dict[str, Any],
    constraints: dict[str, Any],
    *,
    max_hops: int = 2,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Return memory ids discovered by bounded entity/relation expansion."""

    graph = memory.get("graph") or {}
    entities = graph.get("entities") if isinstance(graph, dict) else {}
    relations = graph.get("relations") if isinstance(graph, dict) else []
    if not isinstance(entities, dict) or not isinstance(relations, list):
        return {}, []

    starts = query_entities_from_constraints(constraints)
    hits: dict[str, dict[str, Any]] = {}
    paths: list[dict[str, Any]] = []
    queue = deque((entity, 0, [entity]) for entity in starts)
    visited = {(entity, 0) for entity in starts}

    while queue:
        entity, hop, path = queue.popleft()
        entity_record = entities.get(entity)
        if isinstance(entity_record, dict):
            for memory_id in entity_record.get("memory_ids", []) or []:
                if not memory_id:
                    continue
                hit = hits.setdefault(
                    str(memory_id),
                    {
                        "source": "graph",
                        "hop": hop,
                        "path": list(path),
                    },
                )
                if hop < hit.get("hop", hop):
                    hit.update({"hop": hop, "path": list(path)})

        if hop >= max_hops:
            continue

        for relation in relations:
            if not isinstance(relation, dict):
                continue
            source = str(relation.get("source") or "")
            target = str(relation.get("target") or "")
            if not source or not target:
                continue
            if entity not in {source, target}:
                continue

            next_entity = target if entity == source else source
            next_path = [
                *path,
                str(relation.get("predicate") or "related_to"),
                next_entity,
            ]
            relation_memory_id = relation.get("memory_id")
            if relation_memory_id:
                hits.setdefault(
                    str(relation_memory_id),
                    {
                        "source": "graph",
                        "hop": hop + 1,
                        "path": next_path,
                    },
                )
            paths.append(
                {
                    "memory_id": relation_memory_id,
                    "source": source,
                    "predicate": relation.get("predicate"),
                    "target": target,
                    "hop": hop + 1,
                    "path": next_path,
                }
            )
            visit_key = (next_entity, hop + 1)
            if visit_key not in visited:
                visited.add(visit_key)
                queue.append((next_entity, hop + 1, next_path))

    return hits, paths
