"""Optional LangGraph BaseStore adapter for semantic memory search."""

from __future__ import annotations

from typing import Any

from src.memory.utils import as_list, unique

MEMORY_NAMESPACE_ROOT = ("weekendflow", "memory")


def memory_namespace(user_id: str) -> tuple[str, ...]:
    return (*MEMORY_NAMESPACE_ROOT, str(user_id))


def query_text_from_constraints(constraints: dict[str, Any]) -> str:
    parts = unique(
        [
            constraints.get("scene"),
            *as_list(constraints.get("hard_tags")),
            *as_list(constraints.get("soft_tags")),
            *as_list(constraints.get("avoid")),
            *as_list(constraints.get("active_value_ids")),
            constraints.get("mom_diet"),
            constraints.get("budget_type"),
        ]
    )
    return " ".join(str(part) for part in parts if part not in (None, ""))


def semantic_search_memories(
    store: Any,
    user_id: str,
    constraints: dict[str, Any],
    *,
    limit: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search BaseStore if available; callers can always fall back to sparse."""

    if store is None:
        return [], {
            "source": "semantic",
            "status": "skipped",
            "reason": "no_base_store",
        }

    query = query_text_from_constraints(constraints)
    if not query:
        return [], {
            "source": "semantic",
            "status": "skipped",
            "reason": "empty_query",
        }

    try:
        results = store.search(memory_namespace(user_id), query=query, limit=limit)
    except Exception as exc:  # pragma: no cover - exact adapters vary by environment
        return [], {
            "source": "semantic",
            "status": "error",
            "reason": exc.__class__.__name__,
        }

    memories: list[dict[str, Any]] = []
    for result in results or []:
        value = getattr(result, "value", None)
        if not isinstance(value, dict):
            continue
        atom = dict(value)
        atom.setdefault("memory_id", getattr(result, "key", None))
        score = getattr(result, "score", None)
        if score is not None:
            atom["semantic_score"] = round(float(score), 3)
        memories.append(atom)

    return memories, {
        "source": "semantic",
        "status": "ok",
        "query": query,
        "count": len(memories),
    }


def persist_memory_to_store(store: Any, memory: dict[str, Any]) -> dict[str, Any]:
    """Persist profile and atoms to a LangGraph BaseStore-like backend."""

    if store is None:
        return {
            "backend": "base_store",
            "status": "skipped",
            "reason": "no_base_store",
        }

    user_id = str(memory.get("user_id") or "u001")
    namespace = memory_namespace(user_id)
    atoms = [atom for atom in memory.get("atoms", []) or [] if isinstance(atom, dict)]
    stored = 0
    errors: list[str] = []

    try:
        store.put(
            namespace,
            "__profile__",
            {
                "memory_id": "__profile__",
                "user_id": user_id,
                "kind": "profile",
                "summary": "WeekendFlow memory profile",
                "text": "WeekendFlow memory profile",
                "profile": memory.get("profile", {}),
            },
            index=["summary", "text"],
        )
    except Exception as exc:  # pragma: no cover - adapter-specific
        errors.append(exc.__class__.__name__)

    for atom in atoms:
        memory_id = atom.get("memory_id")
        if not memory_id:
            continue
        try:
            store.put(namespace, str(memory_id), dict(atom), index=["summary", "text"])
            stored += 1
        except Exception as exc:  # pragma: no cover - adapter-specific
            errors.append(exc.__class__.__name__)

    if errors:
        return {
            "backend": "base_store",
            "status": "partial_error" if stored else "error",
            "stored_atoms": stored,
            "errors": errors[:3],
        }
    return {
        "backend": "base_store",
        "status": "ok",
        "stored_atoms": stored,
        "namespace": namespace,
    }
