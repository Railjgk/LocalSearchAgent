"""Thin orchestration node for the WeekendFlow memory subsystem."""

from __future__ import annotations

from src.memory.ingestion import (
    apply_memory_updates,
    build_memory_trace,
    current_turn_memory_atoms,
)
from src.memory.policy import (
    apply_value_memory,
    build_user_profile,
    select_active_value_memory,
)
from src.memory.retrieval import (
    retrieve_relevant_memories,
    retrieve_relevant_memories_with_trace,
)
from src.memory.schema import (
    DEFAULT_MEMORY,
    DEFAULT_USER_ID,
    DEFAULT_VALUE_MEMORY,
    MEMORY_POLICY,
    MEMORY_STORE_VERSION,
)
from src.memory.semantic import persist_memory_to_store
from src.memory.storage import (
    JSONMemoryStore,
    decay_short_term_items,
    load_memory,
    save_memory,
)
from src.nodes._utils import append_log, merge_tool_results
from src.state import PlanState


def _runtime_store():
    try:
        from langgraph.config import get_store

        return get_store()
    except Exception:
        return None


def _load_decayed_memory(state: PlanState) -> dict:
    return decay_short_term_items(
        load_memory(state.get("user_id"), state.get("memory") or None)
    )


def _short_term_memory_with_turn(state: PlanState) -> list:
    short_term_memory = list(state.get("short_term_memory", []))
    if state.get("user_input"):
        short_term_memory.append(state["user_input"])
    return short_term_memory


def _memory_tool_payload(
    *,
    updated_memory: dict,
    merged_constraints: dict,
    active_value_memory: dict,
    retrieved_memories: list,
    memory_updates: list,
    retrieval_trace: dict,
    semantic_sync: dict,
    user_profile: dict,
) -> dict:
    return {
        "memory": updated_memory,
        "merged_constraints": merged_constraints,
        "active_value_memory": active_value_memory,
        "retrieved_memories": retrieved_memories,
        "memory_updates": memory_updates,
        "retrieval_trace": retrieval_trace,
        "semantic_sync": semantic_sync,
        "user_profile": user_profile,
    }


def memory_manager_node(state: PlanState) -> dict:
    constraints = state.get("constraints", {})
    base_store = _runtime_store()
    memory = _load_decayed_memory(state)
    merged_constraints = apply_value_memory(constraints, memory)
    active_value_memory = select_active_value_memory(merged_constraints, memory)
    retrieved_memories, retrieval_trace = retrieve_relevant_memories_with_trace(
        merged_constraints,
        memory,
        semantic_store=base_store,
    )
    user_profile = build_user_profile(
        merged_constraints,
        memory,
        active_value_memory,
        retrieved_memories,
    )
    turn_atoms = current_turn_memory_atoms(
        state,
        merged_constraints,
        active_value_memory,
    )
    updated_memory, memory_updates = apply_memory_updates(memory, turn_atoms)
    save_memory(updated_memory)
    semantic_sync = persist_memory_to_store(base_store, updated_memory)

    memory_trace = build_memory_trace(
        retrieved_memories,
        memory_updates,
        active_value_memory,
        retrieval_trace,
        semantic_sync,
    )
    tool_payload = _memory_tool_payload(
        updated_memory=updated_memory,
        merged_constraints=merged_constraints,
        active_value_memory=active_value_memory,
        retrieved_memories=retrieved_memories,
        memory_updates=memory_updates,
        retrieval_trace=retrieval_trace,
        semantic_sync=semantic_sync,
        user_profile=user_profile,
    )
    return {
        "constraints": merged_constraints,
        "memory": updated_memory,
        "user_profile": user_profile,
        "value_memory": active_value_memory,
        "retrieved_memories": retrieved_memories,
        "memory_updates": memory_updates,
        "memory_trace": memory_trace,
        "short_term_memory": _short_term_memory_with_turn(state),
        "tool_results": merge_tool_results(state, "memory_manager", tool_payload),
        "execution_log": append_log(
            state,
            "[memory_manager] retrieved profile memory, applied values, and persisted turn facts",
        ),
    }


__all__ = [
    "DEFAULT_MEMORY",
    "DEFAULT_USER_ID",
    "DEFAULT_VALUE_MEMORY",
    "JSONMemoryStore",
    "MEMORY_POLICY",
    "MEMORY_STORE_VERSION",
    "apply_memory_updates",
    "apply_value_memory",
    "build_memory_trace",
    "build_user_profile",
    "current_turn_memory_atoms",
    "decay_short_term_items",
    "load_memory",
    "memory_manager_node",
    "retrieve_relevant_memories",
    "retrieve_relevant_memories_with_trace",
    "save_memory",
    "select_active_value_memory",
]
