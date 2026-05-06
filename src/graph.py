"""Graph assembly for the WeekendFlow A-stage demo."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.nodes.intent_parser import intent_parser_node
from src.nodes.memory_manager import memory_manager_node
from src.state import PlanState


NODE_SEQUENCE: list[tuple[str, Callable[[PlanState], dict[str, Any]]]] = [
    ("intent_parser", intent_parser_node),
    ("memory_manager", memory_manager_node),
]


class SimpleWeekendFlowAStageApp:
    """Small fallback runner with the same `invoke` shape as a compiled graph."""

    def invoke(self, state: PlanState) -> PlanState:
        current: PlanState = dict(state)
        for _name, node in NODE_SEQUENCE:
            updates = node(current)
            current.update(updates)
        return current


def build_graph() -> Any:
    """Build a LangGraph workflow when available, otherwise use a simple runner."""

    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return SimpleWeekendFlowAStageApp()

    workflow = StateGraph(PlanState)
    for name, node in NODE_SEQUENCE:
        workflow.add_node(name, node)

    workflow.set_entry_point("intent_parser")
    for (source, _), (target, _) in zip(NODE_SEQUENCE, NODE_SEQUENCE[1:]):
        workflow.add_edge(source, target)
    workflow.add_edge("memory_manager", END)
    return workflow.compile()


def get_graph() -> Any:
    return build_graph()
