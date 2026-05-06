"""Shared state and output types for the WeekendFlow A-stage demo."""

from __future__ import annotations

from typing import Any, TypedDict


class ValueMemoryItem(TypedDict):
    """A calculable personal value used to steer planning."""

    value_id: str
    label: str
    score: float
    confidence: float
    ttl: str
    source: str
    planning_effect: str
    evidence: list[str]


class PlanState(TypedDict, total=False):
    """Global state for input parsing and value-aware memory."""

    user_input: str
    user_id: str
    scene_type: str
    intent: dict[str, Any]
    memory: dict[str, Any]
    constraints: dict[str, Any]
    user_profile: dict[str, Any]
    value_memory: list[ValueMemoryItem]
    short_term_memory: list[str]
    tool_results: dict[str, Any]
    execution_log: list[str]
