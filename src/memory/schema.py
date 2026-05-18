"""Memory Atom schema and deterministic seed memory."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, TypedDict

from src.memory.utils import now_iso, unique
from src.state import ValueMemoryItem


MEMORY_POLICY = "explicit_current_input_first"
MEMORY_STORE_VERSION = 2
DEFAULT_USER_ID = "u001"
MAX_EPISODIC_ITEMS = 20
MAX_SHORT_TERM_ITEMS = 12


class MemoryRelation(TypedDict, total=False):
    source: str
    predicate: str
    target: str
    confidence: float


class MemoryAtom(TypedDict, total=False):
    memory_id: str
    user_id: str
    modality: Literal["text"]
    kind: str
    summary: str
    text: str
    raw_ref: dict[str, Any]
    tags: list[Any]
    entities: list[str]
    relations: list[MemoryRelation]
    scope: str
    ttl_turns: int | None
    confidence: float
    source: str
    payload: dict[str, Any]
    relevance: float
    created_at: str
    updated_at: str


DEFAULT_MEMORY: dict[str, Any] = {
    "user_id": DEFAULT_USER_ID,
    "stable_profile": {
        "home_area": "unknown",
        "consumption_level": "middle",
        "default_transport": "drive_or_taxi",
    },
    "companion_profile": {
        "child": {
            "age": 5,
            "needs": ["kid_friendly", "low_intensity"],
            "confidence": 0.9,
            "source": "historical_profile",
            "ttl": "long_term",
        },
        "wife": {
            "state": "dieting",
            "needs": ["low_calorie", "light_food"],
            "ttl": "short_term",
            "confidence": 0.8,
            "source": "recent_user_input",
        },
    },
    "preference_profile": {
        "food": ["light_food", "japanese"],
        "activity": ["indoor", "parent_child", "light_activity"],
        "avoid": ["long_queue", "crowded_mall"],
    },
    "history_feedback": [
        {
            "plan_id": "p001",
            "positive": ["kid_happy"],
            "negative": ["too_crowded", "too_far"],
            "confidence": 0.82,
            "source": "history_feedback",
        },
    ],
    "derived_defaults": {
        "max_distance_km": 8.0,
        "max_queue_time_min": 15,
        "preferred_duration_hours": [4, 6],
    },
    "episodic_memory": [],
    "short_term_items": [],
}


DEFAULT_VALUE_MEMORY: list[ValueMemoryItem] = [
    {
        "value_id": "family_care",
        "label": "儿童优先和家庭舒适",
        "score": 0.92,
        "confidence": 0.9,
        "ttl": "long_term",
        "source": "historical_profile",
        "planning_effect": "increase group_fit and require kid_friendly activities",
        "evidence": ["常与5岁孩子同行", "历史反馈偏好低强度活动"],
    },
    {
        "value_id": "health",
        "label": "健康饮食",
        "score": 0.82,
        "confidence": 0.78,
        "ttl": "short_term",
        "source": "recent_user_input",
        "planning_effect": "prefer low_calorie and light_food restaurants",
        "evidence": ["妻子处于减脂状态", "偏好轻食或日料"],
    },
    {
        "value_id": "convenience",
        "label": "少排队和少折腾",
        "score": 0.76,
        "confidence": 0.84,
        "ttl": "long_term",
        "source": "history_feedback",
        "planning_effect": "penalize long_queue, crowded_mall and far routes",
        "evidence": ["用户历史负反馈: too_crowded", "当前输入包含别太远"],
    },
    {
        "value_id": "cost_sensitivity",
        "label": "中等预算",
        "score": 0.45,
        "confidence": 0.55,
        "ttl": "long_term",
        "source": "stable_profile",
        "planning_effect": "keep budget reasonable but do not force cheapest option",
        "evidence": ["消费层级为 middle"],
    },
]


def build_memory_atom(
    memory_id: str,
    kind: str,
    scope: str,
    summary: str,
    tags: Any,
    payload: dict[str, Any] | None = None,
    *,
    user_id: str = DEFAULT_USER_ID,
    source: str = "memory",
    confidence: float = 0.7,
    ttl_turns: int | None = None,
    relevance: float = 0.0,
    raw_ref: dict[str, Any] | None = None,
    entities: Any = None,
    relations: list[MemoryRelation] | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> MemoryAtom:
    timestamp = created_at or now_iso()
    return {
        "memory_id": memory_id,
        "user_id": user_id,
        "modality": "text",
        "kind": kind,
        "summary": summary,
        "text": summary,
        "raw_ref": raw_ref or {"type": "payload"},
        "tags": unique(tags),
        "entities": [str(item) for item in unique(entities)],
        "relations": relations or [],
        "scope": scope,
        "ttl_turns": ttl_turns,
        "confidence": round(float(confidence), 3),
        "source": source,
        "payload": payload or {},
        "relevance": round(float(relevance), 3),
        "created_at": timestamp,
        "updated_at": updated_at or timestamp,
    }


def atom_from_legacy_item(
    item: dict[str, Any],
    *,
    user_id: str,
) -> MemoryAtom:
    summary = str(item.get("summary") or item.get("text") or "")
    return build_memory_atom(
        str(item.get("memory_id") or item.get("id") or "legacy_atom"),
        str(item.get("kind") or "episode"),
        str(item.get("scope") or "short_term"),
        summary,
        item.get("tags", []),
        item.get("payload", {}),
        user_id=user_id,
        source=str(item.get("source") or "legacy_memory"),
        confidence=float(item.get("confidence", 0.7) or 0.7),
        ttl_turns=item.get("ttl_turns"),
        relevance=float(item.get("relevance", 0.0) or 0.0),
        raw_ref=item.get("raw_ref"),
        entities=item.get("entities", []),
        relations=item.get("relations", []),
        created_at=item.get("created_at"),
        updated_at=item.get("updated_at"),
    )


def default_value_memory() -> list[ValueMemoryItem]:
    return deepcopy(DEFAULT_VALUE_MEMORY)
