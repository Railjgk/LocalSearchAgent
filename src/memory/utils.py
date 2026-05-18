"""Shared helpers for the WeekendFlow memory subsystem."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def as_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def unique(values: Any) -> list[Any]:
    seen = set()
    result = []
    for value in as_list(values):
        if value in (None, ""):
            continue
        key = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        elif isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = unique([*merged[key], *value])
        else:
            merged[key] = deepcopy(value)
    return merged


def to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            value = digits
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
