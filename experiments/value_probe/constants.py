"""Shared constants for value probe experiments."""

from __future__ import annotations

VALUE_IDS = ("family_care", "health", "convenience", "cost_sensitivity")
VALUE_TO_INDEX = {value_id: index for index, value_id in enumerate(VALUE_IDS)}
RELATIONS = ("related", "opposite", "unrelated")

POSITIVE_RELATION = "related"
NEGATIVE_RELATION = "opposite"

