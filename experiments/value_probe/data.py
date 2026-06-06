"""Dataset utilities for weakly supervised value probe training."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Iterable

from experiments.value_probe.constants import (
    NEGATIVE_RELATION,
    POSITIVE_RELATION,
    RELATIONS,
    VALUE_IDS,
    VALUE_TO_INDEX,
    canonical_value_id,
)


@dataclass(frozen=True)
class ProbeExample:
    """One weak supervision row for a target value."""

    example_id: str
    text: str
    target_value: str
    relation: str
    labels: dict[str, float] = field(default_factory=dict)
    value_relations: dict[str, str] = field(default_factory=dict)
    supervision_values: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, object] = field(default_factory=dict)
    split: str = ""

    @property
    def target_index(self) -> int:
        return VALUE_TO_INDEX[self.target_value]

    def positive_label_vector(self) -> list[float]:
        if self.labels:
            return [1.0 if float(self.labels.get(value_id, 0.0)) > 0 else 0.0 for value_id in VALUE_IDS]
        vector = [0.0] * len(VALUE_IDS)
        vector[self.target_index] = 1.0 if self.relation == POSITIVE_RELATION else 0.0
        return vector

    def negative_label_vector(self) -> list[float]:
        if self.labels:
            return [1.0 if float(self.labels.get(value_id, 0.0)) < 0 else 0.0 for value_id in VALUE_IDS]
        vector = [0.0] * len(VALUE_IDS)
        vector[self.target_index] = 1.0 if self.relation == NEGATIVE_RELATION else 0.0
        return vector

    def supervision_mask_vector(self) -> list[float]:
        if self.supervision_values:
            supervised = set(self.supervision_values)
            return [1.0 if value_id in supervised else 0.0 for value_id in VALUE_IDS]
        if self.labels:
            return [1.0 if value_id in self.labels else 0.0 for value_id in VALUE_IDS]
        mask = [0.0] * len(VALUE_IDS)
        mask[self.target_index] = 1.0
        return mask

    def to_json(self) -> dict[str, object]:
        row = {
            "example_id": self.example_id,
            "text": self.text,
            "target_value": self.target_value,
            "value_id": self.target_value,
            "target_index": self.target_index,
            "relation": self.relation,
            "split": self.split,
            "positive_labels": self.positive_label_vector(),
            "negative_labels": self.negative_label_vector(),
            "supervision_mask": self.supervision_mask_vector(),
        }
        if self.labels:
            row["labels"] = {value_id: self.labels.get(value_id, 0.0) for value_id in VALUE_IDS}
        if self.value_relations:
            row["value_relations"] = {
                value_id: self.value_relations.get(value_id, "unrelated")
                for value_id in VALUE_IDS
            }
        if self.supervision_values:
            row["supervision_values"] = list(self.supervision_values)
        for key, value in self.metadata.items():
            if key not in row:
                row[key] = value
        return row


def _canonical_labels(raw_labels: object, target_value: str, relation: str) -> dict[str, float]:
    labels = {value_id: 0.0 for value_id in VALUE_IDS}
    if isinstance(raw_labels, dict):
        for raw_value, raw_score in raw_labels.items():
            value_id = canonical_value_id(str(raw_value))
            if value_id in VALUE_TO_INDEX:
                labels[value_id] = float(raw_score or 0.0)
        return labels
    labels[target_value] = 6.0 if relation == POSITIVE_RELATION else -6.0 if relation == NEGATIVE_RELATION else 0.0
    return labels


def _canonical_relations(
    raw_relations: object,
    labels: dict[str, float],
    target_value: str,
    relation: str,
) -> dict[str, str]:
    relations = {value_id: "unrelated" for value_id in VALUE_IDS}
    if isinstance(raw_relations, dict):
        for raw_value, raw_relation in raw_relations.items():
            value_id = canonical_value_id(str(raw_value))
            if value_id in VALUE_TO_INDEX and str(raw_relation) in RELATIONS:
                relations[value_id] = str(raw_relation)
    for value_id, score in labels.items():
        if score > 0 and relations[value_id] == "unrelated":
            relations[value_id] = "related"
        elif score < 0 and relations[value_id] == "unrelated":
            relations[value_id] = "opposite"
    relations[target_value] = relation
    return relations


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_raw_examples(path: Path) -> list[ProbeExample]:
    examples: list[ProbeExample] = []
    seen: set[tuple[str, str, str]] = set()
    for row_index, row in enumerate(read_jsonl(path)):
        text = str(row.get("text", "")).strip()
        raw_value_id = row.get("target_value", row.get("value_id", ""))
        value_id = canonical_value_id(str(raw_value_id))
        relation = str(row.get("relation", "")).strip()
        if not text:
            raise ValueError(f"Missing text at input row {row_index}")
        if value_id not in VALUE_TO_INDEX:
            raise ValueError(f"Unsupported value_id {value_id!r} at input row {row_index}")
        if relation not in RELATIONS:
            raise ValueError(f"Unsupported relation {relation!r} at input row {row_index}")
        text_type = str(row.get("text_type", "concrete_query")).strip() or "concrete_query"
        key = (value_id, relation, text, text_type)
        if key in seen:
            continue
        seen.add(key)
        labels = _canonical_labels(row.get("labels"), value_id, relation)
        value_relations = _canonical_relations(row.get("value_relations"), labels, value_id, relation)
        raw_supervision_values = row.get("supervision_values")
        if isinstance(raw_supervision_values, list):
            supervision_values = tuple(
                value_id
                for value_id in (canonical_value_id(str(value)) for value in raw_supervision_values)
                if value_id in VALUE_TO_INDEX
            )
        else:
            supervision_values = ()
        metadata = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "example_id",
                "text",
                "target_value",
                "value_id",
                "target_index",
                "relation",
                "split",
                "positive_labels",
                "negative_labels",
                "supervision_mask",
                "labels",
                "value_relations",
                "supervision_values",
            }
        }
        example_id = str(row.get("example_id") or f"{value_id}-{relation}-{row_index:05d}")
        examples.append(
            ProbeExample(
                example_id=example_id,
                text=text,
                target_value=value_id,
                relation=relation,
                labels=labels,
                value_relations=value_relations,
                supervision_values=supervision_values,
                metadata=metadata,
            )
        )
    return examples


def stratified_split(
    examples: list[ProbeExample],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> list[ProbeExample]:
    """Split by `(target_value, relation)` so all labels stay balanced."""

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must satisfy train > 0, val > 0, train + val < 1")
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[ProbeExample]] = defaultdict(list)
    for example in examples:
        buckets[(example.target_value, example.relation)].append(example)

    split_examples: list[ProbeExample] = []
    for key in sorted(buckets):
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        n_total = len(bucket)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        n_train = max(1, min(n_train, n_total - 2))
        n_val = max(1, min(n_val, n_total - n_train - 1))
        for index, example in enumerate(bucket):
            if index < n_train:
                split = "train"
            elif index < n_train + n_val:
                split = "val"
            else:
                split = "test"
            split_examples.append(
                ProbeExample(
                    example_id=example.example_id,
                    text=example.text,
                    target_value=example.target_value,
                    relation=example.relation,
                    labels=dict(example.labels),
                    value_relations=dict(example.value_relations),
                    supervision_values=tuple(example.supervision_values),
                    metadata=dict(example.metadata),
                    split=split,
                )
            )
    split_examples.sort(key=lambda item: item.example_id)
    return split_examples


def summarize_examples(examples: Iterable[ProbeExample]) -> dict[str, object]:
    examples = list(examples)
    counts = Counter((example.split or "unsplit", example.target_value, example.relation) for example in examples)
    text_type_counts = Counter(
        (example.split or "unsplit", str(example.metadata.get("text_type", "")) or "unknown")
        for example in examples
    )
    text_counts = Counter(example.text for example in examples)
    suspicious_short = [
        example.example_id
        for example in examples
        if len(example.text) < 12
    ]
    duplicate_texts = [text for text, count in text_counts.items() if count > 1]
    return {
        "num_examples": len(examples),
        "num_duplicate_texts": len(duplicate_texts),
        "duplicate_text_examples": duplicate_texts[:20],
        "num_suspicious_short": len(suspicious_short),
        "suspicious_short_example_ids": suspicious_short[:20],
        "counts": [
            {
                "split": split,
                "value_id": value_id,
                "relation": relation,
                "count": count,
            }
            for (split, value_id, relation), count in sorted(counts.items())
        ],
        "text_type_counts": [
            {
                "split": split,
                "text_type": text_type,
                "count": count,
            }
            for (split, text_type), count in sorted(text_type_counts.items())
        ],
        "value_ids": list(VALUE_IDS),
    }


def rows_for_split(path: Path, split: str | None = None) -> list[dict[str, object]]:
    rows = read_jsonl(path)
    if split is None:
        return rows
    return [row for row in rows if row.get("split") == split]
