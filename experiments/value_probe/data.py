"""Dataset utilities for weakly supervised value probe training."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from experiments.value_probe.constants import (
    NEGATIVE_RELATION,
    POSITIVE_RELATION,
    RELATIONS,
    VALUE_IDS,
    VALUE_TO_INDEX,
)


@dataclass(frozen=True)
class ProbeExample:
    """One weak supervision row for a target value."""

    example_id: str
    text: str
    target_value: str
    relation: str
    split: str = ""

    @property
    def target_index(self) -> int:
        return VALUE_TO_INDEX[self.target_value]

    def positive_label_vector(self) -> list[float]:
        labels = [0.0] * len(VALUE_IDS)
        labels[self.target_index] = 1.0 if self.relation == POSITIVE_RELATION else 0.0
        return labels

    def negative_label_vector(self) -> list[float]:
        labels = [0.0] * len(VALUE_IDS)
        labels[self.target_index] = 1.0 if self.relation == NEGATIVE_RELATION else 0.0
        return labels

    def supervision_mask_vector(self) -> list[float]:
        mask = [0.0] * len(VALUE_IDS)
        mask[self.target_index] = 1.0
        return mask

    def to_json(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "text": self.text,
            "target_value": self.target_value,
            "target_index": self.target_index,
            "relation": self.relation,
            "split": self.split,
            "positive_labels": self.positive_label_vector(),
            "negative_labels": self.negative_label_vector(),
            "supervision_mask": self.supervision_mask_vector(),
        }


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
        value_id = str(row.get("value_id", "")).strip()
        relation = str(row.get("relation", "")).strip()
        if not text:
            raise ValueError(f"Missing text at input row {row_index}")
        if value_id not in VALUE_TO_INDEX:
            raise ValueError(f"Unsupported value_id {value_id!r} at input row {row_index}")
        if relation not in RELATIONS:
            raise ValueError(f"Unsupported relation {relation!r} at input row {row_index}")
        key = (value_id, relation, text)
        if key in seen:
            continue
        seen.add(key)
        examples.append(
            ProbeExample(
                example_id=f"{value_id}-{relation}-{row_index:05d}",
                text=text,
                target_value=value_id,
                relation=relation,
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
                    split=split,
                )
            )
    split_examples.sort(key=lambda item: item.example_id)
    return split_examples


def summarize_examples(examples: Iterable[ProbeExample]) -> dict[str, object]:
    examples = list(examples)
    counts = Counter((example.split or "unsplit", example.target_value, example.relation) for example in examples)
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
    }


def rows_for_split(path: Path, split: str | None = None) -> list[dict[str, object]]:
    rows = read_jsonl(path)
    if split is None:
        return rows
    return [row for row in rows if row.get("split") == split]

