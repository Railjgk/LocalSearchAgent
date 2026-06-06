"""Build a balanced JSONL subset from LLM-generated value probe samples."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.value_probe.constants import RELATIONS, VALUE_IDS, canonical_value_id
from experiments.value_probe.data import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument(
        "--exclude-example-ids",
        type=Path,
        help="Optional newline, CSV, or JSONL file containing example_ids to skip.",
    )
    return parser.parse_args()


def group_key(row: dict[str, Any]) -> tuple[str, str]:
    value_id = canonical_value_id(str(row.get("target_value", row.get("value_id", ""))))
    relation = str(row.get("relation", ""))
    return value_id, relation


def read_excluded_example_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    excluded: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                example_id = str(row.get("example_id", "")).strip()
            elif "," in line:
                example_id = line.split(",", 1)[0].strip()
            else:
                example_id = line
            if example_id and example_id != "example_id":
                excluded.add(example_id)
    return excluded


def main() -> int:
    args = parse_args()
    if args.per_group <= 0:
        raise ValueError("--per-group must be positive")

    rows = [row for path in args.input for row in read_jsonl(path)]
    excluded_example_ids = read_excluded_example_ids(args.exclude_example_ids)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if str(row.get("example_id", "")).strip() in excluded_example_ids:
            continue
        value_id, relation = group_key(row)
        if value_id not in VALUE_IDS or relation not in RELATIONS:
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        dedupe_key = (value_id, relation, text)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized = dict(row)
        normalized["target_value"] = value_id
        normalized["value_id"] = value_id
        normalized["relation"] = relation
        buckets[(value_id, relation)].append(normalized)

    rng = random.Random(args.seed)
    balanced: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []
    for value_id in VALUE_IDS:
        for relation in RELATIONS:
            bucket = list(buckets.get((value_id, relation), []))
            if len(bucket) < args.per_group:
                raise ValueError(
                    f"Not enough rows for {value_id}/{relation}: "
                    f"{len(bucket)} < {args.per_group}"
                )
            rng.shuffle(bucket)
            selected = bucket[: args.per_group]
            selected.sort(key=lambda row: str(row.get("example_id", "")))
            for index, row in enumerate(selected, start=1):
                output_row = dict(row)
                output_row["example_id"] = f"balanced-{value_id}-{relation}-{index:04d}"
                balanced.append(output_row)
            counts.append(
                {
                    "value_id": value_id,
                    "relation": relation,
                    "available": len(bucket),
                    "selected": len(selected),
                }
            )

    balanced.sort(
        key=lambda row: (
            str(row.get("target_value", "")),
            str(row.get("relation", "")),
            str(row.get("example_id", "")),
        )
    )
    write_jsonl(args.output, balanced)
    summary = {
        "input": [str(path) for path in args.input],
        "exclude_example_ids": str(args.exclude_example_ids)
        if args.exclude_example_ids
        else None,
        "num_excluded_example_ids": len(excluded_example_ids),
        "output": str(args.output),
        "num_input_rows": len(rows),
        "num_output_rows": len(balanced),
        "per_group": args.per_group,
        "counts": counts,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
