"""Validate whether unrelated samples leak target-value cues."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.value_probe.constants import VALUE_IDS, canonical_value_id
from experiments.value_probe.control_tasks import OPPOSITE_KEYWORDS, VALUE_KEYWORDS
from experiments.value_probe.data import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit with a non-zero status when keyword leakage is found.",
    )
    return parser.parse_args()


def keyword_hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [keyword for keyword in keywords if keyword and keyword in text]


def row_relation(row: dict[str, Any], value_id: str) -> str:
    relations = row.get("value_relations")
    if isinstance(relations, dict) and value_id in relations:
        return str(relations[value_id])
    return str(row.get("relation", ""))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    totals = {
        value_id: {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "positive_keyword_hit_count": 0,
            "opposite_keyword_hit_count": 0,
        }
        for value_id in VALUE_IDS
    }
    hit_keywords: dict[str, dict[str, int]] = {
        value_id: defaultdict(int) for value_id in VALUE_IDS
    }
    failures: list[dict[str, Any]] = []
    num_input_rows = 0
    num_unrelated_rows = 0

    for path in args.input:
        for row in read_jsonl(path):
            num_input_rows += 1
            value_id = canonical_value_id(str(row.get("target_value", row.get("value_id", ""))))
            if value_id not in VALUE_IDS or row_relation(row, value_id) != "unrelated":
                continue

            num_unrelated_rows += 1
            text = str(row.get("text", ""))
            positive_hits = keyword_hits(text, VALUE_KEYWORDS.get(value_id, ()))
            opposite_hits = keyword_hits(text, OPPOSITE_KEYWORDS.get(value_id, ()))
            total_hits = positive_hits + opposite_hits

            value_totals = totals[value_id]
            value_totals["total"] += 1
            value_totals["positive_keyword_hit_count"] += len(positive_hits)
            value_totals["opposite_keyword_hit_count"] += len(opposite_hits)
            for keyword in total_hits:
                hit_keywords[value_id][keyword] += 1

            if total_hits:
                value_totals["failed"] += 1
                failures.append(
                    {
                        "input": str(path),
                        "example_id": row.get("example_id", ""),
                        "target_value": value_id,
                        "positive_hits": "|".join(positive_hits),
                        "opposite_hits": "|".join(opposite_hits),
                        "text": text,
                    }
                )
            else:
                value_totals["passed"] += 1

    by_value: list[dict[str, Any]] = []
    for value_id in VALUE_IDS:
        value_totals = totals[value_id]
        total = int(value_totals["total"])
        failed = int(value_totals["failed"])
        by_value.append(
            {
                "value_id": value_id,
                **value_totals,
                "failure_rate": failed / total if total else None,
                "hit_keywords": dict(sorted(hit_keywords[value_id].items())),
            }
        )

    summary = {
        "input": [str(path) for path in args.input],
        "num_input_rows": num_input_rows,
        "num_unrelated_rows": num_unrelated_rows,
        "num_failures": len(failures),
        "failure_rate": len(failures) / num_unrelated_rows if num_unrelated_rows else None,
        "by_value": by_value,
    }

    summary_path = args.output_dir / "unrelated_validation.json"
    failures_path = args.output_dir / "unrelated_validation_failures.csv"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with failures_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "input",
                "example_id",
                "target_value",
                "positive_hits",
                "opposite_hits",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(failures)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if failures and args.fail_on_issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
