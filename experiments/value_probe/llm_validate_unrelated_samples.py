"""LLM semantic validation for unrelated value-probe samples."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.generate_value_probe_paragraphs import (  # noqa: E402
    VALUE_SPECS,
    load_dotenv,
    load_generation_config,
)
from experiments.value_probe.constants import VALUE_IDS, canonical_value_id  # noqa: E402
from experiments.value_probe.data import read_jsonl  # noqa: E402
from src.nodes.longcat_client import chat_completion, sanitize_longcat_error  # noqa: E402


VALID_STATUSES = {"pass", "fail_related", "fail_opposite", "ambiguous"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        help="Optional CSV with example_id column. When set, validate only those rows.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--request-sleep", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def candidate_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            str(row.get("example_id", "")).strip()
            for row in csv.DictReader(handle)
            if str(row.get("example_id", "")).strip()
        }


def relation_for(row: dict[str, Any], value_id: str) -> str:
    value_relations = row.get("value_relations")
    if isinstance(value_relations, dict) and value_id in value_relations:
        return str(value_relations[value_id])
    return str(row.get("relation", ""))


def load_rows(path: Path, wanted_ids: set[str] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        example_id = str(row.get("example_id", "")).strip()
        if wanted_ids is not None and example_id not in wanted_ids:
            continue
        value_id = canonical_value_id(str(row.get("target_value", row.get("value_id", ""))))
        if value_id not in VALUE_IDS or relation_for(row, value_id) != "unrelated":
            continue
        rows.append(
            {
                "example_id": example_id,
                "target_value": value_id,
                "text": str(row.get("text", "")),
            }
        )
    return rows


def parse_json_array(content: str) -> list[dict[str, Any]]:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        raise ValueError("LLM response must be a JSON array")
    return [item for item in parsed if isinstance(item, dict)]


def build_prompt(batch: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in batch:
        grouped.setdefault(row["target_value"], []).append(
            {"example_id": row["example_id"], "text": row["text"]}
        )

    value_blocks = []
    for value_id, rows in grouped.items():
        spec = VALUE_SPECS[value_id]
        value_blocks.append(
            {
                "target_value": value_id,
                "definition": spec.definition,
                "related_rule": spec.related_requirement,
                "opposite_rule": spec.opposite_requirement,
                "supposed_unrelated_rows": rows,
            }
        )

    return f"""你是中文本地生活推荐数据质检员。请判断每条样本是否真的与目标价值无关。

判定标准:
- pass: 文本既不支持也不反对目标价值，只是在表达其他偏好。
- fail_related: 文本明显支持目标价值，或给出了目标价值的正向证据。
- fail_opposite: 文本明显反对目标价值，或给出了目标价值的反向证据。
- ambiguous: 文本可能牵涉目标价值但证据弱，无法稳定判定。

注意:
- 不要只按单字词机械判断，要看语义。例如“最近想”里的“近”不等于省心便利。
- 如果文本明显表达其他价值，但同时支持或反对目标价值，应判为 fail_related 或 fail_opposite。
- 只输出 JSON 数组，不要 Markdown，不要解释正文。
- 每个对象必须包含 example_id、status、reason 三个字段。
- status 只能是 pass、fail_related、fail_opposite、ambiguous。

待质检数据:
{json.dumps(value_blocks, ensure_ascii=False, indent=2)}
"""


def validate_batch(
    batch: list[dict[str, Any]],
    *,
    config: Any,
    retries: int,
    retry_sleep: float,
) -> list[dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": "你只输出严格 JSON 数组，用于数据质检。",
        },
        {"role": "user", "content": build_prompt(batch)},
    ]
    last_error: BaseException | None = None
    expected_ids = {row["example_id"] for row in batch}
    for attempt in range(retries + 1):
        try:
            result = chat_completion(messages, config=config)
            parsed = parse_json_array(result["content"])
            by_id = {
                str(item.get("example_id", "")).strip(): item
                for item in parsed
                if str(item.get("example_id", "")).strip() in expected_ids
            }
            judgments: list[dict[str, Any]] = []
            for row in batch:
                item = by_id.get(row["example_id"], {})
                status = str(item.get("status", "ambiguous")).strip()
                if status not in VALID_STATUSES:
                    status = "ambiguous"
                judgments.append(
                    {
                        **row,
                        "status": status,
                        "reason": str(item.get("reason", "")).strip(),
                    }
                )
            return judgments
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep * (attempt + 1))
    message = sanitize_longcat_error(last_error or RuntimeError("unknown validation error"))
    raise RuntimeError(message) from last_error


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary(output_dir: Path, judgments: list[dict[str, Any]], args: argparse.Namespace) -> None:
    by_value: dict[str, Counter[str]] = {value_id: Counter() for value_id in VALUE_IDS}
    status_counts = Counter(row["status"] for row in judgments)
    for row in judgments:
        by_value[row["target_value"]][row["status"]] += 1

    summary = {
        "input": str(args.input),
        "candidate_csv": str(args.candidate_csv) if args.candidate_csv else None,
        "num_validated": len(judgments),
        "status_counts": dict(sorted(status_counts.items())),
        "by_value": {
            value_id: dict(sorted(counter.items()))
            for value_id, counter in by_value.items()
            if counter
        },
    }
    (output_dir / "llm_unrelated_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    load_dotenv(args.env_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output_dir / "llm_unrelated_validation.jsonl"
    if args.overwrite and output_path.exists():
        output_path.unlink()

    wanted_ids = candidate_ids(args.candidate_csv)
    rows = load_rows(args.input, wanted_ids)
    existing_ids: set[str] = set()
    if output_path.exists():
        for row in read_jsonl(output_path):
            existing_ids.add(str(row.get("example_id", "")))
    rows = [row for row in rows if row["example_id"] not in existing_ids]

    config = load_generation_config()
    config = replace(config, temperature=0.0, max_tokens=max(config.max_tokens, 4000))

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        judgments = validate_batch(
            batch,
            config=config,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        append_jsonl(output_path, judgments)
        print(f"validated {min(start + len(batch), len(rows))}/{len(rows)}")
        if args.request_sleep:
            time.sleep(args.request_sleep)

    all_judgments = read_jsonl(output_path)
    write_summary(args.output_dir, all_judgments, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
