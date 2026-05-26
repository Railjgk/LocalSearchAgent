"""Annotate value-probe examples with grammar-level Chinese word scores."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.generate_value_probe_paragraphs import VALUE_SPECS, load_dotenv
from experiments.value_probe.data import read_jsonl, write_jsonl
from src.nodes.longcat_client import (
    DEFAULT_LONGCAT_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
    LongCatConfig,
    chat_completion,
    sanitize_longcat_error,
)


DEFAULT_INPUT_PATH = Path("experiments/value_probe_runs/qwen3_4b_token_probe/dataset/all.jsonl")
DEFAULT_OUTPUT_DIR = Path("experiments/value_probe_runs/qwen3_4b_token_probe/fluent_level")
DEFAULT_MODEL = "LongCat-2.0-Preview"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0
SCORE_MIN = -6
SCORE_MAX = 6


def _read_float(keys: Iterable[str], default: float) -> float:
    for key in keys:
        raw_value = os.environ.get(key)
        if not raw_value:
            continue
        try:
            return float(raw_value)
        except ValueError:
            continue
    return default


def _read_int(keys: Iterable[str], default: int) -> int:
    for key in keys:
        raw_value = os.environ.get(key)
        if not raw_value:
            continue
        try:
            return int(raw_value)
        except ValueError:
            continue
    return default


def load_annotation_config() -> LongCatConfig:
    """Load an OpenAI-compatible model config for fluent-level annotation."""

    api_key = (
        os.environ.get("VALUE_PROBE_FLUENT_API_KEY")
        or os.environ.get("VALUE_PROBE_GEN_API_KEY")
        or os.environ.get("WF_A_LLM_API_KEY")
        or os.environ.get("WF_A_LLM_APP_KEY")
        or os.environ.get("LONGCAT_API_KEY")
        or os.environ.get("LONGCAT_APP_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "Missing model API key. Set VALUE_PROBE_FLUENT_API_KEY, VALUE_PROBE_GEN_API_KEY, "
            "WF_A_LLM_API_KEY, WF_A_LLM_APP_KEY, LONGCAT_API_KEY, or LONGCAT_APP_KEY in .env."
        )

    base_url = (
        os.environ.get("VALUE_PROBE_FLUENT_BASE_URL")
        or os.environ.get("VALUE_PROBE_GEN_BASE_URL")
        or os.environ.get("WF_A_LLM_BASE_URL")
        or os.environ.get("LONGCAT_BASE_URL")
        or DEFAULT_LONGCAT_BASE_URL
    ).strip()
    model = (
        os.environ.get("VALUE_PROBE_FLUENT_MODEL")
        or os.environ.get("VALUE_PROBE_GEN_MODEL")
        or os.environ.get("WF_A_LLM_MODEL")
        or os.environ.get("LONGCAT_MODEL")
        or DEFAULT_MODEL
    ).strip()

    return LongCatConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=_read_float(
            (
                "VALUE_PROBE_FLUENT_TIMEOUT_SECONDS",
                "VALUE_PROBE_GEN_TIMEOUT_SECONDS",
                "WF_A_LLM_TIMEOUT_SECONDS",
                "LONGCAT_TIMEOUT_SECONDS",
            ),
            max(DEFAULT_TIMEOUT_SECONDS, 60.0),
        ),
        max_tokens=_read_int(
            (
                "VALUE_PROBE_FLUENT_MAX_TOKENS",
                "VALUE_PROBE_GEN_MAX_TOKENS",
                "WF_A_LLM_MAX_TOKENS",
                "LONGCAT_MAX_TOKENS",
            ),
            DEFAULT_MAX_TOKENS,
        ),
        temperature=_read_float(
            (
                "VALUE_PROBE_FLUENT_TEMPERATURE",
                "VALUE_PROBE_GEN_TEMPERATURE",
                "WF_A_LLM_TEMPERATURE",
                "LONGCAT_TEMPERATURE",
            ),
            DEFAULT_TEMPERATURE,
        ),
    )


def value_name(value_id: str) -> str:
    spec = VALUE_SPECS.get(value_id)
    if spec is None:
        raise ValueError(f"Unsupported target_value: {value_id!r}")
    return spec.name


def value_definition(value_id: str) -> str:
    spec = VALUE_SPECS.get(value_id)
    if spec is None:
        raise ValueError(f"Unsupported target_value: {value_id!r}")
    return spec.definition


def build_prompt(*, text: str, value_id: str) -> str:
    name = value_name(value_id)
    definition = value_definition(value_id)
    coverage_reference = "|".join(normalize_for_coverage(text))
    return f"""你是中文文本标注专家。请对一条本地生活用户需求文本进行中文分词，并为每个词标注它与目标价值的相关性分数。

目标价值:
{name}

目标价值定义:
{definition}

用户文本:
{text}

去标点字符检查串:
{coverage_reference}

标注要求:

1. 先按照中文语法和词法边界分词。
   - 保持原文顺序。
   - 不要为了表达语义证据而合并短语。
   - 例如“少排队”应切为“少”“排队”，“别太贵”应切为“别”“太”“贵”，“带孩子”应切为“带”“孩子”。
   - 标点符号可以省略；如果保留，分数必须为 0。
   - 分词结果按顺序拼接并去掉标点、符号、空格后，必须与上面的“去标点字符检查串”去掉 `|` 后完全一致。不要漏字、改字或增字。
   - 不要把叠词、口语词或固定表达规范化改写，例如“聊聊天”不能写成“聊天”。如果某处不确定，可以拆成更小的词或单字。

2. 对每个词打一个整数分，范围是 -6 到 6。
   - 正分表示该词正向体现目标价值。
   - 负分表示该词反向体现目标价值。
   - 0 表示该词与目标价值无关。
   - 1 到 2 表示弱相关。
   - 3 到 4 表示明确相关。
   - 5 到 6 表示强相关或核心证据。
   - -1 到 -2 表示弱反向相关。
   - -3 到 -4 表示明确反向相关。
   - -5 到 -6 表示强反向相关或核心反证。

3. 只输出 JSON，不要解释，不要 Markdown，不要额外文本。

输出格式必须是:

{{
  "text": "原文字符串",
  "target_value": "{value_id}",
  "target_value_name": "{name}",
  "tokens": [
    {{
      "token": "分词结果",
      "score": 整数
    }}
  ]
}}
"""


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "你只输出符合要求的 JSON 对象，不要输出解释、Markdown 或额外文本。",
        },
        {"role": "user", "content": prompt},
    ]


def extract_json_object(content: str) -> dict[str, object]:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Annotation response must be a JSON object")
    return payload


def _is_ignored_char(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in {"P", "S", "Z"} or char.isspace()


def normalize_for_coverage(text: str) -> str:
    return "".join(char for char in text if not _is_ignored_char(char))


def validate_annotation(payload: dict[str, object], *, source_text: str, value_id: str) -> list[str]:
    errors: list[str] = []
    if payload.get("text") != source_text:
        errors.append("text does not match source")
    if payload.get("target_value") != value_id:
        errors.append("target_value does not match source")
    expected_name = value_name(value_id)
    if payload.get("target_value_name") != expected_name:
        errors.append("target_value_name does not match source")

    raw_tokens = payload.get("tokens")
    if not isinstance(raw_tokens, list) or not raw_tokens:
        errors.append("tokens must be a non-empty list")
        return errors

    normalized_tokens: list[str] = []
    for index, raw_token in enumerate(raw_tokens):
        if not isinstance(raw_token, dict):
            errors.append(f"tokens[{index}] must be an object")
            continue
        token = raw_token.get("token")
        score = raw_token.get("score")
        if not isinstance(token, str) or not token.strip():
            errors.append(f"tokens[{index}].token must be a non-empty string")
        else:
            normalized_tokens.append(token)
        if not isinstance(score, int):
            errors.append(f"tokens[{index}].score must be an integer")
        elif score < SCORE_MIN or score > SCORE_MAX:
            errors.append(f"tokens[{index}].score is outside [-6, 6]")

    source_norm = normalize_for_coverage(source_text)
    tokens_norm = normalize_for_coverage("".join(normalized_tokens))
    if source_norm != tokens_norm:
        errors.append("tokens do not cover source text after removing punctuation/symbols/spaces")
    return errors


def annotate_one(
    *,
    row: dict[str, object],
    config: LongCatConfig,
    retries: int,
    retry_sleep: float,
) -> dict[str, object]:
    text = str(row.get("text", "")).strip()
    value_id = str(row.get("target_value") or row.get("value_id") or "").strip()
    if not text:
        raise ValueError("Source row is missing text")
    if value_id not in VALUE_SPECS:
        raise ValueError(f"Source row has unsupported target value: {value_id!r}")

    prompt = build_prompt(text=text, value_id=value_id)
    last_error: BaseException | None = None
    validation_feedback = ""
    for attempt in range(retries + 1):
        try:
            retry_prompt = prompt
            if validation_feedback:
                retry_prompt = (
                    f"{prompt}\n\n上一次输出未通过校验: {validation_feedback}\n"
                    "请重新输出完整 JSON，并特别检查 tokens 是否逐字覆盖原文；"
                    "不要规范化改写原文，不确定的片段可以拆成单字。"
                )
            response = chat_completion(build_messages(retry_prompt), config=config)
            payload = extract_json_object(response["content"])
            errors = validate_annotation(payload, source_text=text, value_id=value_id)
            if errors:
                validation_feedback = "; ".join(errors)
                raise ValueError("; ".join(errors))
            return {
                "example_id": row.get("example_id"),
                "split": row.get("split"),
                "relation": row.get("relation"),
                "text": payload["text"],
                "target_value": payload["target_value"],
                "target_value_name": payload["target_value_name"],
                "tokens": payload["tokens"],
                "annotator_model": response["model"],
                "annotation_level": "fluent",
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep * (attempt + 1))
    message = sanitize_longcat_error(last_error or RuntimeError("unknown annotation error"))
    example_id = str(row.get("example_id") or "<missing-example-id>")
    raise RuntimeError(f"{example_id}: {message}") from last_error


def completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        example_id = str(row.get("example_id") or "")
        target_value = str(row.get("target_value") or "")
        if example_id and target_value:
            keys.add((example_id, target_value))
    return keys


def summarize(rows: list[dict[str, object]], source_count: int) -> dict[str, object]:
    invalid_rows: list[dict[str, object]] = []
    score_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    value_counts: dict[str, int] = {}
    total_tokens = 0
    nonzero_tokens = 0

    for row in rows:
        text = str(row.get("text", ""))
        value_id = str(row.get("target_value", ""))
        errors = validate_annotation(row, source_text=text, value_id=value_id)
        if errors:
            invalid_rows.append({"example_id": row.get("example_id"), "errors": errors})
        split = str(row.get("split") or "")
        value_counts[value_id] = value_counts.get(value_id, 0) + 1
        split_counts[split] = split_counts.get(split, 0) + 1
        for token in row.get("tokens", []):
            if not isinstance(token, dict):
                continue
            score = token.get("score")
            score_counts[str(score)] = score_counts.get(str(score), 0) + 1
            total_tokens += 1
            if score != 0:
                nonzero_tokens += 1

    return {
        "annotation_level": "fluent",
        "source_examples": source_count,
        "annotated_examples": len(rows),
        "missing_examples": max(source_count - len(rows), 0),
        "invalid_examples": len(invalid_rows),
        "invalid_example_details": invalid_rows[:20],
        "total_tokens": total_tokens,
        "nonzero_tokens": nonzero_tokens,
        "split_counts": dict(sorted(split_counts.items())),
        "value_counts": dict(sorted(value_counts.items())),
        "score_counts": dict(sorted(score_counts.items(), key=lambda item: int(item[0]))),
    }


def write_outputs(output_dir: Path, rows: list[dict[str, object]], source_count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: str(row.get("example_id") or ""))
    write_jsonl(output_dir / "all.jsonl", rows)
    for split in ("train", "val", "test"):
        write_jsonl(
            output_dir / f"{split}.jsonl",
            [row for row in rows if row.get("split") == split],
        )
    summary = summarize(rows, source_count)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Annotate at most this many source rows; 0 means all.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--request-sleep", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent annotation requests.")
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in output all.jsonl.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output files before annotating.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first prompt and exit.")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing output files without API calls.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = read_jsonl(args.input)
    if args.limit > 0:
        source_rows = source_rows[: args.limit]

    if args.dry_run:
        first = source_rows[0]
        print(
            build_prompt(
                text=str(first["text"]),
                value_id=str(first.get("target_value") or first.get("value_id")),
            )
        )
        return 0

    output_all_path = args.output_dir / "all.jsonl"
    if args.overwrite and output_all_path.exists():
        for name in ("all.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "summary.json"):
            path = args.output_dir / name
            if path.exists():
                path.unlink()

    existing_rows = read_jsonl(output_all_path) if output_all_path.exists() else []
    if args.validate_only:
        write_outputs(args.output_dir, existing_rows, len(source_rows))
        summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 1 if summary["invalid_examples"] or summary["missing_examples"] else 0

    load_dotenv(args.env_file)
    config = load_annotation_config()
    done = completed_keys(output_all_path) if args.resume else set()
    output_rows = list(existing_rows) if args.resume else []

    rows_to_annotate = []
    for row in source_rows:
        key = (str(row.get("example_id") or ""), str(row.get("target_value") or row.get("value_id") or ""))
        if key not in done:
            rows_to_annotate.append(row)

    if args.workers <= 1:
        for row in rows_to_annotate:
            annotated = annotate_one(
                row=row,
                config=config,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
            output_rows.append(annotated)
            write_outputs(args.output_dir, output_rows, len(source_rows))
            print(f"annotated {len(output_rows):>4}/{len(source_rows)}: {annotated['example_id']}")
            if args.request_sleep:
                time.sleep(args.request_sleep)
    else:
        if args.request_sleep:
            raise ValueError("--request-sleep is only supported when --workers is 1")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_row = {
                executor.submit(
                    annotate_one,
                    row=row,
                    config=config,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                ): row
                for row in rows_to_annotate
            }
            for future in concurrent.futures.as_completed(future_to_row):
                try:
                    annotated = future.result()
                except Exception:
                    for pending in future_to_row:
                        pending.cancel()
                    raise
                output_rows.append(annotated)
                write_outputs(args.output_dir, output_rows, len(source_rows))
                print(f"annotated {len(output_rows):>4}/{len(source_rows)}: {annotated['example_id']}")

    write_outputs(args.output_dir, output_rows, len(source_rows))
    summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if summary["invalid_examples"] or summary["missing_examples"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
