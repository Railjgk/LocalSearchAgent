"""Generate natural-language paragraphs for value-probe training.

The generated dataset keeps labels as JSONL metadata, but the probe input is
the plain `text` field: one natural Chinese user-demand paragraph per sample.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nodes.longcat_client import (  # noqa: E402
    DEFAULT_LONGCAT_BASE_URL,
    DEFAULT_LONGCAT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    LongCatConfig,
    chat_completion,
    sanitize_longcat_error,
)


DEFAULT_OUTPUT_PATH = Path("experiments/value_probe_data/paragraph_samples.jsonl")
DEFAULT_GENERATION_MAX_TOKENS = 10000
RELATIONS = ("related", "opposite", "unrelated")


@dataclass(frozen=True)
class ValueSpec:
    value_id: str
    name: str
    definition: str
    related_requirement: str
    opposite_requirement: str
    unrelated_requirement: str


VALUE_SPECS: dict[str, ValueSpec] = {
    "family_care": ValueSpec(
        value_id="family_care",
        name="家庭照顾",
        definition=(
            "用户在活动、餐饮或路线选择中明确考虑孩子、伴侣、老人、家庭成员、"
            "亲子友好、安全、低强度、群体适配等需求。只有低强度但没有家庭或"
            "同行人证据时，不算家庭照顾。"
        ),
        related_requirement=(
            "必须自然提到孩子、老人、伴侣、家人或亲子同行，并体现照顾他们的选择标准。"
        ),
        opposite_requirement=(
            "必须自然表达不需要照顾家庭成员、只想一个人或成年人自由安排、强度可以高、"
            "不考虑亲子或老人友好。"
        ),
        unrelated_requirement=(
            "不能出现孩子、老人、爸妈、父母、伴侣、家人、亲子、全家、老人腿脚、"
            "家庭照顾或反向拒绝家庭照顾；可以体现健康、便利或预算等其他诉求。"
        ),
    ),
    "health": ValueSpec(
        value_id="health",
        name="健康",
        definition=(
            "用户在餐饮或活动选择中偏向低油、清淡、低卡、减脂、营养均衡、"
            "身体负担小，或明确提到自己/同行人当前有健康饮食需求。"
        ),
        related_requirement=(
            "必须自然体现清淡、少油、低卡、减脂、健康饮食、身体负担小等证据。"
        ),
        opposite_requirement=(
            "必须自然表达放纵、重口味、高热量、火锅烧烤炸物甜品、不想管热量等反向诉求。"
        ),
        unrelated_requirement=(
            "不能出现健康、减脂、清淡、少油、少盐、低卡、热量、营养、养生、医生、"
            "体检、肠胃、血脂、牙口、放纵、重口味等健康相关或反向证据；"
            "可以体现家庭、便利或预算等其他诉求。"
        ),
    ),
    "convenience": ValueSpec(
        value_id="convenience",
        name="便利",
        definition=(
            "用户重视距离近、路线短、省时间、少排队、可预约、可订座、别折腾、"
            "确定性高等便利性。"
        ),
        related_requirement=(
            "必须自然体现附近、路线短、少排队、可订座、别折腾、省时间或确定性高。"
        ),
        opposite_requirement=(
            "必须自然表达愿意多走、多换乘、排队也可以、想探索远一点或复杂路线也能接受。"
        ),
        unrelated_requirement=(
            "不能出现附近、近、远、离家、离公司、地铁、走路、交通、少排队、可订座、"
            "路线短、别折腾、省时间、方便或反向愿意折腾等便利相关证据；"
            "可以体现家庭、健康或预算等其他诉求。"
        ),
    ),
    "cost_sensitivity": ValueSpec(
        value_id="cost_sensitivity",
        name="预算敏感",
        definition=(
            "用户明确在意预算、人均上限、省钱、别太贵、有优惠券、团购、性价比，"
            "但不是单纯追求最低价。"
        ),
        related_requirement=(
            "必须自然体现预算、人均上限、省钱、别太贵、有券、团购或性价比。"
        ),
        opposite_requirement=(
            "必须自然表达愿意花钱、预算不是问题、纪念日/商务宴请优先品质或体验。"
        ),
        unrelated_requirement=(
            "不能出现预算、价格、人均、便宜、贵、优惠、性价比或反向不差钱等预算相关证据；"
            "可以体现家庭、健康或便利等其他诉求。"
        ),
    ),
}


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE pairs without printing secrets."""

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


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


def load_generation_config() -> LongCatConfig:
    """Load an OpenAI-compatible model config from `.env` variables."""

    api_key = (
        os.environ.get("VALUE_PROBE_GEN_API_KEY")
        or os.environ.get("WF_A_LLM_API_KEY")
        or os.environ.get("WF_A_LLM_APP_KEY")
        or os.environ.get("LONGCAT_API_KEY")
        or os.environ.get("LONGCAT_APP_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "Missing model API key. Set VALUE_PROBE_GEN_API_KEY, WF_A_LLM_API_KEY, "
            "WF_A_LLM_APP_KEY, LONGCAT_API_KEY, or LONGCAT_APP_KEY in .env."
        )

    base_url = (
        os.environ.get("VALUE_PROBE_GEN_BASE_URL")
        or os.environ.get("WF_A_LLM_BASE_URL")
        or os.environ.get("LONGCAT_BASE_URL")
        or DEFAULT_LONGCAT_BASE_URL
    ).strip()
    model = (
        os.environ.get("VALUE_PROBE_GEN_MODEL")
        or os.environ.get("WF_A_LLM_MODEL")
        or os.environ.get("LONGCAT_MODEL")
        or DEFAULT_LONGCAT_MODEL
    ).strip()

    return LongCatConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=_read_float(
            ("VALUE_PROBE_GEN_TIMEOUT_SECONDS", "WF_A_LLM_TIMEOUT_SECONDS", "LONGCAT_TIMEOUT_SECONDS"),
            DEFAULT_TIMEOUT_SECONDS,
        ),
        max_tokens=_read_int(
            ("VALUE_PROBE_GEN_MAX_TOKENS", "WF_A_LLM_MAX_TOKENS", "LONGCAT_MAX_TOKENS"),
            max(DEFAULT_MAX_TOKENS, DEFAULT_GENERATION_MAX_TOKENS),
        ),
        temperature=_read_float(
            ("VALUE_PROBE_GEN_TEMPERATURE", "WF_A_LLM_TEMPERATURE", "LONGCAT_TEMPERATURE"),
            max(DEFAULT_TEMPERATURE, 0.8),
        ),
    )


def build_prompt(spec: ValueSpec, relation: str, count: int, seed_offset: int) -> str:
    if relation == "related":
        relation_name = "相关"
        relation_definition = (
            f"用户需求明确体现 `{spec.name}`。{spec.related_requirement}"
        )
    elif relation == "opposite":
        relation_name = "相反"
        relation_definition = (
            f"用户需求明确表达与 `{spec.name}` 冲突或相反的偏好。{spec.opposite_requirement}"
        )
    elif relation == "unrelated":
        relation_name = "无关"
        relation_definition = (
            f"用户需求与 `{spec.name}` 无关，既不支持也不反对该价值。{spec.unrelated_requirement}"
            "优先自然体现其他目标价值之一，用作对比学习的 hard negative。"
        )
    else:
        raise ValueError(f"Unsupported relation: {relation}")

    return f"""你是中文本地生活推荐场景的数据合成专家。请生成用于训练 hidden-state probe 的自然语言用户需求段落。

目标价值: {spec.name}
目标价值定义: {spec.definition}
样本类型: {relation_name}
样本类型定义: {relation_definition}

请生成 {count} 条中文用户需求段落。

硬性要求:
- 每条只能是一段自然中文用户需求，像真实用户在对推荐助手说话。
- 每条 1-3 句话，适合直接作为 probe 的输入文本。
- 不要输出 JSON、Markdown、编号、标签、解释、引号或项目符号。
- 不要直接说“我重视{spec.name}”“这体现{spec.name}”“价值观”等元话语。
- 场景要覆盖餐饮、周末活动、亲子/朋友/约会、短途出行、商圈选择、饭后安排等本地生活请求。
- 样本之间必须明显不同，不要只替换地名、人数、预算或菜品。
- 可以写口语、简短命令、完整句、隐式表达、带冲突偏好或多约束的需求。
- 不要生成真实姓名、电话、身份证、精确住址等敏感个人信息。
- 每条之间用一个空行分隔。

多样性种子: {seed_offset}
"""


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你只生成训练样本原文。不要解释，不要编号，不要输出结构化数据。"
            ),
        },
        {"role": "user", "content": prompt},
    ]


def clean_sample(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[-*•]\s*", "", text)
    text = re.sub(r"^\d+[\.、)]\s*", "", text)
    text = text.strip().strip('"“”')
    return re.sub(r"\s+", " ", text).strip()


def parse_samples(content: str) -> list[str]:
    samples: list[str] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    if len(blocks) == 1:
        blocks = [line for line in content.splitlines() if line.strip()]

    for block in blocks:
        sample = clean_sample(block)
        if not sample:
            continue
        if sample.startswith("[") and sample.endswith("]"):
            continue
        if sample.lower().startswith(("related", "opposite", "unrelated")):
            continue
        samples.append(sample)
    return samples


def generate_batch(
    *,
    config: LongCatConfig,
    spec: ValueSpec,
    relation: str,
    count: int,
    seed_offset: int,
    retries: int,
    retry_sleep: float,
) -> list[str]:
    prompt = build_prompt(spec, relation, count, seed_offset)
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            result = chat_completion(build_messages(prompt), config=config)
            samples = parse_samples(result["content"])
            if len(samples) < count:
                raise RuntimeError(
                    f"Model returned {len(samples)} samples, expected at least {count}."
                )
            return samples[:count]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep * (attempt + 1))
    message = sanitize_longcat_error(last_error or RuntimeError("unknown generation error"))
    raise RuntimeError(message) from last_error


def append_records(
    output_path: Path,
    *,
    records: Iterable[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def existing_keys(output_path: Path) -> set[tuple[str, str, str]]:
    if not output_path.exists():
        return set()

    keys: set[tuple[str, str, str]] = set()
    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            value_id = str(record.get("value_id", ""))
            relation = str(record.get("relation", ""))
            text = str(record.get("text", ""))
            if value_id and relation and text:
                keys.add((value_id, relation, text))
    return keys


def existing_counts(output_path: Path) -> dict[tuple[str, str], int]:
    if not output_path.exists():
        return {}

    counts: dict[tuple[str, str], int] = {}
    seen: set[tuple[str, str, str]] = set()
    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            value_id = str(record.get("value_id", ""))
            relation = str(record.get("relation", ""))
            text = str(record.get("text", ""))
            if not value_id or not relation or not text:
                continue
            unique_key = (value_id, relation, text)
            if unique_key in seen:
                continue
            seen.add(unique_key)
            count_key = (value_id, relation)
            counts[count_key] = counts.get(count_key, 0) + 1
    return counts


def selected_value_specs(values: list[str] | None) -> list[ValueSpec]:
    if not values:
        return list(VALUE_SPECS.values())

    specs: list[ValueSpec] = []
    for value_id in values:
        if value_id not in VALUE_SPECS:
            raise ValueError(
                f"Unknown value_id {value_id!r}. Choose from: {', '.join(VALUE_SPECS)}"
            )
        specs.append(VALUE_SPECS[value_id])
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate natural-language paragraph samples for value-probe training."
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--per-relation", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--values", nargs="*", choices=sorted(VALUE_SPECS))
    parser.add_argument("--relations", nargs="*", choices=RELATIONS, default=list(RELATIONS))
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep after each successful model call.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output file before generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first prompt instead of calling the model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)

    specs = selected_value_specs(args.values)
    if args.per_relation <= 0:
        raise ValueError("--per-relation must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.request_sleep < 0:
        raise ValueError("--request-sleep must be non-negative")

    if args.dry_run:
        prompt = build_prompt(specs[0], args.relations[0], min(args.batch_size, args.per_relation), 0)
        print(prompt)
        return 0

    config = load_generation_config()
    output_path = args.output
    if args.overwrite and output_path.exists():
        output_path.unlink()

    seen = existing_keys(output_path)
    counts = existing_counts(output_path)
    total_written = 0

    for spec in specs:
        for relation in args.relations:
            existing_count = counts.get((spec.value_id, relation), 0)
            if existing_count >= args.per_relation:
                print(
                    f"skip {spec.value_id}/{relation} "
                    f"({existing_count}/{args.per_relation} already present)"
                )
                continue
            remaining = args.per_relation - existing_count
            seed_offset = 0
            empty_batches = 0
            while remaining > 0:
                batch_count = min(args.batch_size, remaining)
                samples = generate_batch(
                    config=config,
                    spec=spec,
                    relation=relation,
                    count=batch_count,
                    seed_offset=seed_offset,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                )
                records = []
                for text in samples:
                    key = (spec.value_id, relation, text)
                    if key in seen:
                        continue
                    seen.add(key)
                    counts[(spec.value_id, relation)] = counts.get((spec.value_id, relation), 0) + 1
                    records.append(
                        {
                            "text": text,
                            "value_id": spec.value_id,
                            "value_name": spec.name,
                            "relation": relation,
                            "probe_input_field": "text",
                            "generator_model": config.model,
                        }
                    )
                append_records(output_path, records=records)
                total_written += len(records)
                remaining -= len(records)
                seed_offset += batch_count
                empty_batches = empty_batches + 1 if not records else 0
                print(
                    f"wrote {len(records):>2} records for {spec.value_id}/{relation} "
                    f"({counts.get((spec.value_id, relation), 0)}/{args.per_relation})"
                )
                if empty_batches >= args.retries + 1:
                    raise RuntimeError(
                        f"Could not get new unique samples for {spec.value_id}/{relation}."
                    )
                if remaining > 0 and args.request_sleep:
                    time.sleep(args.request_sleep)

    print(f"done: wrote {total_written} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
