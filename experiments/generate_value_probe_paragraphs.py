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
from experiments.value_probe.constants import (  # noqa: E402
    RELATIONS,
    VALUE_IDS,
    VALUE_LABELS,
    canonical_value_id,
)


DEFAULT_OUTPUT_PATH = Path("experiments/value_probe_data/paragraph_samples.jsonl")
DEFAULT_GENERATION_MAX_TOKENS = 10000


@dataclass(frozen=True)
class ValueSpec:
    value_id: str
    alias: str
    name: str
    definition: str
    related_requirement: str
    opposite_requirement: str
    unrelated_requirement: str
    semantic_pattern: str


VALUE_SPECS: dict[str, ValueSpec] = {
    "家庭照护": ValueSpec(
        value_id="家庭照护",
        alias="family_care",
        name="家庭照护",
        definition=VALUE_LABELS["家庭照护"].definition + VALUE_LABELS["家庭照护"].boundary,
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
        semantic_pattern="protect_family_comfort",
    ),
    "健康克制": ValueSpec(
        value_id="健康克制",
        alias="health",
        name="健康克制",
        definition=VALUE_LABELS["健康克制"].definition + VALUE_LABELS["健康克制"].boundary,
        related_requirement=(
            "必须自然体现清淡、少油、低卡、减脂、健康饮食、身体负担小等证据。"
        ),
        opposite_requirement=(
            "必须自然表达放纵、重口味、高热量、火锅烧烤炸物甜品、不想管热量等反向诉求。"
        ),
        unrelated_requirement=(
            "不能出现健康、减脂、清淡、少油、少盐、低卡、热量、营养、养生、医生、"
            "体检、肠胃、血脂、牙口、别太辣、不辣、舒服、放纵、重口味等健康相关或反向证据；"
            "可以体现家庭、便利或预算等其他诉求。"
        ),
        semantic_pattern="avoid_unhealthy_food",
    ),
    "省心便利": ValueSpec(
        value_id="省心便利",
        alias="convenience",
        name="省心便利",
        definition=VALUE_LABELS["省心便利"].definition + VALUE_LABELS["省心便利"].boundary,
        related_requirement=(
            "必须自然体现附近、路线短、少排队、可订座、别折腾、省时间或确定性高。"
        ),
        opposite_requirement=(
            "必须自然表达愿意多走、多换乘、排队也可以、想探索远一点或复杂路线也能接受。"
        ),
        unrelated_requirement=(
            "不能出现附近、近、远、离家、离公司、地铁、走路、交通、少排队、可订座、"
            "路线短、别折腾、省时间、方便或反向愿意折腾等便利相关证据；"
            "也不要写不用等、慢慢逛、远一点也没关系、不赶时间等容易构成反向便利的表达；"
            "可以体现家庭、健康或预算等其他诉求。"
        ),
        semantic_pattern="reduce_friction",
    ),
    "价格敏感": ValueSpec(
        value_id="价格敏感",
        alias="cost_sensitivity",
        name="价格敏感",
        definition=VALUE_LABELS["价格敏感"].definition + VALUE_LABELS["价格敏感"].boundary,
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
        semantic_pattern="save_money",
    ),
    "品质可靠": ValueSpec(
        value_id="品质可靠",
        alias="quality_reliability",
        name="品质可靠",
        definition=VALUE_LABELS["品质可靠"].definition + VALUE_LABELS["品质可靠"].boundary,
        related_requirement="必须自然体现靠谱、评价稳定、评分高、卫生服务稳、老店或少踩雷。",
        opposite_requirement="必须自然表达愿意尝试新店、小众店、评价少也可以，不把稳定可靠放在首位。",
        unrelated_requirement=(
            "不能出现评分、评价、口碑、靠谱、踩雷、老店、新店试错、评价好等可靠性证据；"
            "可以体现其他诉求。"
        ),
        semantic_pattern="trust_and_certainty",
    ),
    "体验享受": ValueSpec(
        value_id="体验享受",
        alias="experience_enjoyment",
        name="体验享受",
        definition=VALUE_LABELS["体验享受"].definition + VALUE_LABELS["体验享受"].boundary,
        related_requirement="必须自然体现好玩、沉浸、体验感强、活动丰富、玩得尽兴或主观满足。",
        opposite_requirement="必须自然表达体验普通也没关系，只要完成吃饭/办事/休息等实用目标。",
        unrelated_requirement=(
            "不能出现好玩、沉浸、体验感、尽兴、活动丰富、菜品丰富、无聊也行等体验享受或反向证据；"
            "可以体现其他诉求。"
        ),
        semantic_pattern="seek_enjoyment",
    ),
    "新奇探索": ValueSpec(
        value_id="新奇探索",
        alias="novelty_exploration",
        name="新奇探索",
        definition=VALUE_LABELS["新奇探索"].definition + VALUE_LABELS["新奇探索"].boundary,
        related_requirement="必须自然体现想试新店、小众、本地探索、没去过、隐藏宝藏或新鲜感。",
        opposite_requirement="必须自然表达不想冒险、不想试新，优先熟悉稳妥或常去的地方。",
        unrelated_requirement="不能出现新店、小众、探索、没去过、熟悉稳妥等新奇或反向证据；可以体现其他诉求。",
        semantic_pattern="novelty_exploration",
    ),
    "社交连接": ValueSpec(
        value_id="社交连接",
        alias="social_connection",
        name="社交连接",
        definition=VALUE_LABELS["社交连接"].definition + VALUE_LABELS["社交连接"].boundary,
        related_requirement="必须自然体现朋友聚会、多人互动、适合聊天、热闹、共同参与或拉近关系。",
        opposite_requirement="必须自然表达想一个人待着、少社交、安静独处或不需要互动。",
        unrelated_requirement=(
            "不能出现朋友、同学、团建、聚会、聊天、热闹、社交、多人互动、互动体验等正向证据；"
            "也不能出现一个人、自己待着、不聊天、各自休息、独处、单人座、独立空间等反向证据；"
            "不要描述同行人数、关系身份、孩子、亲子、男朋友、女朋友、情侣、约会、包间、是否独处、是否聊天或是否互动；"
            "优先写预算、健康、可靠、距离、营业时间、菜品口味等非人际偏好，句子里不要出现人物关系。"
        ),
        semantic_pattern="social_connection",
    ),
    "氛围仪式": ValueSpec(
        value_id="氛围仪式",
        alias="atmosphere_ritual",
        name="氛围仪式",
        definition=VALUE_LABELS["氛围仪式"].definition + VALUE_LABELS["氛围仪式"].boundary,
        related_requirement="必须自然体现纪念日、浪漫、氛围好、出片、仪式感或审美场景。",
        opposite_requirement="必须自然表达不在意氛围和仪式感，普通、朴素、实用即可。",
        unrelated_requirement=(
            "不能出现氛围、仪式感、浪漫、纪念日、生日、出片、拍照、精致、商务宴请、包间、朴素实用等相关或反向证据；"
            "可以体现其他诉求。"
        ),
        semantic_pattern="ritual_atmosphere",
    ),
    "舒适安全": ValueSpec(
        value_id="舒适安全",
        alias="comfort_safety",
        name="舒适安全",
        definition=VALUE_LABELS["舒适安全"].definition + VALUE_LABELS["舒适安全"].boundary,
        related_requirement="必须自然体现不累、不挤、安全、低强度、无烟、少风险或适合身体状态。",
        opposite_requirement="必须自然表达可以累一点、拥挤也行、刺激强度高也可以或不怕风险。",
        unrelated_requirement=(
            "不能出现不累、安全、不挤、低强度、无烟、安静、舒服、刺激、冒险、拥挤也行、"
            "够辣够刺激、人少、放松、休息、待一下午、看书、发呆、咖啡馆坐坐、隔音好、"
            "环境安静、温泉、按摩等舒适安全或反向证据；可以体现其他诉求。"
        ),
        semantic_pattern="comfort_and_safety",
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
            value_id = canonical_value_id(str(record.get("target_value", record.get("value_id", ""))))
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
            value_id = canonical_value_id(str(record.get("target_value", record.get("value_id", ""))))
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
    alias_to_value = {spec.alias: spec.value_id for spec in VALUE_SPECS.values()}
    for raw_value_id in values:
        value_id = canonical_value_id(alias_to_value.get(raw_value_id, raw_value_id))
        if value_id not in VALUE_SPECS:
            raise ValueError(
                f"Unknown value_id {raw_value_id!r}. Choose from: {', '.join(VALUE_SPECS)}"
            )
        specs.append(VALUE_SPECS[value_id])
    return specs


def signed_score_for_relation(relation: str) -> int:
    if relation == "related":
        return 6
    if relation == "opposite":
        return -6
    return 0


def labels_for(spec: ValueSpec, relation: str) -> dict[str, int]:
    labels = {value_id: 0 for value_id in VALUE_IDS}
    labels[spec.value_id] = signed_score_for_relation(relation)
    return labels


def value_relations_for(spec: ValueSpec, relation: str) -> dict[str, str]:
    value_relations = {value_id: "unrelated" for value_id in VALUE_IDS}
    value_relations[spec.value_id] = relation
    return value_relations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate natural-language paragraph samples for value-probe training."
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--per-relation", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--values", nargs="*")
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
                            "example_id": f"llm-{spec.value_id}-{relation}-{counts[(spec.value_id, relation)]:05d}",
                            "text": text,
                            "value_id": spec.value_id,
                            "target_value": spec.value_id,
                            "value_name": spec.name,
                            "relation": relation,
                            "labels": labels_for(spec, relation),
                            "value_relations": value_relations_for(spec, relation),
                            "supervision_values": [spec.value_id],
                            "language": "zh",
                            "domain": "local_life",
                            "text_type": "concrete_query",
                            "template_id": f"llm_{spec.alias}_{relation}",
                            "semantic_pattern": spec.semantic_pattern,
                            "probe_input_field": "text",
                            "source": {
                                "generator": "llm_synthetic",
                                "generator_model": config.model,
                                "generator_version": "value_probe_cn10_v1",
                            },
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
