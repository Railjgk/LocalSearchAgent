"""Generate ambiguous local-life user inputs for probing trained value heads."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.generate_value_probe_paragraphs import (  # noqa: E402
    load_dotenv,
    load_generation_config,
)
from experiments.value_probe.constants import VALUE_DEFINITIONS  # noqa: E402
from src.nodes.longcat_client import chat_completion, sanitize_longcat_error  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_json_array(content: str) -> list[dict[str, Any]]:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    payload = json.loads(cleaned)
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON array")
    return [item for item in payload if isinstance(item, dict)]


def build_prompt(count: int) -> str:
    value_lines = "\n".join(
        f"- {item.value_id}: {item.definition}"
        for item in VALUE_DEFINITIONS
    )
    return f"""你是中文本地生活推荐场景的数据合成专家。请生成 {count} 条“模糊要求”的真实用户输入，用于测试 value probe。

价值维度如下，只作为你设计隐含偏好的参考:
{value_lines}

生成要求:
- 每条是 1-2 句自然中文用户输入，像用户直接对推荐助手说的话。
- 输入要模糊、隐含、不完整或带轻微冲突；不要像训练样本那样明确说出单一价值。
- 例子应覆盖餐饮、周末活动、约会、亲子/老人、短途出行、商圈、饭后安排等本地生活场景。
- 有些样本可以没有明显偏好，有些可以同时隐含 2-3 个价值。
- 不要写真实姓名、电话、身份证、精确住址等敏感个人信息。
- 不要输出 Markdown，不要编号。

只输出 JSON 数组。每个对象包含:
- text: 用户输入原文
- note: 简短说明为什么它模糊
- likely_values: 你认为可能被激活的价值中文名数组，可以为空
"""


def generate_rows(*, count: int, retries: int, retry_sleep: float) -> list[dict[str, Any]]:
    config = load_generation_config()
    messages = [
        {"role": "system", "content": "你只输出严格 JSON 数组。"},
        {"role": "user", "content": build_prompt(count)},
    ]
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            result = chat_completion(messages, config=config)
            rows = parse_json_array(result["content"])
            cleaned = []
            seen: set[str] = set()
            for row in rows:
                text = str(row.get("text", "")).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                cleaned.append(
                    {
                        "text": text,
                        "note": str(row.get("note", "")).strip(),
                        "likely_values": [
                            str(value).strip()
                            for value in row.get("likely_values", [])
                            if str(value).strip()
                        ]
                        if isinstance(row.get("likely_values"), list)
                        else [],
                    }
                )
            if len(cleaned) < count:
                raise RuntimeError(f"Model returned {len(cleaned)} unique rows, expected {count}.")
            return cleaned[:count]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep * (attempt + 1))
    message = sanitize_longcat_error(last_error or RuntimeError("unknown generation error"))
    raise RuntimeError(message) from last_error


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    load_dotenv(args.env_file)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists. Use --overwrite to replace it.")
    rows = generate_rows(count=args.count, retries=args.retries, retry_sleep=args.retry_sleep)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            payload = {
                "example_id": f"ambiguous-{index:04d}",
                "text": row["text"],
                "note": row["note"],
                "likely_values": row["likely_values"],
                "source": {"generator": "llm_synthetic", "purpose": "ambiguous_probe_test"},
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "count": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
