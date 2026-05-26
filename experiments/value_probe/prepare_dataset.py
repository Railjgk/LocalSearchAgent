"""Prepare train/val/test JSONL files for value probe training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.value_probe.data import (
    load_raw_examples,
    stratified_split,
    summarize_examples,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/value_probe_data/paragraph_samples.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen34b_token_probe/dataset"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260524)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples = load_raw_examples(args.input)
    examples = stratified_split(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "all.jsonl", [example.to_json() for example in examples])
    for split in ("train", "val", "test"):
        write_jsonl(
            args.output_dir / f"{split}.jsonl",
            [example.to_json() for example in examples if example.split == split],
        )
    summary = summarize_examples(examples)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
