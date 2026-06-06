#!/usr/bin/env python3
"""Prepare LocalSearchBench data for local evaluation scripts.

The official Hugging Face artifact is a parquet file, while the upstream
evaluation scripts primarily consume JSON. This script downloads the parquet
file and writes JSON/JSONL views with stable field aliases.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd


HF_PARQUET_URL = (
    "https://huggingface.co/datasets/localsearchbench/localsearchbench/"
    "resolve/main/data/train-00000-of-00001.parquet"
)
HF_SHA = "bb1c2c5dc45705bcbb5bfcb1ac1cc54ede4c221e"


def extract_boxed_answer(answer: str) -> str | None:
    """Return the short LaTeX boxed answer if present."""

    match = re.search(r"\\boxed\{([^}]+)\}", answer or "")
    return match.group(1).strip() if match else None


def extract_hop_queries(search_path: str) -> list[str]:
    """Extract serialized hop query strings from the Chinese path text."""

    if not search_path:
        return []
    return [
        match.group(1).strip()
        for match in re.finditer(r"搜索[\"“](.*?)[\"”]", search_path)
        if match.group(1).strip()
    ]


def download_parquet(path: Path, force: bool = False) -> None:
    """Download the official parquet file if it is missing."""

    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(HF_PARQUET_URL, timeout=120) as response:
        path.write_bytes(response.read())


def convert(parquet_path: Path, data_dir: Path) -> dict:
    """Convert parquet data into JSON and JSONL files."""

    df = pd.read_parquet(parquet_path)
    records = []
    for index, row in df.iterrows():
        answer = row["Answer"]
        search_path = row["Multi-hop search path"]
        records.append(
            {
                "id": f"lsb_{index + 1:04d}",
                "hop_count": int(row["Hop Count"]),
                "difficulty": row["Difficulty"],
                "city": row["City"],
                "question": row["Question"],
                "query": row["Question"],
                "multi_hop_search_path": search_path,
                "search_path": search_path,
                "answer": answer,
                "reference_answer": answer,
                "ground_truth": extract_boxed_answer(answer) or answer,
                "hop_used_queries": extract_hop_queries(search_path),
            }
        )

    json_path = data_dir / "localsearchbench_train.json"
    jsonl_path = data_dir / "localsearchbench_train.jsonl"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "source": "https://huggingface.co/datasets/localsearchbench/localsearchbench",
        "arxiv": "https://arxiv.org/abs/2512.07436",
        "official_code": "https://github.com/localsearchbench/localsearchbench",
        "hf_sha": HF_SHA,
        "downloaded_file": str(parquet_path),
        "converted_json": str(json_path),
        "converted_jsonl": str(jsonl_path),
        "num_rows": int(len(df)),
        "columns": list(df.columns),
        "city_counts": {
            str(key): int(value)
            for key, value in df["City"].value_counts().sort_index().items()
        },
        "difficulty_counts": {
            str(key): int(value)
            for key, value in df["Difficulty"].value_counts().sort_index().items()
        },
        "hop_counts": {
            str(int(key)): int(value)
            for key, value in df["Hop Count"].value_counts().sort_index().items()
        },
    }
    (data_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare LocalSearchBench dataset files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    parquet_path = args.data_dir / "train-00000-of-00001.parquet"
    download_parquet(parquet_path, force=args.force_download)
    summary = convert(parquet_path, args.data_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
