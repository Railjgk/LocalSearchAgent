#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate whether a Gaode supply extraction is ready for B optimizer testing."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO_ROOT / "experiments" / "b_eval_cases_gaode_v2.jsonl"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    print("\n$ " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a generated Gaode supply mock data directory.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--eval-cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir
    required = ["activities.json", "restaurants.json", "availability.json", "deals.json", "products.json", "merchants.json"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        print(f"Extraction output is incomplete. Missing: {', '.join(missing)}")
        return 2

    build_report = load_json(data_dir / "build_report.json", {})
    coverage_report = load_json(data_dir / "coverage_report.json", {})
    dedupe_report = load_json(data_dir / "dedupe_report.json", {})
    quota_ledger = load_json(data_dir / "quota_ledger.json", {})
    quota_totals = quota_ledger.get("totals", {}) if isinstance(quota_ledger, dict) else {}

    print("Gaode supply run summary")
    print(f"- data_dir: {data_dir}")
    print(f"- attempted_calls: {build_report.get('attempted_calls', 'unknown')}")
    print(f"- quota_attempted_calls: {build_report.get('quota_attempted_calls', build_report.get('attempted_calls', 'unknown'))}")
    print(f"- successful_calls: {build_report.get('successful_calls', 'unknown')}")
    if quota_totals:
        print(f"- quota_attempted_total: {quota_totals.get('attempted_calls', 'unknown')}")
        print(f"- quota_http_attempted_total: {quota_totals.get('quota_attempted_calls', quota_totals.get('attempted_calls', 'unknown'))}")
        print(f"- quota_successful_total: {quota_totals.get('successful_calls', 'unknown')}")
        print(f"- estimated_remaining_calls: {quota_ledger.get('estimated_remaining_calls', 'unknown')}")
    print(f"- raw_search_request_lines: {count_jsonl(data_dir / 'raw_search_requests.jsonl')}")
    print(f"- activity_count: {build_report.get('activity_count', 'unknown')}")
    print(f"- restaurant_count: {build_report.get('restaurant_count', 'unknown')}")
    print(f"- deduped_records: {build_report.get('deduped_records', 'unknown')}")
    print(f"- duplicate_groups: {dedupe_report.get('duplicate_groups', 'unknown')}")
    if coverage_report:
        print(f"- activity_categories: {coverage_report.get('activity_categories', {})}")
        print(f"- restaurant_categories: {coverage_report.get('restaurant_categories', {})}")

    quality_code = run(
        [
            sys.executable,
            "experiments/check_mock_data_quality.py",
            "--mock-dir",
            str(data_dir),
        ]
    )
    if quality_code != 0:
        return quality_code

    if args.skip_eval:
        return 0

    env = os.environ.copy()
    env["WF_MOCK_DATA_DIR"] = str(data_dir)
    return run(
        [
            sys.executable,
            "experiments/run_b_eval.py",
            "--cases",
            str(args.eval_cases),
            "--report-out",
            str(REPO_ROOT / "experiments" / "artifacts" / f"b_eval_{data_dir.name}.json"),
        ],
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
