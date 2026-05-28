#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Nightly autopilot for WeekendFlow Gaode supply extraction.

This script intentionally wraps `build_supply_from_gaode_v2.py` instead of
duplicating extraction logic. It owns scheduling-friendly guardrails: process
locking, quota budgets, validation, stop conditions, and daily summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.build_supply_from_gaode_v2 import (  # noqa: E402
    build_search_plan,
    completed_plan_ids,
    load_config,
    merged_settings,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "mock_data" / "gaode_supply_shanghai_v2_20260525_smoke"
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "gaode_supply_extraction_config.yaml"
DEFAULT_EVAL_CASES = REPO_ROOT / "experiments" / "b_eval_cases_gaode_v2.jsonl"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "experiments" / "artifacts" / "gaode_runs"
DEFAULT_EXPIRY_STOP = "2026-06-03T18:00:00"

LIMIT_ERROR_TOKENS = (
    "invalid_user_key",
    "invalid user key",
    "daily_query_over_limit",
    "over quota",
    "quota",
    "limit",
    "too many",
    "10001",
    "10003",
    "10004",
    "10021",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run guarded Gaode supply extraction batches.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--eval-cases", type=Path, default=DEFAULT_EVAL_CASES)
    parser.add_argument("--preferred-mode", default="gapfill_scene_semantics_v2")
    parser.add_argument("--fallback-mode", default="main_balanced")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--daily-call-budget", type=int, default=5000)
    parser.add_argument("--budget-safety-margin", type=int, default=100)
    parser.add_argument("--total-call-limit", type=int, default=39000)
    parser.add_argument("--expiry-stop", default=DEFAULT_EXPIRY_STOP)
    parser.add_argument("--sleep-sec", type=float, default=0.34)
    parser.add_argument("--request-timeout-sec", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-sec", type=float, default=1.5)
    parser.add_argument("--min-successful-calls", type=int, default=480)
    parser.add_argument("--max-error-calls", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not call Gaode or run validation.")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--allow-concurrent", action="store_true", help="Bypass extractor process guard.")
    parser.add_argument("--today", default=None, help="Testing override in YYYY-MM-DD format.")
    return parser.parse_args()


def load_local_env() -> None:
    for path in (REPO_ROOT / ".env.local", REPO_ROOT / ".env"):
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lstrip("\ufeff")
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_date(value: str | None) -> date:
    if value:
        return datetime.fromisoformat(value).date()
    return date.today()


def parse_expiry_stop(value: str) -> datetime:
    return datetime.fromisoformat(value)


def current_metrics(output_dir: Path) -> dict[str, Any]:
    build_report = read_json(output_dir / "build_report.json", {})
    coverage_report = read_json(output_dir / "coverage_report.json", {})
    quota_ledger = read_json(output_dir / "quota_ledger.json", {})
    totals = quota_ledger.get("totals", {}) if isinstance(quota_ledger, dict) else {}
    normalized = coverage_report.get("normalized_counts", {}) if isinstance(coverage_report, dict) else {}
    rich_mock = rich_mock_metrics(output_dir)
    return {
        "quota_attempted_total": int(totals.get("quota_attempted_calls") or 0),
        "quota_successful_total": int(totals.get("successful_calls") or 0),
        "quota_error_total": int(totals.get("error_calls") or 0),
        "deduped_records": int(build_report.get("deduped_records") or 0),
        "activity_count": int(build_report.get("activity_count") or normalized.get("activities") or 0),
        "restaurant_count": int(build_report.get("restaurant_count") or normalized.get("restaurants") or 0),
        "with_phone": int(normalized.get("with_phone") or 0),
        "activity_categories": coverage_report.get("activity_categories", {}) if isinstance(coverage_report, dict) else {},
        "restaurant_categories": coverage_report.get("restaurant_categories", {}) if isinstance(coverage_report, dict) else {},
        "rich_mock": rich_mock,
    }


def rich_mock_metrics(output_dir: Path) -> dict[str, Any]:
    activities = read_json(output_dir / "activities.json", [])
    restaurants = read_json(output_dir / "restaurants.json", [])
    items = []
    if isinstance(activities, list):
        items.extend(activities)
    if isinstance(restaurants, list):
        items.extend(restaurants)
    fields = [
        "review_breakdown",
        "package_options",
        "promotion_highlights",
        "decision_profile",
        "fulfillment_actions",
        "substitution_strategy",
        "peak_risk_profile",
        "reservation_slots",
        "parking_available",
        "parking_fee_policy",
        "mock_detail_sources",
    ]
    total = len(items)
    counts = {
        field: sum(1 for item in items if item.get(field) not in (None, "", [], {}))
        for field in fields
    }
    restaurant_fields = {
        "signature_dishes": sum(1 for item in restaurants if isinstance(item, dict) and item.get("signature_dishes")),
        "recommended_dishes": sum(1 for item in restaurants if isinstance(item, dict) and item.get("recommended_dishes")),
        "dietary_options": sum(1 for item in restaurants if isinstance(item, dict) and item.get("dietary_options")),
        "baby_chair_available": sum(1 for item in restaurants if isinstance(item, dict) and item.get("baby_chair_available")),
    } if isinstance(restaurants, list) else {}
    return {
        "total_items": total,
        "field_counts": counts,
        "restaurant_field_counts": restaurant_fields,
    }


def quota_used_on(day: date, output_dir: Path) -> int:
    ledger = read_json(output_dir / "quota_ledger.json", {})
    runs = ledger.get("runs", []) if isinstance(ledger, dict) else []
    total = 0
    for run in runs:
        raw_started_at = str(run.get("run_started_at") or "")
        if not raw_started_at:
            continue
        try:
            run_day = datetime.fromisoformat(raw_started_at).date()
        except ValueError:
            continue
        if run_day == day:
            total += int(run.get("quota_attempted_calls", run.get("attempted_calls", 0)) or 0)
    return total


def running_extractors() -> list[dict[str, Any]]:
    command = (
        "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | "
        "Where-Object { $_.CommandLine -match 'build_supply_from_gaode_v2' } | "
        "Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return []
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    items = parsed if isinstance(parsed, list) else [parsed]
    current_pid = os.getpid()
    return [item for item in items if int(item.get("ProcessId") or -1) != current_pid]


def namespace_for_mode(args: argparse.Namespace, mode: str, max_calls: int | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        config=args.config,
        mode=mode,
        city=None,
        output_dir=args.output_dir,
        api_key=None,
        max_calls=max_calls,
        page_size=None,
        radius=None,
        max_pages_per_query=None,
        sleep_sec=args.sleep_sec,
        request_timeout_sec=args.request_timeout_sec,
        max_retries=args.max_retries,
        retry_backoff_sec=args.retry_backoff_sec,
        quota_limit=args.total_call_limit,
        dry_run_plan=False,
        rebuild_from_raw=False,
        no_resume=False,
        reset_logs=False,
        allow_empty=False,
    )


def remaining_plan_calls(args: argparse.Namespace, mode: str) -> int:
    config = load_config(args.config)
    mode_args = namespace_for_mode(args, mode)
    settings = merged_settings(mode_args, config)
    plan = build_search_plan(config, settings)
    done = completed_plan_ids(args.output_dir / "raw_search_requests.jsonl")
    return sum(1 for query in plan if query.get("plan_id") not in done)


def choose_mode(args: argparse.Namespace) -> tuple[str, dict[str, int]]:
    preferred_remaining = remaining_plan_calls(args, args.preferred_mode)
    fallback_remaining = remaining_plan_calls(args, args.fallback_mode)
    mode = args.preferred_mode if preferred_remaining >= args.batch_size else args.fallback_mode
    return mode, {
        args.preferred_mode: preferred_remaining,
        args.fallback_mode: fallback_remaining,
    }


def run_command(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    log_prefix: Path,
) -> dict[str, Any]:
    started_at = datetime.now().isoformat(timespec="seconds")
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout_path = log_prefix.with_suffix(".out.log")
    stderr_path = log_prefix.with_suffix(".err.log")
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return {
        "cmd": redact_command(cmd),
        "returncode": completed.returncode,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def redact_command(cmd: list[str]) -> list[str]:
    redacted = []
    skip_next = False
    for item in cmd:
        if skip_next:
            redacted.append("<redacted>")
            skip_next = False
            continue
        redacted.append(item)
        if item in {"--api-key", "--key"}:
            skip_next = True
    return redacted


def run_extraction_batch(args: argparse.Namespace, mode: str, max_calls: int, batch_index: int) -> dict[str, Any]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_prefix = args.artifacts_dir / f"autopilot_batch_{batch_index:02d}_{mode}_{stamp}"
    cmd = [
        sys.executable,
        "experiments/build_supply_from_gaode_v2.py",
        "--mode",
        mode,
        "--output-dir",
        str(args.output_dir),
        "--max-calls",
        str(max_calls),
        "--sleep-sec",
        str(args.sleep_sec),
        "--max-retries",
        str(args.max_retries),
        "--retry-backoff-sec",
        str(args.retry_backoff_sec),
        "--quota-limit",
        str(args.total_call_limit),
    ]
    if args.request_timeout_sec is not None:
        cmd.extend(["--request-timeout-sec", str(args.request_timeout_sec)])
    env = os.environ.copy()
    return run_command(cmd, env=env, log_prefix=log_prefix)


def run_validation(args: argparse.Namespace, batch_index: int) -> dict[str, Any]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_prefix = args.artifacts_dir / f"autopilot_validate_{batch_index:02d}_{stamp}"
    cmd = [
        sys.executable,
        "experiments/validate_gaode_supply_run.py",
        "--data-dir",
        str(args.output_dir),
        "--eval-cases",
        str(args.eval_cases),
    ]
    env = os.environ.copy()
    env["WF_MOCK_DATA_DIR"] = str(args.output_dir)
    return run_command(cmd, env=env, log_prefix=log_prefix)


def contains_limit_error(build_report: dict[str, Any], command_result: dict[str, Any]) -> bool:
    blob = json.dumps(build_report.get("errors", []), ensure_ascii=False).lower()
    blob += "\n" + str(command_result.get("stdout_tail", "")).lower()
    blob += "\n" + str(command_result.get("stderr_tail", "")).lower()
    return any(token in blob for token in LIMIT_ERROR_TOKENS)


def warning_lines(validation_result: dict[str, Any]) -> list[str]:
    lines = []
    for source in (validation_result.get("stdout_tail", ""), validation_result.get("stderr_tail", "")):
        for line in str(source).splitlines():
            if "[WARN]" in line:
                lines.append(line)
    return lines


def category_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(before) | set(after))
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys if int(after.get(key, 0)) - int(before.get(key, 0))}


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "quota_attempted_total",
        "quota_successful_total",
        "quota_error_total",
        "deduped_records",
        "activity_count",
        "restaurant_count",
        "with_phone",
    ]
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys}


def validate_batch_stop_conditions(
    args: argparse.Namespace,
    extraction_result: dict[str, Any],
    validation_result: dict[str, Any] | None,
) -> tuple[bool, str]:
    build_report = read_json(args.output_dir / "build_report.json", {})
    successful_calls = int(build_report.get("successful_calls") or 0)
    error_calls = int(build_report.get("error_calls") or 0)
    attempted_calls = int(build_report.get("attempted_calls") or 0)

    if extraction_result["returncode"] != 0:
        return True, "extraction_command_failed"
    if contains_limit_error(build_report, extraction_result):
        return True, "api_key_or_quota_error"
    if attempted_calls > 0 and error_calls > args.max_error_calls:
        return True, "too_many_errors"
    min_successful = min(args.min_successful_calls, max(1, int(attempted_calls * 0.96)))
    if attempted_calls > 0 and successful_calls < min_successful:
        return True, "too_few_successful_calls"
    if validation_result and validation_result["returncode"] != 0:
        return True, "validation_failed"
    return False, ""


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    final_metrics = summary.get("final_metrics", {})
    deltas = summary.get("deltas", {})
    lines = [
        f"# WeekendFlow Gaode Autopilot Summary {summary.get('run_date')}",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Stop reason: `{summary.get('stop_reason')}`",
        f"- Batches attempted: {len(summary.get('batches', []))}",
        f"- Mode preference: `{summary.get('preferred_mode')}` -> fallback `{summary.get('fallback_mode')}`",
        "",
        "## Metrics",
        "",
        f"- Quota attempted total: {final_metrics.get('quota_attempted_total', 0)} ({deltas.get('quota_attempted_total', 0):+d} today delta in this run)",
        f"- Deduped POIs: {final_metrics.get('deduped_records', 0)} ({deltas.get('deduped_records', 0):+d})",
        f"- Activities: {final_metrics.get('activity_count', 0)} ({deltas.get('activity_count', 0):+d})",
        f"- Restaurants: {final_metrics.get('restaurant_count', 0)} ({deltas.get('restaurant_count', 0):+d})",
        f"- With phone: {final_metrics.get('with_phone', 0)} ({deltas.get('with_phone', 0):+d})",
        "",
        "## Rich Mock Coverage",
        "",
    ]
    rich_mock = final_metrics.get("rich_mock", {}) if isinstance(final_metrics.get("rich_mock"), dict) else {}
    field_counts = rich_mock.get("field_counts", {}) if isinstance(rich_mock.get("field_counts"), dict) else {}
    restaurant_field_counts = rich_mock.get("restaurant_field_counts", {}) if isinstance(rich_mock.get("restaurant_field_counts"), dict) else {}
    total_items = int(rich_mock.get("total_items") or 0)
    if field_counts:
        lines.extend(
            f"- {field}: {count}/{total_items}"
            for field, count in sorted(field_counts.items())
        )
    if restaurant_field_counts:
        lines.append("")
        lines.append("### Restaurant-specific")
        lines.extend(
            f"- {field}: {count}"
            for field, count in sorted(restaurant_field_counts.items())
        )
    lines.extend([
        "",
        "## Category Deltas",
        "",
    ])
    activity_delta = summary.get("activity_category_delta", {})
    restaurant_delta = summary.get("restaurant_category_delta", {})
    if activity_delta:
        lines.append("### Activities")
        lines.extend(f"- {name}: {delta:+d}" for name, delta in sorted(activity_delta.items(), key=lambda kv: -abs(kv[1]))[:20])
        lines.append("")
    if restaurant_delta:
        lines.append("### Restaurants")
        lines.extend(f"- {name}: {delta:+d}" for name, delta in sorted(restaurant_delta.items(), key=lambda kv: -abs(kv[1]))[:20])
        lines.append("")
    warnings = summary.get("warnings", [])
    lines.extend(["## Warnings", ""])
    if warnings:
        lines.extend(f"- {line}" for line in warnings[:30])
    else:
        lines.append("- None")
    lines.append("")
    lines.extend(["## Batch Records", ""])
    for batch in summary.get("batches", []):
        lines.append(
            f"- Batch {batch.get('batch_index')}: mode `{batch.get('mode')}`, "
            f"max_calls={batch.get('max_calls')}, extraction_rc={batch.get('extraction', {}).get('returncode')}, "
            f"validation_rc={batch.get('validation', {}).get('returncode') if batch.get('validation') else 'skipped'}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(args: argparse.Namespace, summary: dict[str, Any]) -> tuple[Path, Path]:
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    day = summary["run_date"].replace("-", "")
    json_path = args.artifacts_dir / f"autopilot_summary_{day}.json"
    md_path = args.artifacts_dir / f"autopilot_summary_{day}.md"
    write_json(json_path, summary)
    write_markdown_summary(md_path, summary)
    return json_path, md_path


def main() -> int:
    args = parse_args()
    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    args.artifacts_dir = args.artifacts_dir.resolve()
    args.eval_cases = args.eval_cases.resolve()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    load_local_env()
    run_day = resolve_date(args.today)
    initial_metrics = current_metrics(args.output_dir)
    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_date": run_day.isoformat(),
        "status": "started",
        "stop_reason": "",
        "output_dir": str(args.output_dir),
        "preferred_mode": args.preferred_mode,
        "fallback_mode": args.fallback_mode,
        "daily_call_budget": args.daily_call_budget,
        "batch_size": args.batch_size,
        "total_call_limit": args.total_call_limit,
        "expiry_stop": args.expiry_stop,
        "initial_metrics": initial_metrics,
        "batches": [],
        "warnings": [],
    }

    running = running_extractors()
    if running and not args.allow_concurrent:
        summary["status"] = "skipped"
        summary["stop_reason"] = "extractor_already_running"
        summary["running_processes"] = running
        summary["final_metrics"] = current_metrics(args.output_dir)
        summary["deltas"] = metric_delta(initial_metrics, summary["final_metrics"])
        json_path, md_path = write_summary(args, summary)
        print(f"Existing extractor is running; autopilot skipped. Summary: {json_path}")
        print(f"Markdown summary: {md_path}")
        return 0

    if datetime.now() >= parse_expiry_stop(args.expiry_stop):
        summary["status"] = "stopped"
        summary["stop_reason"] = "past_expiry_stop"
        summary["final_metrics"] = current_metrics(args.output_dir)
        summary["deltas"] = metric_delta(initial_metrics, summary["final_metrics"])
        json_path, md_path = write_summary(args, summary)
        print(f"Past expiry stop; summary: {json_path}")
        print(f"Markdown summary: {md_path}")
        return 0

    if not args.dry_run and not os.environ.get("GAODE_API_KEY"):
        summary["status"] = "stopped"
        summary["stop_reason"] = "missing_gaode_api_key"
        summary["final_metrics"] = current_metrics(args.output_dir)
        summary["deltas"] = metric_delta(initial_metrics, summary["final_metrics"])
        json_path, md_path = write_summary(args, summary)
        print(f"GAODE_API_KEY is missing; summary: {json_path}")
        print(f"Markdown summary: {md_path}")
        return 2

    batch_index = 0
    while True:
        final_metrics = current_metrics(args.output_dir)
        daily_used = quota_used_on(run_day, args.output_dir)
        total_remaining = args.total_call_limit - int(final_metrics.get("quota_attempted_total", 0))
        daily_remaining = args.daily_call_budget - daily_used
        mode, mode_remaining = choose_mode(args)
        max_calls = min(args.batch_size, max(0, daily_remaining), max(0, total_remaining - args.budget_safety_margin))

        if args.max_batches is not None and batch_index >= args.max_batches:
            summary["status"] = "completed"
            summary["stop_reason"] = "max_batches_reached"
            break
        if max_calls <= 0:
            summary["status"] = "completed"
            summary["stop_reason"] = "quota_budget_exhausted"
            break
        if mode_remaining.get(mode, 0) <= 0:
            summary["status"] = "completed"
            summary["stop_reason"] = "no_remaining_plan_calls"
            break

        batch_index += 1
        batch_record: dict[str, Any] = {
            "batch_index": batch_index,
            "mode": mode,
            "mode_remaining_before": mode_remaining,
            "daily_used_before": daily_used,
            "daily_remaining_before": daily_remaining,
            "total_remaining_before": total_remaining,
            "max_calls": max_calls,
        }
        summary["batches"].append(batch_record)

        if args.dry_run:
            batch_record["dry_run"] = True
            summary["status"] = "completed"
            summary["stop_reason"] = "dry_run"
            break

        extraction_result = run_extraction_batch(args, mode, max_calls, batch_index)
        batch_record["extraction"] = extraction_result

        validation_result = None
        if not args.skip_validation:
            validation_result = run_validation(args, batch_index)
            batch_record["validation"] = validation_result
            summary["warnings"].extend(warning_lines(validation_result))

        should_stop, reason = validate_batch_stop_conditions(args, extraction_result, validation_result)
        batch_record["build_report_after"] = read_json(args.output_dir / "build_report.json", {})
        batch_record["metrics_after"] = current_metrics(args.output_dir)
        if should_stop:
            summary["status"] = "stopped"
            summary["stop_reason"] = reason
            break

    final = current_metrics(args.output_dir)
    summary.setdefault("status", "completed")
    summary.setdefault("stop_reason", "completed")
    summary["final_metrics"] = final
    summary["deltas"] = metric_delta(initial_metrics, final)
    summary["activity_category_delta"] = category_delta(
        dict(Counter(initial_metrics.get("activity_categories", {}))),
        dict(Counter(final.get("activity_categories", {}))),
    )
    summary["restaurant_category_delta"] = category_delta(
        dict(Counter(initial_metrics.get("restaurant_categories", {}))),
        dict(Counter(final.get("restaurant_categories", {}))),
    )
    json_path, md_path = write_summary(args, summary)
    print(f"Autopilot status: {summary['status']} ({summary['stop_reason']})")
    print(f"Summary: {json_path}")
    print(f"Markdown summary: {md_path}")
    return 0 if summary["status"] in {"completed", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
