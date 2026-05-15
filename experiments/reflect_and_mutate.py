#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from run_b_eval import (
    DEFAULT_CASES_PATH,
    DEFAULT_POLICY_PATH,
    build_aggregate_report,
    load_cases,
    load_policy,
    run_pipeline,
    seed_for_case,
    summarize_case,
    validate_case,
)


DEFAULT_PROMPT_PATH = Path(__file__).with_name("reflection_prompt.md")
DEFAULT_OUT_DIR = Path(__file__).with_name("artifacts")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def select_cases(
    cases: list[dict[str, Any]],
    target_ids: set[str] | None,
) -> list[dict[str, Any]]:
    if not target_ids:
        return cases
    return [case for case in cases if str(case.get("case_id")) in target_ids]


def build_case_trace(case: dict[str, Any], state: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "tags": case.get("tags", []),
        "input_state": case.get("input_state", {}),
        "expected": case.get("expected", {}),
        "errors": errors,
        "execution_log": state.get("execution_log", []),
        "candidates": state.get("candidates", []),
        "filtered_candidates": state.get("filtered_candidates", []),
        "filter_reasons": state.get("filter_reasons", {}),
        "selected_plan": state.get("selected_plan", {}),
        "alternative_plans": state.get("alternative_plans", []),
        "optimization_score": state.get("optimization_score", 0.0),
        "explanation_text": state.get("explanation_text", ""),
    }


def evaluate_cases_for_reflection(
    policy: dict[str, Any],
    cases: list[dict[str, Any]],
    base_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    del policy  # reserved for future policy-driven runtime

    summaries: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    import random

    for index, case in enumerate(cases):
        case_id = str(case.get("case_id", f"case_{index}"))
        random.seed(seed_for_case(base_seed, case_id))
        state = run_pipeline(case.get("input_state", {}) or {})
        errors = validate_case(case, state)
        summaries.append(summarize_case(case, state, errors))
        traces.append(build_case_trace(case, state, errors))

    return summaries, traces


def choose_trace_bundle(
    summaries: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    max_cases: int,
    include_passes: bool,
) -> list[dict[str, Any]]:
    trace_by_id = {trace["case_id"]: trace for trace in traces}

    failed = [s for s in summaries if not s["passed"]]
    passed = [s for s in summaries if s["passed"]]

    chosen_ids: list[str] = [str(s["case_id"]) for s in failed[:max_cases]]

    if include_passes and len(chosen_ids) < max_cases:
        for item in passed:
            if len(chosen_ids) >= max_cases:
                break
            chosen_ids.append(str(item["case_id"]))

    return [trace_by_id[case_id] for case_id in chosen_ids if case_id in trace_by_id]


def build_reflection_bundle(
    policy: dict[str, Any],
    prompt_text: str,
    summaries: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policy": policy,
        "eval_summary": build_aggregate_report(summaries),
        "case_results": summaries,
        "trace_bundle": traces,
        "reflection_prompt": prompt_text,
    }


def render_prompt_package(bundle: dict[str, Any]) -> str:
    prompt_text = bundle.get("reflection_prompt", "").rstrip()
    payload = {
        "policy": bundle.get("policy", {}),
        "eval_summary": bundle.get("eval_summary", {}),
        "case_results": bundle.get("case_results", []),
        "trace_bundle": bundle.get("trace_bundle", []),
    }
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        f"{prompt_text}\n\n"
        "## Reflection Input Payload\n\n"
        "```json\n"
        f"{payload_json}\n"
        "```\n"
    )


def get_path(container: dict[str, Any], path: str) -> Any:
    current: Any = container
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_path(container: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: dict[str, Any] = container
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def append_unique_path(container: dict[str, Any], path: str, value: Any) -> None:
    existing = get_path(container, path)
    if existing is None:
        set_path(container, path, [value])
        return
    if not isinstance(existing, list):
        raise ValueError(f"Path is not a list: {path}")
    if value not in existing:
        existing.append(value)


def apply_single_change(policy: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    proposed_change = proposal.get("proposed_change", {}) or {}
    change_type = proposed_change.get("type")
    path = proposed_change.get("path")
    new_value = proposed_change.get("new_value")

    if not path or not isinstance(path, str):
        raise ValueError(f"Missing proposed_change.path in proposal {proposal.get('change_id')}")

    old_value = get_path(policy, path)

    if change_type in {
        "adjust_value",
        "relax_rule",
        "tighten_rule",
        "reweight_objectives",
    }:
        set_path(policy, path, new_value)
    elif change_type in {"add_rule", "add_template_bias", "improve_preference_inference"}:
        if isinstance(new_value, list):
            for item in new_value:
                append_unique_path(policy, path, item)
        else:
            append_unique_path(policy, path, new_value)
    else:
        raise ValueError(f"Unsupported change type: {change_type}")

    return {
        "change_id": proposal.get("change_id"),
        "path": path,
        "type": change_type,
        "old_value": old_value,
        "new_value": get_path(policy, path),
        "priority": proposal.get("priority"),
    }


def apply_reflection_to_policy(
    policy: dict[str, Any],
    reflection: dict[str, Any],
    max_changes: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mutated = copy.deepcopy(policy)
    proposals = reflection.get("policy_change_proposals", []) or []
    applied: list[dict[str, Any]] = []

    sortable = {"high": 0, "medium": 1, "low": 2}
    proposals = sorted(
        proposals,
        key=lambda item: sortable.get(str(item.get("priority", "medium")).lower(), 9),
    )

    for proposal in proposals:
        if max_changes is not None and len(applied) >= max_changes:
            break
        applied.append(apply_single_change(mutated, proposal))

    base_name = str(policy.get("policy_name", "planner_policy"))
    mutated["policy_name"] = f"{base_name}_mutated"
    mutated["version"] = str(policy.get("version", "0.1"))
    mutated["mutation_meta"] = {
        "source_policy_name": policy.get("policy_name"),
        "source_policy_version": policy.get("version"),
        "applied_change_count": len(applied),
        "applied_changes": applied,
        "reflection_summary": reflection.get("reflection_summary", {}),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    return mutated, applied


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def save_json(path: Path, data: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_prepare(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy)
    cases = load_cases(args.cases)
    cases = select_cases(cases, set(args.case_ids or []))
    prompt_text = args.prompt.read_text(encoding="utf-8")

    summaries, traces = evaluate_cases_for_reflection(policy, cases, args.seed)
    trace_bundle = choose_trace_bundle(
        summaries,
        traces,
        max_cases=args.max_cases,
        include_passes=args.include_passes,
    )

    bundle = build_reflection_bundle(policy, prompt_text, summaries, trace_bundle)
    save_json(args.bundle_out, bundle)

    prompt_package = render_prompt_package(bundle)
    ensure_parent(args.prompt_package_out)
    args.prompt_package_out.write_text(prompt_package, encoding="utf-8")

    failed = sum(1 for item in summaries if not item["passed"])
    print(f"Bundle written to: {args.bundle_out}")
    print(f"Prompt package written to: {args.prompt_package_out}")
    print(f"Cases evaluated: {len(summaries)}")
    print(f"Failed cases: {failed}")
    print(f"Trace bundle size: {len(trace_bundle)}")
    return 0


def run_apply(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy)
    with args.reflection_json.open("r", encoding="utf-8-sig") as f:
        reflection = json.load(f)

    mutated, applied = apply_reflection_to_policy(policy, reflection, args.max_changes)
    save_yaml(args.policy_out, mutated)

    mutation_report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_policy": str(args.policy),
        "reflection_json": str(args.reflection_json),
        "applied_change_count": len(applied),
        "applied_changes": applied,
    }
    save_json(args.mutation_report_out, mutation_report)

    print(f"Mutated policy written to: {args.policy_out}")
    print(f"Mutation report written to: {args.mutation_report_out}")
    print(f"Applied changes: {len(applied)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare reflection bundles and mutate planner policy.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Run eval and package traces for reflection.")
    prepare.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    prepare.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    prepare.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--max-cases", type=int, default=5)
    prepare.add_argument("--include-passes", action="store_true")
    prepare.add_argument("--case-ids", nargs="*", default=None)
    prepare.add_argument(
        "--bundle-out",
        type=Path,
        default=DEFAULT_OUT_DIR / "reflection_bundle.json",
    )
    prepare.add_argument(
        "--prompt-package-out",
        type=Path,
        default=DEFAULT_OUT_DIR / "reflection_prompt_package.md",
    )

    apply_cmd = subparsers.add_parser("apply", help="Apply a reflection JSON to create a mutated policy.")
    apply_cmd.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    apply_cmd.add_argument("--reflection-json", type=Path, required=True)
    apply_cmd.add_argument("--max-changes", type=int, default=5)
    apply_cmd.add_argument(
        "--policy-out",
        type=Path,
        default=DEFAULT_OUT_DIR / "planner_policy_mutated.yaml",
    )
    apply_cmd.add_argument(
        "--mutation-report-out",
        type=Path,
        default=DEFAULT_OUT_DIR / "mutation_report.json",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        return run_prepare(args)
    if args.command == "apply":
        return run_apply(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
