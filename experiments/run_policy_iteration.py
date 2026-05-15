#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from reflect_and_mutate import (
    DEFAULT_OUT_DIR,
    DEFAULT_PROMPT_PATH,
    apply_reflection_to_policy,
    build_reflection_bundle,
    choose_trace_bundle,
    ensure_parent,
    render_prompt_package,
    save_json,
    save_yaml,
)
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


def evaluate_policy(
    policy: dict[str, Any],
    policy_path: Path,
    cases: list[dict[str, Any]],
    seed: int,
    report_out: Path | None = None,
) -> dict[str, Any]:
    del policy  # current runtime consumes policy mainly through plan_optimizer

    summaries: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    for index, case in enumerate(cases):
        case_id = str(case.get("case_id", f"case_{index}"))
        random.seed(seed_for_case(seed, case_id))
        state = run_pipeline(case.get("input_state", {}) or {}, policy_path=policy_path)
        errors = validate_case(case, state)
        summaries.append(summarize_case(case, state, errors))
        traces.append(
            {
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
        )

    aggregate = build_aggregate_report(summaries)
    payload = {
        "aggregate": aggregate,
        "cases": summaries,
        "traces": traces,
    }

    if report_out:
        ensure_parent(report_out)
        with report_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


def compare_reports(
    baseline_report: dict[str, Any],
    mutated_report: dict[str, Any],
) -> dict[str, Any]:
    base_agg = baseline_report.get("aggregate", {})
    mut_agg = mutated_report.get("aggregate", {})

    baseline_cases = {
        item["case_id"]: item for item in baseline_report.get("cases", [])
    }
    mutated_cases = {
        item["case_id"]: item for item in mutated_report.get("cases", [])
    }

    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []

    for case_id, base_case in baseline_cases.items():
        mut_case = mutated_cases.get(case_id)
        if mut_case is None:
            continue

        if (not base_case["passed"]) and mut_case["passed"]:
            improved.append(case_id)
        elif base_case["passed"] and (not mut_case["passed"]):
            regressed.append(case_id)
        else:
            unchanged.append(case_id)

    return {
        "baseline": base_agg,
        "mutated": mut_agg,
        "delta": {
            "pass_rate": round(
                float(mut_agg.get("pass_rate", 0.0)) - float(base_agg.get("pass_rate", 0.0)),
                2,
            ),
            "avg_optimization_score": round(
                float(mut_agg.get("avg_optimization_score", 0.0))
                - float(base_agg.get("avg_optimization_score", 0.0)),
                2,
            ),
            "execution_ready_rate": round(
                float(mut_agg.get("execution_ready_rate", 0.0))
                - float(base_agg.get("execution_ready_rate", 0.0)),
                2,
            ),
        },
        "case_changes": {
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
        },
    }


def load_reflection_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_iteration_dir(root: Path, label: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    path = root / f"iteration_{stamp}{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_phase(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_aggregate(label: str, aggregate: dict[str, Any]) -> None:
    print(
        f"{label}: pass_rate={aggregate.get('pass_rate', 0.0)}% "
        f"({aggregate.get('passed_cases', 0)}/{aggregate.get('total_cases', 0)}) "
        f"avg_score={aggregate.get('avg_optimization_score', 0.0)} "
        f"execution_ready_rate={aggregate.get('execution_ready_rate', 0.0)}%"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one manual GEPA-style policy iteration for WeekendFlow B."
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--reflection-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-traces", type=int, default=5)
    parser.add_argument("--include-passes", action="store_true")
    parser.add_argument("--max-changes", type=int, default=5)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    policy = load_policy(args.policy)
    cases = load_cases(args.cases)
    prompt_text = args.prompt.read_text(encoding="utf-8")
    iteration_dir = build_iteration_dir(args.out_dir, args.label)

    print_phase("Phase 1: Baseline Eval")
    baseline_report_path = iteration_dir / "baseline_eval_report.json"
    baseline_report = evaluate_policy(
        policy=policy,
        policy_path=args.policy,
        cases=cases,
        seed=args.seed,
        report_out=baseline_report_path,
    )
    print_aggregate("Baseline", baseline_report["aggregate"])
    print(f"Baseline report written to: {baseline_report_path}")
    print()

    print_phase("Phase 2: Reflection Package")
    trace_bundle = choose_trace_bundle(
        baseline_report["cases"],
        baseline_report["traces"],
        max_cases=args.max_traces,
        include_passes=args.include_passes,
    )
    reflection_bundle = build_reflection_bundle(
        policy=policy,
        prompt_text=prompt_text,
        summaries=baseline_report["cases"],
        traces=trace_bundle,
    )
    reflection_bundle_path = iteration_dir / "reflection_bundle.json"
    reflection_prompt_package_path = iteration_dir / "reflection_prompt_package.md"
    save_json(reflection_bundle_path, reflection_bundle)
    ensure_parent(reflection_prompt_package_path)
    reflection_prompt_package_path.write_text(
        render_prompt_package(reflection_bundle),
        encoding="utf-8",
    )
    print(f"Reflection bundle written to: {reflection_bundle_path}")
    print(f"Prompt package written to: {reflection_prompt_package_path}")
    print(f"Trace bundle size: {len(trace_bundle)}")
    print()

    if args.reflection_json is None:
        print_phase("Phase 3: Await Reflection JSON")
        print("No --reflection-json provided.")
        print("Next step:")
        print("1. Send reflection_prompt_package.md to an LLM")
        print("2. Save the JSON response locally")
        print("3. Re-run this script with --reflection-json <path>")
        return 0

    print_phase("Phase 3: Apply Reflection")
    reflection = load_reflection_json(args.reflection_json)
    mutated_policy, applied_changes = apply_reflection_to_policy(
        policy=policy,
        reflection=reflection,
        max_changes=args.max_changes,
    )
    mutated_policy_path = iteration_dir / "planner_policy_mutated.yaml"
    mutation_report_path = iteration_dir / "mutation_report.json"
    save_yaml(mutated_policy_path, mutated_policy)
    save_json(
        mutation_report_path,
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_policy": str(args.policy),
            "reflection_json": str(args.reflection_json),
            "applied_change_count": len(applied_changes),
            "applied_changes": applied_changes,
        },
    )
    print(f"Mutated policy written to: {mutated_policy_path}")
    print(f"Mutation report written to: {mutation_report_path}")
    print(f"Applied changes: {len(applied_changes)}")
    print()

    print_phase("Phase 4: Mutated Eval")
    mutated_report_path = iteration_dir / "mutated_eval_report.json"
    mutated_report = evaluate_policy(
        policy=mutated_policy,
        policy_path=mutated_policy_path,
        cases=cases,
        seed=args.seed,
        report_out=mutated_report_path,
    )
    print_aggregate("Mutated", mutated_report["aggregate"])
    print(f"Mutated report written to: {mutated_report_path}")
    print()

    print_phase("Phase 5: Compare")
    comparison = compare_reports(baseline_report, mutated_report)
    comparison_path = iteration_dir / "iteration_comparison.json"
    save_json(comparison_path, comparison)
    print_aggregate("Baseline", comparison["baseline"])
    print_aggregate("Mutated", comparison["mutated"])
    print(
        f"Delta: pass_rate={comparison['delta']['pass_rate']} "
        f"avg_score={comparison['delta']['avg_optimization_score']} "
        f"execution_ready_rate={comparison['delta']['execution_ready_rate']}"
    )
    print(f"Improved cases: {comparison['case_changes']['improved']}")
    print(f"Regressed cases: {comparison['case_changes']['regressed']}")
    print(f"Comparison written to: {comparison_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
