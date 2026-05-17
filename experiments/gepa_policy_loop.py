#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run one GEPA-style reflective policy iteration for WeekendFlow B.

This script keeps B's runtime deterministic. The LLM is used offline only to
propose small planner_policy.yaml mutations from eval traces.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from reflect_and_mutate import (  # noqa: E402
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
from run_b_eval import (  # noqa: E402
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
from src.nodes.longcat_client import (  # noqa: E402
    DEFAULT_LONGCAT_BASE_URL,
    DEFAULT_LONGCAT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    LongCatConfig,
    chat_completion,
    sanitize_longcat_error,
)


DEFAULT_GEPA_MODEL = os.getenv("LONGCAT_MODEL", DEFAULT_LONGCAT_MODEL)


@contextmanager
def mock_data_dir_context(mock_dir: Path | None):
    key = "WF_MOCK_DATA_DIR"
    previous = os.environ.get(key)
    if mock_dir is not None:
        os.environ[key] = str(mock_dir)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def build_iteration_dir(root: Path, label: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    path = root / f"gepa_iteration_{stamp}{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_phase(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def evaluate_policy(
    *,
    policy_path: Path,
    cases: list[dict[str, Any]],
    seed: int,
    mock_dir: Path | None,
    report_out: Path | None = None,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    with mock_data_dir_context(mock_dir):
        for index, case in enumerate(cases):
            case_id = str(case.get("case_id", f"case_{index}"))
            random.seed(seed_for_case(seed, case_id))
            state = run_pipeline(case.get("input_state", {}) or {}, policy_path=policy_path)
            errors = validate_case(case, state)
            summaries.append(summarize_case(case, state, errors))
            traces.append(build_trace(case, state, errors))

    payload = {
        "aggregate": build_aggregate_report(summaries),
        "cases": summaries,
        "traces": traces,
        "mock_dir": str(mock_dir) if mock_dir else None,
    }
    if report_out:
        save_json(report_out, payload)
    return payload


def build_trace(case: dict[str, Any], state: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    selected_plan = state.get("selected_plan", {}) or {}
    return {
        "case_id": case.get("case_id"),
        "tags": case.get("tags", []),
        "input_state": case.get("input_state", {}),
        "expected": case.get("expected", {}),
        "errors": errors,
        "execution_log": state.get("execution_log", []),
        "filter_reasons": state.get("filter_reasons", {}),
        "selected_plan": selected_plan,
        "selected_objective_vector": selected_plan.get("objective_vector", {}),
        "selected_score_breakdown": selected_plan.get("score_breakdown", {}),
        "alternative_plans": state.get("alternative_plans", []),
        "optimization_score": state.get("optimization_score", 0.0),
        "explanation_text": state.get("explanation_text", ""),
        "candidate_count": len(state.get("candidates", []) or []),
        "filtered_count": len(state.get("filtered_candidates", []) or []),
    }


def compare_reports(baseline_report: dict[str, Any], mutated_report: dict[str, Any]) -> dict[str, Any]:
    baseline = baseline_report.get("aggregate", {})
    mutated = mutated_report.get("aggregate", {})
    baseline_cases = {item["case_id"]: item for item in baseline_report.get("cases", [])}
    mutated_cases = {item["case_id"]: item for item in mutated_report.get("cases", [])}

    improved: list[str] = []
    regressed: list[str] = []
    changed_selection: list[str] = []
    for case_id, base_case in baseline_cases.items():
        mut_case = mutated_cases.get(case_id)
        if not mut_case:
            continue
        if not base_case["passed"] and mut_case["passed"]:
            improved.append(case_id)
        elif base_case["passed"] and not mut_case["passed"]:
            regressed.append(case_id)
        if base_case.get("selected_plan_id") != mut_case.get("selected_plan_id"):
            changed_selection.append(case_id)

    return {
        "baseline": baseline,
        "mutated": mutated,
        "delta": {
            "pass_rate": round(float(mutated.get("pass_rate", 0.0)) - float(baseline.get("pass_rate", 0.0)), 2),
            "avg_optimization_score": round(
                float(mutated.get("avg_optimization_score", 0.0))
                - float(baseline.get("avg_optimization_score", 0.0)),
                2,
            ),
            "execution_ready_rate": round(
                float(mutated.get("execution_ready_rate", 0.0))
                - float(baseline.get("execution_ready_rate", 0.0)),
                2,
            ),
        },
        "case_changes": {
            "improved": improved,
            "regressed": regressed,
            "changed_selection": changed_selection,
        },
    }


def guardrail_decision(
    comparison: dict[str, Any],
    *,
    min_pass_rate: float,
    min_execution_ready_rate: float,
    allow_avg_score_drop: float,
) -> dict[str, Any]:
    mutated = comparison.get("mutated", {})
    delta = comparison.get("delta", {})
    regressed = comparison.get("case_changes", {}).get("regressed", [])

    reasons: list[str] = []
    if regressed:
        reasons.append(f"regressed cases: {regressed}")
    if float(mutated.get("pass_rate", 0.0)) < min_pass_rate:
        reasons.append(f"pass_rate below {min_pass_rate}")
    if float(mutated.get("execution_ready_rate", 0.0)) < min_execution_ready_rate:
        reasons.append(f"execution_ready_rate below {min_execution_ready_rate}")
    if float(delta.get("avg_optimization_score", 0.0)) < -abs(allow_avg_score_drop):
        reasons.append(f"avg score dropped more than {allow_avg_score_drop}")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "guardrails": {
            "min_pass_rate": min_pass_rate,
            "min_execution_ready_rate": min_execution_ready_rate,
            "allow_avg_score_drop": allow_avg_score_drop,
        },
    }


def parse_jsonish(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("LLM response did not contain a JSON object")


def load_longcat_config_for_gepa() -> LongCatConfig:
    api_key = (os.getenv("LONGCAT_API_KEY") or os.getenv("LONGCAT_APP_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Set LONGCAT_API_KEY or LONGCAT_APP_KEY before using --call-llm")
    return LongCatConfig(
        api_key=api_key,
        base_url=(os.getenv("LONGCAT_BASE_URL") or DEFAULT_LONGCAT_BASE_URL).strip().rstrip("/"),
        model=(os.getenv("LONGCAT_MODEL") or DEFAULT_GEPA_MODEL).strip(),
        timeout_seconds=float(os.getenv("LONGCAT_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
        max_tokens=int(os.getenv("LONGCAT_MAX_TOKENS", "2400")),
        temperature=float(os.getenv("LONGCAT_TEMPERATURE", "0.2")),
    )


def call_reflection_llm(prompt_package: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_longcat_config_for_gepa()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a GEPA-style reflective optimizer for WeekendFlow B. "
                "Return valid JSON only. Do not edit code. Propose small planner_policy.yaml changes."
            ),
        },
        {"role": "user", "content": prompt_package},
    ]
    response = chat_completion(messages, config=config)
    reflection = parse_jsonish(response["content"])
    return reflection, {
        "provider": "longcat",
        "model": response.get("model") or config.model,
        "usage": response.get("usage", {}),
        "finish_reason": response.get("finish_reason"),
    }


def print_aggregate(label: str, aggregate: dict[str, Any]) -> None:
    print(
        f"{label}: pass_rate={aggregate.get('pass_rate', 0.0)}% "
        f"({aggregate.get('passed_cases', 0)}/{aggregate.get('total_cases', 0)}) "
        f"avg_score={aggregate.get('avg_optimization_score', 0.0)} "
        f"execution_ready_rate={aggregate.get('execution_ready_rate', 0.0)}%"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an automated GEPA-style reflective policy loop for WeekendFlow B."
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--reflection-json", type=Path, default=None)
    parser.add_argument("--call-llm", action="store_true", help="Call LongCat to generate reflection JSON")
    parser.add_argument("--mock-dir", type=Path, default=None, help="Optional mock data dir for this iteration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-traces", type=int, default=5)
    parser.add_argument("--include-passes", action="store_true")
    parser.add_argument("--max-changes", type=int, default=5)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-pass-rate", type=float, default=100.0)
    parser.add_argument("--min-execution-ready-rate", type=float, default=100.0)
    parser.add_argument("--allow-avg-score-drop", type=float, default=1.0)
    args = parser.parse_args()

    policy = load_policy(args.policy)
    cases = load_cases(args.cases)
    prompt_text = args.prompt.read_text(encoding="utf-8")
    iteration_dir = build_iteration_dir(args.out_dir, args.label)

    print_phase("Phase 1: Baseline Eval")
    baseline_report_path = iteration_dir / "baseline_eval_report.json"
    baseline_report = evaluate_policy(
        policy_path=args.policy,
        cases=cases,
        seed=args.seed,
        mock_dir=args.mock_dir,
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
    prompt_package_path = iteration_dir / "reflection_prompt_package.md"
    prompt_package = render_prompt_package(reflection_bundle)
    save_json(reflection_bundle_path, reflection_bundle)
    ensure_parent(prompt_package_path)
    prompt_package_path.write_text(prompt_package, encoding="utf-8")
    print(f"Reflection bundle written to: {reflection_bundle_path}")
    print(f"Prompt package written to: {prompt_package_path}")
    print(f"Trace bundle size: {len(trace_bundle)}")
    print()

    print_phase("Phase 3: Reflection")
    llm_meta: dict[str, Any] | None = None
    if args.reflection_json:
        reflection = json.loads(args.reflection_json.read_text(encoding="utf-8-sig"))
        print(f"Loaded reflection JSON: {args.reflection_json}")
    elif args.call_llm:
        try:
            reflection, llm_meta = call_reflection_llm(prompt_package)
        except Exception as exc:
            print(f"LLM reflection failed: {sanitize_longcat_error(exc)}")
            return 2
        print(f"LLM reflection generated by {llm_meta.get('model')}")
    else:
        print("No reflection source provided.")
        print("Next: rerun with --call-llm, or save a JSON response and pass --reflection-json <path>.")
        return 0

    reflection_path = iteration_dir / "reflection_response.json"
    save_json(reflection_path, {"reflection": reflection, "llm_meta": llm_meta})
    print(f"Reflection response written to: {reflection_path}")
    print()

    print_phase("Phase 4: Mutate Policy")
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
            "source_policy": str(args.policy),
            "reflection_response": str(reflection_path),
            "applied_change_count": len(applied_changes),
            "applied_changes": applied_changes,
        },
    )
    print(f"Mutated policy written to: {mutated_policy_path}")
    print(f"Applied changes: {len(applied_changes)}")
    print()

    print_phase("Phase 5: Mutated Eval")
    mutated_report_path = iteration_dir / "mutated_eval_report.json"
    mutated_report = evaluate_policy(
        policy_path=mutated_policy_path,
        cases=cases,
        seed=args.seed,
        mock_dir=args.mock_dir,
        report_out=mutated_report_path,
    )
    print_aggregate("Mutated", mutated_report["aggregate"])
    print(f"Mutated report written to: {mutated_report_path}")
    print()

    print_phase("Phase 6: Compare And Guardrail")
    comparison = compare_reports(baseline_report, mutated_report)
    decision = guardrail_decision(
        comparison,
        min_pass_rate=args.min_pass_rate,
        min_execution_ready_rate=args.min_execution_ready_rate,
        allow_avg_score_drop=args.allow_avg_score_drop,
    )
    comparison_path = iteration_dir / "iteration_comparison.json"
    save_json(
        comparison_path,
        {
            "comparison": comparison,
            "guardrail_decision": decision,
            "mutated_policy_path": str(mutated_policy_path),
        },
    )
    print_aggregate("Baseline", comparison["baseline"])
    print_aggregate("Mutated", comparison["mutated"])
    print(f"Delta: {comparison['delta']}")
    print(f"Regressed cases: {comparison['case_changes']['regressed']}")
    print(f"Changed selections: {comparison['case_changes']['changed_selection']}")
    print(f"Guardrail accepted: {decision['accepted']}")
    if decision["reasons"]:
        print(f"Reasons: {decision['reasons']}")
    print(f"Comparison written to: {comparison_path}")

    return 0 if decision["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
