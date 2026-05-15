#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nodes.candidate_generator import candidate_generator_node
from src.nodes.constraint_filter import constraint_filter_node
from src.nodes.plan_optimizer import plan_optimizer_node
from src.nodes.explainability import explainability_node


DEFAULT_POLICY_PATH = Path(__file__).with_name("planner_policy.yaml")
DEFAULT_CASES_PATH = Path(__file__).with_name("b_eval_cases.jsonl")


@contextmanager
def planner_policy_path_context(policy_path: Path | None):
    key = "WF_PLANNER_POLICY_PATH"
    previous = os.environ.get(key)

    if policy_path is not None:
        os.environ[key] = str(policy_path.resolve())

    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def load_policy(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Policy file must load to dict: {path}")
    return data


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Case line {line_no} must be JSON object")
            cases.append(item)
    return cases


def parse_time_to_minutes(slot: str | None) -> int | None:
    if not slot or ":" not in slot:
        return None
    hour, minute = slot.split(":", 1)
    try:
        return int(hour) * 60 + int(minute)
    except ValueError:
        return None


def run_pipeline(input_state: dict[str, Any], policy_path: Path | None = None) -> dict[str, Any]:
    state = deepcopy(input_state)

    with planner_policy_path_context(policy_path):
        for node in (
            candidate_generator_node,
            constraint_filter_node,
            plan_optimizer_node,
            explainability_node,
        ):
            state.update(node(state))

    return state


def seed_for_case(base_seed: int, case_id: str) -> int:
    stable_offset = sum(ord(ch) for ch in case_id) % 100000
    return base_seed + stable_offset


def find_selected_plan_base(state: dict[str, Any]) -> dict[str, Any]:
    selected_plan = state.get("selected_plan") or {}
    selected_plan_id = selected_plan.get("plan_id")
    if not selected_plan_id:
        return {}

    candidate_plan_id = selected_plan_id.replace("plan_", "cand_", 1)

    for collection_name in ("filtered_candidates", "candidates"):
        for item in state.get(collection_name, []) or []:
            if item.get("plan_id") == candidate_plan_id:
                return item
    return {}


def collect_selected_tags(plan_base: dict[str, Any], selected_plan: dict[str, Any]) -> list[str]:
    tags = list(plan_base.get("tags", []) or [])
    for node in plan_base.get("nodes", []) or []:
        tags.extend(node.get("tags", []) or [])
    for item in selected_plan.get("timeline", []) or []:
        notes = item.get("notes", []) or []
        tags.extend(str(note) for note in notes)
    deduped: list[str] = []
    seen = set()
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    return deduped


def validate_case(case: dict[str, Any], state: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    expected = case.get("expected", {}) or {}
    selected_plan = state.get("selected_plan") or {}
    filter_reasons = state.get("filter_reasons") or {}
    explanation_text = state.get("explanation_text", "")
    plan_base = find_selected_plan_base(state)
    selected_tags = collect_selected_tags(plan_base, selected_plan)

    feasible = bool(selected_plan)
    if "feasible" in expected and feasible != bool(expected["feasible"]):
        errors.append(
            f"expected feasible={expected['feasible']}, got feasible={feasible}"
        )

    requires_selected_plan = bool(expected.get("requires_selected_plan"))
    if requires_selected_plan and not selected_plan:
        errors.append("expected selected_plan, got empty selected_plan")
    if not requires_selected_plan and expected.get("feasible") is False and selected_plan:
        errors.append("expected no selected_plan for infeasible case")

    if expected.get("requires_relaxation_suggestions"):
        suggestions = filter_reasons.get("_relaxation_suggestions") or []
        if not suggestions:
            errors.append("expected relaxation suggestions, got none")

    if selected_plan:
        if "max_total_distance_km" in expected:
            actual = float(selected_plan.get("total_distance_km", 0.0))
            if actual > float(expected["max_total_distance_km"]):
                errors.append(
                    f"expected total_distance <= {expected['max_total_distance_km']}, got {actual}"
                )

        if "max_total_price" in expected:
            actual = float(selected_plan.get("total_price", 0.0))
            if actual > float(expected["max_total_price"]):
                errors.append(
                    f"expected total_price <= {expected['max_total_price']}, got {actual}"
                )

        if "max_queue_time_min" in expected:
            availability = selected_plan.get("availability", {}) or {}
            actual = float(availability.get("max_queue_time_min", 0.0))
            if actual > float(expected["max_queue_time_min"]):
                errors.append(
                    f"expected max_queue_time_min <= {expected['max_queue_time_min']}, got {actual}"
                )

        required_timeline_types = set(expected.get("timeline_types", []))
        if required_timeline_types:
            actual_types = {
                item.get("type")
                for item in selected_plan.get("timeline", [])
                if item.get("type") not in {None, "transition"}
            }
            missing = sorted(required_timeline_types - actual_types)
            if missing:
                errors.append(f"missing timeline types: {missing}")

        required_action_types = set(expected.get("must_include_action_types", []))
        if required_action_types:
            actual_action_types = {
                item.get("action_type")
                for item in selected_plan.get("action_hints", [])
                if item.get("action_type")
            }
            missing = sorted(required_action_types - actual_action_types)
            if missing:
                errors.append(f"missing action_hints types: {missing}")

        forbidden_tags = set(expected.get("must_not_have_tags", []))
        if forbidden_tags:
            hits = sorted(tag for tag in selected_tags if tag in forbidden_tags)
            if hits:
                errors.append(f"selected plan contains forbidden tags: {hits}")

        if expected.get("valid_time_slots_only"):
            for hint in selected_plan.get("action_hints", []):
                slot = hint.get("time")
                if parse_time_to_minutes(slot) is None:
                    errors.append(f"invalid action time format: {slot}")

        preferred_traits = expected.get("preferred_plan_traits", [])
        if preferred_traits:
            trait_checks = {
                "kid_friendly_activity": "kid_friendly" in selected_tags,
                "low_calorie_restaurant": (
                    "low_calorie" in selected_tags or "light_food" in selected_tags
                ),
                "nearby_route": float(selected_plan.get("total_distance_km", 999.0)) <= 8.0,
                "short_queue": float(
                    (selected_plan.get("availability", {}) or {}).get("max_queue_time_min", 999.0)
                ) <= 30.0,
                "indoor_activity": "indoor" in selected_tags,
                "experience_balance": state.get("optimization_score", 0.0) > 0,
                "high_experience": float(
                    (selected_plan.get("objective_vector", {}) or {}).get("experience", 0.0)
                ) >= 0.5,
                "smooth_route": float(
                    (selected_plan.get("objective_vector", {}) or {}).get("route", 0.0)
                ) >= 0.5,
                "light_food": "light_food" in selected_tags or "low_calorie" in selected_tags,
                "budget_first": float(selected_plan.get("total_price", 9999.0)) <= float(
                    expected.get("max_total_price", 300.0)
                ),
            }
            missing_traits = sorted(
                trait for trait in preferred_traits if not trait_checks.get(trait, False)
            )
            if missing_traits:
                errors.append(f"missing preferred traits: {missing_traits}")

    if not explanation_text:
        errors.append("missing explanation_text")

    return errors


def summarize_case(case: dict[str, Any], state: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    selected_plan = state.get("selected_plan") or {}
    return {
        "case_id": case.get("case_id", "unknown"),
        "scene_type": (case.get("input_state", {}) or {}).get("scene_type"),
        "passed": not errors,
        "errors": errors,
        "candidates_count": len(state.get("candidates", []) or []),
        "filtered_count": len(state.get("filtered_candidates", []) or []),
        "selected_plan_id": selected_plan.get("plan_id"),
        "optimization_score": state.get("optimization_score", 0.0),
        "execution_ready": bool(selected_plan.get("execution_ready")),
        "alternative_plans_count": len(state.get("alternative_plans", []) or []),
    }


def print_header(policy: dict[str, Any], cases_path: Path, case_count: int) -> None:
    print("=" * 80)
    print("WeekendFlow B Offline Eval")
    print("=" * 80)
    print(f"Policy: {policy.get('policy_name', 'unknown')} v{policy.get('version', 'n/a')}")
    print(f"Cases:  {cases_path}")
    print(f"Count:  {case_count}")
    print("Note: policy path is injected into policy-aware B nodes; unsupported policy fields are ignored.")
    print()


def print_case_result(summary: dict[str, Any]) -> None:
    status = "PASS" if summary["passed"] else "FAIL"
    print(f"[{status}] {summary['case_id']} ({summary.get('scene_type')})")
    print(
        f"  candidates={summary['candidates_count']} filtered={summary['filtered_count']} "
        f"selected={summary.get('selected_plan_id')} score={summary['optimization_score']} "
        f"execution_ready={summary['execution_ready']} alts={summary['alternative_plans_count']}"
    )
    for error in summary["errors"]:
        print(f"  - {error}")
    print()


def build_aggregate_report(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(summaries)
    passed = sum(1 for item in summaries if item["passed"])
    selected = [item for item in summaries if item.get("selected_plan_id")]
    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "pass_rate": round((passed / total) * 100, 2) if total else 0.0,
        "avg_optimization_score": round(
            sum(float(item.get("optimization_score", 0.0)) for item in selected) / len(selected), 2
        ) if selected else 0.0,
        "execution_ready_rate": round(
            (sum(1 for item in selected if item.get("execution_ready")) / len(selected)) * 100, 2
        ) if selected else 0.0,
    }


def save_report(path: Path, policy: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    payload = {
        "policy_name": policy.get("policy_name"),
        "policy_version": policy.get("version"),
        "aggregate": build_aggregate_report(summaries),
        "cases": summaries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline evaluation for WeekendFlow B planner.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--case-id", type=str, default=None, help="Run a single case by case_id")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for deterministic evaluation")
    args = parser.parse_args()

    policy = load_policy(args.policy)
    cases = load_cases(args.cases)

    if args.case_id:
        cases = [case for case in cases if case.get("case_id") == args.case_id]
        if not cases:
            print(f"No case found for case_id={args.case_id}")
            return 2

    print_header(policy, args.cases, len(cases))

    summaries: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_id = str(case.get("case_id", f"case_{index}"))
        random.seed(seed_for_case(args.seed, case_id))
        state = run_pipeline(case.get("input_state", {}) or {}, policy_path=args.policy)
        errors = validate_case(case, state)
        summary = summarize_case(case, state, errors)
        summaries.append(summary)
        print_case_result(summary)

    aggregate = build_aggregate_report(summaries)
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(
        f"pass_rate={aggregate['pass_rate']}% "
        f"({aggregate['passed_cases']}/{aggregate['total_cases']}) "
        f"avg_score={aggregate['avg_optimization_score']} "
        f"execution_ready_rate={aggregate['execution_ready_rate']}%"
    )

    if args.report_out:
        save_report(args.report_out, policy, summaries)
        print(f"Report written to: {args.report_out}")

    return 0 if aggregate["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
