#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze WeekendFlow B plan quality beyond pass/fail eval assertions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_b_eval import (  # noqa: E402
    DEFAULT_CASES_PATH,
    DEFAULT_POLICY_PATH,
    load_cases,
    run_pipeline,
    seed_for_case,
)
from src.nodes.b_utils import collect_preference_sources, expand_preference_tags  # noqa: E402


SERIOUS_ISSUES = {
    "intent_category_miss",
    "health_intent_miss",
    "missing_execution_target_ids",
}


def _selected_tags(state: dict[str, Any]) -> list[str]:
    selected_plan = state.get("selected_plan") or {}
    selected_id = str(selected_plan.get("plan_id") or "").replace("plan_", "cand_", 1)
    tags: list[str] = []

    for plan in state.get("filtered_candidates", []) or state.get("candidates", []) or []:
        if selected_id and plan.get("plan_id") != selected_id:
            continue
        tags.extend(plan.get("tags", []) or [])
        for node in plan.get("nodes", []) or []:
            tags.extend(node.get("tags", []) or [])
        break

    for item in selected_plan.get("timeline", []) or []:
        tags.extend(item.get("notes", []) or [])

    return expand_preference_tags(tags)


def _has_intent_category(input_state: dict[str, Any]) -> bool:
    constraints = input_state.get("constraints", {}) or {}
    user_profile = input_state.get("user_profile", {}) or {}
    scenario_activities = input_state.get("scenario_activities", []) or []
    sources = collect_preference_sources(constraints, user_profile, scenario_activities)
    return bool(sources)


def _health_intent_present(input_state: dict[str, Any]) -> bool:
    constraints = input_state.get("constraints", {}) or {}
    text = str(input_state.get("user_input") or "")
    sources = collect_preference_sources(
        constraints,
        input_state.get("user_profile", {}) or {},
        input_state.get("scenario_activities", []) or [],
    )
    health_tags = {"low_calorie", "light_food", "low_oil", "healthy"}
    return (
        constraints.get("mom_diet") == "low_calorie"
        or bool(health_tags & set(sources))
        or any(word in text for word in ("减肥", "减脂", "控卡", "低卡", "轻食", "少油", "清淡"))
    )


def _action_target_ids_missing(state: dict[str, Any]) -> bool:
    selected_plan = state.get("selected_plan") or {}
    for hint in selected_plan.get("action_hints", []) or []:
        action_type = hint.get("action_type")
        if action_type in {"order_activity_ticket", "reserve_restaurant"} and not hint.get("poi_id"):
            return True
    return False


def analyze_case(case: dict[str, Any], state: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    input_state = case.get("input_state", {}) or {}
    selected_plan = state.get("selected_plan") or {}
    selected_tags = set(_selected_tags(state))

    if not _has_intent_category(input_state):
        issues.append(
            {
                "issue": "intent_category_miss",
                "severity": "error",
                "detail": "No canonical or mappable scenario/preference tags were found.",
            }
        )

    if selected_plan and _health_intent_present(input_state):
        if not ({"low_calorie", "light_food", "low_oil", "healthy"} & selected_tags):
            issues.append(
                {
                    "issue": "health_intent_miss",
                    "severity": "error",
                    "detail": "Health intent was present but selected plan lacks health food tags.",
                }
            )

    if selected_plan and _action_target_ids_missing(state):
        issues.append(
            {
                "issue": "missing_execution_target_ids",
                "severity": "error",
                "detail": "Executable action_hints must include poi_id.",
            }
        )

    if selected_plan and not selected_plan.get("action_hints"):
        issues.append(
            {
                "issue": "missing_action_hints",
                "severity": "warning",
                "detail": "Selected plan has no action_hints for C.",
            }
        )

    constraints = input_state.get("constraints", {}) or {}
    if constraints.get("budget") is not None and "人均" in str(input_state.get("user_input") or ""):
        if constraints.get("budget_type") != "per_person":
            issues.append(
                {
                    "issue": "budget_type_miss",
                    "severity": "warning",
                    "detail": "Input mentions per-person budget but budget_type is not per_person.",
                }
            )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze WeekendFlow B plan quality.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    case_reports = []

    for index, case in enumerate(cases):
        case_id = str(case.get("case_id", f"case_{index}"))
        random.seed(seed_for_case(args.seed, case_id))
        state = run_pipeline(case.get("input_state", {}) or {}, policy_path=args.policy)
        issues = analyze_case(case, state)
        case_reports.append(
            {
                "case_id": case_id,
                "selected_plan_id": (state.get("selected_plan") or {}).get("plan_id"),
                "issues": issues,
            }
        )

    issue_counts: dict[str, int] = {}
    serious_count = 0
    for report in case_reports:
        for issue in report["issues"]:
            issue_counts[issue["issue"]] = issue_counts.get(issue["issue"], 0) + 1
            if issue["issue"] in SERIOUS_ISSUES and issue["severity"] == "error":
                serious_count += 1

    payload = {
        "total_cases": len(cases),
        "serious_issue_count": serious_count,
        "issue_counts": issue_counts,
        "cases": case_reports,
    }

    print("=" * 80)
    print("WeekendFlow B Plan Quality Analysis")
    print("=" * 80)
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report written to: {args.report_out}")

    return 0 if serious_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
