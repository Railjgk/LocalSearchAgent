#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_b_eval import (  # noqa: E402
    DEFAULT_CASES_PATH,
    DEFAULT_POLICY_PATH,
    load_cases,
    run_pipeline,
    seed_for_case,
    validate_case,
)


HEALTH_TAGS = {"low_calorie", "light_food", "low_oil", "low_sugar", "high_protein", "vegetable_rich"}
MICRO_VACATION_TAGS = {"micro_vacation", "wellness_micro_vacation", "wellness_spa", "spa", "wellness"}
LOCAL_CULTURE_TAGS = {"citywalk", "local_culture", "city_limited", "local_market", "local_experience"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _collect_tags(value: Any) -> list[str]:
    tags: list[str] = []
    if isinstance(value, dict):
        for nested in value.values():
            tags.extend(_collect_tags(nested))
    elif isinstance(value, list):
        for nested in value:
            tags.extend(_collect_tags(nested))
    elif value not in (None, ""):
        tags.append(str(value).strip())

    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        if tag and tag not in seen:
            seen.add(tag)
            result.append(tag)
    return result


def _selected_base_plan(state: dict[str, Any]) -> dict[str, Any]:
    selected = state.get("selected_plan") or {}
    selected_plan_id = selected.get("plan_id")
    if not selected_plan_id:
        return {}
    candidate_id = str(selected_plan_id).replace("plan_", "cand_", 1)
    for collection_name in ("filtered_candidates", "candidates"):
        for item in state.get(collection_name, []) or []:
            if item.get("plan_id") == candidate_id:
                return item
    return {}


def _nodes_by_type(plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    nodes = plan.get("nodes", []) or []
    activity = next((item for item in nodes if item.get("type") == "activity"), {})
    restaurant = next((item for item in nodes if item.get("type") == "restaurant"), {})
    return activity, restaurant


def _has_execution_targets(selected_plan: dict[str, Any]) -> bool:
    hints = selected_plan.get("action_hints", []) or []
    relevant = [
        hint
        for hint in hints
        if hint.get("action_type") in {"order_activity_ticket", "reserve_restaurant"}
    ]
    if not relevant:
        return False
    return all(
        hint.get("poi_id")
        and hint.get("merchant_id")
        and (hint.get("product_id") or hint.get("deal_id"))
        for hint in relevant
    )


def _timeline_transition_gap(selected_plan: dict[str, Any]) -> int:
    for item in selected_plan.get("timeline", []) or []:
        if item.get("type") == "transition":
            try:
                return int(item.get("duration_min") or 0)
            except (TypeError, ValueError):
                return 0
    return 0


def _case_tags(case: dict[str, Any]) -> set[str]:
    input_state = case.get("input_state", {}) or {}
    tags = set(_collect_tags(case.get("tags")))
    tags.update(_collect_tags(input_state.get("scenario_activities")))
    tags.update(_collect_tags(input_state.get("user_profile", {}).get("activity_preference")))
    tags.update(_collect_tags(input_state.get("user_profile", {}).get("emotion_need")))
    tags.update(_collect_tags(input_state.get("constraints", {}).get("planning_preferences")))
    return tags


def analyze_case(case: dict[str, Any], *, policy_path: Path, seed: int) -> dict[str, Any]:
    case_id = str(case.get("case_id", "unknown"))
    random.seed(seed_for_case(seed, case_id))
    state = run_pipeline(case.get("input_state", {}) or {}, policy_path=policy_path)
    eval_errors = validate_case(case, state)

    selected = state.get("selected_plan") or {}
    base_plan = _selected_base_plan(state)
    activity, restaurant = _nodes_by_type(base_plan)
    selected_tags = set(_collect_tags(base_plan.get("tags")))
    selected_tags.update(_collect_tags(activity))
    selected_tags.update(_collect_tags(restaurant))
    intent_tags = _case_tags(case)

    flags: list[dict[str, str]] = []
    for error in eval_errors:
        flags.append({"severity": "error", "code": "eval_expectation_failed", "message": error})

    if selected:
        objective = selected.get("objective_vector", {}) or {}
        preference_score = float(objective.get("preference", 0.0) or 0.0)
        route_score = float(objective.get("route", 0.0) or 0.0)
        transition_gap = _timeline_transition_gap(selected)
        execution_contract = selected.get("execution_contract") or {}

        if not selected.get("execution_ready"):
            flags.append({"severity": "warn", "code": "not_execution_ready", "message": "selected plan has warning constraint status"})
        if not _has_execution_targets(selected):
            flags.append({"severity": "warn", "code": "missing_execution_target_ids", "message": "action_hints should include poi, merchant, and product/deal ids"})
        if not execution_contract:
            flags.append({"severity": "warn", "code": "missing_execution_contract", "message": "selected plan should expose execution_contract checks"})
        elif not execution_contract.get("ready"):
            failed = [
                check.get("name")
                for check in execution_contract.get("checks", [])
                if check.get("status") == "fail"
            ]
            flags.append({"severity": "warn", "code": "execution_contract_failed", "message": f"failed checks: {failed}"})
        if preference_score < 0.5:
            flags.append({"severity": "info", "code": "weak_preference_match", "message": f"preference objective is low: {preference_score:.3f}"})
        if route_score < 0.45:
            flags.append({"severity": "info", "code": "route_tradeoff", "message": f"route objective is low: {route_score:.3f}"})
        if transition_gap > 75:
            flags.append({"severity": "info", "code": "long_transition_gap", "message": f"transition gap is {transition_gap} minutes"})
        if "micro_vacation" in intent_tags and not selected_tags.intersection(MICRO_VACATION_TAGS):
            flags.append({"severity": "warn", "code": "intent_category_miss", "message": "micro-vacation intent did not select a micro-vacation activity"})
        if intent_tags.intersection(LOCAL_CULTURE_TAGS) and not selected_tags.intersection(LOCAL_CULTURE_TAGS):
            flags.append({"severity": "warn", "code": "local_culture_miss", "message": "local-culture intent did not select local/citywalk supply"})
        if intent_tags.intersection(HEALTH_TAGS) and not selected_tags.intersection(HEALTH_TAGS):
            flags.append({"severity": "warn", "code": "health_intent_miss", "message": "health/light-food intent did not select health-tagged restaurant"})

    return {
        "case_id": case_id,
        "scene_type": (case.get("input_state", {}) or {}).get("scene_type"),
        "passed_eval": not eval_errors,
        "flags": flags,
        "selected_plan_id": selected.get("plan_id"),
        "optimization_score": state.get("optimization_score", 0.0),
        "execution_ready": bool(selected.get("execution_ready")),
        "activity": {
            "poi_id": activity.get("poi_id"),
            "name": activity.get("name"),
            "category": activity.get("category"),
        },
        "restaurant": {
            "poi_id": restaurant.get("poi_id"),
            "name": restaurant.get("name"),
            "category": restaurant.get("restaurant_category") or restaurant.get("category"),
        },
        "total_price": selected.get("total_price"),
        "total_distance_km": selected.get("total_distance_km"),
        "max_queue_time_min": (selected.get("availability", {}) or {}).get("max_queue_time_min"),
        "objective_vector": selected.get("objective_vector", {}),
        "action_hints": selected.get("action_hints", []),
        "execution_contract": selected.get("execution_contract", {}),
        "alternative_plans": state.get("alternative_plans", []),
    }


def build_report(cases: list[dict[str, Any]], *, policy_path: Path, seed: int) -> dict[str, Any]:
    case_reports = [analyze_case(case, policy_path=policy_path, seed=seed) for case in cases]
    selected_pair_keys = [
        (
            item.get("activity", {}).get("poi_id"),
            item.get("restaurant", {}).get("poi_id"),
        )
        for item in case_reports
        if item.get("selected_plan_id")
    ]
    selected_pairs = Counter(selected_pair_keys)
    pair_scene_counts: dict[tuple[str, str], Counter] = {}
    for item in case_reports:
        if not item.get("selected_plan_id"):
            continue
        pair = (
            item.get("activity", {}).get("poi_id"),
            item.get("restaurant", {}).get("poi_id"),
        )
        pair_scene_counts.setdefault(pair, Counter())[str(item.get("scene_type") or "unknown")] += 1

    flag_counts = Counter(flag["code"] for item in case_reports for flag in item.get("flags", []))
    severity_counts = Counter(flag["severity"] for item in case_reports for flag in item.get("flags", []))
    selected_total = sum(1 for item in case_reports if item.get("selected_plan_id"))
    top_pair, top_pair_count = (None, 0)
    if selected_pairs:
        top_pair, top_pair_count = selected_pairs.most_common(1)[0]
    top_pair_scene_counts = pair_scene_counts.get(top_pair, Counter()) if top_pair else Counter()
    top_pair_ratio = round(top_pair_count / selected_total, 3) if selected_total else 0.0
    diversity_flags = []
    if top_pair_count >= 5 and top_pair_ratio >= 0.30 and len(top_pair_scene_counts) > 1:
        diversity_flags.append(
            {
                "severity": "info",
                "code": "over_repeated_cross_scene_pair",
                "message": (
                    f"top selected pair covers {top_pair_count}/{selected_total} selected cases; "
                    "it appears across multiple scenes, so add per-intent diversity policy or supply"
                ),
            }
        )

    return {
        "policy_path": str(policy_path),
        "total_cases": len(case_reports),
        "passed_eval_cases": sum(1 for item in case_reports if item.get("passed_eval")),
        "flag_counts": dict(flag_counts),
        "severity_counts": dict(severity_counts),
        "diversity_summary": {
            "selected_total": selected_total,
            "top_pair": {
                "activity_id": top_pair[0] if top_pair else None,
                "restaurant_id": top_pair[1] if top_pair else None,
                "count": top_pair_count,
                "ratio": top_pair_ratio,
                "scene_counts": dict(top_pair_scene_counts),
            },
            "flags": diversity_flags,
        },
        "repeated_selected_pairs": [
            {
                "activity_id": pair[0],
                "restaurant_id": pair[1],
                "count": count,
                "scene_counts": dict(pair_scene_counts.get(pair, Counter())),
            }
            for pair, count in selected_pairs.most_common()
            if count > 1
        ],
        "cases": case_reports,
    }


def print_report(report: dict[str, Any]) -> None:
    print("=" * 80)
    print("WeekendFlow B Plan Quality Diagnostics")
    print("=" * 80)
    print(
        f"cases={report['total_cases']} eval_passed={report['passed_eval_cases']} "
        f"flags={report.get('flag_counts', {})}"
    )
    if report.get("repeated_selected_pairs"):
        print("Repeated selected pairs:")
        for item in report["repeated_selected_pairs"][:5]:
            print(f"  - {item['activity_id']} + {item['restaurant_id']}: {item['count']} cases")
    for flag in (report.get("diversity_summary", {}) or {}).get("flags", []):
        print(f"Diversity: {flag['severity']} {flag['code']}: {flag['message']}")
    print()

    for item in report["cases"]:
        flags = item.get("flags", [])
        if not flags:
            continue
        print(f"[{item['case_id']}] {item['activity']['poi_id']} + {item['restaurant']['poi_id']}")
        for flag in flags:
            print(f"  - {flag['severity']} {flag['code']}: {flag['message']}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose B selected-plan quality beyond pass/fail eval.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.case_id:
        cases = [case for case in cases if case.get("case_id") == args.case_id]
        if not cases:
            print(f"No case found for case_id={args.case_id}")
            return 2

    report = build_report(cases, policy_path=args.policy, seed=args.seed)
    print_report(report)

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report written to: {args.report_out}")

    return 1 if report.get("severity_counts", {}).get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
