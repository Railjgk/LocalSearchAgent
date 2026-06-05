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
from src.nodes.b_poi_rag import b_poi_rag_node
from src.nodes.constraint_filter import constraint_filter_node
from src.nodes.plan_optimizer import plan_optimizer_node
from src.nodes.b_replan_loop import b_replan_loop_node
from src.nodes.explainability import explainability_node
from src.nodes.b_utils import expand_preference_tags
from src.nodes.b_semantics import b_semantic_terms, flatten_semantic_values


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


@contextmanager
def mock_data_dir_context(mock_data_dir: Path | None):
    key = "WF_MOCK_DATA_DIR"
    previous = os.environ.get(key)

    if mock_data_dir is not None:
        os.environ[key] = str(mock_data_dir.resolve())

    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@contextmanager
def env_flag_context(key: str, value: str | None):
    previous = os.environ.get(key)
    if value is not None:
        os.environ[key] = value
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


def safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def run_pipeline(
    input_state: dict[str, Any],
    policy_path: Path | None = None,
    mock_data_dir: Path | None = None,
    enable_rag: bool = False,
) -> dict[str, Any]:
    state = deepcopy(input_state)

    with (
        planner_policy_path_context(policy_path),
        mock_data_dir_context(mock_data_dir),
        env_flag_context("WF_B_RAG_ENABLED", "1" if enable_rag else None),
    ):
        for node in (
            b_poi_rag_node,
            candidate_generator_node,
            constraint_filter_node,
            plan_optimizer_node,
            b_replan_loop_node,
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
        for field_name in (
            "name",
            "category",
            "sub_category",
            "experience_type",
            "restaurant_category",
            "primary_category",
            "primary_keyword",
            "gaode_keyword",
            "signature_dishes",
            "recommended_dishes",
            "dish_tags",
            "review_keywords",
        ):
            tags.extend(flatten_semantic_values(node.get(field_name)))
    for item in selected_plan.get("timeline", []) or []:
        notes = item.get("notes", []) or []
        tags.extend(str(note) for note in notes)
    deduped: list[str] = []
    seen = set()
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    for tag in expand_preference_tags(tags):
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    for tag in b_semantic_terms(tags, include_auxiliary=True):
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    return deduped


def collect_node_tags(node: dict[str, Any]) -> list[str]:
    tags = list(node.get("tags", []) or [])
    for field_name in (
        "name",
        "category",
        "sub_category",
        "experience_type",
        "restaurant_category",
        "primary_category",
        "primary_keyword",
        "gaode_keyword",
        "signature_dishes",
        "recommended_dishes",
        "dish_tags",
        "review_keywords",
    ):
        tags.extend(flatten_semantic_values(node.get(field_name)))

    deduped: list[str] = []
    seen = set()
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    for tag in expand_preference_tags(tags):
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    for tag in b_semantic_terms(tags, include_auxiliary=True):
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
    selected_activity = next((node for node in plan_base.get("nodes", []) if node.get("type") == "activity"), {})
    selected_restaurant = next((node for node in plan_base.get("nodes", []) if node.get("type") == "restaurant"), {})
    selected_activity_tags = collect_node_tags(selected_activity)
    selected_restaurant_tags = collect_node_tags(selected_restaurant)

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
            actual = safe_float(selected_plan.get("total_distance_km"), 0.0)
            if actual > safe_float(expected["max_total_distance_km"], 0.0):
                errors.append(
                    f"expected total_distance <= {expected['max_total_distance_km']}, got {actual}"
                )

        if "max_total_price" in expected:
            actual = safe_float(selected_plan.get("total_price"), 0.0)
            if actual > safe_float(expected["max_total_price"], 0.0):
                errors.append(
                    f"expected total_price <= {expected['max_total_price']}, got {actual}"
                )

        if "max_queue_time_min" in expected:
            availability = selected_plan.get("availability", {}) or {}
            actual = safe_float(availability.get("max_queue_time_min"), 0.0)
            if actual > safe_float(expected["max_queue_time_min"], 0.0):
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

        expected_restaurant_role = expected.get("restaurant_role")
        if expected_restaurant_role:
            actual_restaurant_role = selected_plan.get("restaurant_role")
            if actual_restaurant_role != expected_restaurant_role:
                errors.append(
                    f"expected restaurant_role={expected_restaurant_role}, got {actual_restaurant_role}"
                )

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
            barbecue_signals = (
                "barbecue",
                "bbq",
                "烤肉",
                "烧烤",
                "烤串",
                "羊肉串",
                "肉串",
                "串烧",
                "炭火",
                "炭烤",
                "日式烧肉",
                "日式烤肉",
                "韩式烤肉",
                "韩式烧肉",
            )
            coffee_dessert_signals = (
                "咖啡甜品",
                "咖啡",
                "咖啡馆",
                "咖啡店",
                "精品咖啡",
                "甜品",
                "甜点",
                "蛋糕",
                "面包",
                "烘焙",
                "下午茶",
                "茶饮",
                "coffee",
                "cafe",
                "specialty_coffee",
                "dessert",
                "cake",
                "bakery",
                "afternoon_tea",
                "tea_drink",
            )
            museum_exhibition_signals = (
                "博物馆展览",
                "看展",
                "展览",
                "展馆",
                "博物馆",
                "美术馆",
                "艺术馆",
                "科技馆",
                "影像艺术",
                "museum",
                "gallery",
                "art_museum",
                "exhibition",
                "art_exhibition",
                "cultural",
                "educational",
            )
            board_game_escape_signals = (
                "密室桌游",
                "桌游",
                "棋牌",
                "狼人杀",
                "剧本杀",
                "推理馆",
                "密室",
                "密室逃脱",
                "board_game",
                "chess_cards",
                "script_murder",
                "escape_room",
                "party_game",
            )
            trait_checks = {
                "kid_friendly_activity": "kid_friendly" in selected_tags,
                "low_calorie_restaurant": (
                    any(
                        signal in selected_tags
                        for signal in ("low_calorie", "light_food", "low_oil", "low_sugar", "high_protein", "vegetable_rich")
                    )
                ),
                "nearby_route": safe_float(selected_plan.get("total_distance_km"), 999.0) <= 8.0,
                "short_queue": safe_float(
                    (selected_plan.get("availability", {}) or {}).get("max_queue_time_min"), 999.0
                ) <= 30.0,
                "indoor_activity": "indoor" in selected_tags,
                "experience_balance": state.get("optimization_score", 0.0) > 0,
                "high_experience": safe_float(
                    (selected_plan.get("objective_vector", {}) or {}).get("experience"), 0.0
                ) >= 0.5,
                "smooth_route": safe_float(
                    (selected_plan.get("objective_vector", {}) or {}).get("route"), 0.0
                ) >= 0.5,
                "light_food": "light_food" in selected_tags or "low_calorie" in selected_tags,
                "budget_first": safe_float(selected_plan.get("total_price"), 9999.0) <= safe_float(
                    expected.get("max_total_price"), 300.0
                ),
                "minimum_experience_floor": safe_float(
                    (selected_plan.get("objective_vector", {}) or {}).get("experience"), 0.0
                ) >= 0.45,
                "healthy_menu_option": any(
                    signal in selected_tags
                    for signal in ("low_calorie", "light_food", "low_oil", "low_sugar", "high_protein", "vegetable_rich")
                ),
                "local_experience_activity": any(
                    signal in selected_tags
                    for signal in ("local_experience", "local_culture", "city_limited", "local_market", "citywalk")
                ),
                "emotion_match": any(
                    signal in selected_tags
                    for signal in ("relaxation", "healing", "self_reward", "ritual")
                ),
                "micro_vacation_activity": any(
                    signal in selected_tags
                    for signal in ("micro_vacation", "wellness_micro_vacation", "wellness_spa", "spa")
                ),
                "social_restaurant_ok": any(
                    signal in selected_tags
                    for signal in ("social", "hotpot", "barbecue", "bbq", "chat_friendly", "group_friendly")
                ),
                "barbecue_restaurant": any(
                    signal in selected_tags
                    for signal in barbecue_signals
                ),
                "hotpot_restaurant": any(
                    signal in selected_tags
                    for signal in ("hotpot", "火锅", "涮锅", "牛油锅")
                ),
                "japanese_yakiniku_restaurant": any(
                    signal in selected_tags
                    for signal in ("日式烧肉", "日式烤肉", "yakiniku", "japanese_bbq")
                ),
                "charcoal_barbecue_restaurant": any(
                    signal in selected_tags
                    for signal in ("炭火", "炭烤", "charcoal_grill")
                ),
                "coffee_dessert_restaurant": any(
                    signal in selected_restaurant_tags for signal in coffee_dessert_signals
                ),
                "museum_exhibition_activity": any(
                    signal in selected_activity_tags for signal in museum_exhibition_signals
                ),
                "board_game_escape_activity": any(
                    signal in selected_activity_tags for signal in board_game_escape_signals
                ),
                "commercial_guardrail": not any(
                    signal in selected_tags for signal in ("high_calorie", "crowded_mall", "long_queue")
                ),
                "trust_evidence_present": any(
                    signal in selected_tags for signal in ("kid_friendly", "family_friendly", "low_intensity")
                ) and selected_plan.get("weighted_score", 0.0) > 0,
                "dine_in_available": bool(selected_plan.get("action_hints"))
                and any(
                    hint.get("action_type") == "reserve_restaurant"
                    for hint in selected_plan.get("action_hints", [])
                ),
                "execution_target_ids": all(
                    hint.get("poi_id")
                    and hint.get("merchant_id")
                    and (hint.get("product_id") or hint.get("deal_id"))
                    for hint in selected_plan.get("action_hints", [])
                    if hint.get("action_type") in {"order_activity_ticket", "reserve_restaurant"}
                ),
                "execution_contract_ready": bool(
                    (selected_plan.get("execution_contract") or {}).get("ready")
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
    plan_base = find_selected_plan_base(state)
    filter_reasons = state.get("filter_reasons") or {}
    sample_filter_reasons = [
        {"plan_id": str(plan_id), "reason": str(reason)}
        for plan_id, reason in filter_reasons.items()
        if not str(plan_id).startswith("_")
    ][:12]
    selected_nodes = [
        {
            "type": node.get("type"),
            "poi_id": node.get("poi_id"),
            "name": node.get("name"),
            "category": node.get("category"),
            "restaurant_category": node.get("restaurant_category"),
            "primary_category": node.get("primary_category"),
            "gaode_keyword": node.get("gaode_keyword"),
            "restaurant_role": node.get("restaurant_role"),
            "tags": (node.get("tags") or [])[:12],
        }
        for node in plan_base.get("nodes", []) or []
    ]
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
        "selected_plan_snapshot": {
            "restaurant_role": selected_plan.get("restaurant_role"),
            "total_distance_km": selected_plan.get("total_distance_km"),
            "total_price": selected_plan.get("total_price"),
            "weighted_score": selected_plan.get("weighted_score"),
        },
        "selected_nodes": selected_nodes,
        "candidate_generation_issues": state.get("candidate_generation_issues") or [],
        "candidate_recall_diagnostics": state.get("candidate_recall_diagnostics") or {},
        "filter_summary": filter_reasons.get("_summary"),
        "filter_summary_detail": filter_reasons.get("_summary_detail"),
        "sample_filter_reasons": sample_filter_reasons,
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
    parser.add_argument(
        "--mock-data-dir",
        type=Path,
        default=None,
        help="Override WF_MOCK_DATA_DIR, e.g. a Gaode full supply directory.",
    )
    parser.add_argument(
        "--enable-rag",
        action="store_true",
        help="Run B POI RAG before candidate generation to match the graph path for large local supply.",
    )
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
        state = run_pipeline(
            case.get("input_state", {}) or {},
            policy_path=args.policy,
            mock_data_dir=args.mock_data_dir,
            enable_rag=args.enable_rag,
        )
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
