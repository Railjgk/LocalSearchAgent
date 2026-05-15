#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive test for B module pipeline.
Tests candidate_generator, constraint_filter, plan_optimizer, and explainability.
Includes assertions and metrics output.
"""

import random
from copy import deepcopy

from src.nodes.candidate_generator import candidate_generator_node
from src.nodes.constraint_filter import constraint_filter_node
from src.nodes.plan_optimizer import plan_optimizer_node
from src.nodes.explainability import explainability_node


def run_case(case_name: str, state: dict) -> dict:
    """Run a single test case and return metrics."""
    print("=" * 80)
    print(f"[CASE] {case_name}")
    print("=" * 80)

    # Avoid cross-case mutation.
    state = deepcopy(state)
    random.seed(0)

    metrics = {
        "case": case_name,
        "passed": False,
        "errors": [],
        "candidates_count": 0,
        "filtered_count": 0,
        "has_selected_plan": False,
        "optimization_score": 0.0,
        "alternative_plans_count": 0,
        "execution_ready": False,
        "explanation_generated": False,
    }

    try:
        # Stage 1: Candidate generation
        out1 = candidate_generator_node(state)
        state.update(out1)

        candidates_count = len(state.get("candidates", []))
        metrics["candidates_count"] = candidates_count

        assert candidates_count > 0, f"Expected candidates > 0, got {candidates_count}"
        print(f"  [1/4] Candidates generated: {candidates_count}")

        # Stage 2: Constraint filtering
        out2 = constraint_filter_node(state)
        state.update(out2)

        filtered_count = len(state.get("filtered_candidates", []))
        filter_reasons = state.get("filter_reasons", {})
        metrics["filtered_count"] = filtered_count

        has_summary = "_summary" in filter_reasons
        has_suggestions = "_relaxation_suggestions" in filter_reasons

        assert has_summary, "filter_reasons should have _summary"

        print(f"  [2/4] Constraints filtered: {filtered_count} remaining")
        print(f"       Filter summary: {filter_reasons.get('_summary', '')}")

        if has_suggestions and filter_reasons.get("_relaxation_suggestions"):
            print(
                f"       Relaxation suggestions: "
                f"{len(filter_reasons.get('_relaxation_suggestions', []))} items"
            )

        # Stage 3: Plan optimization
        # Important: even when filtered_count == 0, still run optimizer.
        # This matches the real LangGraph flow: constraint_filter -> plan_optimizer -> explainability.
        out3 = plan_optimizer_node(state)
        state.update(out3)

        selected_plan = state.get("selected_plan", {})
        optimization_score = state.get("optimization_score", 0.0)
        alternative_plans = state.get("alternative_plans", [])

        metrics["has_selected_plan"] = bool(selected_plan)
        metrics["optimization_score"] = optimization_score
        metrics["alternative_plans_count"] = len(alternative_plans)

        if filtered_count == 0:
            assert selected_plan == {}, "no-solution case should return empty selected_plan"
            assert optimization_score == 0.0, "no-solution case should return optimization_score = 0.0"
            assert alternative_plans == [], "no-solution case should return empty alternative_plans"

            print("  [3/4] No feasible plans; optimizer returned empty selected_plan")

        else:
            assert selected_plan, "Expected selected_plan"
            assert optimization_score > 0, "Expected optimization_score > 0"

            # Validate selected_plan structure
            assert "plan_id" in selected_plan, "selected_plan missing plan_id"
            assert "title" in selected_plan, "selected_plan missing title"
            assert "timeline" in selected_plan, "selected_plan missing timeline"
            assert "objective_vector" in selected_plan, "selected_plan missing objective_vector"
            assert "risk_factors" in selected_plan, "selected_plan missing risk_factors"
            assert "constraint_summary" in selected_plan, "selected_plan missing constraint_summary"
            assert "execution_ready" in selected_plan, "selected_plan missing execution_ready"
            assert "action_hints" in selected_plan, "selected_plan missing action_hints"
            assert selected_plan["timeline"][0]["type"] in {"play", "amusement", "museum", "art"}, (
                "timeline activity type should be compatible with C tool_router"
            )
            assert selected_plan["timeline"][-1]["type"] in {"restaurant", "eat"}, (
                "timeline restaurant type should be compatible with C tool_router"
            )

            metrics["execution_ready"] = selected_plan.get("execution_ready", False)

            print(f"  [3/4] Plan optimized: {selected_plan.get('title', 'N/A')}")
            print(f"       Score: {optimization_score:.2f}")
            print(f"       Price: {selected_plan.get('total_price')} RMB")
            print(f"       Distance: {selected_plan.get('total_distance_km')} km")
            print(f"       Duration: {selected_plan.get('total_duration_min')} min")
            print(f"       Execution ready: {metrics['execution_ready']}")

            if selected_plan.get("risk_factors"):
                print(f"       Risk factors: {'; '.join(selected_plan['risk_factors'][:2])}")

            # Validate alternative plans structure
            for alt in alternative_plans:
                assert "plan_id" in alt, "alternative_plan missing plan_id"
                assert "title" in alt, "alternative_plan missing title"
                assert "dominant_dimension" in alt, "alternative_plan missing dominant_dimension"
                assert "objective_vector" in alt, "alternative_plan missing objective_vector"

            print(f"       Alternatives: {len(alternative_plans)} options")

            if alternative_plans:
                for i, alt in enumerate(alternative_plans[:2]):
                    print(
                        f"         [{i + 1}] {alt.get('title')} "
                        f"(dominant: {alt.get('dominant_dimension')})"
                    )

        # Stage 4: Explainability
        out4 = explainability_node(state)
        state.update(out4)

        explanation_text = state.get("explanation_text", "")
        assert explanation_text, "Expected explanation_text"

        metrics["explanation_generated"] = True

        print(f"  [4/4] Explanation generated ({len(explanation_text)} chars)")
        print(f"       {explanation_text[:120]}...")

        metrics["passed"] = True

    except AssertionError as e:
        metrics["errors"].append(f"Assertion: {str(e)}")
        print(f"  [FAILED] {str(e)}")

    except Exception as e:
        metrics["errors"].append(f"Exception: {type(e).__name__}: {str(e)}")
        print(f"  [ERROR] {type(e).__name__}: {str(e)}")

    print()
    return metrics


def main():
    """Run all test cases and report metrics."""
    cases = [
        (
            "family 正常场景",
            {
                "user_input": "今天下午想和老婆孩子出去玩几个小时，别离家太远，孩子5岁，老婆最近在减肥。",
                "scene_type": "family",
                "constraints": {
                    "child_age": "5岁",
                    "mom_diet": "low_calorie",
                    "max_distance_km": 8,
                    "max_queue_time": 30,
                    "duration": 4,
                    "budget": 500,
                    "people_count": 3,
                },
                "user_profile": {
                    "avoid": ["long_queue", "crowded_mall"],
                    "food_preference": ["light_food", "japanese"],
                },
                "short_term_memory": [],
                "scenario_activities": ["亲子乐园", "轻食餐厅"],
                "execution_log": [],
            },
        ),
        (
            "没有 child_age",
            {
                "user_input": "今天想和朋友随便吃个饭，看看附近有什么好推荐。",
                "scene_type": "friends",
                "constraints": {
                    "mom_diet": "low_calorie",
                    "max_distance_km": 8,
                    "max_queue_time": 30,
                    "duration_range": [4, 6],
                    "budget": 600,
                },
                "user_profile": {
                    "avoid": ["crowded_mall"],
                    "food_preference": ["healthy", "light_food"],
                    "people_count": 2,
                },
                "short_term_memory": [],
                "scenario_activities": ["轻食", "室内"],
                "execution_log": [],
            },
        ),
        (
            "A-stage constraints 兼容",
            {
                "user_input": "今天下午想和老婆孩子出去玩，孩子5岁，老婆最近在减肥，别太远。",
                "scene_type": "family",
                "constraints": {
                    "scene": "family",
                    "duration_range": [4, 6],
                    "people_count": 3,
                    "max_distance_km": 8,
                    "max_queue_time_min": 15,
                    "budget": None,
                    "companions": [
                        {"role": "wife", "state": "dieting", "needs": ["low_calorie", "light_food"]},
                        {"role": "child", "age": 5, "needs": ["kid_friendly", "low_intensity"]},
                    ],
                    "hard_tags": ["kid_friendly"],
                    "soft_tags": ["low_intensity", "low_calorie", "light_food"],
                    "planning_preferences": {
                        "activity_type": ["parent_child", "light_activity", "indoor"],
                        "food_type": ["low_calorie", "light_food"],
                        "pace": "relaxed",
                    },
                    "avoid": ["long_queue", "crowded_mall"],
                },
                "user_profile": {
                    "preference_profile": {
                        "food": ["light_food", "japanese"],
                        "activity": ["indoor", "parent_child", "light_activity"],
                        "avoid": ["long_queue", "crowded_mall"],
                    },
                    "companion_profile": {
                        "child": {"age": 5, "needs": ["kid_friendly", "low_intensity"]},
                        "wife": {"state": "dieting", "needs": ["low_calorie", "light_food"]},
                    },
                },
                "short_term_memory": [],
                "execution_log": [],
            },
        ),
        (
            "low_budget 场景",
            {
                "user_input": "预算有限，想找附近便宜又能玩的地方。",
                "scene_type": "low_budget",
                "constraints": {
                    "max_distance_km": 8,
                    "max_queue_time": 30,
                    "duration_range": [4, 6],
                    "budget": 250,
                },
                "user_profile": {
                    "avoid": ["long_queue"],
                    "food_preference": ["budget", "fast_food"],
                    "people_count": 2,
                },
                "short_term_memory": [],
                "scenario_activities": ["附近", "便宜"],
                "execution_log": [],
            },
        ),
        (
            "无解场景",
            {
                "user_input": "只想找离得最近、排队最短、预算最少的方案。",
                "scene_type": "family",
                "constraints": {
                    "child_age": 5,
                    "mom_diet": "low_calorie",
                    "max_distance_km": 1,
                    "max_queue_time": 1,
                    "duration_range": [4, 6],
                    "budget": 50,
                },
                "user_profile": {
                    "avoid": ["crowded_mall", "long_queue"],
                    "food_preference": ["light_food"],
                    "people_count": 3,
                },
                "short_term_memory": [],
                "scenario_activities": ["儿童", "低卡"],
                "execution_log": [],
            },
        ),
    ]

    results = []
    for case_name, case_state in cases:
        result = run_case(case_name, case_state)
        results.append(result)

    # Summary report
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed_count = sum(1 for result in results if result["passed"])
    total_count = len(results)

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} - {result['case']}")

        if result["errors"]:
            for error in result["errors"]:
                print(f"    └─ {error}")
        else:
            print(
                f"    └─ Candidates: {result['candidates_count']}, "
                f"Filtered: {result['filtered_count']}, "
                f"Selected: {result['has_selected_plan']}, "
                f"Score: {result['optimization_score']:.2f}, "
                f"Ready: {result['execution_ready']}"
            )

    print()
    print(f"Total: {passed_count}/{total_count} passed")
    print()

    if passed_count == total_count:
        print("All tests passed!")
        exit(0)

    print(f"{total_count - passed_count} test(s) failed")
    exit(1)


if __name__ == "__main__":
    main()
