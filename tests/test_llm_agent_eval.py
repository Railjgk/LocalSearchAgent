from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import llm_agent_eval


def test_parse_jsonish_accepts_fenced_json():
    parsed = llm_agent_eval.parse_jsonish(
        """```json
{"scores": {"overall": 88}, "pass": true}
```"""
    )

    assert parsed["scores"]["overall"] == 88
    assert parsed["pass"] is True


def test_fallback_generation_covers_three_horizons():
    profile = {
        "profile_id": "p1",
        "city": "上海",
        "partial_profile": {
            "group": "一家三口",
            "preferences": ["亲子", "轻食"],
            "constraints": ["预算 500"],
        },
    }

    cases = llm_agent_eval.generate_cases([profile], cases_per_horizon=1, call_llm=False)

    assert {case["horizon"] for case in cases} == {"short", "one_day", "two_day"}
    assert all(case["case_id"].startswith("llmcase_p1_") for case in cases)
    assert all(case["expected"]["must_satisfy"] for case in cases)
    assert "具体" not in json.dumps(cases[0]["expected"], ensure_ascii=False)


def test_summarize_actual_state_keeps_semantic_plan_fields():
    state = {
        "scene_type": "family",
        "constraints": {"budget": 500},
        "selected_plan": {
            "plan_id": "plan_1",
            "timeline": [
                {
                    "time": "14:00",
                    "activity": "亲子活动",
                    "poi_id": "poi_1",
                    "duration_min": 90,
                    "notes": ["室内"],
                }
            ],
            "nodes": [
                {
                    "type": "activity",
                    "name": "某活动",
                    "category": "parent_child",
                    "tags": ["kid_friendly"],
                    "price": 120,
                }
            ],
            "budget": {"total_price": 320},
        },
        "execution_status": "completed",
        "payment_status": "paid",
        "explanation_text": "适合孩子",
        "tool_results": {"reserve": {"success": True}},
    }

    summary = llm_agent_eval.summarize_actual_state(state, steps=[{"node": "x"}])

    assert summary["selected_plan"]["plan_id"] == "plan_1"
    assert summary["selected_plan"]["timeline"][0]["activity"] == "亲子活动"
    assert summary["selected_plan"]["nodes"][0]["category"] == "parent_child"
    assert summary["selected_plan"]["estimated_total_price"] == 320
    assert summary["step_count"] == 1


def test_heuristic_evaluator_outputs_quantitative_report():
    case = llm_agent_eval.normalize_case(
        {
            "horizon": "short",
            "user_request": "今天下午亲子轻食",
            "expected": {
                "must_satisfy": ["亲子", "轻食", "预算"],
                "should_satisfy": ["解释理由"],
                "avoid": ["长时间排队"],
                "result_shape": {"min_nodes": 1, "max_nodes": 2},
            },
        },
        {"profile_id": "p1"},
    )
    run = {
        "case": case,
        "actual_summary": {
            "selected_plan": {
                "timeline": [{"activity": "亲子活动"}, {"activity": "轻食餐厅"}]
            },
            "execution_status": "completed",
            "final_share_message": "亲子和轻食都已考虑",
        },
    }

    evaluation = llm_agent_eval.heuristic_evaluate_run(run)
    report = llm_agent_eval.build_report([case], [run], [evaluation])

    assert evaluation["scores"]["overall"] > 70
    assert evaluation["pass"] is True
    assert report["summary"]["pass_rate"] == 1.0
    assert report["summary"]["by_horizon"]["short"]["count"] == 1
