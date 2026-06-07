from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import llm_agent_eval
from src.nodes.longcat_client import LongCatConfig


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
    assert all(case["expected"]["poi_reference"] for case in cases)
    assert cases[0]["expected"]["result_shape"]["max_primary_nodes"] == 2
    assert "具体" not in json.dumps(cases[0]["expected"], ensure_ascii=False)


def test_call_llm_json_retries_before_success():
    calls = []
    sleeps = []
    original_chat_completion = llm_agent_eval.chat_completion
    original_sleep = llm_agent_eval.time.sleep
    original_read_retry_config = llm_agent_eval.load_eval_llm_retry_config

    def fake_chat_completion(messages, *, config=None):
        calls.append(messages)
        if len(calls) == 1:
            raise TimeoutError("slow")
        return {
            "content": '{"ok": true}',
            "model": "test-model",
            "usage": {"total_tokens": 3},
            "finish_reason": "stop",
        }

    try:
        llm_agent_eval.chat_completion = fake_chat_completion
        llm_agent_eval.time.sleep = lambda seconds: sleeps.append(seconds)
        llm_agent_eval.load_eval_llm_retry_config = lambda: (1, 0.5)

        parsed, metadata = llm_agent_eval.call_llm_json(
            "system",
            {"payload": "x"},
            config=LongCatConfig(api_key="test-key", base_url="https://example.test", model="test-model"),
        )
    finally:
        llm_agent_eval.chat_completion = original_chat_completion
        llm_agent_eval.time.sleep = original_sleep
        llm_agent_eval.load_eval_llm_retry_config = original_read_retry_config

    assert parsed == {"ok": True}
    assert len(calls) == 2
    assert sleeps == [0.5]
    assert metadata["attempt_count"] == 2
    assert metadata["retry_errors"] == ["slow"]


def test_strict_llm_generation_backfills_missing_horizons():
    profile = {
        "profile_id": "p1",
        "city": "上海",
        "partial_profile": {"group": "朋友", "preferences": ["拍照"], "constraints": ["下雨"]},
    }
    calls = []
    original_call_llm_json = llm_agent_eval.call_llm_json

    def fake_call_llm_json(system_prompt, payload, config=None):
        calls.append(payload)
        horizons = payload["horizons"]
        if calls and len(calls) == 1:
            horizons = ["short"]
        return (
            {
                "cases": [
                    {
                        "horizon": horizon,
                        "user_request": f"{horizon} LLM request {len(calls)}",
                        "expected": {
                            "must_satisfy": ["a", "b", "c"],
                            "should_satisfy": ["d"],
                            "avoid": ["e"],
                            "result_shape": {"min_nodes": 1, "max_primary_nodes": 2},
                        },
                    }
                    for horizon in horizons
                ]
            },
            {"provider": "longcat_openai_compatible", "model": "test-model"},
        )

    try:
        llm_agent_eval.call_llm_json = fake_call_llm_json
        cases = llm_agent_eval.generate_cases(
            [profile],
            cases_per_horizon=1,
            call_llm=True,
            strict_llm_generation=True,
            max_generation_attempts=3,
        )
    finally:
        llm_agent_eval.call_llm_json = original_call_llm_json

    assert [case["horizon"] for case in cases] == ["short", "one_day", "two_day"]
    assert len(calls) == 2
    assert calls[1]["horizons"] == ["one_day", "two_day"]
    assert all(llm_agent_eval.is_llm_generated_case(case) for case in cases)


def test_strict_llm_generation_raises_instead_of_fallback_after_backfill_exhaustion():
    profile = {
        "profile_id": "p1",
        "city": "上海",
        "partial_profile": {"group": "朋友", "preferences": ["拍照"], "constraints": ["下雨"]},
    }
    original_call_llm_json = llm_agent_eval.call_llm_json

    def fake_call_llm_json(system_prompt, payload, config=None):
        return (
            {
                "cases": [
                    {
                        "horizon": "short",
                        "user_request": "only short LLM request",
                        "expected": {"must_satisfy": ["a", "b", "c"]},
                    }
                ]
            },
            {"provider": "longcat_openai_compatible", "model": "test-model"},
        )

    try:
        llm_agent_eval.call_llm_json = fake_call_llm_json
        try:
            llm_agent_eval.generate_cases(
                [profile],
                cases_per_horizon=1,
                call_llm=True,
                strict_llm_generation=True,
                max_generation_attempts=2,
            )
        except RuntimeError as exc:
            assert "missing=" in str(exc)
            assert "local_fallback" not in str(exc)
        else:
            raise AssertionError("strict LLM generation should fail without fallback cases")
    finally:
        llm_agent_eval.call_llm_json = original_call_llm_json


def test_load_seed_cases_marks_codex_provider_and_appends_without_duplicates():
    seed_cases = llm_agent_eval.load_seed_cases(llm_agent_eval.DEFAULT_HARD_SEED_CASES_PATH)

    assert len(seed_cases) == 4
    assert {case["horizon"] for case in seed_cases} == {"short", "one_day", "two_day"}
    assert all(
        (case["generation_metadata"] or {}).get("provider") == "codex_seeded_realistic_case"
        for case in seed_cases
    )
    assert all(llm_agent_eval.is_allowed_generated_case(case) for case in seed_cases)

    merged = llm_agent_eval.append_seed_cases([seed_cases[0]], seed_cases)

    assert len(merged) == 4
    assert merged[0]["case_id"] == seed_cases[0]["case_id"]


def test_load_external_cases_supports_codex_architecture_provider(tmp_path):
    raw_path = tmp_path / "codex_arch_cases.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "case_id": "codexarch_fixed_anchor_conflict",
                "profile_id": "codex_arch_fixed_anchor",
                "profile": {
                    "profile_id": "codex_arch_fixed_anchor",
                    "city": "上海",
                    "partial_profile": {
                        "group": "两位大人和一个孩子",
                        "preferences": ["少排队"],
                        "constraints": ["固定演出时间"],
                    },
                },
                "horizon": "one_day",
                "user_request": "周日有固定演出时间，帮我安排轻松一天，孩子要午睡。",
                "expected": {
                    "must_satisfy": ["保留固定演出时间", "保留午睡窗口", "少排队"],
                    "result_shape": {"min_nodes": 2, "max_primary_nodes": 4},
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = llm_agent_eval.load_external_cases(
        raw_path,
        provider="codex_architecture_challenge_case",
        model="codex-test",
        source="architecture_report.md",
    )

    assert len(cases) == 1
    assert cases[0]["generation_metadata"] == {
        "provider": "codex_architecture_challenge_case",
        "model": "codex-test",
        "source": "architecture_report.md",
    }
    assert llm_agent_eval.is_allowed_generated_case(cases[0])


def test_summarize_actual_state_keeps_semantic_plan_fields():
    state = {
        "scene_type": "family",
        "intent": {
            "task_type": "local_life_plan",
            "scene": "family",
            "people_count": 3,
            "missing_slots": ["origin"],
        },
        "constraints": {"budget": 500},
        "user_profile": {
            "retrieved_memory_ids": ["preference_profile"],
            "stable_profile": {"default_transport": "drive_or_taxi"},
        },
        "value_memory": [
            {
                "value_id": "family_care",
                "label": "儿童优先",
                "score": 0.9,
                "confidence": 0.8,
                "source": "history",
                "ttl": "long_term",
            }
        ],
        "selected_plan": {
            "plan_id": "plan_1",
            "total_distance_km": 2.4,
            "weighted_score": 0.88,
            "execution_ready": True,
            "objective_vector": {"group_fit": 0.9, "route": 0.7},
            "supply_identity": {"activity_id": "poi_1"},
            "action_hints": [{"action_type": "reserve_restaurant"}],
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
        "action_sequence": [{"step": 1, "action_type": "reserve_restaurant", "poi_id": "poi_1"}],
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
    assert summary["component_summaries"]["intent"]["scene"] == "family"
    assert summary["component_summaries"]["memory"]["retrieved_memory_ids"] == ["preference_profile"]
    assert summary["component_summaries"]["planner"]["selected_plan_id"] == "plan_1"
    assert summary["component_summaries"]["execution"]["action_sequence"][0]["action_type"] == "reserve_restaurant"


def test_summarize_component_state_uses_blueprint_horizon_fallback():
    state = {
        "constraints": {
            "b_itinerary_blueprint": {
                "planning_horizon": "full_day",
                "planning_days": 1,
                "template_mode": "expanded_day",
                "node_count": 4,
                "route_pattern": ["start", "activity", "meal", "rest"],
            }
        },
        "selected_plan": {"plan_id": "plan_1", "timeline": []},
    }

    summary = llm_agent_eval.summarize_component_state(state)
    planner = summary["planner"]

    assert planner["planning_horizon"] == "full_day"
    assert planner["planning_days"] == 1
    assert planner["plan_shape"] == {
        "template_mode": "expanded_day",
        "node_count": 4,
        "route_pattern": ["start", "activity", "meal", "rest"],
    }


def test_summarize_component_state_compacts_execution_blocker_chinese_first():
    state = {
        "execution_status": "failed",
        "execution_failure_type": "no_executable_actions",
        "payment_status": "not_required",
        "action_sequence": [],
        "tool_results": {},
        "execution_blocker": {
            "reason_zh": "缺少满足硬约束的具体候选和确认依据",
            "blocker_summary_zh": "缺少满足硬约束的具体候选和确认依据；暂不能进入预订。",
            "node_reasons_zh": [f"节点{i} 缺少可执行证据" for i in range(10)],
            "candidate_evidence_counts": {
                "raw_candidate_count": 10,
                "normalized_candidate_count": 8,
            },
            "selected_plan_status": "needs_rag_candidate_evidence",
            "source": "execution_handoff",
            "protected_non_executable_anchors_zh": [
                "13:30-15:00 午睡/休息",
                "15:00-15:15 缓冲",
            ],
            "raw_candidates": [{"name": "不应复制"}],
            "selected_plan": {"timeline": [{"activity": "不应复制"}]},
            "tool_results": {"reserve": {"success": False}},
        },
    }

    summary = llm_agent_eval.summarize_component_state(state)
    execution = summary["execution"]
    blocker = execution["execution_blocker"]

    assert execution["execution_failure_type"] == "no_executable_actions"
    assert execution["action_sequence"] == []
    assert execution["failed_tools"] == []
    assert blocker["reason_zh"] == "缺少满足硬约束的具体候选和确认依据"
    assert blocker["selected_plan_status"] == "needs_rag_candidate_evidence"
    assert blocker["candidate_evidence_counts"] == {
        "raw_candidate_count": 10,
        "normalized_candidate_count": 8,
    }
    assert blocker["protected_non_executable_anchors_zh"] == [
        "13:30-15:00 午睡/休息",
        "15:00-15:15 缓冲",
    ]
    assert len(blocker["node_reasons_zh"]) == 8
    assert "raw_candidates" not in blocker
    assert "selected_plan" not in blocker
    assert "tool_results" not in blocker


def test_write_compact_run_summary_merges_final_state_blocker(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    runs_path = tmp_path / "runs.jsonl"
    out_json = tmp_path / "compact.json"
    out_md = tmp_path / "compact.md"
    case = {
        "case_id": "case_fixed_anchor",
        "profile_id": "p1",
        "horizon": "one_day",
        "generation_metadata": {"provider": "codex_seeded_realistic_case"},
        "difficulty_tags": ["fixed_anchors"],
        "architecture_targets": [],
        "user_request": "周日带孩子和老人，13:30-15:00 午睡/休息。",
        "expected": {
            "must_satisfy": ["午睡窗口不可安排主要活动"],
            "should_satisfy": [],
            "avoid": ["硬塞行程"],
            "poi_reference": [],
            "result_shape": {},
        },
    }
    run = {
        "case_id": "case_fixed_anchor",
        "case": case,
        "actual_summary": {
            "component_summaries": {
                "intent": {
                    "scene": "family",
                    "people_count": 4,
                    "time": {"start_time": "10:00"},
                    "budget": {"amount": 1000, "type": "total"},
                    "hard_tags": ["低强度"],
                    "soft_tags": ["亲子"],
                    "avoid": ["商场拥挤"],
                    "missing_slots": [],
                },
                "memory": {"retrieved_memory_ids": ["value_family_care"]},
                "planner": {
                    "planning_horizon": "full_day",
                    "planning_days": 1,
                    "selected_plan_id": "plan_1",
                },
                "execution": {
                    "execution_status": "failed",
                    "payment_status": "not_required",
                    "failed_tools": [],
                    "tool_results": [],
                    "action_sequence": [],
                },
            },
            "selected_plan": {
                "plan_id": "plan_1",
                "timeline": [
                    {"time": "13:30-15:00", "activity": "午睡/休息", "poi_id": None}
                ],
            },
            "execution_status": "failed",
            "payment_status": "not_required",
            "final_share_message": "暂时没有形成可执行的预订动作。",
            "execution_log_tail": [],
        },
        "final_state": {
            "execution_status": "failed",
            "execution_failure_type": "no_executable_actions",
            "payment_status": "not_required",
            "selected_plan": {
                "planning_horizon": "full_day",
                "planning_days": 1,
                "execution_ready": False,
                "nodes": [{"name": "被拒POI推荐"}],
            },
            "execution_blocker": {
                "reason_zh": "时间骨架与候选营业/时段不匹配",
                "blocker_summary_zh": "午睡硬窗口不可转成可执行动作。",
                "node_reasons_zh": ["13:30-15:00 午睡/休息 是保护锚点"],
                "candidate_evidence_counts": {
                    "raw_candidate_count": 20,
                    "normalized_candidate_count": 20,
                },
                "selected_plan_status": "needs_rag_candidate_evidence",
                "source": "execution_handoff",
                "protected_non_executable_anchors_zh": ["13:30-15:00 午睡/休息"],
                "execution_ready": False,
                "raw_candidates": [{"name": "RAW_CANDIDATE_SECRET"}],
                "raw_rag_evidence": ["RAW_RAG_SECRET"],
                "tool_results": {"reserve": {"internal_payload": "TOOL_INTERNAL_SECRET"}},
            },
            "tool_results": {"reserve": {"internal_payload": "TOOL_INTERNAL_SECRET"}},
        },
    }
    llm_agent_eval.write_jsonl(cases_path, [case])
    llm_agent_eval.write_jsonl(runs_path, [run])

    llm_agent_eval.write_compact_run_summary(cases_path, runs_path, out_json, out_md)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    md = out_md.read_text(encoding="utf-8")
    item = payload["items"][0]
    execution = item["execution"]
    blocker = execution["execution_blocker"]
    rendered = json.dumps(payload, ensure_ascii=False) + md

    assert execution["execution_failure_type"] == "no_executable_actions"
    assert blocker["reason_zh"] == "时间骨架与候选营业/时段不匹配"
    assert blocker["selected_plan_status"] == "needs_rag_candidate_evidence"
    assert blocker["candidate_evidence_counts"] == {
        "raw_candidate_count": 20,
        "normalized_candidate_count": 20,
    }
    assert blocker["protected_non_executable_anchors_zh"] == ["13:30-15:00 午睡/休息"]
    assert "reason_zh=时间骨架与候选营业/时段不匹配" in md
    assert "selected_plan_status=needs_rag_candidate_evidence" in md
    assert "candidate_evidence_counts=" in md
    assert "\"raw_candidate_count\": 20" in md
    assert "\"normalized_candidate_count\": 20" in md
    assert "protected_non_executable_anchors_zh=['13:30-15:00 午睡/休息']" in md
    assert "RAW_CANDIDATE_SECRET" not in rendered
    assert "RAW_RAG_SECRET" not in rendered
    assert "TOOL_INTERNAL_SECRET" not in rendered
    assert "execution_ready" not in rendered
    assert "被拒POI推荐" not in rendered
    assert "预订成功" not in rendered
    assert "已预约" not in rendered


def test_write_compact_run_summary_fills_completed_replan_metadata(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    runs_path = tmp_path / "runs.jsonl"
    out_json = tmp_path / "compact.json"
    out_md = tmp_path / "compact.md"
    case = {
        "case_id": "case_completed_replan",
        "profile_id": "p1",
        "horizon": "one_day",
        "generation_metadata": {"provider": "codex_seeded_realistic_case"},
        "difficulty_tags": ["rain", "budget"],
        "architecture_targets": [],
        "user_request": "明天朋友下雨 citywalk，预算有限，能订的先订。",
        "expected": {
            "must_satisfy": ["雨天风险", "预算"],
            "should_satisfy": [],
            "avoid": ["商场拥挤"],
            "poi_reference": [],
            "result_shape": {},
        },
    }
    run = {
        "case_id": "case_completed_replan",
        "case": case,
        "actual_summary": {
            "component_summaries": {
                "intent": {
                    "scene": "friends",
                    "people_count": 4,
                    "time": {"start_time": "10:30", "end_time": "20:30"},
                    "budget": {"amount": 180, "type": "per_person"},
                    "hard_tags": ["素食", "不辣"],
                    "soft_tags": ["Citywalk"],
                    "avoid": ["商场拥挤"],
                    "missing_slots": [],
                },
                "memory": {"retrieved_memory_ids": ["value_cost_sensitivity"]},
                "planner": {
                    "planning_horizon": "full_day",
                    "planning_days": 1,
                    "selected_plan_id": "plan_1",
                },
                "execution": {
                    "execution_status": "failed",
                    "execution_failure_type": "no_executable_actions",
                    "execution_blocker": {
                        "reason_zh": "组件侧中文原因优先",
                        "blocker_summary_zh": "组件侧阻断摘要优先。",
                        "node_reasons_zh": ["组件节点原因"],
                        "candidate_evidence_counts": {
                            "raw_candidate_count": 14,
                            "normalized_candidate_count": 14,
                        },
                        "selected_plan_status": "needs_rag_candidate_evidence",
                        "source": "component_execution",
                        "protected_non_executable_anchors_zh": [],
                        "raw_candidates": [{"name": "COMPONENT_RAW_SECRET"}],
                        "raw_rag_evidence": ["COMPONENT_RAG_SECRET"],
                    },
                    "payment_status": "not_required",
                    "failed_tools": [],
                    "tool_results": [],
                    "action_sequence": [],
                },
            },
            "selected_plan": {"plan_id": "plan_1", "timeline": []},
            "execution_status": "failed",
            "payment_status": "not_required",
            "final_share_message": "已完成一次有界候选证据/时段修复，仍未形成可执行动作。",
            "execution_log_tail": [],
        },
        "final_state": {
            "execution_status": "failed",
            "execution_failure_type": "no_executable_actions",
            "payment_status": "not_required",
            "selected_plan": {
                "planning_horizon": "full_day",
                "planning_days": 1,
                "execution_ready": False,
                "nodes": [{"name": "被拒POI推荐"}],
            },
            "execution_blocker": {
                "reason_zh": "最终态旧原因不应覆盖组件原因",
                "blocker_summary_zh": "最终态旧摘要不应覆盖组件摘要。",
                "node_reasons_zh": ["最终态节点原因不应覆盖"],
                "candidate_evidence_counts": {
                    "raw_candidate_count": 18,
                    "normalized_candidate_count": 18,
                },
                "selected_plan_status": "needs_rag_candidate_evidence",
                "source": "execution_handoff",
                "completed_replan_attempt": True,
                "replan_source": "skeleton_candidate_evidence",
                "replan_candidate_count": 18,
                "replan_filtered_count": 0,
                "replan_result_zh": (
                    "已完成一次有界候选证据/时段修复，候选计数为18、"
                    "过滤后计数为0，仍未形成可执行动作。"
                ),
                "upstream_replan_request": False,
                "execution_ready": False,
                "raw_candidates": [{"name": "FINAL_RAW_SECRET"}],
                "raw_rag_evidence": ["FINAL_RAG_SECRET"],
                "tool_results": {"reserve": {"internal_payload": "TOOL_INTERNAL_SECRET"}},
            },
            "tool_results": {"reserve": {"internal_payload": "TOOL_INTERNAL_SECRET"}},
        },
    }
    llm_agent_eval.write_jsonl(cases_path, [case])
    llm_agent_eval.write_jsonl(runs_path, [run])

    llm_agent_eval.write_compact_run_summary(cases_path, runs_path, out_json, out_md)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    md = out_md.read_text(encoding="utf-8")
    blocker = payload["items"][0]["execution"]["execution_blocker"]
    rendered = json.dumps(payload, ensure_ascii=False) + md

    assert blocker["reason_zh"] == "组件侧中文原因优先"
    assert blocker["blocker_summary_zh"] == "组件侧阻断摘要优先。"
    assert blocker["node_reasons_zh"] == ["组件节点原因"]
    assert blocker["candidate_evidence_counts"] == {
        "raw_candidate_count": 14,
        "normalized_candidate_count": 14,
    }
    assert blocker["source"] == "component_execution"
    assert blocker["completed_replan_attempt"] is True
    assert blocker["replan_source"] == "skeleton_candidate_evidence"
    assert blocker["replan_candidate_count"] == 18
    assert blocker["replan_filtered_count"] == 0
    assert blocker["upstream_replan_request"] is False
    assert "仍未形成可执行动作" in blocker["replan_result_zh"]
    assert "bounded_replan_zh=已完成一次有界候选证据/时段修复" in md
    assert "completed_replan_attempt=True" in md
    assert "replan_source=skeleton_candidate_evidence" in md
    assert "replan_candidate_count=18" in md
    assert "replan_filtered_count=0" in md
    assert "COMPONENT_RAW_SECRET" not in rendered
    assert "COMPONENT_RAG_SECRET" not in rendered
    assert "FINAL_RAW_SECRET" not in rendered
    assert "FINAL_RAG_SECRET" not in rendered
    assert "TOOL_INTERNAL_SECRET" not in rendered
    assert "execution_ready" not in rendered
    assert "被拒POI推荐" not in rendered
    assert "预订成功" not in rendered
    assert "已预约" not in rendered


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
    assert {item["component"] for item in evaluation["component_findings"]} == {
        "intent",
        "memory",
        "planner",
        "execution",
        "explanation",
    }
    assert report["summary"]["pass_rate"] == 1.0
    assert report["summary"]["by_horizon"]["short"]["count"] == 1
    assert report["summary"]["component_status_counts"]["execution"]["pass"] == 1


def test_heuristic_evaluator_uses_broad_poi_reference_and_primary_nodes():
    case = llm_agent_eval.normalize_case(
        {
            "horizon": "short",
            "user_request": "今天下午亲子活动后吃健康轻食，少排队少绕路",
            "expected": {
                "must_satisfy": ["符合同行人画像", "控制总时长和预算", "减少排队和绕路"],
                "should_satisfy": ["解释推荐理由"],
                "avoid": ["长时间排队"],
                "poi_reference": [
                    {
                        "name": "child_friendly_activity",
                        "expectation": "活动 POI 应体现儿童友好、安全、有趣或低强度",
                        "evidence_terms": ["儿童友好", "亲子", "低强度"],
                    },
                    {
                        "name": "healthy_or_restricted_dining",
                        "expectation": "餐饮 POI 应体现健康、低卡、少油、轻食或满足饮食限制",
                        "evidence_terms": ["低卡", "轻食", "少油"],
                    },
                    {
                        "name": "route_and_queue_control",
                        "expectation": "路线、排队和预约风险应被控制",
                        "evidence_terms": ["排队风险低", "距离", "路线控制合理"],
                    },
                ],
                "result_shape": {"min_nodes": 1, "max_primary_nodes": 2},
            },
        },
        {"profile_id": "p1"},
    )
    run = {
        "case": case,
        "actual_summary": {
            "selected_plan": {
                "estimated_total_price": 418,
                "total_distance_km": 3.3,
                "timeline": [
                    {
                        "activity": "树屋亲子陶艺体验馆",
                        "poi_id": "act_family_ceramic",
                        "notes": ["儿童友好", "低强度"],
                    },
                    {"activity": "附近休息与转场", "poi_id": None, "notes": ["避免行程过满"]},
                    {
                        "activity": "禾间轻食日料",
                        "poi_id": "res_light_japanese",
                        "notes": ["低卡/少油选项", "轻食"],
                    },
                ],
            },
            "execution_status": "success",
            "explanation_text": "可预约性好，排队风险低；路线控制合理；预算使用合理。",
        },
    }

    evaluation = llm_agent_eval.heuristic_evaluate_run(run)

    assert evaluation["pass"] is True
    assert "shape_gap" not in evaluation["failure_categories"]
    assert "活动 POI 应体现儿童友好、安全、有趣或低强度" in evaluation["met_expectations"]
