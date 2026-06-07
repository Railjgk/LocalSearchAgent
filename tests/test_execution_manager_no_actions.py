from src.nodes.execution_manager import execution_manager_node


def test_no_action_skeleton_gets_chinese_blocker_metadata() -> None:
    state = {
        "selected_plan": {
            "plan_status": "needs_rag_candidate_evidence",
            "execution_ready": False,
            "execution_blockers": [
                "硬时间窗、坚果过敏和少步行约束缺少可执行候选证据。"
            ],
            "candidate_evidence_grade": {
                "raw_candidate_count": 10,
                "normalized_candidate_count": 10,
                "hard_filter_reason": (
                    "已召回活动和餐饮候选，但尚未确认任何活动加晚餐组合"
                    "同时满足硬时间窗、过敏、少步行等硬约束"
                ),
                "nodes": [
                    {
                        "time": "16:20-18:13",
                        "label": "亲子活动",
                        "execution_status": "raw_candidates_failed_hard_constraints",
                        "hard_filter_reason": "硬时间窗、过敏、少步行约束未同时满足",
                    }
                ],
            },
            "timeline": [
                {"time": "16:20-18:13", "activity": "亲子活动", "poi_id": None},
                {"time": "18:13-19:30", "activity": "晚餐", "poi_id": None},
            ],
        },
        "action_sequence": [],
        "raw_api_results": {},
        "execution_commit_result": {},
        "execution_log": [],
    }

    result = execution_manager_node(state)

    assert result["execution_status"] == "failed"
    assert result["tool_results"] == {}
    assert result["raw_api_results"] == {}
    assert result["execution_failure_type"] == "no_executable_actions"

    blocker = result["execution_blocker"]
    assert blocker["source"] == "execution_handoff"
    assert blocker["reason_code"] == "no_executable_actions"
    assert "硬时间窗、坚果过敏和少步行约束" in blocker["reason_zh"]
    assert "硬时间窗、过敏、少步行" in blocker["blocker_summary_zh"]
    assert "计划缺少可执行候选证据" in blocker["blocker_summary_zh"]
    assert (
        "16:20-18:13 亲子活动 缺少可执行证据：硬时间窗、过敏、少步行约束未同时满足"
        in blocker["node_reasons_zh"]
    )
    assert blocker["candidate_evidence_counts"]["raw_candidate_count"] == 10
    assert blocker["protected_non_executable_anchors_zh"] == []
    assert any("没有可执行动作" in item for item in result["execution_log"])


def test_no_action_blocker_preserves_protected_non_executable_anchor() -> None:
    state = {
        "selected_plan": {
            "plan_status": "needs_rag_candidate_evidence",
            "execution_ready": False,
            "timeline": [
                {"time": "10:00-12:00", "activity": "亲子活动", "poi_id": None},
                {"time": "13:30-15:00", "activity": "午睡/休息", "poi_id": None},
                {"time": "19:00-20:40", "activity": "脱口秀/演出", "poi_id": None},
            ],
        },
        "action_sequence": [],
        "raw_api_results": {},
        "execution_commit_result": {},
        "execution_log": [],
    }

    result = execution_manager_node(state)
    blocker = result["execution_blocker"]

    assert "13:30-15:00 午睡/休息" in blocker["protected_non_executable_anchors_zh"]
    assert all("午睡/休息 缺少" not in item for item in blocker["node_reasons_zh"])
    assert (
        "10:00-12:00 亲子活动 缺少可执行候选或动作证据。"
        in blocker["node_reasons_zh"]
    )
    assert (
        "19:00-20:40 脱口秀/演出 缺少可执行候选或动作证据。"
        in blocker["node_reasons_zh"]
    )


def test_no_action_blocker_surfaces_completed_bounded_replan_metadata() -> None:
    state = {
        "selected_plan": {
            "plan_status": "needs_rag_candidate_evidence",
            "execution_ready": False,
            "timeline": [
                {"time": "12:00-13:10", "activity": "餐饮", "poi_id": None},
                {"time": "20:30-次日10:00", "activity": "住宿", "poi_id": None},
            ],
            "candidate_evidence_grade": {
                "raw_candidate_count": 35,
                "normalized_candidate_count": 37,
                "hard_filter_reason": "缺少活动和餐厅均宠物友好的证据",
            },
        },
        "b_replan_attempt": {
            "status": "completed",
            "source_request": "skeleton_candidate_evidence",
            "candidate_count": 18,
            "filtered_count": 0,
            "used_rag_candidates": True,
        },
        "constraints": {
            "b_replan_trace": {
                "status": "completed",
                "last_completed_request": {
                    "status": "completed",
                    "trace_only": True,
                    "post_replan_trace_status": "completed",
                    "source": "skeleton_candidate_evidence",
                    "rag_request": {
                        "target_nodes": [
                            {
                                "label": "餐饮",
                                "poi_id": "hidden_poi",
                                "query_terms": ["不应暴露"],
                            }
                        ]
                    },
                },
            }
        },
        "action_sequence": [],
        "raw_api_results": {},
        "execution_commit_result": {},
        "execution_log": [],
    }

    result = execution_manager_node(state)
    blocker = result["execution_blocker"]

    assert blocker["completed_replan_attempt"] is True
    assert blocker["replan_source"] == "skeleton_candidate_evidence"
    assert blocker["replan_candidate_count"] == 18
    assert blocker["replan_filtered_count"] == 0
    assert "已完成一次有界候选证据/时段修复" in blocker["replan_result_zh"]
    assert "仍未形成可执行动作" in blocker["replan_result_zh"]
    assert "hidden_poi" not in str(blocker)
    assert "不应暴露" not in str(blocker)
    assert blocker["upstream_replan_request"] is False


def test_no_action_blocker_does_not_mark_active_replan_request_completed() -> None:
    state = {
        "selected_plan": {
            "plan_status": "needs_rag_candidate_evidence",
            "execution_ready": False,
            "timeline": [
                {"time": "16:20-18:13", "activity": "亲子活动", "poi_id": None}
            ],
        },
        "b_replan_request": {
            "status": "pending",
            "reason_zh": "等待补候选证据",
        },
        "action_sequence": [],
        "raw_api_results": {},
        "execution_commit_result": {},
        "execution_log": [],
    }

    result = execution_manager_node(state)
    blocker = result["execution_blocker"]

    assert blocker["upstream_replan_request"] is True
    assert "completed_replan_attempt" not in blocker
    assert "replan_result_zh" not in blocker


def test_no_action_blocker_accepts_completed_trace_only_replan() -> None:
    state = {
        "selected_plan": {
            "plan_status": "needs_rag_candidate_evidence",
            "execution_ready": False,
            "timeline": [
                {"time": "16:20-18:13", "activity": "亲子活动", "poi_id": None}
            ],
        },
        "constraints": {
            "b_replan_trace": {
                "status": "completed",
                "last_completed_request": {
                    "status": "completed",
                    "trace_only": True,
                    "post_replan_trace_status": "completed",
                    "source": "skeleton_candidate_evidence",
                },
            }
        },
        "action_sequence": [],
        "raw_api_results": {},
        "execution_commit_result": {},
        "execution_log": [],
    }

    result = execution_manager_node(state)
    blocker = result["execution_blocker"]

    assert blocker["completed_replan_attempt"] is True
    assert blocker["replan_source"] == "skeleton_candidate_evidence"
    assert blocker["replan_candidate_count"] is None
    assert blocker["replan_filtered_count"] is None
    assert blocker["upstream_replan_request"] is False


def test_failed_raw_api_result_is_not_no_action_blocker() -> None:
    state = {
        "action_sequence": [],
        "raw_api_results": {
            "reserve_restaurant_1": {
                "action": "reserve_restaurant",
                "name": "青禾沙拉碗",
                "result": {"success": False, "error": "库存不足"},
            }
        },
        "execution_commit_result": {},
        "execution_log": [],
    }

    result = execution_manager_node(state)

    assert result["execution_status"] == "failed"
    assert result["tool_results"]["reserve_restaurant_1"]["success"] is False
    assert "execution_failure_type" not in result
    assert "execution_blocker" not in result


def test_failed_commit_result_is_not_no_action_blocker() -> None:
    state = {
        "action_sequence": [],
        "raw_api_results": {},
        "execution_commit_result": {
            "success": False,
            "execution_id": "exec_failed",
            "error": "commit failed",
        },
        "execution_log": [],
    }

    result = execution_manager_node(state)

    assert result["execution_status"] == "failed"
    assert result["tool_results"]["execution_commit"]["success"] is False
    assert "execution_failure_type" not in result
    assert "execution_blocker" not in result
