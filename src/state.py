"""Shared state definitions for the full planning workflow."""

from typing import Any, Dict, List, TypedDict


class ValueMemoryItem(TypedDict):
    """A calculable personal value used to steer planning."""

    value_id: str
    label: str
    score: float
    confidence: float
    ttl: str
    source: str
    planning_effect: str
    evidence: List[str]


class PlanState(TypedDict, total=False):
    """Global state passed through A, B, and C stages."""

    # Input fields
    user_input: str
    messages: List[Dict[str, Any]]
    input: Any
    query: Any
    user_id: str
    scene_type: str
    requested_city: str
    mock_data_dir: str

    # A-stage: intent, memory, and scenario modeling
    intent: Dict[str, Any]
    a_llm_intent: Dict[str, Any]
    memory: Dict[str, Any]
    retrieved_memories: List[Dict[str, Any]]
    memory_updates: List[Dict[str, Any]]
    memory_trace: Dict[str, Any]
    constraints: Dict[str, Any]
    user_profile: Dict[str, Any]
    short_term_memory: List[str]
    value_memory: List[ValueMemoryItem]
    scenario_activities: List[str]
    scenario_template: Dict[str, Any]
    route_pattern_hints: Dict[str, Any]

    # B-stage: candidate generation and plan selection
    candidates: List[Dict[str, Any]]
    candidate_generation_issues: List[Dict[str, Any]]
    filtered_candidates: List[Dict[str, Any]]
    filter_reasons: Dict[str, Any]
    selected_plan: Dict[str, Any]
    optimization_score: float
    alternative_plans: List[Dict[str, Any]]
    explanation_text: str
    b_itinerary_blueprint: Dict[str, Any]
    b_rag_node_candidates: Dict[str, Any]
    rag_node_candidates: Dict[str, Any]
    b_rag_candidate_evidence: Dict[str, Any]
    rag_candidate_evidence: Dict[str, Any]
    b_rag_candidate_metadata: Dict[str, Any]
    b_rag_candidate_coverage: Dict[str, Any]
    b_poi_rag_metadata: Dict[str, Any]
    b_requirement_contract: Dict[str, Any]
    b_ai_semantic_hints: Dict[str, Any]
    b_ai_requirement_compiler: Dict[str, Any]
    b_ai_explanation: Dict[str, Any]
    b_ai_plan_critic: Dict[str, Any]
    b_ai_planning_review: Dict[str, Any]
    b_ai_repair_plan: Dict[str, Any]
    b_ai_trace: Dict[str, Any]
    b_replan_request: Dict[str, Any]
    b_replan_attempt: Dict[str, Any]
    b_replan_attempt_count: int
    b_repair_plan: Dict[str, Any]
    weather_context: Dict[str, Any]

    # C-stage: tool routing, mock execution, payment, and sharing
    action_sequence: List[Dict[str, Any]]
    raw_api_results: Dict[str, Any]
    execution_commit_result: Dict[str, Any]
    execution_status: str
    execution_failure_type: str
    execution_blocker: Dict[str, Any]
    tool_results: Dict[str, Any]
    payment_order: Dict[str, Any]
    payment_results: Dict[str, Any]
    payment_status: str
    retry_history: List[Dict[str, Any]]
    final_share_message: str

    # Control fields
    execution_log: List[str]
    retry_count: int
    need_confirm: bool
    payment_ui_mode: str
    payment_auto_confirm: bool
    payment_auto_pay: bool
    payment_method: str
