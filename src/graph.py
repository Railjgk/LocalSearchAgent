"""
LangGraph 工作流编排
A-stage + B-stage + C-stage 全链路
"""

from src.state import PlanState

try:
    from langgraph.graph import StateGraph, END
except ModuleNotFoundError:  # pragma: no cover - used in lightweight demo envs
    StateGraph = None
    END = "__end__"

# ========== A的节点（已接入真实实现）==========
from src.nodes.intent_parser import intent_parser_node
from src.nodes.memory_manager import memory_manager_node
from src.nodes.scenario_planner import scenario_planner_node

# ========== B的节点（真实实现）==========
from src.nodes.b_poi_rag import b_poi_rag_node
from src.nodes.candidate_generator import candidate_generator_node
from src.nodes.constraint_filter import constraint_filter_node
from src.nodes.plan_optimizer import plan_optimizer_node
from src.nodes.b_replan_loop import b_replan_loop_node
from src.nodes.explainability import explainability_node
from src.nodes.b_repair_planner import repair_planner_node
from src.nodes.b_ai_trace import b_ai_trace_node

# ========== C的节点（真实实现）==========
from src.nodes.tool_router import tool_router_node
from src.nodes.mock_api_layer import mock_api_layer_node
from src.nodes.execution_manager import execution_manager_node
from src.nodes.payment_layer import payment_layer_node
from src.nodes.share_generator import share_generator_node


WORKFLOW_STEPS = [
    ("intent_parser", intent_parser_node),
    ("memory_manager", memory_manager_node),
    ("scenario_planner", scenario_planner_node),
    ("b_poi_rag", b_poi_rag_node),
    ("candidate_generator", candidate_generator_node),
    ("constraint_filter", constraint_filter_node),
    ("plan_optimizer", plan_optimizer_node),
    ("b_replan_loop", b_replan_loop_node),
    ("explainability", explainability_node),
    ("tool_router", tool_router_node),
    ("mock_api_layer", mock_api_layer_node),
    ("execution_manager", execution_manager_node),
    ("repair_planner", repair_planner_node),
    ("b_ai_trace", b_ai_trace_node),
    ("payment_layer", payment_layer_node),
    ("share_generator", share_generator_node),
]

WORKFLOW_NODES = [node for _, node in WORKFLOW_STEPS]


class SequentialGraph:
    """Small `invoke` compatible fallback when LangGraph is not installed."""

    def __init__(self, nodes):
        self.nodes = nodes

    def invoke(self, state: PlanState) -> PlanState:
        current_state = dict(state)
        for node in self.nodes:
            updates = node(current_state)
            if updates:
                current_state.update(updates)
        return current_state


def build_graph(store=None):
    """构建LangGraph工作流"""

    if StateGraph is None:
        return SequentialGraph(WORKFLOW_NODES)

    workflow = StateGraph(PlanState)

    for name, node in WORKFLOW_STEPS:
        workflow.add_node(name, node)

    workflow.set_entry_point(WORKFLOW_STEPS[0][0])
    for (from_name, _), (to_name, _) in zip(WORKFLOW_STEPS, WORKFLOW_STEPS[1:]):
        workflow.add_edge(from_name, to_name)
    workflow.add_edge(WORKFLOW_STEPS[-1][0], END)

    # 编译
    app = workflow.compile(store=store)
    return app


# 便捷函数
def get_graph(store=None):
    return build_graph(store=store)
