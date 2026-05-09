"""
LangGraph 工作流编排
C负责定义执行顺序
"""

from langgraph.graph import StateGraph, END
from src.state import PlanState

# ========== 导入A的节点（用户理解与场景建模）==========
# TODO: A完成后取消注释
# from src.nodes.intent_parser import intent_parser_node
# from src.nodes.memory_manager import memory_manager_node
# from src.nodes.scenario_planner import scenario_planner_node

# ========== 导入B的节点（候选生成与方案决策）==========
from src.nodes.candidate_generator import candidate_generator_node
from src.nodes.constraint_filter import constraint_filter_node
from src.nodes.plan_optimizer import plan_optimizer_node
from src.nodes.explainability import explainability_node

# ========== 导入C的节点（工具调用与执行闭环）==========
from src.nodes.tool_router import tool_router_node
from src.nodes.mock_api_layer import mock_api_layer_node
from src.nodes.execution_manager import execution_manager_node
from src.nodes.share_generator import share_generator_node


# ========== 临时假节点（A完成前使用）==========
def make_dummy_node(node_name: str):
    """生成临时节点，用于在A未完成时跑通流程。"""

    def dummy_node(state: PlanState) -> dict:
        execution_log = state.get("execution_log", [])
        execution_log.append(f"[临时] {node_name} 执行完成")
        return {"execution_log": execution_log}

    return dummy_node


def build_graph():
    """构建LangGraph工作流"""

    workflow = StateGraph(PlanState)

    # ========== 添加工作流节点 ==========
    # A的节点（暂时用假节点，后续替换）
    workflow.add_node("intent_parser", make_dummy_node("intent_parser"))  # TODO: 替换为 intent_parser_node
    workflow.add_node("memory_manager", make_dummy_node("memory_manager"))  # TODO: 替换为 memory_manager_node
    workflow.add_node("scenario_planner", make_dummy_node("scenario_planner"))  # TODO: 替换为 scenario_planner_node

    # B的节点（真实实现）
    workflow.add_node("candidate_generator", candidate_generator_node)
    workflow.add_node("constraint_filter", constraint_filter_node)
    workflow.add_node("plan_optimizer", plan_optimizer_node)
    workflow.add_node("explainability", explainability_node)

    # C的节点（真实实现）
    workflow.add_node("tool_router", tool_router_node)
    workflow.add_node("mock_api_layer", mock_api_layer_node)
    workflow.add_node("execution_manager", execution_manager_node)
    workflow.add_node("share_generator", share_generator_node)

    # ========== 定义边（线性执行顺序）==========
    workflow.set_entry_point("intent_parser")

    # A的链路
    workflow.add_edge("intent_parser", "memory_manager")
    workflow.add_edge("memory_manager", "scenario_planner")
    workflow.add_edge("scenario_planner", "candidate_generator")

    # B的链路
    workflow.add_edge("candidate_generator", "constraint_filter")
    workflow.add_edge("constraint_filter", "plan_optimizer")
    workflow.add_edge("plan_optimizer", "explainability")
    workflow.add_edge("explainability", "tool_router")

    # C的链路
    workflow.add_edge("tool_router", "mock_api_layer")
    workflow.add_edge("mock_api_layer", "execution_manager")
    workflow.add_edge("execution_manager", "share_generator")
    workflow.add_edge("share_generator", END)

    # 编译
    app = workflow.compile()
    return app


# 便捷函数
def get_graph():
    return build_graph()
