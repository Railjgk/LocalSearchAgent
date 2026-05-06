"""
State定义 - 所有12个模块共享的数据结构
C负责维护，A和B基于此开发
"""

from typing import TypedDict, List, Dict, Optional, Any


class PlanState(TypedDict):
    """Agent全局状态"""

    # ========== 输入字段 ==========
    user_input: str  # 用户原始输入
    scene_type: str  # "family" 或 "friends"

    # ========== A产出（用户理解与场景建模）==========
    constraints: Dict[str, Any]  # 提取的约束 {"child_age":5, "mom_diet":"low_calorie"}
    user_profile: Dict[str, Any]  # 用户画像 {"avoid":["spicy"]}
    short_term_memory: List[str]  # 本轮对话历史
    scenario_activities: List[str]  # 候选活动类型 ["亲子乐园","轻食餐厅"]

    # ========== B产出（候选生成与方案决策）==========
    candidates: List[Dict]  # POI候选列表
    filtered_candidates: List[Dict]  # 过滤后的POI
    filter_reasons: Dict[str, str]  # 过滤原因 {"poi_id": "评分过低"}
    selected_plan: Dict[str, Any]  # 最终方案（含timeline）
    optimization_score: float  # 方案满意度评分
    alternative_plans: List[Dict]  # 备选方案列表
    explanation_text: str  # 推荐理由文案

    # ========== C产出（工具调用与执行闭环）==========
    action_sequence: List[Dict]  # 待执行动作列表
    raw_api_results: Dict[str, Any]  # API原始返回
    execution_status: str  # "pending" / "running" / "success" / "partial" / "failed"
    tool_results: Dict[str, Any]  # 工具执行结果汇总
    retry_history: List[Dict]  # 重试记录
    final_share_message: str  # 最终分享文案

    # ========== 控制字段 ==========
    execution_log: List[str]  # 调试日志（所有节点追加）
    retry_count: int  # 当前重试次数
    need_confirm: bool  # 是否需要用户确认