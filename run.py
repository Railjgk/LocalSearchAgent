"""
运行入口 - 测试完整流程
"""

import sys
sys.path.append(".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.state import PlanState
from src.graph import build_graph


def main():
    # 构造初始状态
    initial_state: PlanState = {
        # ========== 输入 ==========
        "user_input": "今天下午和老婆孩子出去玩，孩子5岁，老婆最近在减肥",
        "scene_type": "family",  # "family" 或 "friends"

        # ========== A产出（用户理解与场景建模）==========
        "constraints": {
            "people_count": 3,
            "child_age": 5,
            "mom_diet": "low_calorie",
            "duration": 4,
            "start_time": "14:00",
            "location": "杨浦区",
            "max_distance_km": 20,
            "max_queue_time": 30,
            "budget": 500
        },
        "user_profile": {},
        "short_term_memory": [],
        "scenario_activities": ["亲子乐园", "轻食餐厅", "儿童剧场"],

        # ========== B产出（候选生成与方案决策）==========
        "candidates": [],
        "filtered_candidates": [],
        "filter_reasons": {},

        # B节点会根据 candidates -> filtered_candidates -> selected_plan 自动生成方案
        "selected_plan": {},
        "optimization_score": 0.0,
        "alternative_plans": [],
        "explanation_text": "",

        # ========== C产出（工具调用与执行闭环）- 初始空值 ==========
        "action_sequence": [],
        "raw_api_results": {},
        "execution_status": "pending",
        "tool_results": {},
        "retry_history": [],
        "final_share_message": "",

        # ========== 控制字段 ==========
        "execution_log": [],
        "retry_count": 0,
        "need_confirm": False
    }

    # 构建并运行图
    print("\n" + "="*60)
    print("🚀 美团AI黑客松 - 本地生活Agent启动")
    print("="*60 + "\n")

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    # ========== 输出结果 ==========
    print("\n" + "="*60)
    print("📊 执行结果")
    print("="*60)

    print("\n📝 执行日志:")
    for log in final_state.get("execution_log", []):
        print(f"   {log}")

    print(f"\n📅 最终方案:")
    timeline = final_state.get("selected_plan", {}).get("timeline", [])
    for item in timeline:
        print(f"   {item.get('time')} - {item.get('activity')}")

    print(f"\n📊 执行状态: {final_state.get('execution_status')}")

    print(f"\n💬 分享消息:")
    print(f"   {final_state.get('final_share_message')}")

    # 打印工具执行结果详情（调试用）
    tool_results = final_state.get("tool_results", {})
    if tool_results:
        print(f"\n🔧 工具执行详情:")
        for key, value in tool_results.items():
            status = "✅" if value.get("success") else "❌"
            print(f"   {status} {key}: {value.get('name', '')} - {value.get('data', {}).get('message', '')}")

    print("\n" + "="*60)
    print("✅ 执行完成")
    print("="*60)


def test_friends_scene():
    """测试朋友场景"""
    print("\n" + "=" * 60)
    print("👥 测试朋友场景")
    print("=" * 60)

    initial_state: PlanState = {
        "user_input": "下午和朋友出去玩，4个人",
        "scene_type": "friends",
        "constraints": {
            "people_count": 4,
            "duration": 4,
            "start_time": "14:00",
            "location": "黄浦区",
            "max_distance_km": 20,
            "max_queue_time": 30,
            "budget": 600
        },
        "user_profile": {},
        "short_term_memory": [],
        "scenario_activities": ["艺术展览", "网红餐厅", "桌游吧"],
        "candidates": [],
        "filtered_candidates": [],
        "filter_reasons": {},
        "selected_plan": {},
        "optimization_score": 0.0,
        "alternative_plans": [],
        "explanation_text": "",
        "action_sequence": [],
        "raw_api_results": {},
        "execution_status": "pending",
        "tool_results": {},
        "retry_history": [],
        "final_share_message": "",
        "execution_log": [],
        "retry_count": 0,
        "need_confirm": False
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    print("\n📝 执行日志:")
    for log in final_state.get("execution_log", []):
        print(f"   {log}")

    print(f"\n💬 分享消息: {final_state.get('final_share_message')}")
    print(f"📊 执行状态: {final_state.get('execution_status')}")

if __name__ == "__main__":
    # 运行家庭场景
    main()

    # 可选：运行朋友场景测试
    # test_friends_scene()
