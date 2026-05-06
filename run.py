"""
运行入口 - 测试完整流程
"""

import sys
import json
sys.path.append(".")

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
            "location": "杨浦区"
        },
        "user_profile": {},
        "short_term_memory": [],
        "scenario_activities": ["亲子乐园", "轻食餐厅", "儿童剧场"],

        # ========== B产出（候选生成与方案决策）==========
        "candidates": [],
        "filtered_candidates": [],
        "filter_reasons": {},

        # 关键：selected_plan 必须包含 timeline 字段
        # timeline 中每个元素需要包含 type, poi_id, activity, time
        "selected_plan": {
            "timeline": [
                {
                    "type": "play",
                    "poi_id": "act_001",
                    "activity": "亲子陶艺体验馆",
                    "time": "14:00"
                },
                {
                    "type": "eat",
                    "poi_id": "res_001",
                    "activity": "轻食日料餐厅",
                    "time": "17:30"
                }
            ],
            "total_duration": 4,
            "score": 0.85
        },
        "optimization_score": 0.85,
        "alternative_plans": [],
        "explanation_text": "推荐去亲子陶艺体验馆，因为孩子5岁适合低强度活动；晚餐选择轻食日料餐厅，满足低卡需求。",

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
            "location": "黄浦区"
        },
        "user_profile": {},
        "short_term_memory": [],
        "scenario_activities": ["艺术展览", "网红餐厅", "桌游吧"],
        "candidates": [],
        "filtered_candidates": [],
        "filter_reasons": {},
        "selected_plan": {
            "timeline": [
                {
                    "type": "play",
                    "poi_id": "act_003",
                    "activity": "上海自然博物馆",
                    "time": "13:00"  # ← 修正：act_003 支持 13:00
                },
                {
                    "type": "eat",
                    "poi_id": "res_003",
                    "activity": "海底捞火锅",
                    "time": "18:00"  # ← 修正：res_003 支持 18:00
                }
            ],
            "total_duration": 4,
            "score": 0.90
        },
        "optimization_score": 0.90,
        "alternative_plans": [],
        "explanation_text": "推荐自然博物馆和海底捞，适合朋友聚会。",
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