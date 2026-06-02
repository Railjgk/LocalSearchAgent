#!/usr/bin/env python3
"""
轨迹质量评估脚本

从 *_search_details.json 中提取每个问题的完整搜索轨迹，
结合对应的 *_agent_results_*.json 中的问题和标准答案，
使用 LLM 评估轨迹的四个维度质量。

用法示例:
  python evaluate_trajectories.py \
    --search-details ./evaluation_results/data_cons_rag_results_bj_5_agent_results_20251211_031417_search_details.json \
    --agent-results ./evaluation_results/data_cons_rag_results_bj_5_agent_results_20251211_031417.json \
    --output ./evaluation_results/trajectory_evaluation_20251211_031417.json \
    --model gpt-4o
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional
from collections import defaultdict
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from trajectory_prompt import (
    TRAJECTORY_EVAL_SYSTEM_PROMPT,
    TRAJECTORY_EVAL_USER_PROMPT_TEMPLATE,
)

# 导入配置管理器
try:
    from llm_utils import ConfigManager
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    print("警告: 未找到 llm_utils，将使用默认配置")

# 生成阶段同款客户端（带限流、切 key、重试）
try:
    from llm_utils import FridayClient
    HAS_FRIDAY = True
except ImportError:
    HAS_FRIDAY = False
    print("警告: 未找到 FridayClient (llm_utils.py)，无法使用生成阶段一致的限流/重试逻辑")

# tqdm（进度条，风格与生成阶段一致）
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def load_search_details(path: str) -> List[Dict[str, Any]]:
    """加载搜索详情文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_agent_results(path: str) -> Dict[str, Dict[str, Any]]:
    """加载 agent 结果文件，返回 question_id -> question_info 的映射"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results_map = {}
    for item in data.get("results", []):
        qid = item.get("question_id")
        if qid is not None:
            results_map[qid] = {
                "question": item.get("question", ""),
                "ground_truth": item.get("ground_truth", ""),
                "reference_answer": item.get("reference_answer", ""),
                "search_path": item.get("search_path", ""),
                "difficulty": item.get("difficulty", ""),
            }
    return results_map


def build_search_details_from_agent_results(agent_results_path: str) -> List[Dict[str, Any]]:
    """
    当 *_search_details.json 不存在时，从 agent_results.json 中构造近似的 search_details 结构，
    以便后续按 question_id 分组并进行轨迹评估。生成的每个 step 包含至少以下字段：
    - question_id, timestamp, round, tool_type, query, success
    """
    if not os.path.isfile(agent_results_path):
        return []

    with open(agent_results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items: List[Dict[str, Any]] = []
    for entry in data.get("results", []):
        qid = entry.get("question_id")
        if qid is None:
            # try alternative keys
            qid = entry.get("question_idx") or entry.get("id")
        # prefer structured 'trajectory' if present (list), else try conversation_history
        traj = entry.get("trajectory")
        conv = entry.get("conversation_history", [])
        if isinstance(traj, list) and traj:
            for i, step in enumerate(traj):
                # step may be a dict or a string; normalize to a query string
                if isinstance(step, dict):
                    query = step.get("query") or step.get("prompt") or json.dumps(step, ensure_ascii=False)
                else:
                    query = str(step)
                items.append(
                    {
                        "question_id": qid,
                        "timestamp": f"{i:06d}",
                        "round": i + 1,
                        "tool_type": "llm",
                        "query": query,
                        "success": True,
                    }
                )
        elif isinstance(conv, list) and conv:
            # extract llm responses as steps
            for i, round_item in enumerate(conv):
                # round_item may contain llm_response or model_output
                llm_resp = round_item.get("llm_response") if isinstance(round_item, dict) else None
                if not llm_resp:
                    llm_resp = round_item.get("model_output") if isinstance(round_item, dict) else None
                query = llm_resp if llm_resp else (round_item if isinstance(round_item, str) else json.dumps(round_item, ensure_ascii=False))
                items.append(
                    {
                        "question_id": qid,
                        "timestamp": f"{i:06d}",
                        "round": i + 1,
                        "tool_type": "llm",
                        "query": query,
                        "success": True,
                    }
                )
        else:
            # fallback: create a single step containing final_response
            final_answer = entry.get("final_response") or entry.get("model_output") or ""
            items.append(
                {
                    "question_id": qid,
                    "timestamp": "000000",
                    "round": 1,
                    "tool_type": "llm",
                    "query": final_answer if final_answer is not None else "",
                    "success": True,
                }
            )

    return items


def group_trajectory_by_question(search_details: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """按 question_id 对搜索详情进行分组"""
    grouped = defaultdict(list)
    for item in search_details:
        qid = item.get("question_id")
        if qid is not None:
            grouped[qid].append(item)
    
    # 按时间戳排序
    for qid in grouped:
        grouped[qid].sort(key=lambda x: x.get("timestamp", ""))
    
    return dict(grouped)


def format_trajectory(trajectory: List[Dict[str, Any]]) -> str:
    """格式化轨迹为可读文本"""
    lines = []
    for idx, step in enumerate(trajectory, 1):
        round_num = step.get("round", idx)
        tool_type = step.get("tool_type", "unknown")
        query = step.get("query", "")
        success = step.get("success", False)
        
        lines.append(f"### 步骤 {round_num} ({tool_type})")
        lines.append(f"**查询**: {query}")
        lines.append(f"**执行状态**: {'成功' if success else '失败'}")
        
        # 提取关键结果信息
        if tool_type == "rag" and "rag" in step:
            rag_data = step["rag"]
            total = rag_data.get("total_results", 0)
            lines.append(f"**检索结果数**: {total}")
            
            # 提取前3个商户名称作为摘要
            merchants = rag_data.get("merchants", [])[:3]
            if merchants:
                merchant_names = [m.get("name", "未知") for m in merchants]
                lines.append(f"**主要商户**: {', '.join(merchant_names)}")
        
        lines.append("")
    
    return "\n".join(lines)

def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出中尽可能提取 JSON 对象。"""
    if not text:
        return None
    s = text.strip()
    # 去除可能的 markdown fence
    if s.startswith("```json"):
        s = s[7:].strip()
    if s.startswith("```"):
        s = s[3:].strip()
    if s.endswith("```"):
        s = s[:-3].strip()

    # 1) 直接解析
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) 兜底：尝试提取最外层 {...}
    import re
    json_pattern = r"\{[\s\S]*\}"
    m = re.search(json_pattern, s)
    if not m:
        return None
    candidate = m.group(0).strip()
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None

def _subset_api_keys(api_keys: List[str], worker_id: int, worker_count: int) -> List[str]:
    """按 worker_id/worker_count 对 keys 做稳定切分，减少多进程抢同一 key。"""
    if not api_keys:
        return []
    if worker_count <= 1:
        return api_keys
    # 取 i % worker_count == worker_id 的子集
    subset = [k for i, k in enumerate(api_keys) if (i % worker_count) == worker_id]
    # 极端情况兜底：确保非空
    if not subset:
        subset = [api_keys[worker_id % len(api_keys)]]
    return subset


def call_llm_judge_friday(
    question: str,
    ground_truth: str,
    trajectory: str,
    model: str = "deepseek-v32-meituan",
    base_url: Optional[str] = None,
    config_path: str = "config/model_rpm.yaml",
    worker_id: int = 0,
    worker_count: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 18000,
) -> Dict[str, Any]:
    """调用 LLM 进行轨迹评估（生成阶段一致：FridayClient + 内置限流/切key/重试）"""
    if not HAS_FRIDAY:
        return {"error": "未找到 FridayClient，无法使用生成阶段一致的限流/重试逻辑", "reasoning": {}, "scores": {}}
    
    user_prompt = TRAJECTORY_EVAL_USER_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        trajectory=trajectory,
    )

    try:
        # FridayClient 会从 config_path 加载 keys/base_url/rpm，并包含 429 切key + 重试
        client = FridayClient(
            model_name=model,
            api_url=base_url or "https://api.example.com/v1/openai/native",
            api_token=None,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=300,
            max_retries=5,
            config_path=config_path,
            use_api_key_manager=True,
        )

        # 关键：对多进程并发做 key 子集切分，尽量做到“每个并发槽位固定一组 key”
        keys = _subset_api_keys(client.api_keys, worker_id=worker_id, worker_count=worker_count)
        client.api_keys = keys
        client.api_token = keys[0] if keys else client.api_token

        content, _, _ = client.single_request(
            messages=[
                {"role": "system", "content": TRAJECTORY_EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        parsed = _extract_json_object(content)
        if not parsed:
            return {"error": "JSON 解析失败", "raw_content": content, "reasoning": {}, "scores": {}}
        return parsed
    except Exception as e:
        return {"error": f"LLM 调用失败: {e}", "reasoning": {}, "scores": {}}


def evaluate_all_trajectories(
    search_details_path: str,
    agent_results_path: str,
    output_path: str,
    model: str = "deepseek-v32-meituan",
    base_url: Optional[str] = None,
    limit: Optional[int] = None,
    config_path: str = "config/model_rpm.yaml",
    worker_id: int = 0,
    worker_count: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 18000,
    parallel_workers: int = 1,
) -> None:
    """评估所有轨迹"""
    print(f"📂 加载搜索详情: {search_details_path}")
    # 尝试加载 search_details；若文件缺失或格式不符合预期，则从 agent_results 回退生成
    if not os.path.isfile(search_details_path):
        print(f"⚠️  未找到 search_details 文件: {search_details_path}，将从 agent_results 回退构造轨迹")
        search_details = build_search_details_from_agent_results(agent_results_path)
    else:
        search_details = load_search_details(search_details_path)
        # 如果内容看起来不是期望的 dict 列表（例如字符串列表），也回退
        if not isinstance(search_details, list) or (len(search_details) > 0 and not isinstance(search_details[0], dict)):
            print("⚠️  search_details 格式不符合预期，将从 agent_results 回退构造轨迹")
            search_details = build_search_details_from_agent_results(agent_results_path)
    
    print(f"📂 加载 Agent 结果: {agent_results_path}")
    agent_results = load_agent_results(agent_results_path)
    
    print(f"🔄 按问题分组轨迹...")
    trajectories = group_trajectory_by_question(search_details)
    
    print(f"📊 共找到 {len(trajectories)} 条轨迹")
    print(f"🌐 API Base URL: {base_url}")
    print(f"🧩 worker_id/worker_count: {worker_id}/{worker_count}")
    print(f"🧠 temperature/max_tokens: {temperature}/{max_tokens}")
    print(f"⚡ parallel_workers: {parallel_workers}")
    
    evaluations: List[Dict[str, Any]] = []
    question_ids = sorted(trajectories.keys())
    
    if limit:
        question_ids = question_ids[:limit]
        print(f"⚠️  仅评估前 {limit} 条轨迹")

    # 只评测能在 agent_results 里找到的 qid
    eval_qids = [qid for qid in question_ids if qid in agent_results]
    missing_qids = [qid for qid in question_ids if qid not in agent_results]
    if missing_qids:
        print(f"⚠️  {len(missing_qids)} 条轨迹缺少 agent_results 对应项，将跳过")

    def _eval_one(qid: int) -> Dict[str, Any]:
        question_info = agent_results[qid]
        question = question_info["question"]
        ground_truth = question_info["ground_truth"]
        trajectory_steps = trajectories.get(qid, [])
        trajectory_text = format_trajectory(trajectory_steps)
        evaluation = call_llm_judge_friday(
            question=question,
            ground_truth=ground_truth,
            trajectory=trajectory_text,
            model=model,
            base_url=base_url,
            config_path=config_path,
            worker_id=worker_id,
            worker_count=worker_count,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "question_id": qid,
            "question": question,
            "ground_truth": ground_truth,
            "difficulty": question_info.get("difficulty", ""),
            "trajectory_steps": len(trajectory_steps),
            "evaluation": evaluation,
        }

    def _format_done_line(done: int, total: int, result: Dict[str, Any]) -> str:
        qid = result.get("question_id", "NA")
        ev = result.get("evaluation", {}) or {}
        if "error" in ev:
            return f"[{done}/{total}] Q{qid} ❌ {ev.get('error')}"
        scores = ev.get("scores", {}) or {}
        ar = scores.get("action_relevance", 0)
        es = scores.get("evidence_sufficiency", 0)
        cc = scores.get("causal_coherence", 0)
        se = scores.get("search_efficiency", 0)
        overall = (ar + es + cc + se) / 4 if all(isinstance(x, (int, float)) for x in [ar, es, cc, se]) else 0
        return f"[{done}/{total}] Q{qid} ✅ AR={ar} ES={es} CC={cc} SE={se} Overall={overall:.2f}"

    # 并发评测：同一个模型内对 question 并发调用（70并行）
    if parallel_workers and parallel_workers > 1:
        print(f"🚀 开始并行评测 (工作线程数: {parallel_workers})")
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {executor.submit(_eval_one, qid): qid for qid in eval_qids}
            if tqdm is not None:
                with tqdm(total=len(eval_qids), desc="评测进度") as pbar:
                    done = 0
                    for fut in as_completed(futures):
                        qid = futures[fut]
                        try:
                            res = fut.result()
                        except Exception as e:
                            res = {"question_id": qid, "evaluation": {"error": f"评测异常: {e}"}}
                        evaluations.append(res)
                        done += 1
                        # 每条轨迹评测完成都输出一行（不打乱进度条）
                        tqdm.write(_format_done_line(done, len(eval_qids), res))
                        pbar.update(1)
            else:
                done = 0
                for fut in as_completed(futures):
                    qid = futures[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = {"question_id": qid, "evaluation": {"error": f"评测异常: {e}"}}
                    evaluations.append(res)
                    done += 1
                    print(_format_done_line(done, len(eval_qids), res))
                    if done % 5 == 0 or done == len(eval_qids):
                        print(f"评测进度: {done}/{len(eval_qids)}")
    else:
        print("🚀 开始顺序评测")
        iterable = tqdm(eval_qids, desc="评测进度") if tqdm is not None else eval_qids
        done = 0
        total = len(eval_qids)
        for qid in iterable:
            try:
                res = _eval_one(qid)
            except Exception as e:
                res = {"question_id": qid, "evaluation": {"error": f"评测异常: {e}"}}
            evaluations.append(res)
            done += 1
            # 每条轨迹评测完成都输出一行（顺序模式直接 print 即可）
            if tqdm is not None and hasattr(iterable, "write"):
                iterable.write(_format_done_line(done, total, res))
            else:
                print(_format_done_line(done, total, res))
    
    # 保存结果
    output_data = {
        "metadata": {
            "search_details_file": os.path.basename(search_details_path),
            "agent_results_file": os.path.basename(agent_results_path),
            "evaluation_model": model,
            "total_evaluated": len(evaluations),
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "parallel_workers": parallel_workers,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        "evaluations": evaluations,
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ 评估完成！结果已保存至: {output_path}")
    print(f"📊 共评估 {len(evaluations)} 条轨迹")
    
    # 统计成功率
    success_count = sum(1 for e in evaluations if "error" not in e["evaluation"])
    print(f"✅ 成功: {success_count}/{len(evaluations)} ({success_count/len(evaluations)*100:.1f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估搜索轨迹质量")
    parser.add_argument(
        "--search-details",
        required=True,
        help="搜索详情 JSON 文件路径 (*_search_details.json)",
    )
    parser.add_argument(
        "--agent-results",
        required=True,
        help="Agent 结果 JSON 文件路径 (*_agent_results_*.json)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出评估结果的 JSON 文件路径",
    )
    parser.add_argument(
        "--model",
        default="deepseek-v32-meituan",
        help="用于评估的 LLM 模型名称 (默认: deepseek-v32-meituan)",
    )
    parser.add_argument(
        "--base-url",
        help="API Base URL (如果不提供，则从配置文件加载)",
    )
    parser.add_argument(
        "--config",
        default="config/model_rpm.yaml",
        help="配置文件路径 (默认: config/model_rpm.yaml)",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="并发槽位ID（用于稳定分配API key子集，默认0）",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help="并发槽位总数（用于稳定分配API key子集，默认1）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="评测温度参数（默认0.0）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=18000,
        help="评测 max_tokens（默认18000）",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="同一个模型内并发评测的线程数（默认1；如需70并行设置为70）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="仅评估前 N 条轨迹 (用于测试)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # 从配置文件或参数中加载 API 配置
    base_url = args.base_url
    
    # base_url 默认从配置文件取（与生成阶段一致）
    if HAS_CONFIG and not base_url:
        try:
            config_manager = ConfigManager(config_path=args.config)
            base_url = config_manager.get_base_url()
        except Exception:
            base_url = base_url
    
    evaluate_all_trajectories(
        search_details_path=args.search_details,
        agent_results_path=args.agent_results,
        output_path=args.output,
        model=args.model,
        base_url=base_url,
        limit=args.limit,
        config_path=args.config,
        worker_id=args.worker_id,
        worker_count=args.worker_count,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        parallel_workers=args.parallel,
    )


if __name__ == "__main__":
    main()

