#!/usr/bin/env python3
"""
离线评估已完成的 Agent 轨迹，直接为结果文件补齐 LLM Judge 评分。

用法示例：
  python evaluate_existing_results_agent.py --input ./evaluation_results/data_cons_rag_results_bj_5_agent_results_20251211_031417.json
  python evaluate_existing_results_agent.py --input ./evaluation_results/xxx.json --output ./evaluation_results/xxx_judged.json --judge-model anthropic.claude-opus-4.1

依赖：
- 配置文件 `config/model_rpm.yaml`（用于读取 judge API base_url 与 api_keys；若缺失则回退默认值）
- 已安装的 `tqdm`
"""

import argparse
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from llm_judge import LLMJudge
from llm_utils import ConfigManager


logger = logging.getLogger(__name__)


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


def extract_rag_context(tool_calls: List[Dict[str, Any]]) -> str:
    """从工具调用记录中提取并拼接 RAG 上下文。"""
    contexts: List[str] = []
    for call in tool_calls or []:
        if call.get("tool_type") != "rag":
            continue
        context = call.get("result", {}).get("context")
        if context:
            contexts.append(context)
    return "\n\n".join(contexts)


def summarize_judge_scores(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对一组 Judge 结果求平均。"""
    if not scores:
        return {}

    summary = {
        "correctness": sum(s["correctness"]["score"] for s in scores) / len(scores),
        "completeness": sum(s["completeness"]["score"] for s in scores) / len(scores),
        "fluency": sum(s["fluency"]["score"] for s in scores) / len(scores),
        "safety": sum(s["safety"]["score"] for s in scores) / len(scores),
        "total_score": sum(s["total_score"] for s in scores) / len(scores),
        "max_score": sum(s["max_score"] for s in scores) / len(scores),
        "count": len(scores),
    }

    hallucination_scores = [
        s.get("hallucination", {}).get("score")
        for s in scores
        if isinstance(s.get("hallucination"), dict) and s["hallucination"].get("score") is not None
    ]
    if hallucination_scores:
        summary["hallucination"] = sum(hallucination_scores) / len(hallucination_scores)

    return summary


def calculate_judge_averages(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """计算整体与按难度拆分的 Judge 平均值。"""
    grouped: Dict[str, List[Dict[str, Any]]] = {"__overall__": []}
    for result in results:
        judge_scores = result.get("judge_scores")
        if not (judge_scores and isinstance(judge_scores, dict) and "error" not in judge_scores):
            continue
        grouped["__overall__"].append(judge_scores)
        difficulty = result.get("difficulty")
        if isinstance(difficulty, str) and difficulty.strip():
            grouped.setdefault(difficulty.strip(), []).append(judge_scores)

    if not grouped["__overall__"]:
        return None

    averages = summarize_judge_scores(grouped["__overall__"])
    difficulty_breakdown: Dict[str, Any] = {}
    for diff, scores in grouped.items():
        if diff == "__overall__" or not scores:
            continue
        difficulty_breakdown[diff] = summarize_judge_scores(scores)
    averages["difficulty_breakdown"] = difficulty_breakdown
    return averages


def calculate_tool_and_conversation_stats(results: List[Dict[str, Any]], sample_count: Optional[int]) -> Dict[str, Any]:
    """统计工具调用与对话轮数，平均值使用有效样本数。"""
    total_tool_calls = 0
    total_conversation_rounds = 0
    for result in results:
        total_tool_calls += len(result.get("tool_calls", []))
        total_conversation_rounds += len(result.get("conversation_history", []))

    num_samples = sample_count if sample_count is not None else len(results)
    return {
        "total_tool_calls": total_tool_calls,
        "avg_tool_calls": total_tool_calls / num_samples if num_samples else 0,
        "total_conversation_rounds": total_conversation_rounds,
        "avg_conversation_rounds": total_conversation_rounds / num_samples if num_samples else 0,
    }


def evaluate_results(
    input_file: str,
    output_file: str,
    judge_model: str,
    api_url: str,
    api_keys: List[str],
    resume: bool,
    worker_id: int = 0,
    worker_count: int = 1,
    parallel_workers: int = 1,
    max_tokens: int = 18000,
) -> None:
    """为指定结果文件补齐 Judge 评分并保存。"""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    logger.info("📂 读取结果文件: %s", input_file)
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    metadata = data.get("metadata", {})
    logger.info("📊 样本数: %d", len(results))
    logger.info("🧩 worker_id/worker_count: %d/%d", worker_id, worker_count)
    logger.info("⚡ parallel_workers: %d", parallel_workers)

    # 对多进程并发做 key 子集切分，尽量做到"每个并发槽位固定一组 key"
    subset_keys = _subset_api_keys(api_keys, worker_id=worker_id, worker_count=worker_count)
    logger.info("🔑 使用 %d/%d 个 API Keys (worker_id=%d)", len(subset_keys), len(api_keys), worker_id)

    # 在并发模式下，使用线程本地存储为每个线程创建独立的 judge 实例（线程安全）
    # 在顺序模式下，共享一个 judge 实例
    if parallel_workers > 1:
        thread_local = threading.local()
        thread_counter = [0]  # 使用列表以便在闭包中修改
        lock = threading.Lock()
        
        def _get_thread_judge() -> LLMJudge:
            """为线程创建独立的 judge 实例，使用分配给该线程的 key 子集"""
            if not hasattr(thread_local, 'judge'):
                with lock:
                    thread_id = thread_counter[0]
                    thread_counter[0] += 1
                thread_keys = _subset_api_keys(subset_keys, worker_id=thread_id, worker_count=parallel_workers)
                thread_local.judge = LLMJudge(api_url=api_url, api_keys=thread_keys, model_name=judge_model, max_tokens=max_tokens)
            return thread_local.judge
        
        judge = None  # 顺序模式下不使用
    else:
        judge = LLMJudge(api_url=api_url, api_keys=subset_keys, model_name=judge_model, max_tokens=max_tokens)
        _get_thread_judge = None  # 顺序模式下不使用

    def _evaluate_one(item: Dict[str, Any], idx: int) -> tuple[int, Dict[str, Any], bool]:
        """评估单个样本，返回 (index, item, updated)"""
        if resume and item.get("judge_scores") and "error" not in item["judge_scores"]:
            return idx, item, False

        ground_truth = item.get("ground_truth")
        
        # 在并发模式下，为每个线程创建独立的 judge 实例
        if parallel_workers > 1:
            thread_judge = _get_thread_judge()
        else:
            thread_judge = judge

        rag_context = extract_rag_context(item.get("tool_calls", []))
        
        # 如果没有 ground_truth，优先尝试一次性调用 judge 的 evaluate_all 获取所有维度（同一次 API 调用返回）
        if not ground_truth:
            logger.warning(f"样本 {idx} 缺少 ground_truth，将优先一次性调用 judge.evaluate_all 获取其他维度")
            try:
                # 尝试一次性评估所有维度（如果 judge 支持在 ground_truth 为 None 的情况下返回除 correctness 外的指标）
                score = thread_judge.evaluate_all(
                    query=item.get("question", ""),
                    model_output=item.get("final_response", ""),
                    ground_truth=ground_truth,
                    conversation_history=item.get("conversation_history", []),
                    tool_calls=item.get("tool_calls", []),
                    rag_context=rag_context,
                    enable_hallucination=bool(rag_context),
                )
            except Exception:
                # 回退到逐项评估以保证兼容性（旧逻辑）
                try:
                    completeness = thread_judge.evaluate_completeness(
                        query=item.get("question", ""),
                        model_output=item.get("final_response", ""),
                        conversation_history=item.get("conversation_history", []),
                        tool_calls=item.get("tool_calls", []),
                    )
                    fluency = thread_judge.evaluate_fluency(
                        query=item.get("question", ""),
                        model_output=item.get("final_response", ""),
                    )
                    safety = thread_judge.evaluate_safety(
                        query=item.get("question", ""),
                        model_output=item.get("final_response", ""),
                    )

                    # 汇总结果（正确性标记为错误）
                    total_score = (
                        completeness["score"] / 10 +
                        fluency["score"] / 10 +
                        safety["score"] / 10
                    )

                    score = {
                        "correctness": {"score": 0, "reason": "缺少 ground_truth，无法评估正确性"},
                        "completeness": completeness,
                        "fluency": fluency,
                        "safety": safety,
                        "total_score": total_score,
                        "max_score": 4
                    }

                    # 不再单独发起幻觉评估：若未包含幻觉结果则标记为未返回
                    if "hallucination" not in score:
                        score["hallucination"] = {"score": None, "reason": "未返回幻觉评估"}

                except Exception as exc:  # noqa: BLE001
                    score = {"error": f"评估失败（缺少ground_truth）: {str(exc)}"}
        else:
            # 有 ground_truth，进行完整评估
            try:
                score = thread_judge.evaluate_all(
                    query=item.get("question", ""),
                    model_output=item.get("final_response", ""),
                    ground_truth=ground_truth,
                    conversation_history=item.get("conversation_history", []),
                    tool_calls=item.get("tool_calls", []),
                    rag_context=rag_context,
                    enable_hallucination=bool(rag_context),
                )
            except Exception as exc:  # noqa: BLE001
                score = {"error": str(exc)}

        # 确保返回结果中始终包含幻觉检测字段（五个指标）
        try:
            if isinstance(score, dict) and "hallucination" not in score:
                if rag_context:
                    try:
                        hallucination = thread_judge.evaluate_hallucination(
                            query=item.get("question", ""),
                            model_output=item.get("final_response", ""),
                            rag_context=rag_context,
                        )
                        score["hallucination"] = hallucination
                    except Exception:
                        score["hallucination"] = {"score": None, "reason": "幻觉评估失败"}
                else:
                    score["hallucination"] = {"score": None, "reason": "缺少 rag_context，无法评估幻觉"}
        except Exception:
            # 即使幻觉评估模块出现不可预期错误，也不要阻塞主流程
            pass

        item["judge_scores"] = score
        return idx, item, True

    updated = 0
    skipped = 0

    # 并发评测：同一个模型内对样本并发调用
    if parallel_workers and parallel_workers > 1:
        logger.info("🚀 开始并行评测 (工作线程数: %d)", parallel_workers)
        items_to_eval = [(item, idx) for idx, item in enumerate(results)]
        
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(_evaluate_one, item, idx): idx
                for idx, item in enumerate(results)
            }
            
            if tqdm is not None:
                with tqdm(total=len(results), desc="Running LLM Judge") as pbar:
                    for future in as_completed(futures):
                        idx, item, was_updated = future.result()
                        results[idx] = item
                        if was_updated:
                            updated += 1
                        else:
                            skipped += 1
                        pbar.update(1)
            else:
                done = 0
                for future in as_completed(futures):
                    idx, item, was_updated = future.result()
                    results[idx] = item
                    if was_updated:
                        updated += 1
                    else:
                        skipped += 1
                    done += 1
                    if done % 5 == 0 or done == len(results):
                        logger.info("评测进度: %d/%d", done, len(results))
    else:
        logger.info("🚀 开始顺序评测")
        for idx, item in enumerate(tqdm(results, desc="Running LLM Judge")):
            if resume and item.get("judge_scores") and "error" not in item["judge_scores"]:
                skipped += 1
                continue

            ground_truth = item.get("ground_truth")
            rag_context = extract_rag_context(item.get("tool_calls", []))
            
            # 如果没有 ground_truth，优先尝试一次性调用 judge.evaluate_all 获取所有维度（同一次 API 调用返回）
            if not ground_truth:
                logger.warning(f"样本 {idx + 1} 缺少 ground_truth，将优先一次性调用 judge.evaluate_all 获取其他维度")
                try:
                    score = judge.evaluate_all(
                        query=item.get("question", ""),
                        model_output=item.get("final_response", ""),
                        ground_truth=ground_truth,
                        conversation_history=item.get("conversation_history", []),
                        tool_calls=item.get("tool_calls", []),
                        rag_context=rag_context,
                        enable_hallucination=bool(rag_context),
                    )
                except Exception:
                    # 回退到逐项评估以保证兼容性（旧逻辑）
                    try:
                        completeness = judge.evaluate_completeness(
                            query=item.get("question", ""),
                            model_output=item.get("final_response", ""),
                            conversation_history=item.get("conversation_history", []),
                            tool_calls=item.get("tool_calls", []),
                        )
                        fluency = judge.evaluate_fluency(
                            query=item.get("question", ""),
                            model_output=item.get("final_response", ""),
                        )
                        safety = judge.evaluate_safety(
                            query=item.get("question", ""),
                            model_output=item.get("final_response", ""),
                        )

                        # 汇总结果（正确性标记为错误）
                        total_score = (
                            completeness["score"] / 10 +
                            fluency["score"] / 10 +
                            safety["score"] / 10
                        )

                        score = {
                            "correctness": {"score": 0, "reason": "缺少 ground_truth，无法评估正确性"},
                            "completeness": completeness,
                            "fluency": fluency,
                            "safety": safety,
                            "total_score": total_score,
                            "max_score": 4
                        }

                        # 如果启用幻觉检测且提供了RAG上下文
                        if rag_context:
                            try:
                                hallucination = judge.evaluate_hallucination(
                                    query=item.get("question", ""),
                                    model_output=item.get("final_response", ""),
                                    rag_context=rag_context,
                                )
                                score["hallucination"] = hallucination
                            except Exception:
                                pass  # 幻觉检测失败不影响其他维度

                    except Exception as exc:  # noqa: BLE001
                        score = {"error": f"评估失败（缺少ground_truth）: {str(exc)}"}
            else:
                # 有 ground_truth，进行完整评估
                try:
                    score = judge.evaluate_all(
                        query=item.get("question", ""),
                        model_output=item.get("final_response", ""),
                        ground_truth=ground_truth,
                        conversation_history=item.get("conversation_history", []),
                        tool_calls=item.get("tool_calls", []),
                        rag_context=rag_context,
                        enable_hallucination=bool(rag_context),
                    )
                except Exception as exc:  # noqa: BLE001
                    score = {"error": str(exc)}

            # 确保返回结果中始终包含幻觉检测字段（五个指标），但不再单独调用 API：若未返回则标记为未返回
            try:
                if isinstance(score, dict) and "hallucination" not in score:
                    score["hallucination"] = {"score": None, "reason": "未返回幻觉评估"}
            except Exception:
                pass

            item["judge_scores"] = score
            updated += 1

    logger.info("✅ 评分完成，新增评分: %d, 跳过: %d", updated, skipped)

    judge_averages = calculate_judge_averages(results)
    sample_count = judge_averages.get("count") if judge_averages else len(results)
    metadata.update(calculate_tool_and_conversation_stats(results, sample_count))
    if judge_averages:
        metadata["judge_averages"] = judge_averages
    data["metadata"] = metadata

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("💾 已保存到: %s", output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为已完成的 Agent 结果文件补齐 LLM Judge 评分")
    parser.add_argument("--input", required=True, help="Agent 结果 JSON 文件路径")
    parser.add_argument("--output", help="输出文件路径，默认在文件名后追加 _judged")
    parser.add_argument("--judge-model", default="anthropic.claude-opus-4.1", help="Judge 使用的模型名称")
    parser.add_argument(
        "--api-key",
        action="append",
        dest="api_keys",
        help="Judge API Key，可多次提供；缺省则读取 config/model_rpm.yaml",
    )
    parser.add_argument("--api-url", help="Judge API base URL，缺省读取 config/model_rpm.yaml")
    parser.add_argument("--resume", action="store_true", help="跳过已有 judge_scores 的样本")
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
        "--parallel",
        type=int,
        default=1,
        help="同一个模型内并发评测的线程数（默认1；如需70并行设置为70）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=18000,
        help="评测 max_tokens（默认18000，与轨迹评估一致）",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    cfg = ConfigManager()
    api_url = args.api_url or cfg.get_base_url()
    api_keys = args.api_keys or cfg.get_api_keys()
    if not api_keys:
        raise ValueError("未找到可用的 Judge API Key，请通过 --api-key 指定或在 config/model_rpm.yaml 配置 api_keys")

    input_path = os.path.abspath(args.input)
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        root, ext = os.path.splitext(input_path)
        output_path = f"{root}_judged{ext}"

    evaluate_results(
        input_file=input_path,
        output_file=output_path,
        judge_model=args.judge_model,
        api_url=api_url,
        api_keys=api_keys,
        resume=args.resume,
        worker_id=args.worker_id,
        worker_count=args.worker_count,
        parallel_workers=args.parallel,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()

