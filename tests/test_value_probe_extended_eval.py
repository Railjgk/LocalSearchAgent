from __future__ import annotations

from experiments.value_probe.evaluate_extended import (
    control_and_selectivity_metrics,
    length_bias_metrics,
    paired_consistency_metrics,
    sentence_relation_metrics,
    template_split_metrics,
)


def _score_record(
    example_id: str,
    target_value: str,
    relation: str,
    positive_score: float,
    negative_score: float,
) -> dict[str, object]:
    positive_scores = {
        "家庭照护": 0.1,
        "健康克制": 0.1,
        "省心便利": 0.1,
        "价格敏感": 0.1,
        "品质可靠": 0.1,
        "体验享受": 0.1,
        "新奇探索": 0.1,
        "社交连接": 0.1,
        "氛围仪式": 0.1,
        "舒适安全": 0.1,
    }
    negative_scores = dict(positive_scores)
    positive_scores[target_value] = positive_score
    negative_scores[target_value] = negative_score
    return {
        "example_id": example_id,
        "split": "test",
        "target_value": target_value,
        "relation": relation,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "all_positive_scores": positive_scores,
        "all_negative_scores": negative_scores,
    }


def test_sentence_relation_metrics_include_signed_macro_outputs() -> None:
    records = [
        _score_record("c1", "省心便利", "related", 0.9, 0.1),
        _score_record("c2", "省心便利", "opposite", 0.1, 0.9),
        _score_record("c3", "省心便利", "unrelated", 0.1, 0.1),
        _score_record("h1", "健康克制", "related", 0.8, 0.1),
        _score_record("h2", "健康克制", "opposite", 0.1, 0.8),
        _score_record("h3", "健康克制", "unrelated", 0.1, 0.1),
    ]

    metrics = sentence_relation_metrics(records, threshold=0.5)

    assert metrics["macro_3way_f1"] > 0.0
    assert metrics["macro_positive_vs_rest_f1"] > 0.0
    assert metrics["macro_negative_vs_rest_f1"] > 0.0
    assert "macro_positive_vs_rest_auc" in metrics
    assert "macro_negative_vs_rest_auc" in metrics
    assert metrics["top1_matched_value_accuracy"] == 1.0


def test_control_metrics_include_lexical_baseline_and_template_groups() -> None:
    eval_rows = [
        {
            "example_id": "c1",
            "target_value": "省心便利",
            "relation": "related",
            "split": "test",
            "text": "希望附近方便少排队",
            "template_id": "nearby",
        },
        {
            "example_id": "c2",
            "target_value": "省心便利",
            "relation": "opposite",
            "split": "test",
            "text": "愿意绕路慢慢逛",
            "template_id": "slow",
        },
    ]
    train_rows = [
        {
            "example_id": "t1",
            "target_value": "省心便利",
            "relation": "related",
            "split": "train",
            "text": "附近方便不用等",
        },
        {
            "example_id": "t2",
            "target_value": "省心便利",
            "relation": "opposite",
            "split": "train",
            "text": "远一点绕路也可以",
        },
        {
            "example_id": "t3",
            "target_value": "省心便利",
            "relation": "unrelated",
            "split": "train",
            "text": "想找评分稳定的店",
        },
    ]
    records = [
        _score_record("c1", "省心便利", "related", 0.9, 0.1),
        _score_record("c2", "省心便利", "opposite", 0.1, 0.9),
    ]

    control_metrics, selectivity = control_and_selectivity_metrics(
        records,
        eval_rows,
        train_rows,
        threshold=0.5,
        seed=7,
    )
    template_metrics = template_split_metrics(records, eval_rows, threshold=0.5)

    assert "lexical_nb_baseline" in control_metrics
    assert "lexical_nb_baseline" in selectivity
    assert control_metrics["template_split"]["status"] == "computed"
    assert template_metrics["field"] == "template_id"


def test_robustness_helpers_report_available_and_unavailable_inputs() -> None:
    rows = [
        {
            "example_id": "p1",
            "target_value": "健康克制",
            "relation": "related",
            "text": "想吃清淡健康一点",
            "paraphrase_group_id": "g1",
        },
        {
            "example_id": "p2",
            "target_value": "健康克制",
            "relation": "related",
            "text": "希望少油少糖更健康",
            "paraphrase_group_id": "g1",
        },
    ]
    records = [
        _score_record("p1", "健康克制", "related", 0.8, 0.1),
        _score_record("p2", "健康克制", "related", 0.7, 0.1),
    ]

    length_metrics = length_bias_metrics(records, rows)
    consistency = paired_consistency_metrics(records, rows)
    unavailable_rows = [{**row, "paraphrase_group_id": ""} for row in rows]
    unavailable = paired_consistency_metrics(records, unavailable_rows)

    assert length_metrics["num_examples"] == 2
    assert consistency["status"] == "computed"
    assert consistency["num_groups"] == 1
    assert unavailable["status"] == "unavailable"
