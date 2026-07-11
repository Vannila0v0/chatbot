from eval.memory2_quality.evaluators import (
    aggregate_scores,
    evaluate_recall_ranking,
    normalize_fact,
)


def test_normalize_fact_removes_spacing_and_punctuation() -> None:
    assert normalize_fact(" 用户，喜欢 安静的餐厅。 ") == "用户喜欢安静的餐厅"


def test_recall_metrics_count_forbidden_and_rank() -> None:
    metrics = evaluate_recall_ranking(
        required_ids={"new"},
        forbidden_ids={"old"},
        ranked_ids=["x", "new", "old"],
    )
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["forbidden_recall_rate"] == 1.0
    assert metrics["passed"] is False


def test_recall_metrics_handle_no_required_ids() -> None:
    metrics = evaluate_recall_ranking(
        required_ids=set(), forbidden_ids=set(), ranked_ids=[]
    )
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["passed"] is True


def test_aggregate_scores_groups_categories_and_errors() -> None:
    summary = aggregate_scores(
        [
            {"category": "conflict", "passed": True, "score": 0.8, "error": None},
            {"category": "conflict", "passed": False, "score": 0.2, "error": None},
            {"category": "noise", "passed": False, "score": 0.0, "error": "boom"},
        ]
    )
    assert summary["overall"] == {
        "n": 3,
        "passed": 1,
        "pass_rate": 1 / 3,
        "average_score": 1 / 3,
        "errors": 1,
    }
    assert summary["by_category"]["conflict"]["pass_rate"] == 0.5

