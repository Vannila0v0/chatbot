from eval.memory2_cluster.ablation import (
    compare_ranking_results,
    evaluate_ablation_ranking,
    summarize_ablation,
)
from eval.memory2_cluster.models import ClusterProbe


def test_compare_ranking_results_reports_treatment_delta_and_regression() -> None:
    result = compare_ranking_results(
        baseline={
            "weighted_cluster_coverage": 0.5,
            "cluster_mrr": 0.5,
            "pairwise_accuracy": 0.0,
            "forbidden_cluster_rate": 1.0,
        },
        treatment={
            "weighted_cluster_coverage": 0.8,
            "cluster_mrr": 1.0,
            "pairwise_accuracy": 1.0,
            "forbidden_cluster_rate": 0.0,
        },
    )

    assert result["coverage_delta"] == 0.3
    assert result["mrr_delta"] == 0.5
    assert result["forbidden_rate_delta"] == -1.0
    assert result["treatment_improved"] is True
    assert result["regression"] is False
    assert result["mixed"] is False


def test_compare_ranking_results_marks_regression() -> None:
    result = compare_ranking_results(
        baseline={"weighted_cluster_coverage": 1.0, "cluster_mrr": 1.0},
        treatment={"weighted_cluster_coverage": 0.5, "cluster_mrr": 0.5},
    )

    assert result["regression"] is True


def test_compare_ranking_results_marks_mixed_changes() -> None:
    result = compare_ranking_results(
        baseline={"cluster_mrr": 0.5, "forbidden_cluster_rate": 0.0},
        treatment={"cluster_mrr": 1.0, "forbidden_cluster_rate": 1.0},
    )

    assert result["treatment_improved"] is False
    assert result["regression"] is False
    assert result["mixed"] is True


def test_summarize_ablation_reports_paired_average_and_wins() -> None:
    summary = summarize_ablation(
        [
            {
                "split": "natural",
                "tags": ["benefit"],
                "baseline": {"metrics": {"weighted_cluster_coverage": 0.5, "cluster_mrr": 0.5}},
                "treatment": {"metrics": {"weighted_cluster_coverage": 0.75, "cluster_mrr": 1.0}},
                "comparison": {
                    "treatment_improved": True,
                    "regression": False,
                    "mixed": False,
                },
            },
            {
                "split": "natural",
                "tags": ["guardrail"],
                "baseline": {"metrics": {"weighted_cluster_coverage": 1.0, "cluster_mrr": 1.0}},
                "treatment": {"metrics": {"weighted_cluster_coverage": 1.0, "cluster_mrr": 1.0}},
                "comparison": {
                    "treatment_improved": False,
                    "regression": False,
                    "mixed": False,
                },
            },
        ]
    )

    assert summary["natural"]["n"] == 2
    assert summary["natural"]["treatment_wins"] == 1
    assert summary["natural"]["baseline"]["weighted_cluster_coverage"] == 0.75
    assert summary["natural"]["treatment"]["weighted_cluster_coverage"] == 0.875
    assert summary["natural"]["cohorts"]["benefit"]["n"] == 1
    assert summary["natural"]["cohorts"]["guardrail"]["n"] == 1


def test_evaluate_ablation_ranking_distinguishes_memories_in_same_cluster() -> None:
    probe = ClusterProbe(
        case_id="case_1",
        timeline_id="timeline_1",
        query="现在是什么状态？",
        query_time="2026-01-10T00:00:00+08:00",
        cluster_oracle={"status": "core"},
        memory_oracle={"current_status": "core", "old_status": "forbidden"},
        preferred_memory_pairs=[("current_status", "old_status")],
    )
    hits = [
        {"local_id": "old_status", "cluster_id": "status"},
        {"local_id": "current_status", "cluster_id": "status"},
    ]

    metrics = evaluate_ablation_ranking(probe, hits)

    assert metrics["core_cluster_recall"] == 1.0
    assert metrics["memory_mrr"] == 0.5
    assert metrics["memory_pairwise_accuracy"] == 0.0
    assert metrics["memory_forbidden_rate"] == 1.0
    assert metrics["memory_passed"] is False
