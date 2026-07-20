from eval.memory2_cluster.ablation import (
    _analyze_rank_changes,
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


def test_rank_change_diagnostics_detects_a_b_integrity_and_broken_pair() -> None:
    probe = ClusterProbe(
        case_id="case_1",
        timeline_id="timeline_1",
        query="现在是什么状态？",
        query_time="2026-01-10T00:00:00+08:00",
        cluster_oracle={"status": "core"},
        memory_oracle={"current": "core", "old": "forbidden"},
        preferred_memory_pairs=[("current", "old")],
        top_k=1,
    )
    baseline = [
        {
            "local_id": "current",
            "cluster_id": "status",
            "memory_oracle_role": "core",
            "semantic_score": 0.9,
            "keyword_rank": 1,
            "vector_rank": 1,
            "final_rank": 1,
            "selected": True,
        },
        {
            "local_id": "old",
            "cluster_id": "status",
            "memory_oracle_role": "forbidden",
            "semantic_score": 0.8,
            "keyword_rank": 2,
            "vector_rank": 2,
            "final_rank": 2,
            "selected": False,
        },
    ]
    treatment = [
        {
            **baseline[1],
            "vector_rank": 1,
            "final_rank": 1,
            "selected": True,
        },
        {
            **baseline[0],
            "vector_rank": 2,
            "final_rank": 2,
            "selected": False,
        },
    ]

    diagnostics = _analyze_rank_changes(
        baseline=baseline,
        treatment=treatment,
        probe=probe,
        query_embedding_sha256="HASH",
    )

    assert diagnostics["ablation_integrity"]["ablation_integrity_passed"] is True
    changes = {
        item["local_id"]: item for item in diagnostics["rank_changes"]
    }
    assert changes["current"]["change_type"] == "top_k_dropped"
    assert changes["old"]["change_type"] == "top_k_entered"
    assert diagnostics["preferred_memory_pair_changes"] == [
        {
            "preferred_local_id": "current",
            "less_preferred_local_id": "old",
            "baseline_correct": True,
            "treatment_correct": False,
            "change_type": "preferred_pair_broken",
        }
    ]


def test_rank_change_diagnostics_rejects_noncontrolled_ablation() -> None:
    probe = ClusterProbe(
        case_id="case_1",
        timeline_id="timeline_1",
        query="测试",
        query_time="2026-01-10T00:00:00+08:00",
        cluster_oracle={"cluster": "core"},
    )
    baseline = [
        {
            "local_id": "a",
            "cluster_id": "cluster",
            "memory_oracle_role": "unlabeled",
            "semantic_score": 0.9,
            "keyword_rank": 1,
            "vector_rank": 1,
            "final_rank": 1,
            "selected": True,
        }
    ]
    treatment = [{**baseline[0], "semantic_score": 0.7}]

    diagnostics = _analyze_rank_changes(
        baseline=baseline,
        treatment=treatment,
        probe=probe,
        query_embedding_sha256="HASH",
    )

    assert diagnostics["ablation_integrity"]["semantic_scores_match"] is False
    assert diagnostics["ablation_integrity"]["ablation_integrity_passed"] is False
