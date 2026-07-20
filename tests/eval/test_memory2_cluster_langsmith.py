from eval.memory2_cluster.langsmith import cluster_feedback_scores


def test_cluster_feedback_scores_expose_cluster_metrics() -> None:
    scores = cluster_feedback_scores(
        {
            "passed": True,
            "score": 0.8,
            "metrics": {
                "core_cluster_recall": 1.0,
                "weighted_cluster_coverage": 0.8,
                "cluster_mrr": 0.5,
                "irrelevant_cluster_rate": 0.25,
                "forbidden_cluster_rate": 0.0,
                "duplicate_cluster_rate": 0.4,
                "relevant_cluster_diversity": 3,
                "context_budget_efficiency": 0.75,
            },
        }
    )

    assert scores == {
        "cluster_quality_score": 0.8,
        "case_pass": 1.0,
        "core_cluster_recall": 1.0,
        "weighted_cluster_coverage": 0.8,
        "cluster_mrr": 0.5,
        "irrelevant_cluster_rate": 0.25,
        "forbidden_cluster_rate": 0.0,
        "duplicate_cluster_rate": 0.4,
        "relevant_cluster_diversity": 3.0,
        "context_budget_efficiency": 0.75,
    }
