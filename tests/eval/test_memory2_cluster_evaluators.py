from eval.memory2_cluster.evaluators import evaluate_cluster_ranking
from eval.memory2_cluster.runner import map_ranked_hits
from eval.memory2_cluster.runner import replay_time_at_evaluation


def test_cluster_metrics_reward_coverage_without_double_counting_repeats() -> None:
    metrics = evaluate_cluster_ranking(
        cluster_oracle={
            "symptom": "core",
            "diet": "core",
            "history": "supporting",
            "later_meal": "forbidden",
            "project": "irrelevant",
        },
        ranked_cluster_ids=["diet", "diet", "symptom", "history", "project"],
    )

    assert metrics["core_cluster_recall"] == 1.0
    assert metrics["weighted_cluster_coverage"] == 1.0
    assert metrics["cluster_mrr"] == 1.0
    assert metrics["relevant_cluster_diversity"] == 3
    assert metrics["duplicate_cluster_rate"] == 0.2
    assert metrics["irrelevant_cluster_rate"] == 0.25
    assert metrics["forbidden_cluster_rate"] == 0.0
    assert metrics["passed"] is True


def test_cluster_metrics_flag_forbidden_and_missing_core_clusters() -> None:
    metrics = evaluate_cluster_ranking(
        cluster_oracle={
            "symptom": "core",
            "diet": "core",
            "later_meal": "forbidden",
            "project": "irrelevant",
        },
        ranked_cluster_ids=["project", "later_meal", "project"],
    )

    assert metrics["core_cluster_recall"] == 0.0
    assert metrics["cluster_mrr"] == 0.0
    assert metrics["forbidden_cluster_rate"] == 1.0
    assert metrics["duplicate_cluster_rate"] == 1 / 3
    assert metrics["passed"] is False


def test_cluster_metrics_expose_minimum_gate_separately_from_full_core_coverage() -> None:
    metrics = evaluate_cluster_ranking(
        cluster_oracle={"symptom": "core", "diet": "core", "project": "irrelevant"},
        ranked_cluster_ids=["symptom"],
    )

    assert metrics["minimum_core_gate_passed"] is True
    assert metrics["passed"] is False


def test_map_ranked_hits_keeps_memory_rank_and_cluster_membership() -> None:
    hits = map_ranked_hits(
        records=[
            {"id": "m2", "summary": "second", "score": 0.8},
            {"id": "m1", "summary": "first", "score": 0.6},
        ],
        item_to_local_id={"m1": "diet_01", "m2": "diet_02"},
        local_to_cluster_id={"diet_01": "diet", "diet_02": "diet"},
    )

    assert hits == [
        {"rank": 1, "memory_id": "m2", "local_id": "diet_02", "cluster_id": "diet", "summary": "second", "score": 0.8},
        {"rank": 2, "memory_id": "m1", "local_id": "diet_01", "cluster_id": "diet", "summary": "first", "score": 0.6},
    ]


def test_replay_time_preserves_event_distance_from_historical_query() -> None:
    from datetime import datetime

    event_time = datetime.fromisoformat("2026-04-06T20:00:00+08:00")
    query_time = datetime.fromisoformat("2026-04-07T20:00:00+08:00")
    evaluation_time = datetime.fromisoformat("2026-07-13T12:00:00+08:00")

    assert replay_time_at_evaluation(event_time, query_time, evaluation_time) == datetime.fromisoformat(
        "2026-07-12T12:00:00+08:00"
    )
