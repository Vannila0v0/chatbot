from __future__ import annotations

from typing import Any

from eval.memory2_quality.langsmith_sync import run_experiment


def cluster_feedback_scores(result: dict[str, Any]) -> dict[str, float]:
    metrics = result.get("metrics") or {}
    keys = (
        "core_cluster_recall",
        "weighted_cluster_coverage",
        "cluster_mrr",
        "irrelevant_cluster_rate",
        "forbidden_cluster_rate",
        "duplicate_cluster_rate",
        "relevant_cluster_diversity",
        "context_budget_efficiency",
    )
    scores = {
        "cluster_quality_score": float(result.get("score") or 0.0),
        "case_pass": float(bool(result.get("passed"))),
    }
    scores.update({key: float(metrics[key]) for key in keys if key in metrics})
    return scores


def cluster_metric_evaluator(run: Any, example: Any) -> list[dict[str, float | str]]:
    _ = example
    outputs = getattr(run, "outputs", None) or {}
    return [
        {"key": key, "score": score}
        for key, score in cluster_feedback_scores(outputs).items()
    ]


async def run_cluster_experiment(
    *, target: Any, data: Any, experiment_prefix: str, max_concurrency: int
) -> Any:
    return await run_experiment(
        target=target,
        data=data,
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
        evaluators=[cluster_metric_evaluator],
        metadata={"benchmark": "memory2-event-cluster-retrieval"},
    )
