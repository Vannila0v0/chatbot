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


def ablation_feedback_scores(result: dict[str, Any]) -> dict[str, float]:
    baseline = (result.get("baseline") or {}).get("metrics") or {}
    treatment = (result.get("treatment") or {}).get("metrics") or {}
    comparison = result.get("comparison") or {}
    keys = (
        "weighted_cluster_coverage",
        "core_cluster_recall",
        "cluster_mrr",
        "ndcg_at_k",
        "pairwise_accuracy",
        "forbidden_cluster_rate",
        "irrelevant_cluster_rate",
    )
    scores: dict[str, float] = {}
    for key in keys:
        scores[f"baseline_{key}"] = float(baseline.get(key, 0.0))
        scores[f"treatment_{key}"] = float(treatment.get(key, 0.0))
    scores["coverage_delta"] = float(comparison.get("coverage_delta", 0.0))
    scores["mrr_delta"] = float(comparison.get("mrr_delta", 0.0))
    scores["regression"] = float(bool(comparison.get("regression")))
    return scores


def ablation_metric_evaluator(run: Any, example: Any) -> list[dict[str, float | str]]:
    _ = example
    outputs = getattr(run, "outputs", None) or {}
    return [
        {"key": key, "score": score}
        for key, score in ablation_feedback_scores(outputs).items()
    ]


async def run_ablation_experiment(
    *, target: Any, data: Any, experiment_prefix: str, max_concurrency: int
) -> Any:
    return await run_experiment(
        target=target,
        data=data,
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
        evaluators=[ablation_metric_evaluator],
        metadata={"benchmark": "memory2-ranking-ablation"},
    )
