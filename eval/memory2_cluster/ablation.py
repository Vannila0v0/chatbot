from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from typing import Any

from eval.memory2_quality.langsmith_sync import traced_stage
from eval.memory2_quality.models import MemoryFixture
from eval.memory2_quality.seed import seed_memories
from memory2.retriever import _extract_terms, _rrf_merge

from .evaluators import evaluate_cluster_ranking
from .models import ClusterProbe, EventTimeline
from .runner import map_ranked_hits, replay_time_at_evaluation


_GAIN = {"core": 3.0, "supporting": 2.0, "weak": 1.0}
_SUMMARY_METRICS = (
    "weighted_cluster_coverage",
    "core_cluster_recall",
    "cluster_mrr",
    "ndcg_at_k",
    "pairwise_accuracy",
    "forbidden_cluster_rate",
    "irrelevant_cluster_rate",
    "memory_weighted_coverage",
    "memory_core_recall",
    "memory_mrr",
    "memory_ndcg_at_k",
    "memory_pairwise_accuracy",
    "memory_forbidden_rate",
    "memory_irrelevant_rate",
)


def summarize_ablation(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    splits = sorted({str(result.get("split") or "unknown") for result in results})

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        def average(group: str, key: str) -> float:
            values = [float(row[group]["metrics"].get(key, 0.0)) for row in rows]
            return sum(values) / len(values) if values else 0.0

        return {
            "n": len(rows),
            "baseline": {key: average("baseline", key) for key in _SUMMARY_METRICS},
            "treatment": {key: average("treatment", key) for key in _SUMMARY_METRICS},
            "treatment_wins": sum(
                bool(row["comparison"].get("treatment_improved")) for row in rows
            ),
            "regressions": sum(bool(row["comparison"].get("regression")) for row in rows),
            "mixed": sum(bool(row["comparison"].get("mixed")) for row in rows),
        }

    for split in splits:
        rows = [result for result in results if str(result.get("split") or "unknown") == split]
        split_summary = aggregate(rows)
        cohorts = {}
        for cohort in ("benefit", "guardrail"):
            cohort_rows = [row for row in rows if cohort in (row.get("tags") or [])]
            if cohort_rows:
                cohorts[cohort] = aggregate(cohort_rows)
        split_summary["cohorts"] = cohorts
        summary[split] = split_summary
    return summary


def compare_ranking_results(
    *, baseline: dict[str, Any], treatment: dict[str, Any]
) -> dict[str, float | bool]:
    higher_is_better = {
        "coverage_delta": "weighted_cluster_coverage",
        "core_recall_delta": "core_cluster_recall",
        "mrr_delta": "cluster_mrr",
        "ndcg_delta": "ndcg_at_k",
        "pairwise_delta": "pairwise_accuracy",
        "memory_coverage_delta": "memory_weighted_coverage",
        "memory_core_recall_delta": "memory_core_recall",
        "memory_mrr_delta": "memory_mrr",
        "memory_ndcg_delta": "memory_ndcg_at_k",
        "memory_pairwise_delta": "memory_pairwise_accuracy",
    }
    lower_is_better = {
        "forbidden_rate_delta": "forbidden_cluster_rate",
        "irrelevant_rate_delta": "irrelevant_cluster_rate",
        "memory_forbidden_rate_delta": "memory_forbidden_rate",
        "memory_irrelevant_rate_delta": "memory_irrelevant_rate",
    }
    deltas = {
        name: round(float(treatment.get(metric, 0.0)) - float(baseline.get(metric, 0.0)), 12)
        for name, metric in {**higher_is_better, **lower_is_better}.items()
    }
    directional = [deltas[name] for name in higher_is_better]
    directional.extend(-deltas[name] for name in lower_is_better)
    improved = any(delta > 0 for delta in directional)
    worsened = any(delta < 0 for delta in directional)
    return {
        **deltas,
        "treatment_improved": improved and not worsened,
        "regression": worsened and not improved,
        "mixed": improved and worsened,
    }


def _first_cluster_ranks(ranked_cluster_ids: list[str]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for rank, cluster_id in enumerate(ranked_cluster_ids, 1):
        ranks.setdefault(cluster_id, rank)
    return ranks


def _pairwise_accuracy(
    ranked_cluster_ids: list[str], preferred_pairs: list[tuple[str, str]]
) -> float:
    if not preferred_pairs:
        return 1.0
    ranks = _first_cluster_ranks(ranked_cluster_ids)
    correct = 0
    missing_rank = len(ranked_cluster_ids) + 1
    for better, worse in preferred_pairs:
        better_rank = ranks.get(better, missing_rank)
        worse_rank = ranks.get(worse, missing_rank)
        if better in ranks and better_rank < worse_rank:
            correct += 1
    return correct / len(preferred_pairs)


def _ndcg_at_k(ranked_cluster_ids: list[str], cluster_oracle: dict[str, str]) -> float:
    unique_ranked = list(dict.fromkeys(ranked_cluster_ids))
    gains = [_GAIN.get(cluster_oracle.get(cluster_id, ""), 0.0) for cluster_id in unique_ranked]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal_gains = sorted(
        (_GAIN.get(role, 0.0) for role in cluster_oracle.values()), reverse=True
    )[: len(unique_ranked)]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, 1))
    return dcg / idcg if idcg else 0.0


def evaluate_ablation_ranking(
    probe: ClusterProbe, hits: list[dict[str, Any]]
) -> dict[str, Any]:
    ranked_cluster_ids = [str(hit["cluster_id"]) for hit in hits]
    metrics = evaluate_cluster_ranking(
        cluster_oracle=probe.cluster_oracle,
        ranked_cluster_ids=ranked_cluster_ids,
    )
    metrics["pairwise_accuracy"] = _pairwise_accuracy(
        ranked_cluster_ids, probe.preferred_pairs
    )
    metrics["ndcg_at_k"] = _ndcg_at_k(ranked_cluster_ids, probe.cluster_oracle)
    if probe.memory_oracle:
        ranked_memory_ids = [str(hit["local_id"]) for hit in hits]
        memory_metrics = evaluate_cluster_ranking(
            cluster_oracle=probe.memory_oracle,
            ranked_cluster_ids=ranked_memory_ids,
        )
        metrics.update(
            {
                "memory_weighted_coverage": memory_metrics[
                    "weighted_cluster_coverage"
                ],
                "memory_core_recall": memory_metrics["core_cluster_recall"],
                "memory_mrr": memory_metrics["cluster_mrr"],
                "memory_ndcg_at_k": _ndcg_at_k(
                    ranked_memory_ids, probe.memory_oracle
                ),
                "memory_pairwise_accuracy": _pairwise_accuracy(
                    ranked_memory_ids, probe.preferred_memory_pairs
                ),
                "memory_forbidden_rate": memory_metrics[
                    "forbidden_cluster_rate"
                ],
                "memory_irrelevant_rate": memory_metrics[
                    "irrelevant_cluster_rate"
                ],
                "memory_minimum_core_gate_passed": memory_metrics[
                    "minimum_core_gate_passed"
                ],
                "memory_passed": memory_metrics["passed"],
            }
        )
    return metrics


def _stable_keyword_rank(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (-float(item.get("keyword_score", 0.0)), str(item.get("id", ""))),
    )


def _query_vector_sha256(query_vector: list[float]) -> str:
    """只记录查询向量的指纹，不上传原始 embedding。"""
    payload = json.dumps(query_vector, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _candidate_diagnostics(
    *,
    records: list[dict[str, Any]],
    keyword_items: list[dict[str, Any]],
    item_to_local: dict[str, str],
    memories_by_local: dict[str, Any],
    probe: ClusterProbe,
) -> list[dict[str, Any]]:
    """生成脱敏后的候选评分明细，不包含 embedding、路径或 source_ref。"""
    keyword_scores = {
        str(item.get("id", "")): float(item.get("keyword_score", 0.0))
        for item in keyword_items
    }
    diagnostics: list[dict[str, Any]] = []
    for record in records:
        memory_id = str(record.get("id", ""))
        local_id = item_to_local.get(memory_id, "")
        memory = memories_by_local.get(local_id)
        score_debug = dict(record.get("_score_debug") or {})
        rrf_debug = dict(record.get("_rrf_debug") or {})
        final_rank = int(rrf_debug.get("final_rank") or 0)
        diagnostics.append(
            {
                "memory_id": memory_id,
                "local_id": local_id,
                "cluster_id": getattr(memory, "cluster_id", "unknown"),
                "memory_type": getattr(memory, "memory_type", record.get("memory_type")),
                "summary": getattr(memory, "summary", str(record.get("summary", ""))),
                "cluster_oracle_role": probe.cluster_oracle.get(
                    getattr(memory, "cluster_id", "unknown"), "unlabeled"
                ),
                "memory_oracle_role": probe.memory_oracle.get(local_id, "unlabeled"),
                "semantic_score": score_debug.get("semantic"),
                "reinforcement": score_debug.get(
                    "reinforcement", getattr(memory, "reinforcement", None)
                ),
                "emotional_weight": score_debug.get(
                    "emotional_weight", getattr(memory, "emotional_weight", None)
                ),
                "age_days": score_debug.get(
                    "age_days", getattr(memory, "last_used_days_ago", None)
                ),
                "base_half_life_days": score_debug.get("base_half_life_days"),
                "effective_half_life_days": score_debug.get(
                    "effective_half_life_days"
                ),
                "frequency_factor": score_debug.get("frequency_factor"),
                "recency_factor": score_debug.get("recency_factor"),
                "hotness_score": score_debug.get("hotness"),
                "hotness_alpha": score_debug.get("alpha"),
                "semantic_contribution": score_debug.get("semantic_contribution"),
                "hotness_contribution": score_debug.get("hotness_contribution"),
                "vector_final_score": score_debug.get("final"),
                "vector_rank": rrf_debug.get("vector_rank"),
                "keyword_score": keyword_scores.get(memory_id),
                "keyword_rank": rrf_debug.get("keyword_rank"),
                "vector_rrf_contribution": rrf_debug.get("vector_contribution"),
                "keyword_rrf_contribution": rrf_debug.get(
                    "keyword_contribution"
                ),
                "rrf_score": rrf_debug.get("rrf_score"),
                "final_rank": final_rank,
                "selected": 0 < final_rank <= probe.top_k,
            }
        )
    return diagnostics


def _rank_change_type(role: str, baseline_rank: int, treatment_rank: int) -> str:
    if treatment_rank == baseline_rank:
        return "unchanged"
    direction = "promoted" if treatment_rank < baseline_rank else "demoted"
    if role in {"core", "forbidden", "irrelevant"}:
        return f"{role}_{direction}"
    return direction


def _analyze_rank_changes(
    *,
    baseline: list[dict[str, Any]],
    treatment: list[dict[str, Any]],
    probe: ClusterProbe,
    query_embedding_sha256: str,
) -> dict[str, Any]:
    baseline_by_id = {str(item["local_id"]): item for item in baseline}
    treatment_by_id = {str(item["local_id"]): item for item in treatment}
    baseline_ids = set(baseline_by_id)
    treatment_ids = set(treatment_by_id)
    common_ids = baseline_ids & treatment_ids

    semantic_scores_match = all(
        math.isclose(
            float(baseline_by_id[item_id].get("semantic_score") or 0.0),
            float(treatment_by_id[item_id].get("semantic_score") or 0.0),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for item_id in common_ids
    )
    keyword_ranks_match = all(
        baseline_by_id[item_id].get("keyword_rank")
        == treatment_by_id[item_id].get("keyword_rank")
        for item_id in common_ids
    )
    integrity = {
        "candidate_ids_match": baseline_ids == treatment_ids,
        "semantic_scores_match": semantic_scores_match,
        "keyword_ranks_match": keyword_ranks_match,
        "query_embedding_sha256": query_embedding_sha256,
    }
    integrity["ablation_integrity_passed"] = all(
        bool(value)
        for key, value in integrity.items()
        if key != "query_embedding_sha256"
    )

    rank_changes: list[dict[str, Any]] = []
    for local_id in sorted(common_ids):
        baseline_item = baseline_by_id[local_id]
        treatment_item = treatment_by_id[local_id]
        baseline_rank = int(baseline_item["final_rank"])
        treatment_rank = int(treatment_item["final_rank"])
        if baseline_rank == treatment_rank:
            continue
        role = str(treatment_item.get("memory_oracle_role") or "unlabeled")
        baseline_selected = bool(baseline_item.get("selected"))
        treatment_selected = bool(treatment_item.get("selected"))
        change_type = _rank_change_type(role, baseline_rank, treatment_rank)
        if baseline_selected and not treatment_selected:
            change_type = "top_k_dropped"
        elif treatment_selected and not baseline_selected:
            change_type = "top_k_entered"
        rank_changes.append(
            {
                "local_id": local_id,
                "cluster_id": treatment_item.get("cluster_id"),
                "memory_oracle_role": role,
                "baseline_vector_rank": baseline_item.get("vector_rank"),
                "treatment_vector_rank": treatment_item.get("vector_rank"),
                "baseline_final_rank": baseline_rank,
                "treatment_final_rank": treatment_rank,
                "rank_change_treatment_minus_baseline": treatment_rank
                - baseline_rank,
                "baseline_selected": baseline_selected,
                "treatment_selected": treatment_selected,
                "change_type": change_type,
            }
        )

    pair_changes: list[dict[str, Any]] = []
    for preferred, less_preferred in probe.preferred_memory_pairs:
        if preferred not in common_ids or less_preferred not in common_ids:
            continue
        baseline_correct = (
            int(baseline_by_id[preferred]["final_rank"])
            < int(baseline_by_id[less_preferred]["final_rank"])
        )
        treatment_correct = (
            int(treatment_by_id[preferred]["final_rank"])
            < int(treatment_by_id[less_preferred]["final_rank"])
        )
        if baseline_correct == treatment_correct:
            continue
        pair_changes.append(
            {
                "preferred_local_id": preferred,
                "less_preferred_local_id": less_preferred,
                "baseline_correct": baseline_correct,
                "treatment_correct": treatment_correct,
                "change_type": (
                    "preferred_pair_fixed"
                    if treatment_correct
                    else "preferred_pair_broken"
                ),
            }
        )

    return {
        "ablation_integrity": integrity,
        "rank_changes": rank_changes,
        "preferred_memory_pair_changes": pair_changes,
    }


async def run_paired_ablation(
    runtime: Any,
    timeline: EventTimeline,
    probe: ClusterProbe,
    *,
    tracing: bool = False,
    treatment_alpha: float = 0.2,
    half_life_days: float = 14.0,
) -> dict[str, Any]:
    engine = runtime.core.memory_runtime.engine
    store = engine._v2_store
    embedder = engine._embedder
    evaluation_time = datetime.now(probe.query_time.tzinfo)
    fixtures = [
        MemoryFixture(
            local_id=memory.local_id,
            memory_type=memory.memory_type,
            summary=memory.summary,
            happened_at=replay_time_at_evaluation(
                memory.happened_at, probe.query_time, evaluation_time
            ),
            reinforcement=memory.reinforcement,
            emotional_weight=memory.emotional_weight,
            extra={"cluster_id": memory.cluster_id},
        )
        for memory in timeline.memories
    ]
    with traced_stage(tracing, "seed_ablation_memories", {"count": len(fixtures)}) as stage:
        local_to_item_id = await seed_memories(store, embedder, fixtures)
        for memory in timeline.memories:
            item_id = local_to_item_id[memory.local_id]
            updated_at = evaluation_time - timedelta(days=memory.last_used_days_ago)
            store._db.execute(
                "UPDATE memory_items SET reinforcement=?, updated_at=? WHERE id=?",
                (memory.reinforcement, updated_at.isoformat(), item_id),
            )
        store._db.commit()
        stage.set_outputs(
            {
                "seeded_count": len(local_to_item_id),
                "evaluation_time": evaluation_time.isoformat(),
                "local_to_memory_id": local_to_item_id,
            }
        )

    item_to_local = {item_id: local_id for local_id, item_id in local_to_item_id.items()}
    local_to_cluster = {
        memory.local_id: memory.cluster_id for memory in timeline.memories
    }
    memories_by_local = {memory.local_id: memory for memory in timeline.memories}

    with traced_stage(tracing, "prepare_fixed_query", {"query": probe.query}) as stage:
        query_vector = await embedder.embed(probe.query)
        terms = _extract_terms(probe.query)
        query_embedding_sha256 = _query_vector_sha256(query_vector)
        stage.set_outputs(
            {
                "keyword_terms": terms,
                "query_embedding_sha256": query_embedding_sha256,
                "top_k": probe.top_k,
                "candidate_limit": max(len(timeline.memories), probe.top_k * 4),
            }
        )

    candidate_limit = max(len(timeline.memories), probe.top_k * 4)
    common = {
        "query_vec": query_vector,
        "top_k": candidate_limit,
        "score_threshold": -1.0,
        "time_end": evaluation_time,
    }
    with traced_stage(tracing, "rank_baseline", {"hotness_alpha": 0.0}) as stage:
        baseline_vector = store.vector_search(hotness_alpha=0.0, **common)
        keyword_items = store.keyword_search_summary(
            terms,
            limit=candidate_limit,
            time_end=evaluation_time,
        ) if terms else []
        keyword_rank = _stable_keyword_rank(keyword_items)
        baseline_merged = _rrf_merge(
            baseline_vector,
            keyword_rank,
            top_n=candidate_limit,
            include_score_diagnostics=True,
        )
        baseline_final = baseline_merged[: probe.top_k]
        baseline_diagnostics = _candidate_diagnostics(
            records=baseline_merged,
            keyword_items=keyword_rank,
            item_to_local=item_to_local,
            memories_by_local=memories_by_local,
            probe=probe,
        )
        stage.set_outputs(
            {
                "result_count": len(baseline_final),
                "candidate_count": len(baseline_diagnostics),
                "candidates": baseline_diagnostics,
            }
        )

    with traced_stage(
        tracing,
        "rank_treatment",
        {"hotness_alpha": treatment_alpha, "half_life_days": half_life_days},
    ) as stage:
        treatment_vector = store.vector_search(
            hotness_alpha=treatment_alpha,
            hotness_half_life_days=half_life_days,
            **common,
        )
        treatment_merged = _rrf_merge(
            treatment_vector,
            keyword_rank,
            top_n=candidate_limit,
            include_score_diagnostics=True,
        )
        treatment_final = treatment_merged[: probe.top_k]
        treatment_diagnostics = _candidate_diagnostics(
            records=treatment_merged,
            keyword_items=keyword_rank,
            item_to_local=item_to_local,
            memories_by_local=memories_by_local,
            probe=probe,
        )
        stage.set_outputs(
            {
                "result_count": len(treatment_final),
                "candidate_count": len(treatment_diagnostics),
                "candidates": treatment_diagnostics,
            }
        )

    with traced_stage(tracing, "analyze_rank_changes") as stage:
        ranking_diagnostics = _analyze_rank_changes(
            baseline=baseline_diagnostics,
            treatment=treatment_diagnostics,
            probe=probe,
            query_embedding_sha256=query_embedding_sha256,
        )
        stage.set_outputs(ranking_diagnostics)
    baseline_hits = map_ranked_hits(
        records=baseline_final,
        item_to_local_id=item_to_local,
        local_to_cluster_id=local_to_cluster,
    )
    treatment_hits = map_ranked_hits(
        records=treatment_final,
        item_to_local_id=item_to_local,
        local_to_cluster_id=local_to_cluster,
    )
    baseline_metrics = evaluate_ablation_ranking(probe, baseline_hits)
    treatment_metrics = evaluate_ablation_ranking(probe, treatment_hits)
    comparison = compare_ranking_results(
        baseline=baseline_metrics, treatment=treatment_metrics
    )
    cluster_passed = bool(treatment_metrics["passed"])
    memory_passed = bool(treatment_metrics.get("memory_passed", True))
    overall_passed = cluster_passed and memory_passed
    score_parts = [float(treatment_metrics["weighted_cluster_coverage"])]
    if "memory_weighted_coverage" in treatment_metrics:
        score_parts.append(float(treatment_metrics["memory_weighted_coverage"]))
    return {
        "case_id": probe.case_id,
        "timeline_id": timeline.timeline_id,
        "split": probe.split,
        "tags": probe.tags,
        "query": probe.query,
        "evaluation_time": evaluation_time.isoformat(),
        "baseline": {
            "hits": baseline_hits,
            "metrics": baseline_metrics,
            "vector_rank_ids": [str(item.get("id", "")) for item in baseline_vector],
            "keyword_rank_ids": [str(item.get("id", "")) for item in keyword_rank],
        },
        "treatment": {
            "hits": treatment_hits,
            "metrics": treatment_metrics,
            "vector_rank_ids": [str(item.get("id", "")) for item in treatment_vector],
            "keyword_rank_ids": [str(item.get("id", "")) for item in keyword_rank],
        },
        "comparison": comparison,
        "diagnostics": {
            **ranking_diagnostics,
            "baseline_candidates": baseline_diagnostics,
            "treatment_candidates": treatment_diagnostics,
        },
        "passed": overall_passed,
        "score": sum(score_parts) / len(score_parts),
        "error": None,
    }
