from __future__ import annotations

from typing import Literal


_WEIGHTS = {"core": 3.0, "supporting": 1.0, "weak": 0.25}
_POSITIVE_ROLES = frozenset(_WEIGHTS)
ClusterRole = Literal["core", "supporting", "weak", "irrelevant", "forbidden"]


def evaluate_cluster_ranking(
    *, cluster_oracle: dict[str, ClusterRole], ranked_cluster_ids: list[str]
) -> dict[str, float | int | bool | list[str]]:
    unique_ranked = list(dict.fromkeys(ranked_cluster_ids))
    ranked_set = set(unique_ranked)
    core = {cluster_id for cluster_id, role in cluster_oracle.items() if role == "core"}
    forbidden = {
        cluster_id for cluster_id, role in cluster_oracle.items() if role == "forbidden"
    }
    relevant = {
        cluster_id
        for cluster_id, role in cluster_oracle.items()
        if role in _POSITIVE_ROLES
    }
    covered_core = core & ranked_set
    weighted_total = sum(
        _WEIGHTS[role]
        for role in cluster_oracle.values()
        if role in _WEIGHTS
    )
    weighted_covered = sum(
        _WEIGHTS[role]
        for cluster_id, role in cluster_oracle.items()
        if role in _WEIGHTS and cluster_id in ranked_set
    )
    first_core_rank = next(
        (index + 1 for index, cluster_id in enumerate(ranked_cluster_ids) if cluster_id in core),
        None,
    )
    denominator = len(unique_ranked)
    irrelevant_hits = [
        cluster_id
        for cluster_id in unique_ranked
        if cluster_oracle.get(cluster_id) == "irrelevant"
    ]
    forbidden_hits = [cluster_id for cluster_id in unique_ranked if cluster_id in forbidden]
    duplicate_count = len(ranked_cluster_ids) - len(unique_ranked)
    return {
        "core_cluster_recall": len(covered_core) / len(core),
        "weighted_cluster_coverage": weighted_covered / weighted_total,
        "cluster_mrr": 1.0 / first_core_rank if first_core_rank else 0.0,
        "relevant_cluster_diversity": len(relevant & ranked_set),
        "irrelevant_cluster_rate": len(irrelevant_hits) / denominator if denominator else 0.0,
        "forbidden_cluster_rate": len(forbidden_hits) / len(forbidden) if forbidden else 0.0,
        "duplicate_cluster_rate": duplicate_count / len(ranked_cluster_ids) if ranked_cluster_ids else 0.0,
        "missing_core_clusters": sorted(core - ranked_set),
        "forbidden_cluster_hits": forbidden_hits,
        "minimum_core_gate_passed": bool(covered_core) and not forbidden_hits,
        "passed": covered_core == core and not forbidden_hits,
    }
