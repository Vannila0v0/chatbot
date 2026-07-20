from __future__ import annotations

from datetime import datetime
from typing import Any

from core.memory.engine import MemoryQuery, MemoryQueryFilters, MemoryScope
from eval.memory2_quality.models import MemoryFixture
from eval.memory2_quality.seed import seed_memories
from eval.memory2_quality.langsmith_sync import traced_stage

from .evaluators import evaluate_cluster_ranking
from .models import ClusterProbe, EventTimeline


def replay_time_at_evaluation(
    event_time: datetime, query_time: datetime, evaluation_time: datetime
) -> datetime:
    """Shift a historical timeline so its age is measured from this evaluation."""
    return evaluation_time - (query_time - event_time)


def _field(record: Any, name: str, default: Any = None) -> Any:
    return record.get(name, default) if isinstance(record, dict) else getattr(record, name, default)


def map_ranked_hits(
    *,
    records: list[Any],
    item_to_local_id: dict[str, str],
    local_to_cluster_id: dict[str, str],
) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for rank, record in enumerate(records, 1):
        memory_id = str(_field(record, "id", ""))
        local_id = item_to_local_id.get(memory_id, "")
        mapped.append(
            {
                "rank": rank,
                "memory_id": memory_id,
                "local_id": local_id,
                "cluster_id": local_to_cluster_id.get(local_id, "unknown"),
                "summary": str(_field(record, "summary", "")),
                "score": float(_field(record, "score", 0.0)),
            }
        )
    return mapped


async def run_cluster_probe(
    runtime: Any, timeline: EventTimeline, probe: ClusterProbe, *, tracing: bool = False
) -> dict[str, Any]:
    engine = runtime.core.memory_runtime.engine
    store = engine._v2_store
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
            extra={
                "cluster_id": memory.cluster_id,
                "scope_channel": "benchmark",
                "scope_chat_id": probe.case_id,
            },
        )
        for memory in timeline.memories
    ]
    with traced_stage(
        tracing, "seed_cluster_memories", {"memory_count": len(fixtures)}
    ) as stage:
        local_to_item_id = await seed_memories(store, engine._embedder, fixtures)
        stage.set_outputs({"seeded_count": len(local_to_item_id)})
    item_to_local_id = {item_id: local_id for local_id, item_id in local_to_item_id.items()}
    local_to_cluster_id = {
        memory.local_id: memory.cluster_id for memory in timeline.memories
    }
    with traced_stage(
        tracing, "retrieve_cluster_memories", {"query": probe.query, "top_k": probe.top_k}
    ) as stage:
        response = await engine.query(
            MemoryQuery(
                text=probe.query,
                intent="answer",
                scope=MemoryScope(
                    session_key=f"m2cluster:{probe.case_id}",
                    channel="benchmark",
                    chat_id=probe.case_id,
                ),
                filters=MemoryQueryFilters(time_end=evaluation_time),
                limit=probe.top_k,
            ),
        )
        stage.set_outputs({"record_count": len(response.records)})
    with traced_stage(tracing, "map_hits_to_clusters") as stage:
        hits = map_ranked_hits(
            records=list(response.records[: probe.top_k]),
            item_to_local_id=item_to_local_id,
            local_to_cluster_id=local_to_cluster_id,
        )
        stage.set_outputs({"cluster_ids": [hit["cluster_id"] for hit in hits]})
    with traced_stage(tracing, "evaluate_cluster_metrics") as stage:
        metrics = evaluate_cluster_ranking(
            cluster_oracle=probe.cluster_oracle,
            ranked_cluster_ids=[str(hit["cluster_id"]) for hit in hits],
        )
        total_characters = sum(len(str(hit["summary"])) for hit in hits)
        relevant_characters = sum(
            len(str(hit["summary"]))
            for hit in hits
            if probe.cluster_oracle.get(str(hit["cluster_id"]))
            in {"core", "supporting", "weak"}
        )
        metrics["context_budget_efficiency"] = (
            relevant_characters / total_characters if total_characters else 0.0
        )
        stage.set_outputs(metrics)
    return {
        "case_id": probe.case_id,
        "timeline_id": timeline.timeline_id,
        "category": "event_cluster_retrieval",
        "passed": bool(metrics["passed"]),
        "score": float(metrics["weighted_cluster_coverage"]),
        "error": None,
        "query": probe.query,
        "query_time": probe.query_time.isoformat(),
        "evaluation_time": evaluation_time.isoformat(),
        "hits": hits,
        "metrics": metrics,
        "trace": dict(response.trace or {}),
    }
