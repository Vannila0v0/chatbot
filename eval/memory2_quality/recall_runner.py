from __future__ import annotations

from typing import Any

from core.memory.engine import MemoryQuery, MemoryScope

from .evaluators import evaluate_recall_ranking, fact_is_entailed
from .models import MemoryEvalCase, RecallHit, RecallProbeResult


async def run_recall_probes(
    runtime: Any,
    case: MemoryEvalCase,
    label_to_item_id: dict[str, str],
    local_id_to_item_id: dict[str, str] | None = None,
) -> list[RecallProbeResult]:
    engine = runtime.core.memory_runtime.engine
    local_map = local_id_to_item_id or {}
    results: list[RecallProbeResult] = []
    for probe in case.recall_probes:
        try:
            response = await engine.query(
                MemoryQuery(
                    text=probe.query,
                    intent=probe.intent,  # type: ignore[arg-type]
                    scope=MemoryScope(
                        session_key=f"m2eval:{case.case_id}",
                        channel="benchmark",
                        chat_id=case.case_id,
                    ),
                    limit=probe.top_k,
                )
            )
            hits = [
                RecallHit(
                    id=str(record.id),
                    memory_type=str(record.kind),
                    summary=str(record.summary),
                    score=float(record.score),
                )
                for record in response.records[: probe.top_k]
            ]
            ranked_ids = [hit.id for hit in hits]
            required_ids = {
                item_id
                for label in probe.required_memory_labels
                if (item_id := label_to_item_id.get(label))
            }
            missing_required_labels = sorted(
                label
                for label in probe.required_memory_labels
                if label not in label_to_item_id
            )
            required_ids.update(
                item_id
                for local_id in probe.required_local_ids
                if (item_id := local_map.get(local_id))
            )
            forbidden_ids = {
                item_id
                for local_id in probe.forbidden_local_ids
                if (item_id := local_map.get(local_id))
            }
            metrics = evaluate_recall_ranking(
                required_ids=required_ids,
                forbidden_ids=forbidden_ids,
                ranked_ids=ranked_ids,
            )
            metrics["missing_required_labels"] = missing_required_labels
            missing_required_facts = [
                fact
                for fact in probe.required_facts
                if not any(fact_is_entailed(fact, hit.summary) for hit in hits)
            ]
            forbidden_fact_hits = [
                fact
                for fact in probe.forbidden_facts
                if any(fact_is_entailed(fact, hit.summary) for hit in hits)
            ]
            hit_types = {hit.memory_type for hit in hits}
            missing_required_types = sorted(set(probe.required_types) - hit_types)
            metrics["missing_required_facts"] = missing_required_facts
            metrics["forbidden_fact_hits"] = forbidden_fact_hits
            metrics["missing_required_types"] = missing_required_types
            if missing_required_labels:
                metrics["recall_at_k"] = 0.0
                metrics["mrr"] = 0.0
                metrics["passed"] = False
            if missing_required_facts or forbidden_fact_hits or missing_required_types:
                metrics["passed"] = False
            results.append(
                RecallProbeResult(
                    probe_id=probe.probe_id,
                    query=probe.query,
                    hits=hits,
                    ranked_ids=ranked_ids,
                    metrics=metrics,
                    trace=dict(response.trace or {}),
                )
            )
        except Exception as exc:
            results.append(
                RecallProbeResult(
                    probe_id=probe.probe_id,
                    query=probe.query,
                    error=str(exc),
                    metrics={"passed": False},
                )
            )
    return results
