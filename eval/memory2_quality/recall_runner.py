from __future__ import annotations

from typing import Any

from core.memory.engine import MemoryQuery, MemoryScope

from .evaluators import evaluate_recall_ranking
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
            results.append(
                RecallProbeResult(
                    probe_id=probe.probe_id,
                    query=probe.query,
                    hits=hits,
                    ranked_ids=ranked_ids,
                    metrics=evaluate_recall_ranking(
                        required_ids=required_ids,
                        forbidden_ids=forbidden_ids,
                        ranked_ids=ranked_ids,
                    ),
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
