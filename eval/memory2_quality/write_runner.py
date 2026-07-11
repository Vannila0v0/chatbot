from __future__ import annotations

from typing import Any

from .evaluators import fact_is_entailed
from .models import MemoryEvalCase, WriteRunResult
from .snapshots import diff_snapshots, take_snapshot


async def run_write_case(runtime: Any, case: MemoryEvalCase) -> WriteRunResult:
    engine = runtime.core.memory_runtime.engine
    store = engine._v2_store
    before = take_snapshot(store)
    session_key = f"m2eval:{case.case_id}"
    session_manager = runtime.core.session_manager

    try:
        for eval_session in case.sessions:
            session_manager._cache.pop(session_key, None)
            session = session_manager.get_or_create(session_key)
            session._channel = "benchmark"
            session._chat_id = case.case_id
            for message in eval_session.messages:
                session.add_message(message.role, message.content)
                timestamp = message.timestamp or eval_session.timestamp
                session.messages[-1]["timestamp"] = timestamp.isoformat()
            session_manager.save(session)
            if eval_session.consolidate_after:
                await runtime.consolidation.consolidate(session, archive_all=True)
                session_manager.save(session)
        after = take_snapshot(store)
        state_diff = diff_snapshots(before, after)
        return WriteRunResult(
            before=before,
            after=after,
            state_diff=state_diff,
            label_to_item_id=_resolve_labels(case, state_diff.created),
        )
    except Exception as exc:
        after = take_snapshot(store)
        return WriteRunResult(
            before=before,
            after=after,
            state_diff=diff_snapshots(before, after),
            error=str(exc),
        )


def _resolve_labels(case: MemoryEvalCase, created: list[Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for expected in case.expected_write.required:
        if not expected.label:
            continue
        candidates = [
            item
            for item in created
            if (expected.memory_type is None or item.memory_type == expected.memory_type)
            and any(fact_is_entailed(fact, item.summary) for fact in expected.facts)
        ]
        if len(candidates) == 1:
            mapping[expected.label] = candidates[0].id
    return mapping
