from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.memory2_quality.models import MemoryEvalCase
from eval.memory2_quality.recall_runner import run_recall_probes
from eval.memory2_quality.write_runner import run_write_case
from memory2.store import MemoryStore2


class _Session:
    def __init__(self, key: str):
        self.key = key
        self.messages: list[dict] = []
        self.last_consolidated = 0

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})


class _SessionManager:
    def __init__(self):
        self._cache = {}
        self.session = None

    def get_or_create(self, key: str):
        if self.session is None:
            self.session = _Session(key)
        return self.session

    def save(self, session) -> None:
        self.session = session


class _Consolidation:
    def __init__(self, store: MemoryStore2):
        self.store = store
        self.calls = 0

    async def consolidate(self, session, archive_all: bool = False):
        self.calls += 1
        self.store.upsert_item(
            memory_type="event",
            summary="用户完成了答辩",
            embedding=[0.1, 0.2, 0.3],
            source_ref=f"{session.key}:{self.calls}",
            extra={},
        )


class _Engine:
    def __init__(self, store: MemoryStore2):
        self._v2_store = store

    async def query(self, request):
        record = SimpleNamespace(
            id="m1", kind="event", summary="用户完成了答辩", score=0.9
        )
        return SimpleNamespace(records=[record], raw={"items": []}, trace={})


def _case() -> MemoryEvalCase:
    return MemoryEvalCase.model_validate(
        {
            "case_id": "write_001",
            "category": "type_identification",
            "reference_time": "2026-07-01T10:00:00+08:00",
            "sessions": [
                {
                    "session_id": "s1",
                    "timestamp": "2026-07-01T10:00:00+08:00",
                    "messages": [{"role": "user", "content": "我完成答辩了"}],
                    "consolidate_after": True,
                }
            ],
            "expected_write": {
                "required": [
                    {"label": "defense", "memory_type": "event", "facts": ["完成了答辩"]}
                ]
            },
            "recall_probes": [
                {
                    "probe_id": "p1",
                    "query": "最近完成了什么",
                    "required_memory_labels": ["defense"],
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_write_runner_replays_and_snapshots(tmp_path: Path) -> None:
    store = MemoryStore2(tmp_path / "memory2.db", vec_dim=3)
    consolidation = _Consolidation(store)
    engine = _Engine(store)
    rt = SimpleNamespace(
        core=SimpleNamespace(
            session_manager=_SessionManager(),
            memory_runtime=SimpleNamespace(engine=engine),
        ),
        consolidation=consolidation,
    )
    try:
        result = await run_write_case(rt, _case())
        assert consolidation.calls == 1
        assert len(result.state_diff.created) == 1
        assert result.label_to_item_id == {"defense": result.state_diff.created[0].id}
    finally:
        store.close()


@pytest.mark.asyncio
async def test_recall_runner_resolves_expected_label(tmp_path: Path) -> None:
    store = MemoryStore2(tmp_path / "memory2.db", vec_dim=3)
    engine = _Engine(store)
    rt = SimpleNamespace(core=SimpleNamespace(memory_runtime=SimpleNamespace(engine=engine)))
    try:
        results = await run_recall_probes(rt, _case(), {"defense": "m1"})
        assert results[0].ranked_ids == ["m1"]
        assert results[0].metrics["passed"] is True
    finally:
        store.close()

