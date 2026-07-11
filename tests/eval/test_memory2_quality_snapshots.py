from pathlib import Path

import pytest

from eval.memory2_quality.models import MemoryFixture
from eval.memory2_quality.seed import seed_memories
from eval.memory2_quality.snapshots import diff_snapshots, take_snapshot
from memory2.store import MemoryStore2


class _Embedder:
    async def embed(self, text: str) -> list[float]:
        return [float(len(text)), 0.2, 0.3]


def _add(store: MemoryStore2, summary: str, memory_type: str = "preference") -> str:
    result = store.upsert_item(
        memory_type=memory_type,
        summary=summary,
        embedding=[0.1, 0.2, 0.3],
        source_ref=f"src:{summary}",
        extra={},
    )
    return result.split(":", 1)[1]


def test_diff_snapshots_classifies_create_reinforce_and_supersede(
    tmp_path: Path,
) -> None:
    store = MemoryStore2(tmp_path / "memory2.db", vec_dim=3)
    try:
        reinforced_id = _add(store, "用户喜欢安静餐厅")
        superseded_id = _add(store, "用户喜欢爬山")
        unchanged_id = _add(store, "用户住在桂林", "profile")
        before = take_snapshot(store)

        store.reinforce_items_batch([reinforced_id])
        store.mark_superseded(superseded_id)
        created_id = _add(store, "用户目前不准备爬山")
        after = take_snapshot(store)

        diff = diff_snapshots(before, after)

        assert [item.id for item in diff.created] == [created_id]
        assert [item.id for item in diff.reinforced] == [reinforced_id]
        assert [item.id for item in diff.superseded] == [superseded_id]
        assert unchanged_id in [item.id for item in diff.unchanged]
        assert superseded_id not in [item.id for item in diff.active_after]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_seed_memories_returns_local_id_mapping(tmp_path: Path) -> None:
    store = MemoryStore2(tmp_path / "seed.db", vec_dim=3)
    try:
        mapping = await seed_memories(
            store,
            _Embedder(),
            [
                MemoryFixture(
                    local_id="old_pref",
                    memory_type="preference",
                    summary="用户喜欢爬山",
                    reinforcement=4,
                    emotional_weight=2,
                )
            ],
        )
        snapshot = take_snapshot(store)
        assert set(mapping) == {"old_pref"}
        assert snapshot[0].id == mapping["old_pref"]
        assert snapshot[0].reinforcement == 4
        assert snapshot[0].emotional_weight == 2
    finally:
        store.close()
