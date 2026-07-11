from __future__ import annotations

from typing import Protocol

from memory2.store import MemoryStore2

from .models import MemoryFixture


class EmbedderLike(Protocol):
    async def embed(self, text: str) -> list[float]: ...


async def seed_memories(
    store: MemoryStore2,
    embedder: EmbedderLike,
    memories: list[MemoryFixture],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for fixture in memories:
        embedding = await embedder.embed(fixture.summary)
        result = store.upsert_item(
            memory_type=fixture.memory_type,
            summary=fixture.summary,
            embedding=embedding,
            source_ref=f"eval-seed:{fixture.local_id}",
            extra=fixture.extra,
            happened_at=fixture.happened_at.isoformat() if fixture.happened_at else None,
            emotional_weight=fixture.emotional_weight,
        )
        item_id = result.split(":", 1)[1]
        store._db.execute(
            "UPDATE memory_items SET status=?, reinforcement=?, emotional_weight=? "
            "WHERE id=?",
            (
                fixture.status,
                fixture.reinforcement,
                fixture.emotional_weight,
                item_id,
            ),
        )
        store._db.commit()
        mapping[fixture.local_id] = item_id
    return mapping
