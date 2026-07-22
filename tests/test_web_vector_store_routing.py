from pathlib import Path

import pytest

from core.memory.scope import InvalidMemoryScopeError
from plugins.default_memory.store_resolver import MemoryStore2Resolver


USER_A = "550e8400-e29b-41d4-a716-446655440000"
USER_B = "5d260c61-ecd4-4c11-99bd-2856b9527f6b"


def _resolver(tmp_path: Path) -> MemoryStore2Resolver:
    return MemoryStore2Resolver(
        tmp_path,
        default_db_path=tmp_path / "memory" / "memory2.db",
        vec_dim=2,
    )


def test_non_web_session_uses_personal_vector_store(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    try:
        store = resolver.store_for("telegram:123")

        assert store is resolver.default_store
        assert store.db_path == tmp_path / "memory" / "memory2.db"
    finally:
        resolver.close()


def test_web_sessions_use_separate_vector_databases(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    try:
        store_a = resolver.store_for(f"web:{USER_A}:primary")
        store_b = resolver.store_for(f"web:{USER_B}:primary")

        assert store_a is not store_b
        assert store_a.db_path == (
            tmp_path / "web_users" / USER_A / "memory" / "memory2.db"
        )
        assert store_b.db_path == (
            tmp_path / "web_users" / USER_B / "memory" / "memory2.db"
        )
        assert store_a.db_path.exists()
        assert store_b.db_path.exists()
    finally:
        resolver.close()


def test_same_web_user_reuses_open_vector_store(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    try:
        first = resolver.store_for(f"web:{USER_A}:primary")
        second = resolver.store_for(f"web:{USER_A.upper()}:primary")

        assert second is first
    finally:
        resolver.close()


def test_equal_memories_are_isolated_between_web_users(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    try:
        store_a = resolver.store_for(f"web:{USER_A}:primary")
        store_b = resolver.store_for(f"web:{USER_B}:primary")

        result_a = store_a.upsert_item("preference", "likes tea", [1.0, 0.0])
        result_b = store_b.upsert_item("preference", "likes tea", [1.0, 0.0])

        assert result_a.startswith("new:")
        assert result_b.startswith("new:")
        items_a, total_a = store_a.list_items_for_dashboard()
        items_b, total_b = store_b.list_items_for_dashboard()
        personal_items, personal_total = (
            resolver.default_store.list_items_for_dashboard()
        )
        assert total_a == 1
        assert total_b == 1
        assert items_a[0]["reinforcement"] == 1
        assert items_b[0]["reinforcement"] == 1
        assert personal_items == []
        assert personal_total == 0
    finally:
        resolver.close()


@pytest.mark.parametrize(
    "session_key",
    [
        "web:not-a-uuid:primary",
        "web:../../memory:primary",
        f"web:{USER_A}:secondary",
        f"web:{USER_A}:primary:extra",
    ],
)
def test_invalid_web_session_does_not_create_vector_database(
    tmp_path: Path,
    session_key: str,
) -> None:
    resolver = _resolver(tmp_path)
    try:
        with pytest.raises(InvalidMemoryScopeError):
            resolver.store_for(session_key)

        assert not (tmp_path / "web_users").exists()
    finally:
        resolver.close()


def test_closed_resolver_rejects_new_store_requests(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    resolver.close()

    with pytest.raises(RuntimeError, match="resolver is closed"):
        resolver.store_for(f"web:{USER_A}:primary")
