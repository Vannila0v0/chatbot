from pathlib import Path

import pytest

from agent.memory import DEFAULT_SELF_MD
from agent.context import ContextBuilder
from agent.core.types import ContextRequest
from agent.looping.ports import MemoryServices
from agent.retrieval.default_pipeline import DefaultMemoryRetrievalPipeline
from agent.retrieval.protocol import RetrievalRequest
from core.memory.scope import (
    InvalidMemoryScopeError,
    MarkdownMemoryStoreResolver,
    parse_web_memory_scope,
)


USER_A = "550e8400-e29b-41d4-a716-446655440000"
USER_B = "5d260c61-ecd4-4c11-99bd-2856b9527f6b"


def test_non_web_session_uses_existing_personal_store(tmp_path: Path) -> None:
    resolver = MarkdownMemoryStoreResolver(tmp_path)

    store = resolver.store_for("telegram:123")

    assert store is resolver.default_store
    assert store.memory_dir == tmp_path / "memory"


def test_web_session_uses_canonical_tenant_directory(tmp_path: Path) -> None:
    resolver = MarkdownMemoryStoreResolver(tmp_path)

    store = resolver.store_for(f"web:{USER_A.upper()}:primary")

    assert store.memory_dir == tmp_path / "web_users" / USER_A / "memory"
    assert store.pending_file.exists()
    assert store.memory_file.exists()
    assert store.history_file.exists()
    assert store.recent_context_file.exists()
    assert store.read_self() == DEFAULT_SELF_MD
    assert store.journal_dir.is_dir()
    assert store._consolidation_db.exists()


def test_web_stores_are_cached_and_isolated(tmp_path: Path) -> None:
    resolver = MarkdownMemoryStoreResolver(tmp_path)

    store_a = resolver.store_for(f"web:{USER_A}:primary")
    same_store_a = resolver.store_for(f"web:{USER_A}:primary")
    store_b = resolver.store_for(f"web:{USER_B}:primary")
    store_a.write_long_term("user-a-memory")
    store_b.write_long_term("user-b-memory")

    assert same_store_a is store_a
    assert store_a is not store_b
    assert store_a.read_long_term() == "user-a-memory"
    assert store_b.read_long_term() == "user-b-memory"
    assert resolver.default_store.read_long_term() == ""


def test_context_builder_reads_markdown_for_current_web_user(tmp_path: Path) -> None:
    resolver = MarkdownMemoryStoreResolver(tmp_path)
    resolver.default_store.write_long_term("PERSONAL-SECRET")
    store_a = resolver.store_for(f"web:{USER_A}:primary")
    store_b = resolver.store_for(f"web:{USER_B}:primary")
    store_a.write_long_term("USER-A-LONG-TERM")
    store_a.write_self("USER-A-SELF")
    store_a.write_recent_context("USER-A-RECENT")
    store_b.write_long_term("USER-B-LONG-TERM")

    builder = ContextBuilder(
        tmp_path,
        memory=resolver.default_store,
        memory_resolver=resolver,
    )
    rendered = builder.render(
        ContextRequest(
            history=[],
            current_message="hello",
            session_key=f"web:{USER_A}:primary",
            channel="web",
            chat_id="primary",
        )
    )

    rendered_content = rendered.system_prompt + repr(rendered.messages)
    assert "USER-A-LONG-TERM" in rendered_content
    assert "USER-A-SELF" in rendered_content
    assert "USER-A-RECENT" in rendered_content
    assert "USER-B-LONG-TERM" not in rendered_content
    assert "PERSONAL-SECRET" not in rendered_content


def test_web_context_requires_session_key_when_resolver_is_enabled(
    tmp_path: Path,
) -> None:
    resolver = MarkdownMemoryStoreResolver(tmp_path)
    builder = ContextBuilder(
        tmp_path,
        memory=resolver.default_store,
        memory_resolver=resolver,
    )

    with pytest.raises(ValueError, match="requires a session key"):
        builder.render(
            ContextRequest(
                history=[],
                current_message="hello",
                channel="web",
                chat_id="primary",
            )
        )


@pytest.mark.parametrize(
    ("channel", "session_key"),
    [
        ("web", "telegram:123"),
        ("telegram", f"web:{USER_A}:primary"),
    ],
)
def test_context_builder_rejects_channel_session_mismatch(
    tmp_path: Path,
    channel: str,
    session_key: str,
) -> None:
    resolver = MarkdownMemoryStoreResolver(tmp_path)
    resolver.default_store.write_long_term("PERSONAL-SECRET")
    builder = ContextBuilder(
        tmp_path,
        memory=resolver.default_store,
        memory_resolver=resolver,
    )

    with pytest.raises(ValueError):
        builder.render(
            ContextRequest(
                history=[],
                current_message="hello",
                session_key=session_key,
                channel=channel,
                chat_id="primary",
            )
        )


@pytest.mark.asyncio
async def test_web_retrieval_does_not_query_unscoped_vector_engine() -> None:
    class _UnexpectedEngine:
        async def query(self, request: object) -> object:
            raise AssertionError(f"unexpected vector query: {request!r}")

    pipeline = DefaultMemoryRetrievalPipeline(
        memory=MemoryServices(engine=_UnexpectedEngine()),  # type: ignore[arg-type]
    )

    result = await pipeline.retrieve(
        RetrievalRequest(
            message="hello",
            session_key=f"web:{USER_A}:primary",
            channel="web",
            chat_id="primary",
            history=[],
            session_metadata={},
        )
    )

    assert result.block == ""
    assert result.trace is None
    assert result.metadata == {"disabled_reason": "web_vector_scope_pending"}


@pytest.mark.parametrize(
    "session_key",
    [
        "web:",
        "web:not-a-uuid:primary",
        "web:../../memory:primary",
        f"web:{USER_A}:secondary",
        f"web:{USER_A}:primary:extra",
    ],
)
def test_invalid_web_session_fails_closed(
    tmp_path: Path,
    session_key: str,
) -> None:
    resolver = MarkdownMemoryStoreResolver(tmp_path)

    with pytest.raises(InvalidMemoryScopeError):
        resolver.store_for(session_key)

    assert not (tmp_path / "web_users").exists()


def test_parser_ignores_non_web_sessions_and_normalizes_uuid() -> None:
    assert parse_web_memory_scope("telegram:123") is None

    scope = parse_web_memory_scope(f"web:{USER_A.upper()}:primary")

    assert scope is not None
    assert scope.user_id == USER_A
    assert scope.conversation_id == "primary"


def test_pending_discovery_ignores_invalid_and_empty_tenant_directories(
    tmp_path: Path,
) -> None:
    resolver = MarkdownMemoryStoreResolver(tmp_path)
    store_a = resolver.store_for(f"web:{USER_A}:primary")
    _ = resolver.store_for(f"web:{USER_B}:primary")
    store_a.append_pending("- [preference] pending-a")
    invalid = tmp_path / "web_users" / "not-a-uuid" / "memory"
    invalid.mkdir(parents=True)
    (invalid / "PENDING.md").write_text("should-not-load", encoding="utf-8")

    candidates = resolver.iter_web_stores_with_pending()

    assert [scope.user_id for scope, _ in candidates] == [USER_A]
