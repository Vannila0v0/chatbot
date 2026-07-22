from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from agent.config_models import Config, MemoryConfig
from core.memory.engine import MemoryQuery, MemoryQueryFilters, MemoryScope
from core.memory.scope import InvalidMemoryScopeError
from memory2.store import MemoryStore2
from plugins.default_memory.config import DefaultMemoryConfig
from plugins.default_memory.engine import DefaultMemoryEngine


USER_A = "550e8400-e29b-41d4-a716-446655440000"
USER_B = "5d260c61-ecd4-4c11-99bd-2856b9527f6b"
KEY_A = f"web:{USER_A}:primary"
KEY_B = f"web:{USER_B}:primary"
VECTOR = [1.0, *([0.0] * 1023)]


def _engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> DefaultMemoryEngine:
    provider = cast(
        Any,
        SimpleNamespace(chat=AsyncMock(return_value=SimpleNamespace(content=""))),
    )
    engine = DefaultMemoryEngine(
        config=Config(
            provider="test",
            model="test-model",
            api_key="test-key",
            system_prompt="test",
            memory=MemoryConfig(enabled=True),
        ),
        default_config=DefaultMemoryConfig(),
        workspace=tmp_path,
        provider=provider,
        light_provider=provider,
        http_resources=cast(
            Any,
            SimpleNamespace(external_default=SimpleNamespace()),
        ),
    )
    assert engine._embedder is not None
    monkeypatch.setattr(engine._embedder, "embed", AsyncMock(return_value=VECTOR))
    return engine


def _store(engine: DefaultMemoryEngine, session_key: str) -> MemoryStore2:
    assert engine._store_resolver is not None
    return engine._store_resolver.store_for(session_key)


def _scope(session_key: str) -> MemoryScope:
    if session_key.startswith("web:"):
        return MemoryScope(
            session_key=session_key,
            channel="web",
            chat_id="primary",
        )
    return MemoryScope(
        session_key=session_key,
        channel="telegram",
        chat_id="123",
    )


def _extra(channel: str, chat_id: str) -> dict[str, object]:
    return {"scope_channel": channel, "scope_chat_id": chat_id}


def _close(engine: DefaultMemoryEngine) -> None:
    if engine._store_resolver is not None:
        engine._store_resolver.close()


@pytest.mark.asyncio
async def test_web_hybrid_retrieval_and_injection_are_isolated_per_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(tmp_path, monkeypatch)
    store_a = _store(engine, KEY_A)
    store_b = _store(engine, KEY_B)
    personal = _store(engine, "telegram:123")
    store_a.upsert_item(
        "preference",
        "tenant-a likes alpha-coffee",
        VECTOR,
        source_ref="a",
        extra=_extra("web", "primary"),
    )
    store_b.upsert_item(
        "preference",
        "tenant-b likes alpha-coffee",
        VECTOR,
        source_ref="b",
        extra=_extra("web", "primary"),
    )
    personal.upsert_item(
        "preference",
        "personal likes alpha-coffee",
        VECTOR,
        source_ref="personal",
        extra=_extra("telegram", "123"),
    )
    try:
        result_a = await engine.query(
            MemoryQuery(
                text="alpha-coffee",
                intent="context",
                scope=_scope(KEY_A),
                filters=MemoryQueryFilters(kinds=("preference",)),
            )
        )
        result_b = await engine.query(
            MemoryQuery(
                text="alpha-coffee",
                intent="context",
                scope=_scope(KEY_B),
                filters=MemoryQueryFilters(kinds=("preference",)),
            )
        )

        assert "tenant-a" in result_a.text_block
        assert "tenant-b" not in result_a.text_block
        assert "personal" not in result_a.text_block
        assert "tenant-b" in result_b.text_block
        assert "tenant-a" not in result_b.text_block
        raw_a = cast(list[dict[str, object]], result_a.raw["items"])
        assert raw_a and float(raw_a[0]["rrf_score"]) > 0.02
    finally:
        _close(engine)


@pytest.mark.asyncio
async def test_web_keyword_only_retrieval_uses_tenant_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(tmp_path, monkeypatch)
    _store(engine, KEY_A).upsert_item(
        "preference",
        "tenant-a keyword-only zephyr-token",
        None,
        source_ref="a-keyword",
        extra=_extra("web", "primary"),
    )
    _store(engine, KEY_B).upsert_item(
        "preference",
        "tenant-b keyword-only zephyr-token",
        None,
        source_ref="b-keyword",
        extra=_extra("web", "primary"),
    )
    try:
        result = await engine.query(
            MemoryQuery(
                text="zephyr-token",
                intent="context",
                scope=_scope(KEY_A),
                filters=MemoryQueryFilters(kinds=("preference",)),
            )
        )

        assert [record.summary for record in result.records] == [
            "tenant-a keyword-only zephyr-token"
        ]
        assert "tenant-b" not in result.text_block
    finally:
        _close(engine)


@pytest.mark.asyncio
async def test_empty_web_store_never_falls_back_to_personal_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(tmp_path, monkeypatch)
    _store(engine, "telegram:123").upsert_item(
        "event",
        "personal-only fallback-marker",
        VECTOR,
        source_ref="personal",
        extra=_extra("telegram", "123"),
    )
    try:
        result = await engine.query(
            MemoryQuery(
                text="fallback-marker",
                intent="context",
                scope=_scope(KEY_A),
            )
        )

        assert result.records == []
        assert result.text_block == ""
    finally:
        _close(engine)


@pytest.mark.asyncio
async def test_web_timeline_reads_only_current_tenant_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(tmp_path, monkeypatch)
    for session_key, summary in (
        (KEY_A, "tenant-a timeline event"),
        (KEY_B, "tenant-b timeline event"),
        ("telegram:123", "personal timeline event"),
    ):
        channel = "web" if session_key.startswith("web:") else "telegram"
        chat_id = "primary" if channel == "web" else "123"
        _store(engine, session_key).upsert_item(
            "event",
            summary,
            VECTOR,
            source_ref=summary,
            extra=_extra(channel, chat_id),
            happened_at="2026-07-22T10:00:00+00:00",
        )
    try:
        result = await engine.query(
            MemoryQuery(
                text="",
                intent="timeline",
                scope=_scope(KEY_A),
                filters=MemoryQueryFilters(
                    time_start=datetime(2026, 7, 22, 9, tzinfo=timezone.utc),
                    time_end=datetime(2026, 7, 22, 11, tzinfo=timezone.utc),
                ),
            )
        )

        assert [record.summary for record in result.records] == [
            "tenant-a timeline event"
        ]
    finally:
        _close(engine)


@pytest.mark.parametrize(
    ("intent", "memory_type"),
    [
        ("answer", "event"),
        ("interest", "preference"),
        ("procedure", "procedure"),
    ],
)
@pytest.mark.asyncio
async def test_web_query_intents_use_tenant_retriever(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    intent: str,
    memory_type: str,
) -> None:
    engine = _engine(tmp_path, monkeypatch)
    marker = f"tenant-a {intent}-marker"
    _store(engine, KEY_A).upsert_item(
        memory_type,
        marker,
        VECTOR,
        source_ref=f"a-{intent}",
        extra=_extra("web", "primary"),
    )
    _store(engine, KEY_B).upsert_item(
        memory_type,
        f"tenant-b {intent}-marker",
        VECTOR,
        source_ref=f"b-{intent}",
        extra=_extra("web", "primary"),
    )
    try:
        result = await engine.query(
            MemoryQuery(
                text=f"{intent}-marker",
                intent=cast(Any, intent),
                scope=_scope(KEY_A),
                filters=MemoryQueryFilters(kinds=(memory_type,)),
            )
        )

        assert [record.summary for record in result.records] == [marker]
    finally:
        _close(engine)


@pytest.mark.asyncio
async def test_invalid_web_query_scope_fails_before_personal_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(tmp_path, monkeypatch)
    _store(engine, "telegram:123").upsert_item(
        "event",
        "must not leak",
        VECTOR,
        source_ref="personal",
    )
    try:
        with pytest.raises(InvalidMemoryScopeError):
            await engine.query(
                MemoryQuery(
                    text="must not leak",
                    intent="context",
                    scope=MemoryScope(
                        session_key="telegram:123",
                        channel="web",
                        chat_id="primary",
                    ),
                )
            )
    finally:
        _close(engine)


@pytest.mark.asyncio
async def test_malformed_web_query_scope_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(tmp_path, monkeypatch)
    try:
        with pytest.raises(InvalidMemoryScopeError):
            await engine.query(
                MemoryQuery(
                    text="must not leak",
                    intent="context",
                    scope=MemoryScope(
                        session_key="web:not-a-uuid:primary",
                        channel="web",
                        chat_id="primary",
                    ),
                )
            )
    finally:
        _close(engine)


@pytest.mark.asyncio
async def test_personal_retrieval_still_uses_default_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(tmp_path, monkeypatch)
    _store(engine, "telegram:123").upsert_item(
        "preference",
        "personal likes baseline-token",
        VECTOR,
        source_ref="personal",
        extra=_extra("telegram", "123"),
    )
    try:
        result = await engine.query(
            MemoryQuery(
                text="baseline-token",
                intent="context",
                scope=_scope("telegram:123"),
                filters=MemoryQueryFilters(kinds=("preference",)),
            )
        )

        assert "personal likes baseline-token" in result.text_block
        assert not (tmp_path / "web_users").exists()
    finally:
        _close(engine)
