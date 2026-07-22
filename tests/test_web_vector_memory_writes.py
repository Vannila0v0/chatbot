from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from agent.config_models import Config, MemoryConfig
from core.memory.events import ConsolidationCommitted, TurnIngested
from core.memory.scope import InvalidMemoryScopeError
from plugins.default_memory.config import DefaultMemoryConfig
from plugins.default_memory.engine import DefaultMemoryEngine


USER_A = "550e8400-e29b-41d4-a716-446655440000"
USER_B = "5d260c61-ecd4-4c11-99bd-2856b9527f6b"
VECTOR = [1.0, *([0.0] * 1023)]


def _engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider_responses: list[str],
) -> tuple[DefaultMemoryEngine, AsyncMock]:
    chat = AsyncMock(
        side_effect=[
            SimpleNamespace(content=response)
            for response in provider_responses
        ]
    )
    provider = cast(Any, SimpleNamespace(chat=chat))
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
    return engine, chat


def _event(
    session_key: str,
    *,
    history_entry: str,
    source_ref: str = '["m1", "m2"]',
) -> ConsolidationCommitted:
    channel = "web" if session_key.startswith("web:") else "telegram"
    chat_id = "primary" if channel == "web" else "123"
    return ConsolidationCommitted(
        session_key=session_key,
        history_entry_payloads=[(history_entry, 3)],
        source_ref=source_ref,
        scope_channel=channel,
        scope_chat_id=chat_id,
        conversation="USER: test conversation",
    )


def _items(engine: DefaultMemoryEngine, session_key: str) -> list[dict[str, object]]:
    assert engine._store_resolver is not None
    store = engine._store_resolver.store_for(session_key)
    items, _ = store.list_items_for_dashboard(page_size=200)
    return items


def _close(engine: DefaultMemoryEngine) -> None:
    if engine._store_resolver is not None:
        engine._store_resolver.close()


@pytest.mark.asyncio
async def test_web_consolidation_writes_and_deduplicates_per_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_implicit = '{"profile":[],"preference":[],"procedure":[]}'
    engine, _ = _engine(
        tmp_path,
        monkeypatch,
        provider_responses=[empty_implicit, empty_implicit, empty_implicit],
    )
    key_a = f"web:{USER_A}:primary"
    key_b = f"web:{USER_B}:primary"
    event_a = _event(key_a, history_entry="[2026-07-22 10:00] likes tea")
    event_b = _event(key_b, history_entry="[2026-07-22 10:00] likes tea")
    try:
        await engine._on_consolidation_committed(event_a)
        await engine._on_consolidation_committed(event_b)
        await engine._on_consolidation_committed(event_a)

        items_a = _items(engine, key_a)
        items_b = _items(engine, key_b)
        personal_items = _items(engine, "telegram:123")
        assert len(items_a) == 1
        assert len(items_b) == 1
        assert items_a[0]["memory_type"] == "event"
        assert items_b[0]["memory_type"] == "event"
        assert items_a[0]["reinforcement"] == 1
        assert items_b[0]["reinforcement"] == 1
        assert personal_items == []
    finally:
        _close(engine)


@pytest.mark.asyncio
async def test_web_consolidation_writes_implicit_types_to_tenant_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implicit = (
        '{"profile":[{"summary":"uses Linux","category":"personal_fact"}],'
        '"preference":[{"summary":"prefers concise replies"}],'
        '"procedure":[{"summary":"search before answering",'
        '"tool_requirement":"web_search","steps":["search"]}]}'
    )
    engine, _ = _engine(
        tmp_path,
        monkeypatch,
        provider_responses=[implicit],
    )
    key_a = f"web:{USER_A}:primary"
    try:
        await engine._on_consolidation_committed(
            _event(key_a, history_entry="")
        )

        items = _items(engine, key_a)
        assert {item["memory_type"] for item in items} == {
            "profile",
            "preference",
            "procedure",
        }
        assert _items(engine, "telegram:123") == []
    finally:
        _close(engine)


@pytest.mark.asyncio
async def test_non_web_consolidation_still_writes_personal_vector_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_implicit = '{"profile":[],"preference":[],"procedure":[]}'
    engine, _ = _engine(
        tmp_path,
        monkeypatch,
        provider_responses=[empty_implicit],
    )
    try:
        await engine._on_consolidation_committed(
            _event(
                "telegram:123",
                history_entry="[2026-07-22 10:00] personal event",
            )
        )

        personal_items = _items(engine, "telegram:123")
        assert len(personal_items) == 1
        assert personal_items[0]["summary"] == "[2026-07-22 10:00] personal event"
        assert not (tmp_path / "web_users").exists()
    finally:
        _close(engine)


@pytest.mark.asyncio
async def test_web_post_response_supersedes_only_current_users_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, chat = _engine(
        tmp_path,
        monkeypatch,
        provider_responses=[],
    )
    key_a = f"web:{USER_A}:primary"
    key_b = f"web:{USER_B}:primary"
    assert engine._store_resolver is not None
    store_a = engine._store_resolver.store_for(key_a)
    store_b = engine._store_resolver.store_for(key_b)
    personal_store = engine._store_resolver.default_store
    result_a = store_a.upsert_item(
        "preference", "response style", VECTOR, source_ref="a"
    )
    result_b = store_b.upsert_item(
        "preference", "response style", VECTOR, source_ref="b"
    )
    result_personal = personal_store.upsert_item(
        "preference", "response style", VECTOR, source_ref="personal"
    )
    item_a = result_a.split(":", 1)[1]
    item_b = result_b.split(":", 1)[1]
    item_personal = result_personal.split(":", 1)[1]
    chat.side_effect = [
        SimpleNamespace(content='["response style"]'),
        SimpleNamespace(content=f'["{item_a}"]'),
    ]
    try:
        await engine._on_turn_ingested(
            TurnIngested(
                session_key=key_a,
                channel="web",
                chat_id="primary",
                user_message="That response style is wrong. Forget it.",
                assistant_response="Understood.",
                tool_chain=[],
                source_ref=f"{key_a}@post_response",
            )
        )

        stored_a = store_a.get_item_for_dashboard(item_a)
        stored_b = store_b.get_item_for_dashboard(item_b)
        stored_personal = personal_store.get_item_for_dashboard(item_personal)
        assert stored_a is not None and stored_a["status"] == "superseded"
        assert stored_b is not None and stored_b["status"] == "active"
        assert stored_personal is not None and stored_personal["status"] == "active"
    finally:
        _close(engine)


@pytest.mark.asyncio
async def test_invalid_web_scope_never_falls_back_to_personal_vector_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, chat = _engine(
        tmp_path,
        monkeypatch,
        provider_responses=[],
    )
    try:
        with pytest.raises(InvalidMemoryScopeError):
            await engine._on_consolidation_committed(
                ConsolidationCommitted(
                    session_key="telegram:123",
                    history_entry_payloads=[("must not be written", 0)],
                    source_ref='["bad"]',
                    scope_channel="web",
                    scope_chat_id="primary",
                    conversation="USER: attack",
                )
            )

        assert _items(engine, "telegram:123") == []
        chat.assert_not_awaited()
    finally:
        _close(engine)
