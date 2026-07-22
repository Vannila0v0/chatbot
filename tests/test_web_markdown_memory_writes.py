from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted
from core.memory.events import ConsolidationCommitted
from core.memory.markdown import (
    ConsolidateRequest,
    MarkdownMemoryMaintenance,
    MarkdownMemoryStore,
    MemoryLifecycleBindRequest,
    RefreshRecentTurnsRequest,
    _ConsolidationDraft,
    _ConsolidationWindow,
)
from core.memory.scope import InvalidMemoryScopeError, MarkdownMemoryStoreResolver


USER_A = "550e8400-e29b-41d4-a716-446655440000"
USER_B = "5d260c61-ecd4-4c11-99bd-2856b9527f6b"


def _session(key: str, marker: str, *, message_count: int = 2) -> SimpleNamespace:
    messages: list[dict[str, object]] = []
    for index in range(message_count):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append(
            {
                "id": f"{key}:{index + 1}",
                "role": role,
                "content": f"{marker}-{role}-{index}",
                "timestamp": "2026-07-22T10:00:00+08:00",
            }
        )
    return SimpleNamespace(key=key, messages=messages, last_consolidated=0)


def _turn_committed(
    session_key: str,
    *,
    channel: str = "web",
) -> TurnCommitted:
    return TurnCommitted(
        session_key=session_key,
        channel=channel,
        chat_id="primary",
        input_message="hello",
        persisted_user_message="hello",
        assistant_response="hi",
        tools_used=[],
    )


async def _drain_maintenance(maintenance: MarkdownMemoryMaintenance) -> None:
    for _ in range(10):
        tasks = list(maintenance._maintenance_tasks.values())
        if not tasks:
            return
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)


def _maintenance(
    tmp_path: Path,
    *,
    event_bus: EventBus | None = None,
    keep_count: int = 20,
) -> tuple[
    MarkdownMemoryMaintenance,
    MarkdownMemoryStoreResolver,
    MarkdownMemoryStore,
]:
    default_store = MarkdownMemoryStore(tmp_path)
    resolver = MarkdownMemoryStoreResolver(tmp_path, default_store=default_store)
    maintenance = MarkdownMemoryMaintenance(
        store=default_store,
        provider=cast(Any, SimpleNamespace()),
        model="lm",
        keep_count=keep_count,
        event_bus=event_bus,
        resolver=resolver,
    )
    return maintenance, resolver, default_store


@pytest.mark.asyncio
async def test_turn_committed_refreshes_each_web_recent_context_separately(
    tmp_path: Path,
) -> None:
    event_bus = EventBus()
    maintenance, resolver, default_store = _maintenance(
        tmp_path,
        event_bus=event_bus,
    )
    key_a = f"web:{USER_A}:primary"
    key_b = f"web:{USER_B}:primary"
    sessions = {
        key_a: _session(key_a, "USER-A"),
        key_b: _session(key_b, "USER-B"),
    }
    maintenance.bind_lifecycle(
        MemoryLifecycleBindRequest(
            get_session=sessions.__getitem__,
            save_session=AsyncMock(),
        )
    )

    event_bus.enqueue(_turn_committed(key_a))
    event_bus.enqueue(_turn_committed(key_b))
    await event_bus.drain()
    await _drain_maintenance(maintenance)

    recent_a = resolver.store_for(key_a).read_recent_context()
    recent_b = resolver.store_for(key_b).read_recent_context()
    assert "USER-A" in recent_a
    assert "USER-B" not in recent_a
    assert "USER-B" in recent_b
    assert "USER-A" not in recent_b
    assert default_store.read_recent_context() == ""
    await event_bus.aclose()


@pytest.mark.asyncio
async def test_web_consolidation_reads_and_writes_only_tenant_store(
    tmp_path: Path,
) -> None:
    event_bus = EventBus()
    vector_events: list[ConsolidationCommitted] = []
    event_bus.on(ConsolidationCommitted, vector_events.append)
    maintenance, resolver, default_store = _maintenance(
        tmp_path,
        event_bus=event_bus,
        keep_count=2,
    )
    key_a = f"web:{USER_A}:primary"
    key_b = f"web:{USER_B}:primary"
    store_a = resolver.store_for(key_a)
    store_b = resolver.store_for(key_b)
    session_a = _session(key_a, "USER-A")
    draft = _ConsolidationDraft(
        window=_ConsolidationWindow(
            old_messages=list(session_a.messages),
            keep_count=0,
            consolidate_up_to=2,
        ),
        source_ref=f'["{key_a}:1", "{key_a}:2"]',
        history_entry_payloads=[("[2026-07-22 10:00] USER-A-EVENT", 4)],
        pending_items="- [preference] USER-A-PREFERENCE",
        conversation="USER: USER-A",
        recent_context_text="# Recent Context\n\nUSER-A-RECENT\n",
        scope_channel="web",
        scope_chat_id="primary",
    )
    maintenance._worker.prepare_consolidation = AsyncMock(return_value=draft)

    await maintenance.consolidate(ConsolidateRequest(session=session_a))
    await maintenance.consolidate(ConsolidateRequest(session=session_a))

    prepare_call = maintenance._worker.prepare_consolidation.await_args_list[0]
    assert prepare_call.kwargs["profile_maint"] is store_a
    assert store_a.read_history().count("USER-A-EVENT") == 1
    assert store_a.read_pending().count("USER-A-PREFERENCE") == 1
    assert "USER-A-RECENT" in store_a.read_recent_context()
    assert "USER-A-EVENT" in (
        store_a.journal_dir / "2026-07-22.md"
    ).read_text(encoding="utf-8")
    assert store_b.read_history() == ""
    assert store_b.read_pending() == ""
    assert default_store.read_history() == ""
    assert default_store.read_pending() == ""
    assert session_a.last_consolidated == 2
    assert vector_events == []
    await event_bus.aclose()


@pytest.mark.asyncio
async def test_non_web_consolidation_keeps_personal_store_and_vector_event(
    tmp_path: Path,
) -> None:
    event_bus = EventBus()
    vector_events: list[ConsolidationCommitted] = []
    event_bus.on(ConsolidationCommitted, vector_events.append)
    maintenance, resolver, default_store = _maintenance(
        tmp_path,
        event_bus=event_bus,
        keep_count=2,
    )
    session = _session("telegram:123", "PERSONAL")
    draft = _ConsolidationDraft(
        window=_ConsolidationWindow(
            old_messages=list(session.messages),
            keep_count=0,
            consolidate_up_to=2,
        ),
        source_ref='["telegram:123:1", "telegram:123:2"]',
        history_entry_payloads=[("[2026-07-22 10:00] PERSONAL-EVENT", 0)],
        pending_items="",
        conversation="USER: PERSONAL",
        recent_context_text="# Recent Context\n",
        scope_channel="telegram",
        scope_chat_id="123",
    )
    maintenance._worker.prepare_consolidation = AsyncMock(return_value=draft)

    await maintenance.consolidate(ConsolidateRequest(session=session))

    prepare_call = maintenance._worker.prepare_consolidation.await_args
    assert prepare_call is not None
    assert prepare_call.kwargs["profile_maint"] is default_store
    assert "PERSONAL-EVENT" in default_store.read_history()
    assert len(vector_events) == 1
    assert vector_events[0].scope_channel == "telegram"
    assert resolver.store_for("telegram:123") is default_store
    await event_bus.aclose()


@pytest.mark.asyncio
async def test_web_partial_write_retries_without_advancing_or_duplicating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maintenance, resolver, _ = _maintenance(tmp_path, keep_count=2)
    key_a = f"web:{USER_A}:primary"
    store_a = resolver.store_for(key_a)
    session_a = _session(key_a, "USER-A")
    draft = _ConsolidationDraft(
        window=_ConsolidationWindow(
            old_messages=list(session_a.messages),
            keep_count=0,
            consolidate_up_to=2,
        ),
        source_ref=f'["{key_a}:1", "{key_a}:2"]',
        history_entry_payloads=[("[2026-07-22 10:00] RETRY-EVENT", 0)],
        pending_items="- [preference] RETRY-PREFERENCE",
        conversation="USER: RETRY",
        recent_context_text="# Recent Context\n",
        scope_channel="web",
        scope_chat_id="primary",
    )
    maintenance._worker.prepare_consolidation = AsyncMock(return_value=draft)
    real_append_pending = store_a.append_pending_once
    attempts = 0

    def _flaky_append_pending(*args: object, **kwargs: object) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary write failure")
        return real_append_pending(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(store_a, "append_pending_once", _flaky_append_pending)

    with pytest.raises(OSError, match="temporary write failure"):
        await maintenance.consolidate(ConsolidateRequest(session=session_a))

    assert session_a.last_consolidated == 0
    assert store_a.read_history().count("RETRY-EVENT") == 1

    await maintenance.consolidate(ConsolidateRequest(session=session_a))

    assert session_a.last_consolidated == 2
    assert store_a.read_history().count("RETRY-EVENT") == 1
    assert store_a.read_pending().count("RETRY-PREFERENCE") == 1


@pytest.mark.asyncio
async def test_invalid_web_session_never_falls_back_to_personal_write(
    tmp_path: Path,
) -> None:
    maintenance, _, default_store = _maintenance(tmp_path)
    session = _session("web:../../memory:primary", "ATTACK")

    with pytest.raises(InvalidMemoryScopeError):
        await maintenance.refresh_recent_turns(
            RefreshRecentTurnsRequest(session=session)
        )

    assert default_store.read_recent_context() == ""


@pytest.mark.asyncio
async def test_mismatched_turn_event_never_reaches_personal_store(
    tmp_path: Path,
) -> None:
    event_bus = EventBus()
    maintenance, _, default_store = _maintenance(
        tmp_path,
        event_bus=event_bus,
    )
    personal_session = _session("telegram:123", "PERSONAL")
    maintenance.bind_lifecycle(
        MemoryLifecycleBindRequest(
            get_session=lambda _key: personal_session,
            save_session=AsyncMock(),
        )
    )

    event_bus.enqueue(_turn_committed("telegram:123", channel="web"))
    await event_bus.drain()
    await _drain_maintenance(maintenance)

    assert default_store.read_recent_context() == ""
    await event_bus.aclose()
