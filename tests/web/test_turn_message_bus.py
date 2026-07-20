from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.looping.core import AgentLoop
from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from web.turns.message_bus import WebTurnCompletionHandler, WebTurnDispatcher
from web.turns.models import TurnStatus
from web.turns.sqlite_repository import SQLiteTurnRepository


class _FakeBus:
    def __init__(self) -> None:
        self.inbound: list[InboundMessage] = []
        self.subscriptions = {}
        self.publish_error: Exception | None = None

    async def publish_inbound(self, msg) -> None:
        if self.publish_error is not None:
            raise self.publish_error
        self.inbound.append(msg)

    def subscribe_outbound(self, channel, callback) -> None:
        self.subscriptions[channel] = callback


@pytest.fixture
def repository(tmp_path: Path):
    store = SQLiteTurnRepository(tmp_path / "web.db")
    try:
        yield store
    finally:
        store.close()


@pytest.mark.asyncio
async def test_dispatcher_returns_false_when_no_turn_is_pending(repository) -> None:
    bus = _FakeBus()

    dispatched = await WebTurnDispatcher(repository, bus).run_once()

    assert dispatched is False
    assert bus.inbound == []


@pytest.mark.asyncio
async def test_dispatcher_claims_and_publishes_correlated_web_turn(repository) -> None:
    turn = repository.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-1",
        content="hello",
    )
    bus = _FakeBus()

    dispatched = await WebTurnDispatcher(repository, bus).run_once()

    assert dispatched is True
    assert len(bus.inbound) == 1
    message = bus.inbound[0]
    assert message.channel == "web"
    assert message.sender == "user-1"
    assert message.chat_id == "conversation-1"
    assert message.content == "hello"
    assert message.metadata == {
        "turn_id": turn.id,
        "user_id": "user-1",
        "client_request_id": "request-1",
    }
    stored = repository.get(turn.id)
    assert stored is not None
    assert stored.status is TurnStatus.PROCESSING


@pytest.mark.asyncio
async def test_dispatch_failure_marks_claimed_turn_failed(repository) -> None:
    turn = repository.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-1",
        content="hello",
    )
    bus = _FakeBus()
    bus.publish_error = RuntimeError("bus unavailable")

    dispatched = await WebTurnDispatcher(repository, bus).run_once()

    assert dispatched is True
    stored = repository.get(turn.id)
    assert stored is not None
    assert stored.status is TurnStatus.FAILED
    assert stored.error_code == "dispatch_error"
    assert stored.error_message == "bus unavailable"


@pytest.mark.asyncio
async def test_completion_handler_subscribes_and_persists_answer(repository) -> None:
    turn = repository.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-1",
        content="hello",
    )
    repository.claim_next_pending()
    bus = _FakeBus()
    handler = WebTurnCompletionHandler(repository)
    handler.subscribe(bus)

    await bus.subscriptions["web"](
        OutboundMessage(
            channel="web",
            chat_id="conversation-1",
            content="hi there",
            metadata={"turn_id": turn.id},
        )
    )

    stored = repository.get(turn.id)
    assert stored is not None
    assert stored.status is TurnStatus.DONE
    assert stored.answer == "hi there"


@pytest.mark.asyncio
async def test_completion_handler_ignores_uncorrelated_web_message(repository) -> None:
    turn = repository.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-1",
        content="hello",
    )
    repository.claim_next_pending()

    await WebTurnCompletionHandler(repository).handle(
        OutboundMessage(
            channel="web",
            chat_id="conversation-1",
            content="proactive message",
        )
    )

    stored = repository.get(turn.id)
    assert stored is not None
    assert stored.status is TurnStatus.PROCESSING


@pytest.mark.asyncio
async def test_agent_loop_error_response_preserves_turn_id() -> None:
    bus = MessageBus()
    loop = object.__new__(AgentLoop)
    loop.bus = bus
    loop._llm_config = SimpleNamespace(max_iterations=1)
    loop._active_turn_states = {}
    loop._active_tasks = {}
    loop._build_initial_turn_state = MagicMock(return_value=MagicMock())
    loop._process = AsyncMock(side_effect=RuntimeError("agent failed"))
    received: list[OutboundMessage] = []
    completed = asyncio.Event()

    async def capture(message: OutboundMessage) -> None:
        received.append(message)
        completed.set()
        loop.stop()
        bus.stop()

    bus.subscribe_outbound("web", capture)
    loop_task = asyncio.create_task(loop.run())
    outbound_task = asyncio.create_task(bus.dispatch_outbound())
    await bus.publish_inbound(
        InboundMessage(
            channel="web",
            sender="user-1",
            chat_id="conversation-1",
            content="hello",
            metadata={"turn_id": "turn-1"},
        )
    )

    await asyncio.wait_for(completed.wait(), timeout=2)
    await asyncio.wait_for(loop_task, timeout=2)
    await asyncio.wait_for(outbound_task, timeout=2)

    assert len(received) == 1
    assert received[0].metadata["turn_id"] == "turn-1"
