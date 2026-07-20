from __future__ import annotations

from datetime import datetime

import pytest

from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from web.events.broker import WebTurnEventBroker
from web.events.bridge import WebTurnEventBridge
from web.events.models import WebTurnEventType


@pytest.mark.asyncio
async def test_bridge_correlates_and_orders_thinking_and_text_deltas() -> None:
    broker = WebTurnEventBroker()
    bridge = WebTurnEventBridge(broker)
    subscription = broker.subscribe("turn-1")
    bridge.on_turn_started(
        TurnStarted(
            session_key="web:conversation-1",
            channel="web",
            chat_id="conversation-1",
            content="hello",
            timestamp=datetime.now(),
            turn_id="turn-1",
        )
    )
    bridge.on_stream_delta(
        StreamDeltaReady(
            session_key="web:conversation-1",
            channel="web",
            chat_id="conversation-1",
            thinking_delta="thinking",
            content_delta="answer",
        )
    )

    events = [await subscription.receive() for _ in range(3)]

    assert [event.type for event in events if event is not None] == [
        WebTurnEventType.TURN_STARTED,
        WebTurnEventType.THINKING_DELTA,
        WebTurnEventType.TEXT_DELTA,
    ]
    assert [event.sequence for event in events if event is not None] == [1, 2, 3]
    assert events[1] is not None
    assert events[1].payload == {"delta": "thinking"}
    assert events[2] is not None
    assert events[2].payload == {"delta": "answer"}


@pytest.mark.asyncio
async def test_tool_events_are_correlated_without_sensitive_details() -> None:
    broker = WebTurnEventBroker()
    bridge = WebTurnEventBridge(broker)
    subscription = broker.subscribe("turn-1")
    bridge.on_turn_started(
        TurnStarted(
            session_key="web:conversation-1",
            channel="web",
            chat_id="conversation-1",
            content="hello",
            timestamp=datetime.now(),
            turn_id="turn-1",
        )
    )
    _ = await subscription.receive()

    bridge.on_tool_started(
        ToolCallStarted(
            session_key="web:conversation-1",
            channel="web",
            chat_id="conversation-1",
            iteration=2,
            call_id="call-1",
            tool_name="web_search",
            arguments={"api_key": "secret", "query": "private"},
        )
    )
    bridge.on_tool_completed(
        ToolCallCompleted(
            session_key="web:conversation-1",
            channel="web",
            chat_id="conversation-1",
            iteration=2,
            call_id="call-1",
            tool_name="web_search",
            arguments={"api_key": "secret"},
            final_arguments={"query": "private"},
            status="success",
            result_preview="sensitive result",
        )
    )

    started = await subscription.receive()
    completed = await subscription.receive()

    assert started is not None
    assert started.payload == {
        "call_id": "call-1",
        "tool_name": "web_search",
        "iteration": 2,
        "status": "running",
    }
    assert completed is not None
    assert completed.payload["status"] == "done"
    serialized = str(started.payload) + str(completed.payload)
    assert "secret" not in serialized
    assert "private" not in serialized
    assert "sensitive result" not in serialized


def test_bridge_ignores_other_channels_and_forgets_completed_turn() -> None:
    broker = WebTurnEventBroker()
    bridge = WebTurnEventBridge(broker)
    bridge.on_turn_started(
        TurnStarted(
            session_key="telegram:1",
            channel="telegram",
            chat_id="1",
            content="hello",
            timestamp=datetime.now(),
            turn_id="turn-1",
        )
    )
    assert bridge.mapped_turn_id("telegram:1") is None

    bridge.on_turn_started(
        TurnStarted(
            session_key="web:conversation-1",
            channel="web",
            chat_id="conversation-1",
            content="hello",
            timestamp=datetime.now(),
            turn_id="turn-1",
        )
    )
    assert bridge.mapped_turn_id("web:conversation-1") == "turn-1"
    bridge.forget_turn("turn-1")
    assert bridge.mapped_turn_id("web:conversation-1") is None
