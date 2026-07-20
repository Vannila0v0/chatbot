from __future__ import annotations

import asyncio

import pytest

from web.events.broker import WebTurnEventBroker
from web.events.models import WebTurnEventType


@pytest.mark.asyncio
async def test_broker_broadcasts_ordered_events_to_every_subscriber() -> None:
    broker = WebTurnEventBroker()
    first = broker.subscribe("turn-1")
    second = broker.subscribe("turn-1")

    started = broker.publish("turn-1", WebTurnEventType.TURN_STARTED)
    delta = broker.publish(
        "turn-1",
        WebTurnEventType.TEXT_DELTA,
        {"delta": "hello"},
    )

    assert started.sequence == 1
    assert delta.sequence == 2
    assert await first.receive() == started
    assert await first.receive() == delta
    assert await second.receive() == started
    assert await second.receive() == delta
    first.close()
    second.close()
    assert broker.subscriber_count("turn-1") == 0


def test_subscription_captures_latest_bounded_snapshot() -> None:
    broker = WebTurnEventBroker()
    broker.publish("turn-1", WebTurnEventType.TURN_STARTED)
    broker.publish(
        "turn-1",
        WebTurnEventType.THINKING_DELTA,
        {"delta": "x" * 4_100},
    )
    broker.publish(
        "turn-1",
        WebTurnEventType.TEXT_DELTA,
        {"delta": "answer"},
    )
    subscription = broker.subscribe("turn-1")

    assert subscription.snapshot is not None
    assert subscription.snapshot.status == "processing"
    assert subscription.snapshot.sequence == 3
    assert len(subscription.snapshot.thinking) == 4_000
    assert subscription.snapshot.text == "answer"
    subscription.close()


@pytest.mark.asyncio
async def test_terminal_event_is_delivered_and_durable_state_is_not_retained() -> None:
    broker = WebTurnEventBroker()
    subscription = broker.subscribe("turn-1")

    completed = broker.publish(
        "turn-1",
        WebTurnEventType.TURN_COMPLETED,
        {"answer": "done"},
    )

    assert completed.terminal is True
    assert await subscription.receive() == completed
    late = broker.subscribe("turn-1")
    assert late.snapshot is None
    subscription.close()
    late.close()


@pytest.mark.asyncio
async def test_broker_close_wakes_waiting_subscribers() -> None:
    broker = WebTurnEventBroker()
    subscription = broker.subscribe("turn-1")
    waiting = asyncio.create_task(subscription.receive())

    await broker.aclose()

    assert await asyncio.wait_for(waiting, timeout=1) is None
