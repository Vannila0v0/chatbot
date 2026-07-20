from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import HTTPException, Request, status
from fastapi.responses import StreamingResponse

from web.events.broker import WebTurnEventBroker, WebTurnSubscription
from web.events.models import (
    WebTurnEvent,
    WebTurnEventType,
    WebTurnSnapshot,
    new_web_turn_event,
)
from web.turns.models import Turn, TurnStatus
from web.turns.repository import TurnRepository

_SSE_HEARTBEAT_SECONDS = 15.0


def create_turn_event_response(
    *,
    turn_id: str,
    request: Request,
    repository: TurnRepository,
    broker: WebTurnEventBroker,
) -> StreamingResponse:
    # Subscribe before reading durable state so a transition cannot fall in between.
    subscription = broker.subscribe(turn_id)
    turn = repository.get(turn_id)
    if turn is None:
        subscription.close()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "turn_not_found",
                "message": f"Turn not found: {turn_id}",
            },
        )
    initial = _initial_event(turn, subscription.snapshot)
    return StreamingResponse(
        _stream_turn_events(request, subscription, initial),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_turn_events(
    request: Request,
    subscription: WebTurnSubscription,
    initial: WebTurnEvent,
) -> AsyncIterator[str]:
    try:
        yield encode_sse_event(initial)
        if initial.terminal:
            return
        while True:
            if await request.is_disconnected():
                return
            try:
                event = await asyncio.wait_for(
                    subscription.receive(),
                    timeout=_SSE_HEARTBEAT_SECONDS,
                )
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
                continue
            if event is None:
                return
            yield encode_sse_event(event)
            if event.terminal:
                return
    finally:
        subscription.close()


def encode_sse_event(event: WebTurnEvent) -> str:
    data = json.dumps(event.as_dict(), ensure_ascii=False, separators=(",", ":"))
    return (
        f"id: {event.turn_id}:{event.sequence}\n"
        f"event: {event.type.value}\n"
        f"data: {data}\n\n"
    )


def _initial_event(
    turn: Turn,
    snapshot: WebTurnSnapshot | None,
) -> WebTurnEvent:
    sequence = snapshot.sequence if snapshot is not None else 0
    if turn.status is TurnStatus.DONE:
        event_type = WebTurnEventType.TURN_COMPLETED
        payload = {"answer": turn.answer or ""}
        sequence += 1
    elif turn.status is TurnStatus.FAILED:
        event_type = WebTurnEventType.TURN_FAILED
        payload = {
            "error_code": turn.error_code or "turn_failed",
            "error_message": turn.error_message or "Turn failed",
        }
        sequence += 1
    elif turn.status is TurnStatus.CANCELLED:
        event_type = WebTurnEventType.TURN_CANCELLED
        payload = {}
        sequence += 1
    elif snapshot is not None:
        event_type = WebTurnEventType.TURN_SNAPSHOT
        payload = snapshot.as_payload()
    elif turn.status is TurnStatus.PROCESSING:
        event_type = WebTurnEventType.TURN_STARTED
        payload = {}
    else:
        event_type = WebTurnEventType.TURN_QUEUED
        payload = {}
    return new_web_turn_event(
        turn_id=turn.id,
        sequence=sequence,
        event_type=event_type,
        payload=payload,
    )
