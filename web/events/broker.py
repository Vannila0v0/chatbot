from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, cast

from web.events.models import (
    TERMINAL_WEB_TURN_EVENT_TYPES,
    WebTurnEvent,
    WebTurnEventType,
    WebTurnSnapshot,
    new_web_turn_event,
)

_CLOSED = object()
_THINKING_SNAPSHOT_LIMIT = 4_000
_TEXT_SNAPSHOT_LIMIT = 20_000


class WebTurnSubscription:
    def __init__(
        self,
        broker: "WebTurnEventBroker",
        turn_id: str,
        queue: asyncio.Queue[WebTurnEvent | object],
        snapshot: WebTurnSnapshot | None,
    ) -> None:
        self._broker = broker
        self.turn_id = turn_id
        self._queue = queue
        self.snapshot = snapshot
        self._closed = False

    async def receive(self) -> WebTurnEvent | None:
        item = await self._queue.get()
        if item is _CLOSED:
            return None
        return cast(WebTurnEvent, item)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._broker.unsubscribe(self.turn_id, self._queue)


class WebTurnEventBroker:
    """In-process ordered broadcast for live Web turn events."""

    def __init__(self) -> None:
        self._subscribers: dict[
            str, set[asyncio.Queue[WebTurnEvent | object]]
        ] = {}
        self._sequences: dict[str, int] = {}
        self._snapshots: dict[str, WebTurnSnapshot] = {}
        self._closed = False

    def publish(
        self,
        turn_id: str,
        event_type: WebTurnEventType,
        payload: dict[str, Any] | None = None,
    ) -> WebTurnEvent:
        if self._closed:
            raise RuntimeError("Web turn event broker is closed")
        normalized_turn_id = str(turn_id or "").strip()
        if not normalized_turn_id:
            raise ValueError("turn_id must not be blank")

        sequence = self._sequences.get(normalized_turn_id, 0) + 1
        self._sequences[normalized_turn_id] = sequence
        event = new_web_turn_event(
            turn_id=normalized_turn_id,
            sequence=sequence,
            event_type=event_type,
            payload=payload,
        )
        self._snapshots[normalized_turn_id] = _apply_event(
            self._snapshots.get(normalized_turn_id),
            event,
        )
        for queue in tuple(self._subscribers.get(normalized_turn_id, set())):
            queue.put_nowait(event)

        if event_type in TERMINAL_WEB_TURN_EVENT_TYPES:
            _ = self._snapshots.pop(normalized_turn_id, None)
            _ = self._sequences.pop(normalized_turn_id, None)
        return event

    def subscribe(self, turn_id: str) -> WebTurnSubscription:
        if self._closed:
            raise RuntimeError("Web turn event broker is closed")
        normalized_turn_id = str(turn_id or "").strip()
        if not normalized_turn_id:
            raise ValueError("turn_id must not be blank")
        queue: asyncio.Queue[WebTurnEvent | object] = asyncio.Queue()
        self._subscribers.setdefault(normalized_turn_id, set()).add(queue)
        return WebTurnSubscription(
            self,
            normalized_turn_id,
            queue,
            self._snapshots.get(normalized_turn_id),
        )

    def unsubscribe(
        self,
        turn_id: str,
        queue: asyncio.Queue[WebTurnEvent | object],
    ) -> None:
        subscribers = self._subscribers.get(turn_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            _ = self._subscribers.pop(turn_id, None)

    def subscriber_count(self, turn_id: str) -> int:
        return len(self._subscribers.get(turn_id, set()))

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        for subscribers in self._subscribers.values():
            for queue in subscribers:
                queue.put_nowait(_CLOSED)
        self._subscribers.clear()
        self._sequences.clear()
        self._snapshots.clear()


def _apply_event(
    snapshot: WebTurnSnapshot | None,
    event: WebTurnEvent,
) -> WebTurnSnapshot:
    current = snapshot or WebTurnSnapshot(turn_id=event.turn_id)
    payload = event.payload
    if event.type is WebTurnEventType.TURN_STARTED:
        return replace(current, sequence=event.sequence, status="processing")
    if event.type is WebTurnEventType.THINKING_DELTA:
        delta = str(payload.get("delta") or "")
        return replace(
            current,
            sequence=event.sequence,
            thinking=(current.thinking + delta)[-_THINKING_SNAPSHOT_LIMIT:],
        )
    if event.type is WebTurnEventType.TEXT_DELTA:
        delta = str(payload.get("delta") or "")
        return replace(
            current,
            sequence=event.sequence,
            text=(current.text + delta)[-_TEXT_SNAPSHOT_LIMIT:],
        )
    if event.type is WebTurnEventType.TOOL_STARTED:
        tools = [dict(item) for item in current.tools]
        tools.append(
            {
                "call_id": str(payload.get("call_id") or ""),
                "tool_name": str(payload.get("tool_name") or ""),
                "iteration": int(payload.get("iteration") or 0),
                "status": "running",
            }
        )
        return replace(current, sequence=event.sequence, tools=tuple(tools))
    if event.type is WebTurnEventType.TOOL_COMPLETED:
        call_id = str(payload.get("call_id") or "")
        tools = [dict(item) for item in current.tools]
        for tool in tools:
            if tool.get("call_id") == call_id:
                tool["status"] = str(payload.get("status") or "done")
                break
        return replace(current, sequence=event.sequence, tools=tuple(tools))
    if event.type is WebTurnEventType.TURN_COMPLETED:
        return replace(
            current,
            sequence=event.sequence,
            status="done",
            text=str(payload.get("answer") or current.text),
        )
    if event.type is WebTurnEventType.TURN_FAILED:
        return replace(current, sequence=event.sequence, status="failed")
    if event.type is WebTurnEventType.TURN_CANCELLED:
        return replace(current, sequence=event.sequence, status="cancelled")
    return replace(current, sequence=event.sequence)
