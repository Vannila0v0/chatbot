from __future__ import annotations

from bus.event_bus import EventBus
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from web.events.broker import WebTurnEventBroker
from web.events.models import WebTurnEventType
from web.turns.message_bus import WEB_CHANNEL


class WebTurnEventBridge:
    """Maps existing Agent lifecycle events to the public Web turn protocol."""

    def __init__(self, broker: WebTurnEventBroker) -> None:
        self._broker = broker
        self._turn_by_session: dict[str, str] = {}

    def subscribe(self, event_bus: EventBus) -> None:
        event_bus.on(TurnStarted, self.on_turn_started)
        event_bus.on(StreamDeltaReady, self.on_stream_delta)
        event_bus.on(ToolCallStarted, self.on_tool_started)
        event_bus.on(ToolCallCompleted, self.on_tool_completed)

    def on_turn_started(self, event: TurnStarted) -> None:
        if event.channel != WEB_CHANNEL or not event.turn_id:
            return
        self._turn_by_session[event.session_key] = event.turn_id
        _ = self._broker.publish(event.turn_id, WebTurnEventType.TURN_STARTED)

    def on_stream_delta(self, event: StreamDeltaReady) -> None:
        turn_id = self._web_turn_id(event.channel, event.session_key)
        if turn_id is None:
            return
        if event.thinking_delta:
            _ = self._broker.publish(
                turn_id,
                WebTurnEventType.THINKING_DELTA,
                {"delta": event.thinking_delta},
            )
        if event.content_delta:
            _ = self._broker.publish(
                turn_id,
                WebTurnEventType.TEXT_DELTA,
                {"delta": event.content_delta},
            )

    def on_tool_started(self, event: ToolCallStarted) -> None:
        turn_id = self._web_turn_id(event.channel, event.session_key)
        if turn_id is None:
            return
        _ = self._broker.publish(
            turn_id,
            WebTurnEventType.TOOL_STARTED,
            {
                "call_id": event.call_id,
                "tool_name": event.tool_name,
                "iteration": event.iteration,
                "status": "running",
            },
        )

    def on_tool_completed(self, event: ToolCallCompleted) -> None:
        turn_id = self._web_turn_id(event.channel, event.session_key)
        if turn_id is None:
            return
        _ = self._broker.publish(
            turn_id,
            WebTurnEventType.TOOL_COMPLETED,
            {
                "call_id": event.call_id,
                "tool_name": event.tool_name,
                "iteration": event.iteration,
                "status": "error" if event.status == "error" else "done",
            },
        )

    def forget_turn(self, turn_id: str) -> None:
        stale_sessions = [
            session_key
            for session_key, mapped_turn_id in self._turn_by_session.items()
            if mapped_turn_id == turn_id
        ]
        for session_key in stale_sessions:
            _ = self._turn_by_session.pop(session_key, None)

    def mapped_turn_id(self, session_key: str) -> str | None:
        return self._turn_by_session.get(session_key)

    def _web_turn_id(self, channel: str, session_key: str) -> str | None:
        if channel != WEB_CHANNEL:
            return None
        return self._turn_by_session.get(session_key)
