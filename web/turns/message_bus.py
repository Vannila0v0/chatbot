from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

from bus.events import InboundItem, InboundMessage, OutboundMessage
from web.turns.repository import TurnRepository
from web.events.broker import WebTurnEventBroker
from web.events.models import WebTurnEventType

if TYPE_CHECKING:
    from web.events.bridge import WebTurnEventBridge

WEB_CHANNEL = "web"


def web_session_key(user_id: str) -> str:
    return f"web:{user_id}:primary"


logger = logging.getLogger(__name__)


class WebTurnBus(Protocol):
    async def publish_inbound(self, msg: InboundItem) -> None: ...

    def subscribe_outbound(
        self,
        channel: str,
        callback: Callable[[OutboundMessage], Awaitable[None]],
    ) -> None: ...


class WebTurnDispatcher:
    """Claims one persisted Web turn and publishes it to the Agent message bus."""

    def __init__(
        self,
        repository: TurnRepository,
        bus: WebTurnBus,
        event_broker: WebTurnEventBroker | None = None,
    ) -> None:
        self._repository = repository
        self._bus = bus
        self._event_broker = event_broker
        self._running = False

    async def run(self, poll_interval_seconds: float = 0.25) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero")
        self._running = True
        while self._running:
            if not await self.run_once():
                await asyncio.sleep(poll_interval_seconds)

    def stop(self) -> None:
        self._running = False

    async def run_once(self) -> bool:
        turn = self._repository.claim_next_pending()
        if turn is None:
            return False

        inbound = InboundMessage(
            channel=WEB_CHANNEL,
            sender=turn.user_id,
            chat_id=turn.conversation_id,
            content=turn.content,
            session_key_override=web_session_key(turn.user_id),
            metadata={
                "turn_id": turn.id,
                "user_id": turn.user_id,
                "client_request_id": turn.client_request_id,
            },
        )
        try:
            await self._bus.publish_inbound(inbound)
        except Exception as exc:
            logger.exception("Failed to dispatch Web turn %s", turn.id)
            failed = self._repository.mark_failed(
                turn.id,
                error_code="dispatch_error",
                error_message=str(exc),
            )
            if self._event_broker is not None:
                _publish_event_safely(
                    self._event_broker,
                    failed.id,
                    WebTurnEventType.TURN_FAILED,
                    {
                        "error_code": failed.error_code,
                        "error_message": failed.error_message,
                    },
                )
        return True


class WebTurnCompletionHandler:
    """Persists correlated Web channel responses as completed turns."""

    def __init__(
        self,
        repository: TurnRepository,
        event_broker: WebTurnEventBroker | None = None,
        event_bridge: "WebTurnEventBridge | None" = None,
    ) -> None:
        self._repository = repository
        self._event_broker = event_broker
        self._event_bridge = event_bridge

    def subscribe(self, bus: WebTurnBus) -> None:
        bus.subscribe_outbound(WEB_CHANNEL, self.handle)

    async def handle(self, message: OutboundMessage) -> None:
        if message.channel != WEB_CHANNEL:
            return
        turn_id = str((message.metadata or {}).get("turn_id") or "").strip()
        if not turn_id:
            return
        completed = self._repository.mark_done(turn_id, message.content)
        if self._event_broker is not None:
            _publish_event_safely(
                self._event_broker,
                completed.id,
                WebTurnEventType.TURN_COMPLETED,
                {"answer": completed.answer or ""},
            )
        if self._event_bridge is not None:
            self._event_bridge.forget_turn(completed.id)


def _publish_event_safely(
    broker: WebTurnEventBroker,
    turn_id: str,
    event_type: WebTurnEventType,
    payload: dict[str, object],
) -> None:
    try:
        _ = broker.publish(turn_id, event_type, payload)
    except RuntimeError:
        logger.info(
            "Skipped live Web turn event after broker shutdown: turn_id=%s type=%s",
            turn_id,
            event_type.value,
        )
