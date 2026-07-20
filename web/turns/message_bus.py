from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

from bus.events import InboundItem, InboundMessage, OutboundMessage
from web.turns.repository import TurnRepository

WEB_CHANNEL = "web"

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

    def __init__(self, repository: TurnRepository, bus: WebTurnBus) -> None:
        self._repository = repository
        self._bus = bus
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
            _ = self._repository.mark_failed(
                turn.id,
                error_code="dispatch_error",
                error_message=str(exc),
            )
        return True


class WebTurnCompletionHandler:
    """Persists correlated Web channel responses as completed turns."""

    def __init__(self, repository: TurnRepository) -> None:
        self._repository = repository

    def subscribe(self, bus: WebTurnBus) -> None:
        bus.subscribe_outbound(WEB_CHANNEL, self.handle)

    async def handle(self, message: OutboundMessage) -> None:
        if message.channel != WEB_CHANNEL:
            return
        turn_id = str((message.metadata or {}).get("turn_id") or "").strip()
        if not turn_id:
            return
        _ = self._repository.mark_done(turn_id, message.content)
