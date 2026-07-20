from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class WebTurnEventType(str, Enum):
    TURN_QUEUED = "turn.queued"
    TURN_STARTED = "turn.started"
    TURN_SNAPSHOT = "turn.snapshot"
    THINKING_DELTA = "thinking.delta"
    TEXT_DELTA = "text.delta"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TURN_COMPLETED = "turn.completed"
    TURN_FAILED = "turn.failed"
    TURN_CANCELLED = "turn.cancelled"


TERMINAL_WEB_TURN_EVENT_TYPES = frozenset(
    {
        WebTurnEventType.TURN_COMPLETED,
        WebTurnEventType.TURN_FAILED,
        WebTurnEventType.TURN_CANCELLED,
    }
)


def _empty_payload() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class WebTurnEvent:
    turn_id: str
    sequence: int
    type: WebTurnEventType
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=_empty_payload)

    @property
    def terminal(self) -> bool:
        return self.type in TERMINAL_WEB_TURN_EVENT_TYPES

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "sequence": self.sequence,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class WebTurnSnapshot:
    turn_id: str
    sequence: int = 0
    status: str = "pending"
    thinking: str = ""
    text: str = ""
    tools: tuple[dict[str, Any], ...] = ()

    def as_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "thinking": self.thinking,
            "text": self.text,
            "tools": [dict(item) for item in self.tools],
        }


def new_web_turn_event(
    *,
    turn_id: str,
    sequence: int,
    event_type: WebTurnEventType,
    payload: dict[str, Any] | None = None,
) -> WebTurnEvent:
    return WebTurnEvent(
        turn_id=turn_id,
        sequence=sequence,
        type=event_type,
        timestamp=datetime.now(timezone.utc),
        payload=dict(payload or {}),
    )
