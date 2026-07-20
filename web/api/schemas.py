from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, StringConstraints

from web.turns.models import Turn, TurnStatus

RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CreateTurnRequest(BaseModel):
    user_id: RequiredText
    conversation_id: RequiredText
    client_request_id: RequiredText
    content: RequiredText


class TurnResponse(BaseModel):
    id: str
    user_id: str
    conversation_id: str
    client_request_id: str
    content: str
    status: TurnStatus
    answer: str | None
    error_code: str | None
    error_message: str | None
    attempts: int
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def from_turn(cls, turn: Turn) -> "TurnResponse":
        return cls(
            id=turn.id,
            user_id=turn.user_id,
            conversation_id=turn.conversation_id,
            client_request_id=turn.client_request_id,
            content=turn.content,
            status=turn.status,
            answer=turn.answer,
            error_code=turn.error_code,
            error_message=turn.error_message,
            attempts=turn.attempts,
            created_at=turn.created_at,
            updated_at=turn.updated_at,
            started_at=turn.started_at,
            finished_at=turn.finished_at,
        )
