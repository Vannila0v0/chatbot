from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class TurnStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Turn:
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
