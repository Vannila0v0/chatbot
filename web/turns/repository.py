from __future__ import annotations

from typing import Protocol

from web.turns.models import Turn, TurnStatus


class TurnRepositoryError(RuntimeError):
    """Base error for persistent Turn operations."""


class TurnNotFoundError(TurnRepositoryError):
    def __init__(self, turn_id: str) -> None:
        self.turn_id = turn_id
        super().__init__(f"Turn not found: {turn_id}")


class InvalidTurnTransitionError(TurnRepositoryError):
    def __init__(
        self,
        *,
        turn_id: str,
        current: TurnStatus,
        expected: TurnStatus,
        target: TurnStatus,
    ) -> None:
        self.turn_id = turn_id
        self.current = current
        self.expected = expected
        self.target = target
        super().__init__(
            f"Cannot transition Turn {turn_id} from {current.value} to "
            f"{target.value}; expected {expected.value}"
        )


class IdempotencyConflictError(TurnRepositoryError):
    def __init__(self, *, user_id: str, client_request_id: str) -> None:
        self.user_id = user_id
        self.client_request_id = client_request_id
        super().__init__(
            "The client request ID is already associated with a different payload: "
            f"user={user_id} request={client_request_id}"
        )


class TurnRepository(Protocol):
    def create(
        self,
        *,
        user_id: str,
        conversation_id: str,
        client_request_id: str,
        content: str,
    ) -> Turn: ...

    def get(self, turn_id: str) -> Turn | None: ...

    def get_for_user(self, turn_id: str, user_id: str) -> Turn | None: ...

    def list_for_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str,
        limit: int = 50,
    ) -> list[Turn]: ...

    def claim_next_pending(self) -> Turn | None: ...

    def mark_done(self, turn_id: str, answer: str) -> Turn: ...

    def mark_failed(
        self,
        turn_id: str,
        error_code: str,
        error_message: str,
    ) -> Turn: ...

    def cancel(self, turn_id: str) -> Turn: ...

    def close(self) -> None: ...
