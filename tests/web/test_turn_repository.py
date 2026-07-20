from __future__ import annotations

from pathlib import Path

import pytest

from web.turns.models import TurnStatus
from web.turns.repository import (
    IdempotencyConflictError,
    InvalidTurnTransitionError,
)
from web.turns.sqlite_repository import SQLiteTurnRepository


@pytest.fixture
def repository(tmp_path: Path):
    store = SQLiteTurnRepository(tmp_path / "web.db")
    try:
        yield store
    finally:
        store.close()


def test_create_turn_is_persisted_with_pending_status(repository) -> None:
    turn = repository.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-1",
        content="hello",
    )

    stored = repository.get(turn.id)

    assert stored == turn
    assert stored is not None
    assert stored.status is TurnStatus.PENDING
    assert stored.answer is None
    assert stored.error_code is None
    assert stored.error_message is None
    assert stored.attempts == 0
    assert stored.started_at is None
    assert stored.finished_at is None


def test_reopen_preserves_turn(tmp_path: Path) -> None:
    db_path = tmp_path / "web.db"
    first = SQLiteTurnRepository(db_path)
    turn = first.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-1",
        content="persist me",
    )
    first.close()

    second = SQLiteTurnRepository(db_path)
    try:
        assert second.get(turn.id) == turn
    finally:
        second.close()


def test_create_is_idempotent_per_user_and_client_request(repository) -> None:
    first = repository.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-1",
        content="hello",
    )

    duplicate = repository.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-1",
        content="hello",
    )
    other_user = repository.create(
        user_id="user-2",
        conversation_id="conversation-2",
        client_request_id="request-1",
        content="hello",
    )

    assert duplicate == first
    assert other_user.id != first.id


def test_reusing_idempotency_key_with_different_payload_is_rejected(repository) -> None:
    repository.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-1",
        content="first payload",
    )

    with pytest.raises(IdempotencyConflictError):
        repository.create(
            user_id="user-1",
            conversation_id="conversation-1",
            client_request_id="request-1",
            content="different payload",
        )


def test_claim_skips_conversation_that_is_already_processing(repository) -> None:
    first = repository.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-1",
        content="first",
    )
    repository.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-2",
        content="second",
    )
    other_conversation = repository.create(
        user_id="user-1",
        conversation_id="conversation-2",
        client_request_id="request-3",
        content="parallel",
    )

    claimed_first = repository.claim_next_pending()
    claimed_parallel = repository.claim_next_pending()

    assert claimed_first is not None
    assert claimed_first.id == first.id
    assert claimed_first.status is TurnStatus.PROCESSING
    assert claimed_first.attempts == 1
    assert claimed_first.started_at is not None
    assert claimed_parallel is not None
    assert claimed_parallel.id == other_conversation.id


def test_done_turn_releases_next_turn_in_same_conversation(repository) -> None:
    first = repository.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-1",
        content="first",
    )
    second = repository.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-2",
        content="second",
    )

    assert repository.claim_next_pending().id == first.id
    completed = repository.mark_done(first.id, answer="finished")
    claimed_second = repository.claim_next_pending()

    assert completed.status is TurnStatus.DONE
    assert completed.answer == "finished"
    assert completed.finished_at is not None
    assert claimed_second is not None
    assert claimed_second.id == second.id


def test_mark_failed_records_structured_error(repository) -> None:
    turn = repository.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-1",
        content="hello",
    )
    repository.claim_next_pending()

    failed = repository.mark_failed(
        turn.id,
        error_code="agent_timeout",
        error_message="Agent did not finish in time",
    )

    assert failed.status is TurnStatus.FAILED
    assert failed.error_code == "agent_timeout"
    assert failed.error_message == "Agent did not finish in time"
    assert failed.finished_at is not None


def test_pending_turn_can_be_cancelled(repository) -> None:
    turn = repository.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-1",
        content="hello",
    )

    cancelled = repository.cancel(turn.id)

    assert cancelled.status is TurnStatus.CANCELLED
    assert cancelled.finished_at is not None
    assert repository.claim_next_pending() is None


@pytest.mark.parametrize(
    ("terminal_action", "terminal_status"),
    [
        (lambda store, turn_id: store.mark_done(turn_id, "ok"), TurnStatus.DONE),
        (
            lambda store, turn_id: store.mark_failed(turn_id, "error", "failed"),
            TurnStatus.FAILED,
        ),
    ],
)
def test_terminal_turn_cannot_transition_again(
    repository,
    terminal_action,
    terminal_status: TurnStatus,
) -> None:
    turn = repository.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-1",
        content="hello",
    )
    repository.claim_next_pending()
    terminal = terminal_action(repository, turn.id)
    assert terminal.status is terminal_status

    with pytest.raises(InvalidTurnTransitionError):
        repository.mark_done(turn.id, answer="second result")


def test_processing_turn_cannot_be_cancelled(repository) -> None:
    turn = repository.create(
        user_id="user-1",
        conversation_id="conversation-1",
        client_request_id="request-1",
        content="hello",
    )
    repository.claim_next_pending()

    with pytest.raises(InvalidTurnTransitionError):
        repository.cancel(turn.id)
