from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from web.turns.models import Turn, TurnStatus
from web.turns.repository import (
    IdempotencyConflictError,
    InvalidTurnTransitionError,
    TurnNotFoundError,
)


class SQLiteTurnRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._db.row_factory = sqlite3.Row
        self._closed = False
        with self._lock:
            _ = self._db.execute("PRAGMA journal_mode=WAL")
            _ = self._db.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()

    def __del__(self) -> None:
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._db.close()

    def create(
        self,
        *,
        user_id: str,
        conversation_id: str,
        client_request_id: str,
        content: str,
    ) -> Turn:
        user_id = _required("user_id", user_id)
        conversation_id = _required("conversation_id", conversation_id)
        client_request_id = _required("client_request_id", client_request_id)
        if not content.strip():
            raise ValueError("content must not be blank")

        with self._lock:
            self._begin()
            try:
                existing = self._db.execute(
                    """
                    SELECT * FROM web_turns
                    WHERE user_id = ? AND client_request_id = ?
                    """,
                    (user_id, client_request_id),
                ).fetchone()
                if existing is not None:
                    turn = _row_to_turn(existing)
                    if (
                        turn.conversation_id != conversation_id
                        or turn.content != content
                    ):
                        raise IdempotencyConflictError(
                            user_id=user_id,
                            client_request_id=client_request_id,
                        )
                    self._db.commit()
                    return turn

                now = _utcnow()
                turn_id = str(uuid4())
                _ = self._db.execute(
                    """
                    INSERT INTO web_turns(
                        id, user_id, conversation_id, client_request_id,
                        content, status, attempts, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        turn_id,
                        user_id,
                        conversation_id,
                        client_request_id,
                        content,
                        TurnStatus.PENDING.value,
                        now,
                        now,
                    ),
                )
                row = self._get_row(turn_id)
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return _row_to_turn(_require_row(row, turn_id))

    def get(self, turn_id: str) -> Turn | None:
        with self._lock:
            row = self._get_row(turn_id)
        return _row_to_turn(row) if row is not None else None

    def list_for_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str,
        limit: int = 50,
    ) -> list[Turn]:
        user_id = _required("user_id", user_id)
        conversation_id = _required("conversation_id", conversation_id)
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        with self._lock:
            rows = self._db.execute(
                """
                SELECT * FROM web_turns
                WHERE user_id = ? AND conversation_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (user_id, conversation_id, limit),
            ).fetchall()
        return [_row_to_turn(row) for row in reversed(rows)]

    def claim_next_pending(self) -> Turn | None:
        with self._lock:
            self._begin()
            try:
                candidate = self._db.execute(
                    """
                    SELECT candidate.id
                    FROM web_turns AS candidate
                    WHERE candidate.status = ?
                      AND NOT EXISTS (
                          SELECT 1
                          FROM web_turns AS active
                          WHERE active.user_id = candidate.user_id
                            AND active.conversation_id = candidate.conversation_id
                            AND active.status = ?
                      )
                    ORDER BY candidate.created_at ASC, candidate.rowid ASC
                    LIMIT 1
                    """,
                    (TurnStatus.PENDING.value, TurnStatus.PROCESSING.value),
                ).fetchone()
                if candidate is None:
                    self._db.commit()
                    return None

                turn_id = str(candidate["id"])
                now = _utcnow()
                updated = self._db.execute(
                    """
                    UPDATE web_turns
                    SET status = ?, attempts = attempts + 1,
                        started_at = ?, updated_at = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        TurnStatus.PROCESSING.value,
                        now,
                        now,
                        turn_id,
                        TurnStatus.PENDING.value,
                    ),
                )
                if updated.rowcount != 1:
                    self._db.rollback()
                    return None
                row = self._get_row(turn_id)
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return _row_to_turn(_require_row(row, turn_id))

    def mark_done(self, turn_id: str, answer: str) -> Turn:
        return self._transition(
            turn_id=turn_id,
            expected=TurnStatus.PROCESSING,
            target=TurnStatus.DONE,
            answer=answer,
        )

    def mark_failed(
        self,
        turn_id: str,
        error_code: str,
        error_message: str,
    ) -> Turn:
        return self._transition(
            turn_id=turn_id,
            expected=TurnStatus.PROCESSING,
            target=TurnStatus.FAILED,
            error_code=error_code,
            error_message=error_message,
        )

    def cancel(self, turn_id: str) -> Turn:
        return self._transition(
            turn_id=turn_id,
            expected=TurnStatus.PENDING,
            target=TurnStatus.CANCELLED,
        )

    def _transition(
        self,
        *,
        turn_id: str,
        expected: TurnStatus,
        target: TurnStatus,
        answer: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> Turn:
        with self._lock:
            self._begin()
            try:
                current_row = self._get_row(turn_id)
                if current_row is None:
                    raise TurnNotFoundError(turn_id)
                current = TurnStatus(str(current_row["status"]))
                if current is not expected:
                    raise InvalidTurnTransitionError(
                        turn_id=turn_id,
                        current=current,
                        expected=expected,
                        target=target,
                    )

                now = _utcnow()
                _ = self._db.execute(
                    """
                    UPDATE web_turns
                    SET status = ?, answer = ?, error_code = ?,
                        error_message = ?, finished_at = ?, updated_at = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        target.value,
                        answer,
                        error_code,
                        error_message,
                        now,
                        now,
                        turn_id,
                        expected.value,
                    ),
                )
                row = self._get_row(turn_id)
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return _row_to_turn(_require_row(row, turn_id))

    def _init_schema(self) -> None:
        _ = self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS web_turns (
                id                TEXT PRIMARY KEY,
                user_id           TEXT NOT NULL,
                conversation_id   TEXT NOT NULL,
                client_request_id TEXT NOT NULL,
                content           TEXT NOT NULL,
                status            TEXT NOT NULL CHECK (
                    status IN ('pending', 'processing', 'done', 'failed', 'cancelled')
                ),
                answer            TEXT,
                error_code        TEXT,
                error_message     TEXT,
                attempts          INTEGER NOT NULL DEFAULT 0,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL,
                started_at        TEXT,
                finished_at       TEXT,
                UNIQUE(user_id, client_request_id)
            );

            CREATE INDEX IF NOT EXISTS idx_web_turns_pending
            ON web_turns(status, created_at);

            CREATE INDEX IF NOT EXISTS idx_web_turns_conversation
            ON web_turns(user_id, conversation_id, created_at);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_web_turns_one_processing
            ON web_turns(user_id, conversation_id)
            WHERE status = 'processing';
            """
        )

    def _begin(self) -> None:
        _ = self._db.execute("BEGIN IMMEDIATE")

    def _get_row(self, turn_id: str) -> sqlite3.Row | None:
        return self._db.execute(
            "SELECT * FROM web_turns WHERE id = ?",
            (turn_id,),
        ).fetchone()


def _row_to_turn(row: sqlite3.Row) -> Turn:
    return Turn(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        conversation_id=str(row["conversation_id"]),
        client_request_id=str(row["client_request_id"]),
        content=str(row["content"]),
        status=TurnStatus(str(row["status"])),
        answer=row["answer"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        attempts=int(row["attempts"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
        started_at=_optional_datetime(row["started_at"]),
        finished_at=_optional_datetime(row["finished_at"]),
    )


def _require_row(row: sqlite3.Row | None, turn_id: str) -> sqlite3.Row:
    if row is None:
        raise TurnNotFoundError(turn_id)
    return row


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(str(value))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required(name: str, value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{name} must not be blank")
    return normalized
