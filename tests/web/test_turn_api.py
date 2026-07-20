from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from web.api.routes import create_turn_router
from web.turns.models import TurnStatus
from web.turns.sqlite_repository import SQLiteTurnRepository


@pytest.fixture
def repository(tmp_path: Path):
    store = SQLiteTurnRepository(tmp_path / "web.db")
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def app(repository) -> FastAPI:
    application = FastAPI()
    application.include_router(create_turn_router(repository))
    return application


@pytest.fixture
def transport(app: FastAPI) -> httpx.ASGITransport:
    return httpx.ASGITransport(app=app)


def _payload(**overrides):
    payload = {
        "user_id": "user-1",
        "conversation_id": "conversation-1",
        "client_request_id": "request-1",
        "content": "hello",
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_create_turn_returns_accepted_pending_turn(
    transport: httpx.ASGITransport,
) -> None:
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post("/api/turns", json=_payload())

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["user_id"] == "user-1"
    assert body["conversation_id"] == "conversation-1"
    assert body["client_request_id"] == "request-1"
    assert body["content"] == "hello"
    assert body["answer"] is None


@pytest.mark.asyncio
async def test_repeated_request_returns_same_turn(
    transport: httpx.ASGITransport,
) -> None:
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        first = await client.post("/api/turns", json=_payload())
        repeated = await client.post("/api/turns", json=_payload())

    assert first.status_code == 202
    assert repeated.status_code == 202
    assert repeated.json()["id"] == first.json()["id"]


@pytest.mark.asyncio
async def test_changed_payload_with_same_request_id_returns_conflict(
    transport: httpx.ASGITransport,
) -> None:
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        first = await client.post("/api/turns", json=_payload())
        conflict = await client.post(
            "/api/turns",
            json=_payload(content="different"),
        )

    assert first.status_code == 202
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_conflict"


@pytest.mark.asyncio
async def test_get_turn_returns_latest_persisted_state(
    transport: httpx.ASGITransport,
    repository,
) -> None:
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        created = await client.post("/api/turns", json=_payload())
        turn_id = created.json()["id"]
        claimed = repository.claim_next_pending()
        assert claimed is not None
        assert claimed.id == turn_id
        repository.mark_done(turn_id, "finished")

        response = await client.get(f"/api/turns/{turn_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == TurnStatus.DONE.value
    assert body["answer"] == "finished"
    assert body["attempts"] == 1
    assert body["started_at"] is not None
    assert body["finished_at"] is not None


@pytest.mark.asyncio
async def test_get_missing_turn_returns_not_found(
    transport: httpx.ASGITransport,
) -> None:
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/turns/missing")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "turn_not_found"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("user_id", ""),
        ("conversation_id", "   "),
        ("client_request_id", ""),
        ("content", "\n\t"),
    ],
)
async def test_create_turn_rejects_blank_required_fields(
    transport: httpx.ASGITransport,
    field: str,
    value: str,
) -> None:
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/turns",
            json=_payload(**{field: value}),
        )

    assert response.status_code == 422
