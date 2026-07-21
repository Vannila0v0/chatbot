from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from web.api.identity import WEB_IDENTITY_COOKIE, WebIdentityService
from web.api.routes import create_turn_router
from web.events.broker import WebTurnEventBroker
from web.events.models import WebTurnEventType
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
def event_broker() -> WebTurnEventBroker:
    return WebTurnEventBroker()


@pytest.fixture
def identity() -> WebIdentityService:
    return WebIdentityService(b"test-web-identity-secret".ljust(32, b"!"))


@pytest.fixture
def app(
    repository,
    identity: WebIdentityService,
    event_broker: WebTurnEventBroker,
) -> FastAPI:
    application = FastAPI()
    application.include_router(create_turn_router(repository, identity, event_broker))
    return application


@pytest.fixture
def transport(app: FastAPI) -> httpx.ASGITransport:
    return httpx.ASGITransport(app=app)


def _payload(**overrides):
    payload = {
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
    assert body["conversation_id"] == "primary"
    assert body["client_request_id"] == "request-1"
    assert body["content"] == "hello"
    assert body["answer"] is None
    assert response.cookies.get(WEB_IDENTITY_COOKIE)


@pytest.mark.asyncio
async def test_list_turns_returns_only_current_user_in_order(
    transport: httpx.ASGITransport,
    repository,
) -> None:
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        first = await client.post("/api/turns", json=_payload())
        second = await client.post(
            "/api/turns",
            json=_payload(client_request_id="request-2", content="second"),
        )
        user_id = first.json()["user_id"]
        repository.create(
            user_id="another-user",
            conversation_id="primary",
            client_request_id="request-other",
            content="other",
        )
        response = await client.get(
            "/api/turns",
            params={"limit": 2},
        )

    assert response.status_code == 200
    assert [turn["id"] for turn in response.json()] == [
        first.json()["id"],
        second.json()["id"],
    ]
    assert all(turn["user_id"] == user_id for turn in response.json())


@pytest.mark.asyncio
async def test_list_turns_rejects_invalid_limit(
    transport: httpx.ASGITransport,
) -> None:
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/turns", params={"limit": 101})

    assert response.status_code == 422


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
async def test_turn_and_sse_are_hidden_from_another_web_user(
    transport: httpx.ASGITransport,
) -> None:
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as owner, httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as stranger:
        created = await owner.post("/api/turns", json=_payload())
        turn_id = created.json()["id"]

        turn_response = await stranger.get(f"/api/turns/{turn_id}")
        event_response = await stranger.get(f"/api/turns/{turn_id}/events")

    assert turn_response.status_code == 404
    assert event_response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
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


@pytest.mark.asyncio
async def test_sse_streams_queued_live_deltas_and_terminal_event(
    repository,
    event_broker: WebTurnEventBroker,
    transport: httpx.ASGITransport,
) -> None:
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        created = await client.post("/api/turns", json=_payload())
        turn_id = created.json()["id"]
        response_task = asyncio.create_task(
            client.get(f"/api/turns/{turn_id}/events")
        )
        for _ in range(100):
            if event_broker.subscriber_count(turn_id) == 1:
                break
            await asyncio.sleep(0.01)
        event_broker.publish(turn_id, WebTurnEventType.TURN_STARTED)
        event_broker.publish(
            turn_id,
            WebTurnEventType.THINKING_DELTA,
            {"delta": "thinking"},
        )
        event_broker.publish(
            turn_id,
            WebTurnEventType.TEXT_DELTA,
            {"delta": "answer"},
        )
        completed = repository.claim_next_pending()
        assert completed is not None
        repository.mark_done(turn_id, "answer")
        event_broker.publish(
            turn_id,
            WebTurnEventType.TURN_COMPLETED,
            {"answer": "answer"},
        )
        response = await asyncio.wait_for(response_task, timeout=2)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert "event: turn.queued" in response.text
    assert "event: turn.started" in response.text
    assert "event: thinking.delta" in response.text
    assert "event: text.delta" in response.text
    assert "event: turn.completed" in response.text
    assert response.text.index("event: thinking.delta") < response.text.index(
        "event: text.delta"
    )
    assert event_broker.subscriber_count(turn_id) == 0


@pytest.mark.asyncio
async def test_sse_returns_durable_terminal_state_immediately(
    repository,
    transport: httpx.ASGITransport,
) -> None:
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        created = await client.post("/api/turns", json=_payload())
        turn_id = created.json()["id"]
        repository.claim_next_pending()
        repository.mark_done(turn_id, "finished")
        response = await client.get(f"/api/turns/{turn_id}/events")

    assert response.status_code == 200
    assert "event: turn.completed" in response.text
    assert '"answer":"finished"' in response.text


@pytest.mark.asyncio
async def test_sse_uses_live_snapshot_for_late_subscriber(
    repository,
    event_broker: WebTurnEventBroker,
    transport: httpx.ASGITransport,
) -> None:
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        created = await client.post("/api/turns", json=_payload())
        turn_id = created.json()["id"]
        repository.claim_next_pending()
        event_broker.publish(turn_id, WebTurnEventType.TURN_STARTED)
        event_broker.publish(
            turn_id,
            WebTurnEventType.THINKING_DELTA,
            {"delta": "already thinking"},
        )
        response_task = asyncio.create_task(
            client.get(f"/api/turns/{turn_id}/events")
        )
        for _ in range(100):
            if event_broker.subscriber_count(turn_id) == 1:
                break
            await asyncio.sleep(0.01)
        repository.mark_done(turn_id, "done")
        event_broker.publish(
            turn_id,
            WebTurnEventType.TURN_COMPLETED,
            {"answer": "done"},
        )
        response = await asyncio.wait_for(response_task, timeout=2)

    assert "event: turn.snapshot" in response.text
    assert "already thinking" in response.text
    assert "event: turn.completed" in response.text


@pytest.mark.asyncio
async def test_sse_missing_turn_returns_not_found(
    transport: httpx.ASGITransport,
) -> None:
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/turns/missing/events")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "turn_not_found"
