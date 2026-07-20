from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from web.api.sse import create_turn_event_response
from web.api.schemas import CreateTurnRequest, TurnResponse
from web.events.broker import WebTurnEventBroker
from web.turns.repository import IdempotencyConflictError, TurnRepository


def create_turn_router(
    repository: TurnRepository,
    event_broker: WebTurnEventBroker | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/turns", tags=["web-turns"])

    @router.post("", response_model=TurnResponse, status_code=status.HTTP_202_ACCEPTED)
    def create_turn(request: CreateTurnRequest) -> TurnResponse:
        try:
            turn = repository.create(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                client_request_id=request.client_request_id,
                content=request.content,
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "idempotency_conflict",
                    "message": str(exc),
                },
            ) from exc
        return TurnResponse.from_turn(turn)

    @router.get("/{turn_id}", response_model=TurnResponse)
    def get_turn(turn_id: str) -> TurnResponse:
        turn = repository.get(turn_id)
        if turn is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "turn_not_found",
                    "message": f"Turn not found: {turn_id}",
                },
            )
        return TurnResponse.from_turn(turn)

    if event_broker is not None:

        @router.get("/{turn_id}/events", response_class=StreamingResponse)
        async def stream_turn_events(
            turn_id: str,
            request: Request,
        ) -> StreamingResponse:
            return create_turn_event_response(
                turn_id=turn_id,
                request=request,
                repository=repository,
                broker=event_broker,
            )

    return router
