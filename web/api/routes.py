from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from web.api.identity import WebIdentityService
from web.api.sse import create_turn_event_response
from web.api.schemas import CreateTurnRequest, TurnResponse
from web.events.broker import WebTurnEventBroker
from web.turns.repository import IdempotencyConflictError, TurnRepository

PRIMARY_CONVERSATION_ID = "primary"


def create_turn_router(
    repository: TurnRepository,
    identity: WebIdentityService,
    event_broker: WebTurnEventBroker | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/turns", tags=["web-turns"])

    @router.get("", response_model=list[TurnResponse])
    def list_turns(
        request: Request,
        response: Response,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> list[TurnResponse]:
        principal = identity.resolve(request)
        identity.apply_cookie(response, request, principal)
        turns = repository.list_for_conversation(
            user_id=principal.user_id,
            conversation_id=PRIMARY_CONVERSATION_ID,
            limit=limit,
        )
        return [TurnResponse.from_turn(turn) for turn in turns]

    @router.post("", response_model=TurnResponse, status_code=status.HTTP_202_ACCEPTED)
    def create_turn(
        payload: CreateTurnRequest,
        request: Request,
        response: Response,
    ) -> TurnResponse:
        principal = identity.resolve(request)
        identity.apply_cookie(response, request, principal)
        try:
            turn = repository.create(
                user_id=principal.user_id,
                conversation_id=PRIMARY_CONVERSATION_ID,
                client_request_id=payload.client_request_id,
                content=payload.content,
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
    def get_turn(turn_id: str, request: Request, response: Response) -> TurnResponse:
        principal = identity.resolve(request)
        identity.apply_cookie(response, request, principal)
        turn = repository.get_for_user(turn_id, principal.user_id)
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
            principal = identity.resolve(request)
            response = create_turn_event_response(
                turn_id=turn_id,
                user_id=principal.user_id,
                request=request,
                repository=repository,
                broker=event_broker,
            )
            identity.apply_cookie(response, request, principal)
            return response

    return router
