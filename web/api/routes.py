from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from web.api.schemas import CreateTurnRequest, TurnResponse
from web.turns.repository import IdempotencyConflictError, TurnRepository


def create_turn_router(repository: TurnRepository) -> APIRouter:
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

    return router
