import type { TurnEvent, TurnResponse } from "./types";

interface CreateTurnInput {
  userId: string;
  conversationId: string;
  clientRequestId: string;
  content: string;
}

export async function createTurn(input: CreateTurnInput): Promise<TurnResponse> {
  const response = await fetch("/api/turns", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: input.userId,
      conversation_id: input.conversationId,
      client_request_id: input.clientRequestId,
      content: input.content,
    }),
  });
  if (!response.ok) {
    throw new Error(await responseError(response));
  }
  return response.json() as Promise<TurnResponse>;
}

export async function getTurn(turnId: string): Promise<TurnResponse> {
  const response = await fetch(`/api/turns/${encodeURIComponent(turnId)}`);
  if (!response.ok) {
    throw new Error(await responseError(response));
  }
  return response.json() as Promise<TurnResponse>;
}

export function parseTurnEvent(event: MessageEvent<string>): TurnEvent {
  return JSON.parse(event.data) as TurnEvent;
}

async function responseError(response: Response): Promise<string> {
  const payload = await response.json().catch(() => null) as {
    detail?: string | { message?: string };
  } | null;
  if (typeof payload?.detail === "string") return payload.detail;
  if (payload?.detail?.message) return payload.detail.message;
  return `请求失败（${response.status}）`;
}
