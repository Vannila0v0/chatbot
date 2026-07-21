export type TurnStatus = "pending" | "processing" | "done" | "failed" | "cancelled";

export interface TurnResponse {
  id: string;
  user_id: string;
  conversation_id: string;
  client_request_id: string;
  content: string;
  status: TurnStatus;
  answer: string | null;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
}

export type TurnEventType =
  | "turn.queued"
  | "turn.started"
  | "turn.snapshot"
  | "thinking.delta"
  | "text.delta"
  | "tool.started"
  | "tool.completed"
  | "turn.completed"
  | "turn.failed"
  | "turn.cancelled";

export interface TurnEvent<T extends Record<string, unknown> = Record<string, unknown>> {
  turn_id: string;
  sequence: number;
  type: TurnEventType;
  timestamp: string;
  payload: T;
}

export interface ToolProgress {
  callId: string;
  name: string;
  status: "running" | "done" | "error";
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  thinking: string;
  tools: ToolProgress[];
  status: TurnStatus;
  error: string | null;
}
