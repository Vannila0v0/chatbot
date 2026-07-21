import type {
  ChatMessage,
  ToolProgress,
  TurnEvent,
  TurnResponse,
  TurnStatus,
} from "./types";

export interface ChatState {
  messages: ChatMessage[];
  activeMessageId: string | null;
  activeTurnId: string | null;
  lastSequence: number;
  connection: "idle" | "connecting" | "open" | "retrying";
  hydrating: boolean;
  historyError: string | null;
}

export type ChatAction =
  | { type: "begin"; requestId: string; content: string }
  | { type: "accepted"; turnId: string }
  | { type: "hydrate"; turns: TurnResponse[] }
  | { type: "history_failed"; message: string }
  | { type: "event"; event: TurnEvent }
  | { type: "submit_failed"; message: string }
  | { type: "connection"; value: ChatState["connection"] };

export const initialChatState: ChatState = {
  messages: [],
  activeMessageId: null,
  activeTurnId: null,
  lastSequence: 0,
  connection: "idle",
  hydrating: true,
  historyError: null,
};

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  if (action.type === "begin") {
    return {
      ...state,
      messages: [
        ...state.messages,
        message(`${action.requestId}:user`, "user", action.content, "done"),
        message(action.requestId, "assistant", "", "pending"),
      ],
      activeMessageId: action.requestId,
      activeTurnId: null,
      lastSequence: 0,
      connection: "connecting",
      historyError: null,
    };
  }
  if (action.type === "accepted") {
    return { ...state, activeTurnId: action.turnId };
  }
  if (action.type === "hydrate") {
    const active = action.turns.find((turn) => turn.status === "processing")
      ?? action.turns.find((turn) => turn.status === "pending")
      ?? null;
    return {
      ...state,
      messages: action.turns.flatMap(messagesFromTurn),
      activeMessageId: active?.id ?? null,
      activeTurnId: active?.id ?? null,
      lastSequence: 0,
      connection: active ? "connecting" : "idle",
      hydrating: false,
      historyError: null,
    };
  }
  if (action.type === "history_failed") {
    return {
      ...state,
      hydrating: false,
      historyError: action.message,
    };
  }
  if (action.type === "connection") {
    return { ...state, connection: action.value };
  }
  if (action.type === "submit_failed") {
    return updateActive(state, (current) => ({
      ...current,
      status: "failed",
      error: action.message,
    }), true);
  }
  return applyTurnEvent(state, action.event);
}

export function applyTurnEvent(state: ChatState, event: TurnEvent): ChatState {
  if (!state.activeMessageId) return state;
  if (state.activeTurnId && event.turn_id !== state.activeTurnId) return state;
  const terminal = isTerminal(event.type);
  if (!terminal && event.sequence <= state.lastSequence) return state;

  const next = updateActive(state, (current) => {
    switch (event.type) {
      case "turn.queued":
        return { ...current, status: "pending" };
      case "turn.started":
        return { ...current, status: "processing" };
      case "turn.snapshot":
        return applySnapshot(current, event.payload);
      case "thinking.delta":
        return { ...current, thinking: current.thinking + textValue(event.payload.delta) };
      case "text.delta":
        return { ...current, content: current.content + textValue(event.payload.delta) };
      case "tool.started":
        return { ...current, tools: upsertTool(current.tools, event.payload, "running") };
      case "tool.completed":
        return {
          ...current,
          tools: upsertTool(
            current.tools,
            event.payload,
            event.payload.status === "error" ? "error" : "done",
          ),
        };
      case "turn.completed":
        return {
          ...current,
          content: textValue(event.payload.answer) || current.content,
          status: "done",
          error: null,
        };
      case "turn.failed":
        return {
          ...current,
          status: "failed",
          error: textValue(event.payload.error_message) || "本轮处理失败",
        };
      case "turn.cancelled":
        return { ...current, status: "cancelled", error: "本轮已取消" };
    }
  }, terminal);
  return {
    ...next,
    lastSequence: Math.max(state.lastSequence, event.sequence),
  };
}

function message(
  id: string,
  role: ChatMessage["role"],
  content: string,
  status: TurnStatus,
): ChatMessage {
  return { id, role, content, thinking: "", tools: [], status, error: null };
}

function messagesFromTurn(turn: TurnResponse): ChatMessage[] {
  const error = turn.status === "failed"
    ? turn.error_message || "本轮处理失败"
    : turn.status === "cancelled" ? "本轮已取消" : null;
  return [
    message(`${turn.id}:user`, "user", turn.content, "done"),
    {
      ...message(turn.id, "assistant", turn.answer ?? "", turn.status),
      error,
    },
  ];
}

function updateActive(
  state: ChatState,
  updater: (message: ChatMessage) => ChatMessage,
  terminal = false,
): ChatState {
  if (!state.activeMessageId) return state;
  return {
    ...state,
    messages: state.messages.map((item) => (
      item.id === state.activeMessageId ? updater(item) : item
    )),
    activeMessageId: terminal ? null : state.activeMessageId,
    activeTurnId: terminal ? null : state.activeTurnId,
    connection: terminal ? "idle" : state.connection,
  };
}

function applySnapshot(
  current: ChatMessage,
  payload: Record<string, unknown>,
): ChatMessage {
  const status = isTurnStatus(payload.status) ? payload.status : current.status;
  const rawTools = Array.isArray(payload.tools) ? payload.tools : [];
  return {
    ...current,
    content: textValue(payload.text),
    thinking: textValue(payload.thinking),
    tools: rawTools.map((tool) => toolFromPayload(asRecord(tool), "running")),
    status,
  };
}

function upsertTool(
  tools: ToolProgress[],
  payload: Record<string, unknown>,
  fallbackStatus: ToolProgress["status"],
): ToolProgress[] {
  const incoming = toolFromPayload(payload, fallbackStatus);
  const index = tools.findIndex((tool) => tool.callId === incoming.callId);
  if (index < 0) return [...tools, incoming];
  return tools.map((tool, currentIndex) => currentIndex === index ? incoming : tool);
}

function toolFromPayload(
  payload: Record<string, unknown>,
  fallbackStatus: ToolProgress["status"],
): ToolProgress {
  const status = payload.status === "error" || payload.status === "done" || payload.status === "running"
    ? payload.status
    : fallbackStatus;
  return {
    callId: textValue(payload.call_id),
    name: textValue(payload.tool_name) || "工具",
    status,
  };
}

function isTerminal(type: TurnEvent["type"]): boolean {
  return type === "turn.completed" || type === "turn.failed" || type === "turn.cancelled";
}

function textValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? value as Record<string, unknown> : {};
}

function isTurnStatus(value: unknown): value is TurnStatus {
  return value === "pending"
    || value === "processing"
    || value === "done"
    || value === "failed"
    || value === "cancelled";
}
