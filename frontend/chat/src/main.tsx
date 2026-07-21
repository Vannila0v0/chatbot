import React, { useEffect, useMemo, useReducer, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Asterisk,
  BrainCircuit,
  Check,
  ChevronDown,
  CircleAlert,
  LoaderCircle,
  SendHorizontal,
  Wrench,
} from "lucide-react";
import { createTurn, getTurn, listTurns, parseTurnEvent } from "./api";
import { chatReducer, initialChatState } from "./state";
import type { ChatMessage, TurnEvent, TurnEventType, TurnResponse } from "./types";
import "./styles.css";

const EVENT_TYPES: TurnEventType[] = [
  "turn.queued",
  "turn.started",
  "turn.snapshot",
  "thinking.delta",
  "text.delta",
  "tool.started",
  "tool.completed",
  "turn.completed",
  "turn.failed",
  "turn.cancelled",
];

const STARTERS = [
  "帮我整理一下今天最重要的三件事",
  "根据我们之前的交流，给我一个现在适合做的建议",
  "陪我聊聊最近让我分心的一件事",
];

function ChatApp(): React.JSX.Element {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const [draft, setDraft] = useState("");
  const sourceRef = useRef<EventSource | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const messageEndRef = useRef<HTMLDivElement | null>(null);
  const identity = useMemo(() => loadIdentity(), []);
  const busy = state.activeMessageId !== null;

  useEffect(() => {
    let cancelled = false;
    void listTurns(identity.userId, identity.conversationId).then((turns) => {
      if (!cancelled) dispatch({ type: "hydrate", turns });
    }).catch((error) => {
      if (!cancelled) {
        dispatch({ type: "history_failed", message: errorMessage(error) });
      }
    });
    return () => {
      cancelled = true;
    };
  }, [identity.conversationId, identity.userId]);

  useEffect(() => {
    if (!state.activeTurnId || sourceRef.current) return;
    const source = connectToTurn(state.activeTurnId, dispatch);
    sourceRef.current = source;
    return () => {
      source.close();
      if (sourceRef.current === source) sourceRef.current = null;
    };
  }, [state.activeTurnId]);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({
      behavior: busy ? "auto" : "smooth",
      block: "end",
    });
  }, [busy, state.messages]);

  const submit = async (content = draft): Promise<void> => {
    const normalized = content.trim();
    if (!normalized || busy || state.hydrating) return;
    const requestId = crypto.randomUUID();
    dispatch({ type: "begin", requestId, content: normalized });
    setDraft("");
    resetTextarea(textareaRef.current);
    try {
      const turn = await createTurn({
        userId: identity.userId,
        conversationId: identity.conversationId,
        clientRequestId: requestId,
        content: normalized,
      });
      dispatch({ type: "accepted", turnId: turn.id });
    } catch (error) {
      dispatch({ type: "submit_failed", message: errorMessage(error) });
    }
  };

  return (
    <main className="chat-shell">
      <header className="chat-header">
        <a className="brand" href="/chat" aria-label="Akashic 对话首页">
          <span className="brand-mark"><Asterisk size={18} strokeWidth={2.2} /></span>
          <span>Akashic</span>
        </a>
        <div className="runtime-state" aria-live="polite">
          <span className={`runtime-dot ${busy || state.hydrating ? "working" : "ready"}`} />
          {connectionLabel(state.connection, busy, state.hydrating)}
        </div>
      </header>

      <section className={`conversation ${state.messages.length ? "has-messages" : "empty"}`}>
        {state.hydrating ? (
          <LoadingConversation />
        ) : state.messages.length === 0 ? (
          <EmptyConversation
            error={state.historyError}
            onSelect={(value) => void submit(value)}
          />
        ) : (
          <div className="message-list" aria-live="polite">
            {state.historyError && (
              <div className="history-notice" role="alert">
                <CircleAlert size={16} />
                <span>{state.historyError}</span>
              </div>
            )}
            {state.messages.map((message) => (
              <Message key={message.id} message={message} />
            ))}
            <div ref={messageEndRef} />
          </div>
        )}
      </section>

      <footer className="composer-band">
        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
        >
          <textarea
            ref={textareaRef}
            value={draft}
            rows={1}
            placeholder={state.hydrating ? "正在恢复会话" : busy ? "等待当前回复完成" : "发消息给 Akashic"}
            aria-label="消息内容"
            disabled={busy || state.hydrating}
            onChange={(event) => {
              setDraft(event.target.value);
              resizeTextarea(event.currentTarget);
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                void submit();
              }
            }}
          />
          <button
            className="send-button"
            type="submit"
            disabled={busy || state.hydrating || !draft.trim()}
            aria-label="发送消息"
            title="发送消息"
          >
            {busy ? <LoaderCircle className="spin" size={19} /> : <SendHorizontal size={19} />}
          </button>
        </form>
        <p className="composer-note">Akashic 可能会出错，请核对重要信息。</p>
      </footer>
    </main>
  );
}

function LoadingConversation(): React.JSX.Element {
  return (
    <div className="conversation-loading" role="status">
      <LoaderCircle className="spin" size={20} />
      <span>正在恢复会话</span>
    </div>
  );
}

function EmptyConversation({
  error,
  onSelect,
}: {
  error: string | null;
  onSelect: (value: string) => void;
}): React.JSX.Element {
  return (
    <div className="empty-content">
      <div className="empty-mark"><Asterisk size={26} strokeWidth={1.7} /></div>
      <h1>今天想从哪里开始？</h1>
      {error && (
        <div className="history-notice empty-history-notice" role="alert">
          <CircleAlert size={16} />
          <span>{error}</span>
        </div>
      )}
      <div className="starter-list">
        {STARTERS.map((starter) => (
          <button type="button" key={starter} onClick={() => onSelect(starter)}>
            <span>{starter}</span>
            <SendHorizontal size={16} aria-hidden="true" />
          </button>
        ))}
      </div>
    </div>
  );
}

function Message({ message }: { message: ChatMessage }): React.JSX.Element {
  if (message.role === "user") {
    return (
      <article className="message user-message">
        <div className="message-content">{message.content}</div>
      </article>
    );
  }
  const active = message.status === "pending" || message.status === "processing";
  return (
    <article className="message assistant-message">
      <div className="assistant-sign"><Asterisk size={16} strokeWidth={2} /></div>
      <div className="assistant-body">
        {(active || message.thinking || message.tools.length > 0) && (
          <ActivityPanel message={message} active={active} />
        )}
        {message.content ? (
          <div className="message-content assistant-copy">{message.content}</div>
        ) : active ? (
          <div className="response-placeholder"><span /><span /><span /></div>
        ) : null}
        {message.error && (
          <div className="message-error" role="alert">
            <CircleAlert size={17} />
            <span>{message.error}</span>
          </div>
        )}
      </div>
    </article>
  );
}

function ActivityPanel({ message, active }: { message: ChatMessage; active: boolean }): React.JSX.Element {
  const completedTools = message.tools.filter((tool) => tool.status === "done").length;
  const summary = active
    ? activityLabel(message)
    : `过程记录${message.tools.length ? ` · ${completedTools}/${message.tools.length} 个工具完成` : ""}`;
  return (
    <details className="activity-panel" open={active || undefined}>
      <summary>
        <span className="activity-title">
          {active ? <LoaderCircle className="spin" size={15} /> : <BrainCircuit size={15} />}
          {summary}
        </span>
        <ChevronDown className="activity-chevron" size={16} />
      </summary>
      <div className="activity-content">
        {message.thinking && (
          <div className="thinking-copy">
            <span>思考</span>
            <p>{message.thinking}</p>
          </div>
        )}
        {message.tools.map((tool) => (
          <div className="tool-row" key={tool.callId || tool.name}>
            <Wrench size={14} />
            <span>{tool.name}</span>
            <span className={`tool-status ${tool.status}`}>
              {tool.status === "running" && <LoaderCircle className="spin" size={13} />}
              {tool.status === "done" && <Check size={13} />}
              {tool.status === "error" && <CircleAlert size={13} />}
              {toolStatusLabel(tool.status)}
            </span>
          </div>
        ))}
        {!message.thinking && message.tools.length === 0 && (
          <div className="activity-waiting">正在准备上下文</div>
        )}
      </div>
    </details>
  );
}

function connectToTurn(
  turnId: string,
  dispatch: React.Dispatch<Parameters<typeof chatReducer>[1]>,
): EventSource {
  const source = new EventSource(`/api/turns/${encodeURIComponent(turnId)}/events`);
  source.onopen = () => dispatch({ type: "connection", value: "open" });
  const handle = (rawEvent: Event): void => {
    const event = parseTurnEvent(rawEvent as MessageEvent<string>);
    if (isTerminalEvent(event)) {
      source.close();
    }
    dispatch({ type: "event", event });
  };
  EVENT_TYPES.forEach((eventType) => source.addEventListener(eventType, handle));
  source.onerror = () => {
    dispatch({ type: "connection", value: "retrying" });
    void getTurn(turnId).then((latest) => {
      const terminal = terminalEvent(latest);
      if (!terminal) return;
      source.close();
      dispatch({ type: "event", event: terminal });
    }).catch(() => undefined);
  };
  return source;
}

function terminalEvent(turn: TurnResponse): TurnEvent | null {
  const common = {
    turn_id: turn.id,
    sequence: Number.MAX_SAFE_INTEGER,
    timestamp: new Date().toISOString(),
  };
  if (turn.status === "done") {
    return { ...common, type: "turn.completed", payload: { answer: turn.answer ?? "" } };
  }
  if (turn.status === "failed") {
    return {
      ...common,
      type: "turn.failed",
      payload: { error_message: turn.error_message ?? "本轮处理失败" },
    };
  }
  if (turn.status === "cancelled") {
    return { ...common, type: "turn.cancelled", payload: {} };
  }
  return null;
}

function isTerminalEvent(event: TurnEvent): boolean {
  return event.type === "turn.completed"
    || event.type === "turn.failed"
    || event.type === "turn.cancelled";
}

function loadIdentity(): { userId: string; conversationId: string } {
  return {
    userId: storedId("akashic.web.user_id", "web-user"),
    conversationId: storedId("akashic.web.conversation_id", "web-conversation"),
  };
}

function storedId(key: string, prefix: string): string {
  try {
    const existing = localStorage.getItem(key);
    if (existing) return existing;
    const value = `${prefix}-${crypto.randomUUID()}`;
    localStorage.setItem(key, value);
    return value;
  } catch {
    return `${prefix}-${crypto.randomUUID()}`;
  }
}

function activityLabel(message: ChatMessage): string {
  if (message.tools.some((tool) => tool.status === "running")) return "正在使用工具";
  if (message.thinking) return "正在思考";
  return message.status === "pending" ? "等待处理" : "正在组织回复";
}

function connectionLabel(connection: string, busy: boolean, hydrating: boolean): string {
  if (hydrating) return "正在恢复会话";
  if (!busy) return "准备就绪";
  if (connection === "retrying") return "正在重新连接";
  if (connection === "connecting") return "正在连接";
  return "正在处理";
}

function toolStatusLabel(status: "running" | "done" | "error"): string {
  if (status === "running") return "执行中";
  if (status === "done") return "完成";
  return "失败";
}

function resizeTextarea(textarea: HTMLTextAreaElement): void {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
}

function resetTextarea(textarea: HTMLTextAreaElement | null): void {
  if (textarea) textarea.style.height = "auto";
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "消息提交失败";
}

createRoot(document.getElementById("root")!).render(<ChatApp />);
