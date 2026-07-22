from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from agent.memory import DEFAULT_SELF_MD
from core.memory.markdown import MarkdownMemoryStore


class InvalidMemoryScopeError(ValueError):
    """Raised when a scoped session key cannot be mapped safely."""


@dataclass(frozen=True)
class WebMemoryScope:
    user_id: str
    conversation_id: str = "primary"


def parse_web_memory_scope(session_key: str) -> WebMemoryScope | None:
    """Parse the canonical Web session key without treating it as a path."""
    if not session_key.startswith("web:"):
        return None

    parts = session_key.split(":")
    if len(parts) != 3:
        raise InvalidMemoryScopeError("invalid Web session key format")

    _, raw_user_id, conversation_id = parts
    if conversation_id != "primary":
        raise InvalidMemoryScopeError("unsupported Web conversation scope")
    try:
        user_id = str(UUID(raw_user_id))
    except (ValueError, AttributeError) as exc:
        raise InvalidMemoryScopeError("invalid Web user id") from exc

    return WebMemoryScope(user_id=user_id, conversation_id=conversation_id)


class MarkdownMemoryStoreResolver:
    """Resolve personal and Web-tenant Markdown stores from a session key."""

    def __init__(
        self,
        workspace: Path,
        *,
        default_store: MarkdownMemoryStore | None = None,
    ) -> None:
        self._workspace = workspace.resolve()
        self._web_users_root = (self._workspace / "web_users").resolve()
        self._default_store = default_store or MarkdownMemoryStore(self._workspace)
        self._web_stores: dict[str, MarkdownMemoryStore] = {}
        self._lock = threading.Lock()

    @property
    def default_store(self) -> MarkdownMemoryStore:
        return self._default_store

    def store_for(self, session_key: str) -> MarkdownMemoryStore:
        scope = parse_web_memory_scope(session_key)
        if scope is None:
            return self._default_store

        with self._lock:
            cached = self._web_stores.get(scope.user_id)
            if cached is not None:
                return cached

            tenant_workspace = (self._web_users_root / scope.user_id).resolve()
            if tenant_workspace.parent != self._web_users_root:
                raise InvalidMemoryScopeError("Web memory path escaped tenant root")
            store = MarkdownMemoryStore(tenant_workspace)
            self._initialize_web_store(store)
            self._web_stores[scope.user_id] = store
            return store

    @staticmethod
    def _initialize_web_store(store: MarkdownMemoryStore) -> None:
        for path in (
            store.memory_file,
            store.history_file,
            store.recent_context_file,
        ):
            path.touch(exist_ok=True)
        if not store.self_file.exists():
            store.write_self(DEFAULT_SELF_MD)
