from __future__ import annotations

import threading
from pathlib import Path

from core.memory.scope import InvalidMemoryScopeError, parse_web_memory_scope
from memory2.store import VEC_DIM, MemoryStore2


class MemoryStore2Resolver:
    """Choose the personal or Web-tenant vector store for a session."""

    def __init__(
        self,
        workspace: Path,
        *,
        default_db_path: Path,
        vec_dim: int = VEC_DIM,
    ) -> None:
        self._workspace = workspace.resolve()
        self._web_users_root = (self._workspace / "web_users").resolve()
        self._vec_dim = vec_dim
        self._default_store = MemoryStore2(default_db_path, vec_dim=vec_dim)
        self._web_stores: dict[str, MemoryStore2] = {}
        self._lock = threading.RLock()
        self._closed = False

    @property
    def default_store(self) -> MemoryStore2:
        return self._default_store

    def store_for(self, session_key: str) -> MemoryStore2:
        scope = parse_web_memory_scope(session_key)
        with self._lock:
            if self._closed:
                raise RuntimeError("memory store resolver is closed")
            if scope is None:
                return self._default_store

            cached = self._web_stores.get(scope.user_id)
            if cached is not None:
                return cached

            tenant_workspace = (self._web_users_root / scope.user_id).resolve()
            if tenant_workspace.parent != self._web_users_root:
                raise InvalidMemoryScopeError("Web memory path escaped tenant root")
            store = MemoryStore2(
                tenant_workspace / "memory" / "memory2.db",
                vec_dim=self._vec_dim,
            )
            self._web_stores[scope.user_id] = store
            return store

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            stores = [self._default_store, *self._web_stores.values()]
            self._web_stores.clear()

        for store in stores:
            store.close()
