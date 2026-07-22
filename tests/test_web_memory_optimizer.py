from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from bootstrap.proactive import build_memory_optimizer_task
from core.memory.markdown import MarkdownMemoryStore
from core.memory.scope import MarkdownMemoryStoreResolver
from proactive_v2.memory_optimizer import (
    MemoryOptimizer,
    MemoryOptimizerGroup,
    WebMemoryOptimizerCoordinator,
)


USER_A = "550e8400-e29b-41d4-a716-446655440000"
USER_B = "5d260c61-ecd4-4c11-99bd-2856b9527f6b"


class _TenantProvider:
    def __init__(self, *, fail_markers: set[str] | None = None) -> None:
        self.fail_markers = fail_markers or set()
        self.chat = AsyncMock(side_effect=self._chat)

    async def _chat(self, **kwargs: Any) -> SimpleNamespace:
        prompt = str(kwargs["messages"][-1]["content"])
        marker = "USER-A" if "USER-A-PENDING" in prompt else "USER-B"
        if marker in self.fail_markers:
            raise RuntimeError(f"{marker} provider failure")
        if "当前 SELF.md" in prompt:
            content = (
                "# Akashic 的自我认知\n\n"
                "## 人格与形象\n- 默认人格\n\n"
                "## 我对当前用户的理解\n"
                f"- {marker}-SELF\n\n"
                "## 我们关系的定义\n- 持续协作\n"
            )
        else:
            content = f"# 用户长期记忆\n\n## 用户事实\n- {marker}-MEMORY\n"
        return SimpleNamespace(content=content)


def _resolver(tmp_path: Path) -> MarkdownMemoryStoreResolver:
    default_store = MarkdownMemoryStore(tmp_path)
    return MarkdownMemoryStoreResolver(tmp_path, default_store=default_store)


@pytest.mark.asyncio
async def test_web_optimizer_isolates_memory_self_and_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MemoryOptimizer, "_STEP_DELAY_SECONDS", 0)
    resolver = _resolver(tmp_path)
    key_a = f"web:{USER_A}:primary"
    key_b = f"web:{USER_B}:primary"
    store_a = resolver.store_for(key_a)
    store_b = resolver.store_for(key_b)
    store_a.append_pending("- [preference] USER-A-PENDING")
    store_b.append_pending("- [identity] USER-B-PENDING")
    resolver.default_store.write_long_term("PERSONAL-MEMORY")
    provider = _TenantProvider()
    coordinator = WebMemoryOptimizerCoordinator(
        resolver=resolver,
        provider=provider,  # type: ignore[arg-type]
        model="test-model",
    )

    await coordinator.optimize()

    assert "USER-A-MEMORY" in store_a.read_long_term()
    assert "USER-B-MEMORY" not in store_a.read_long_term()
    assert "USER-A-SELF" in store_a.read_self()
    assert "USER-B-MEMORY" in store_b.read_long_term()
    assert "USER-A-MEMORY" not in store_b.read_long_term()
    assert "USER-B-SELF" in store_b.read_self()
    assert store_a.read_pending() == ""
    assert store_b.read_pending() == ""
    assert resolver.default_store.read_long_term() == "PERSONAL-MEMORY"
    assert provider.chat.await_count == 4


@pytest.mark.asyncio
async def test_web_optimizer_skips_empty_tenants(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    _ = resolver.store_for(f"web:{USER_A}:primary")
    provider = _TenantProvider()
    coordinator = WebMemoryOptimizerCoordinator(
        resolver=resolver,
        provider=provider,  # type: ignore[arg-type]
        model="test-model",
    )

    await coordinator.optimize()

    provider.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_web_optimizer_failure_is_isolated_per_tenant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MemoryOptimizer, "_STEP_DELAY_SECONDS", 0)
    resolver = _resolver(tmp_path)
    store_a = resolver.store_for(f"web:{USER_A}:primary")
    store_b = resolver.store_for(f"web:{USER_B}:primary")
    store_a.append_pending("- [preference] USER-A-PENDING")
    store_b.append_pending("- [identity] USER-B-PENDING")
    provider = _TenantProvider(fail_markers={"USER-A"})
    coordinator = WebMemoryOptimizerCoordinator(
        resolver=resolver,
        provider=provider,  # type: ignore[arg-type]
        model="test-model",
    )

    await coordinator.optimize()

    assert "USER-A-PENDING" in store_a.read_pending()
    assert store_a.read_long_term() == ""
    assert "USER-B-MEMORY" in store_b.read_long_term()
    assert store_b.read_pending() == ""


def test_pending_discovery_recovers_tenant_snapshot(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    key_a = f"web:{USER_A}:primary"
    store_a = resolver.store_for(key_a)
    store_a.append_pending("- [preference] USER-A-PENDING")
    _ = store_a.snapshot_pending()
    assert store_a._snapshot_path.exists()

    restarted = _resolver(tmp_path)
    candidates = restarted.iter_web_stores_with_pending()

    assert [scope.user_id for scope, _ in candidates] == [USER_A]
    recovered_store = candidates[0][1]
    assert "USER-A-PENDING" in recovered_store.read_pending()
    assert not recovered_store._snapshot_path.exists()


@pytest.mark.asyncio
async def test_optimizer_group_continues_after_one_optimizer_fails() -> None:
    first = SimpleNamespace(optimize=AsyncMock(side_effect=RuntimeError("failed")))
    second = SimpleNamespace(optimize=AsyncMock())
    group = MemoryOptimizerGroup(
        [
            ("first", first),  # type: ignore[list-item]
            ("second", second),  # type: ignore[list-item]
        ]
    )

    await group.optimize()

    first.optimize.assert_awaited_once()
    second.optimize.assert_awaited_once()


def test_optimizer_bootstrap_schedules_group_but_returns_personal_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = _resolver(tmp_path)
    captured: list[object] = []

    class _Loop:
        def __init__(self, optimizer: object, interval_seconds: int) -> None:
            captured.extend([optimizer, interval_seconds])

        def run(self) -> str:
            return "optimizer-task"

    monkeypatch.setattr("bootstrap.proactive.MemoryOptimizerLoop", _Loop)
    config = SimpleNamespace(
        memory_optimizer_enabled=True,
        memory_optimizer_interval_seconds=3600,
        model="test-model",
    )
    provider = _TenantProvider()

    tasks, manual_optimizer = build_memory_optimizer_task(
        config,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
        memory_store=resolver.default_store,
        memory_resolver=resolver,
    )

    assert tasks == ["optimizer-task"]
    assert isinstance(captured[0], MemoryOptimizerGroup)
    assert captured[1] == 3600
    assert isinstance(manual_optimizer, MemoryOptimizer)
