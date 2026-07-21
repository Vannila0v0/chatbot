from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Awaitable, Callable

from agent.config_models import Config
from bootstrap.channels import start_channels
from bootstrap.dashboard_api import build_dashboard_server
from bootstrap.memory import build_memory_runtime
from bootstrap.proactive import build_memory_optimizer_task, build_proactive_runtime
from bootstrap.providers import build_providers
from bootstrap.tools import CoreRuntime, build_core_runtime
from bus.event_bus import EventBus
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)
from web.turns.message_bus import WebTurnCompletionHandler, WebTurnDispatcher
from web.turns.sqlite_repository import SQLiteTurnRepository
from web.events.broker import WebTurnEventBroker
from web.events.bridge import WebTurnEventBridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def _run_cleanup_steps(*steps: tuple[str, Callable[[], Awaitable[None]]]) -> None:
    first_error: Exception | None = None
    for name, step in steps:
        try:
            await step()
        except Exception as exc:
            if first_error is None:
                first_error = exc
            logger.warning("shutdown step failed: %s: %s", name, exc)
    if first_error is not None:
        raise first_error


async def _noop_async() -> None:
    return None


class AppRuntime:
    def __init__(
        self,
        config: Config,
        workspace: Path,
        *,
        dashboard_host: str = "0.0.0.0",
        dashboard_port: int = 2236,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.dashboard_host = dashboard_host
        self.dashboard_port = dashboard_port
        self.http_resources = SharedHttpResources()
        self.ipc = None
        self.tg_channel = None
        self.qq_channel = None
        self.qqbot_channel = None
        self.feishu_channel = None
        self.core: CoreRuntime | None = None
        self.agent_loop = None
        self.bus = None
        self.event_bus: EventBus | None = None
        self.tools = None
        self.push_tool = None
        self.session_manager = None
        self.scheduler = None
        self.provider = None
        self.light_provider = None
        self.mcp_registry = None
        self.memory_runtime = None
        self.presence = None
        self.proactive_loop = None
        self.peer_process_manager = None
        self.peer_poller = None
        self.dashboard_server = None
        self.dashboard_task: asyncio.Task[None] | None = None
        self.turn_repository: SQLiteTurnRepository | None = None
        self.turn_dispatcher: WebTurnDispatcher | None = None
        self.turn_completion_handler: WebTurnCompletionHandler | None = None
        self.turn_dispatch_task: asyncio.Task[None] | None = None
        self.turn_event_broker: WebTurnEventBroker | None = None
        self.turn_event_bridge: WebTurnEventBridge | None = None
        self.tasks: list[Awaitable[None]] = []
        self._memory_optimizer = None
        self._shutdown = False
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        configure_default_shared_http_resources(self.http_resources)
        try:
            self.core = build_core_runtime(
                self.config,
                self.workspace,
                self.http_resources,
            )
            self.agent_loop = self.core.loop
            self.bus = self.core.bus
            event_bus = self.core.event_bus
            self.event_bus = event_bus
            self.tools = self.core.tools
            self.push_tool = self.core.push_tool
            self.session_manager = self.core.session_manager
            self.scheduler = self.core.scheduler
            self.provider = self.core.provider
            self.light_provider = self.core.light_provider
            self.mcp_registry = self.core.mcp_registry
            self.memory_runtime = self.core.memory_runtime
            self.presence = self.core.presence
            self.peer_process_manager = self.core.peer_process_manager
            self.peer_poller = self.core.peer_poller
            await self.core.start()

            self.turn_repository = SQLiteTurnRepository(self.workspace / "web.db")
            self.turn_event_broker = WebTurnEventBroker()
            self.turn_event_bridge = WebTurnEventBridge(self.turn_event_broker)
            self.turn_event_bridge.subscribe(event_bus)
            session_cfg = getattr(self.config, "session", None)
            logical_session_key = str(
                getattr(session_cfg, "primary_key", "") or ""
            ).strip()
            self.turn_dispatcher = WebTurnDispatcher(
                self.turn_repository,
                self.bus,
                self.turn_event_broker,
                logical_session_key=logical_session_key,
            )
            self.turn_completion_handler = WebTurnCompletionHandler(
                self.turn_repository,
                self.turn_event_broker,
                self.turn_event_bridge,
            )
            self.turn_completion_handler.subscribe(self.bus)
            self.turn_dispatch_task = asyncio.create_task(
                self.turn_dispatcher.run(),
                name="web_turn_dispatcher",
            )

            plugin_manager = getattr(self.core, "plugin_manager", None)
            (
                self.ipc,
                self.tg_channel,
                self.qq_channel,
                self.qqbot_channel,
                self.feishu_channel,
            ) = await start_channels(
                self.config,
                bus=self.bus,
                session_manager=self.session_manager,
                push_tool=self.push_tool,
                http_resources=self.http_resources,
                event_bus=event_bus,
                bot_commands=(
                    plugin_manager.telegram_bot_commands
                    if plugin_manager
                    else None
                ),
                interrupt_controller=self.agent_loop,
            )

            self.tasks = [
                self.agent_loop.run(),
                self.bus.dispatch_outbound(),
                self.scheduler.run(),
            ]
            optimizer_tasks, self._memory_optimizer = build_memory_optimizer_task(
                self.config,
                provider=self.provider,
                memory_store=self.memory_runtime.markdown.store,
            )
            self.tasks.extend(optimizer_tasks)
            self.dashboard_server = build_dashboard_server(
                workspace=self.workspace,
                host=self.dashboard_host,
                port=self.dashboard_port,
                manual_consolidator=self.agent_loop,
                manual_memory_optimizer=self._memory_optimizer,
                memory_admin=self.memory_runtime.engine,
                memory_store=self.memory_runtime.markdown.store,
                turn_repository=self.turn_repository,
                turn_event_broker=self.turn_event_broker,
            )
            self.dashboard_task = asyncio.create_task(
                self.dashboard_server.serve(),
                name="dashboard_server",
            )
            proactive_tasks, self.proactive_loop = build_proactive_runtime(
                self.config,
                self.workspace,
                session_manager=self.session_manager,
                provider=self.provider,
                light_provider=self.light_provider,
                push_tool=self.push_tool,
                memory_store=self.memory_runtime,
                presence=self.presence,
                agent_loop=self.agent_loop,
                tool_hooks=list(plugin_manager.tool_hooks) if plugin_manager else None,
            )
            self.tasks.extend(proactive_tasks)
            if self.proactive_loop is not None:
                self.ipc.set_proactive_loop(self.proactive_loop)

            self._started = True
        except Exception:
            await self.shutdown()
            raise

    async def run(self) -> None:
        try:
            await self.start()
            await asyncio.gather(*self.tasks)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        try:
            if self.turn_dispatcher is not None:
                self.turn_dispatcher.stop()
            if self.turn_dispatch_task is not None:
                _ = self.turn_dispatch_task.cancel()
                try:
                    await self.turn_dispatch_task
                except asyncio.CancelledError:
                    pass
            if self.dashboard_server is not None:
                self.dashboard_server.should_exit = True
            if self.dashboard_task is not None:
                try:
                    await self.dashboard_task
                except asyncio.CancelledError:
                    pass
            await _run_cleanup_steps(
                ("core.stop", self.core.stop if self.core else _noop_async),
                ("ipc.stop", self.ipc.stop if self.ipc else _noop_async),
                (
                    "telegram.stop",
                    self.tg_channel.stop if self.tg_channel else _noop_async,
                ),
                ("qq.stop", self.qq_channel.stop if self.qq_channel else _noop_async),
                (
                    "qqbot.stop",
                    self.qqbot_channel.stop if self.qqbot_channel else _noop_async,
                ),
                (
                    "feishu.stop",
                    self.feishu_channel.stop if self.feishu_channel else _noop_async,
                ),
                (
                    "memory_runtime.aclose",
                    self.memory_runtime.aclose if self.memory_runtime else _noop_async,
                ),
                ("turn_event_broker.aclose", self._close_turn_event_broker),
                ("turn_repository.close", self._close_turn_repository),
                ("http_resources.aclose", self.http_resources.aclose),
            )
        finally:
            clear_default_shared_http_resources(self.http_resources)

    async def _close_turn_repository(self) -> None:
        if self.turn_repository is not None:
            self.turn_repository.close()

    async def _close_turn_event_broker(self) -> None:
        if self.turn_event_broker is not None:
            await self.turn_event_broker.aclose()


def build_app_runtime(
    config: Config,
    workspace: Path | None = None,
    *,
    dashboard_host: str = "0.0.0.0",
    dashboard_port: int = 2236,
) -> AppRuntime:
    return AppRuntime(
        config,
        workspace or (Path.home() / ".akashic" / "workspace"),
        dashboard_host=dashboard_host,
        dashboard_port=dashboard_port,
    )
