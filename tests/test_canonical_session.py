from typing import Any, cast

import pytest

from agent.tools.memorize import MemorizeTool
from agent.tools.recall_memory import RecallMemoryTool
from bus.events import InboundMessage
from core.memory.engine import (
    MemoryMutationResult,
    MemoryQueryResult,
    MemoryToolSpec,
)
from session.manager import SessionManager


class _CaptureMemory:
    mutation = None
    query_request = None

    async def mutate(self, request):
        self.mutation = request
        return MemoryMutationResult(
            accepted=True,
            item_id="memory-1",
            actual_kind="preference",
            status="new",
        )

    async def query(self, request):
        self.query_request = request
        return MemoryQueryResult()


def _spec() -> MemoryToolSpec:
    return MemoryToolSpec(description="test", parameters={})


def test_web_and_telegram_messages_share_one_session_store(tmp_path) -> None:
    web = InboundMessage(
        channel="web",
        sender="user",
        chat_id="browser-conversation",
        content="from web",
        logical_session_key="companion:primary",
    )
    telegram = InboundMessage(
        channel="telegram",
        sender="user",
        chat_id="123",
        content="from telegram",
        logical_session_key="companion:primary",
    )
    sessions = SessionManager(tmp_path)

    session = sessions.get_or_create(web.session_key)
    session.add_message("user", web.content)
    sessions.save(session)
    same_session = sessions.get_or_create(telegram.session_key)
    same_session.add_message("user", telegram.content)
    sessions.save(same_session)

    stored = sessions.get_or_create("companion:primary")
    assert [message["content"] for message in stored.messages] == [
        "from web",
        "from telegram",
    ]
    assert web.chat_id == "browser-conversation"
    assert telegram.chat_id == "123"


@pytest.mark.asyncio
async def test_memory_tools_use_logical_session_without_losing_route() -> None:
    memory = _CaptureMemory()
    memorize = MemorizeTool(cast(Any, memory), _spec())
    recall = RecallMemoryTool(cast(Any, memory), _spec())

    await memorize.execute(
        summary="prefers concise answers",
        session_key="companion:primary",
        channel="telegram",
        chat_id="123",
    )
    await recall.execute(
        query="answer style",
        session_key="companion:primary",
        channel="web",
        chat_id="browser-conversation",
    )

    assert memory.mutation.scope.session_key == "companion:primary"
    assert memory.mutation.scope.channel == "telegram"
    assert memory.mutation.scope.chat_id == "123"
    assert memory.query_request.scope.session_key == "companion:primary"
    assert memory.query_request.scope.channel == "web"
    assert memory.query_request.scope.chat_id == "browser-conversation"
