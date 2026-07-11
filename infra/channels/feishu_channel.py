from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, cast

import httpx

from agent.looping.interrupt import InterruptController
from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from session.manager import SessionManager

logger = logging.getLogger(__name__)

_CHANNEL = "feishu"
_API_BASE = "https://open.feishu.cn/open-apis"
_TOKEN_PATH = "/auth/v3/tenant_access_token/internal"


@dataclass
class _TokenCache:
    token: str
    expires_at: float


class FeishuChannel:
    def __init__(
        self,
        app_id: str,
        app_secret: str,
        bus: MessageBus,
        session_manager: SessionManager,
        allow_from: list[str] | None = None,
        receive_id_type: str = "chat_id",
        interrupt_controller: InterruptController | None = None,
        channel_name: str = _CHANNEL,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._bus = bus
        self._session_manager = session_manager
        self._allow_from = {str(item) for item in (allow_from or []) if str(item)}
        self._receive_id_type = receive_id_type or "chat_id"
        self._interrupt_controller = interrupt_controller
        self._channel = channel_name
        self._client = httpx.AsyncClient(timeout=30.0)
        self._token: _TokenCache | None = None
        self._thread: threading.Thread | None = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        self._main_loop = asyncio.get_running_loop()
        self._stopped.clear()
        self._thread = threading.Thread(
            target=self._run_ws_client,
            name="feishu_ws_client",
            daemon=True,
        )
        self._thread.start()
        self._bus.subscribe_outbound(self._channel, self._on_response)
        logger.info("[feishu] FeishuChannel 已启动")

    def _run_ws_client(self) -> None:
        import lark_oapi as lark

        event_handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._on_message_event)
            .build()
        )
        client = lark.ws.Client(
            self._app_id,
            self._app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )
        client.start()

    async def stop(self) -> None:
        self._stopped.set()
        await self._client.aclose()
        logger.info("[feishu] FeishuChannel 已停止")

    def _on_message_event(self, data: Any) -> None:
        loop = self._main_loop
        if loop is None:
            logger.warning("[feishu] main loop 尚未就绪，丢弃消息事件")
            return
        asyncio.run_coroutine_threadsafe(self._handle_message_event(data), loop)

    async def _handle_message_event(
        self,
        data: Any,
    ) -> None:
        payload = _event_to_dict(data)
        event = _as_dict(payload.get("event"))
        sender = _as_dict(event.get("sender"))
        sender_id = _as_dict(sender.get("sender_id"))
        message = _as_dict(event.get("message"))

        chat_id = str(message.get("chat_id") or "")
        message_id = str(message.get("message_id") or "")
        message_type = str(message.get("message_type") or "")
        open_id = str(sender_id.get("open_id") or "")
        union_id = str(sender_id.get("union_id") or "")
        user_id = str(sender_id.get("user_id") or "")

        if not chat_id or not open_id:
            logger.debug("[feishu] 忽略缺少 chat_id/open_id 的事件: %s", payload)
            return
        if not self._is_allowed(open_id=open_id, union_id=union_id, user_id=user_id, chat_id=chat_id):
            logger.warning("[feishu] 拒绝未授权用户 open_id=%s chat_id=%s", open_id, chat_id)
            return

        content = _extract_message_text(message)
        if not content and message_type != "text":
            content = f"[暂不支持的飞书消息类型: {message_type}]"
        if content.strip() == "/stop":
            await self._handle_stop(chat_id, open_id)
            return

        session = self._session_manager.get_or_create(f"{self._channel}:{chat_id}")
        changed = False
        for key, value in {
            "feishu_open_id": open_id,
            "feishu_union_id": union_id,
            "feishu_user_id": user_id,
        }.items():
            if value and session.metadata.get(key) != value:
                session.metadata[key] = value
                changed = True
        if changed:
            await self._session_manager.save_async(session)

        preview = content[:60] + "..." if len(content) > 60 else content
        logger.info(
            "[feishu] 收到消息 chat_id=%s open_id=%s type=%s 内容: %r",
            chat_id,
            open_id,
            message_type,
            preview,
        )
        await self._bus.publish_inbound(
            InboundMessage(
                channel=self._channel,
                sender=open_id,
                chat_id=chat_id,
                content=content,
                metadata={
                    "chat_type": str(message.get("chat_type") or ""),
                    "message_id": message_id,
                    "message_type": message_type,
                    "open_id": open_id,
                    "union_id": union_id,
                    "user_id": user_id,
                },
            )
        )

    async def _handle_stop(self, chat_id: str, sender: str) -> None:
        if self._interrupt_controller is None:
            await self.send(chat_id, "当前未启用中断功能。")
            return
        result = self._interrupt_controller.request_interrupt(
            session_key=f"{self._channel}:{chat_id}",
            sender=sender,
            command="/stop",
        )
        await self.send(chat_id, result.message)

    async def _on_response(self, msg: OutboundMessage) -> None:
        content = msg.content.strip()
        if content:
            preview = content[:60] + "..." if len(content) > 60 else content
            logger.info("[feishu] 发送回复 chat_id=%s 内容: %r", msg.chat_id, preview)
            await self.send(msg.chat_id, content)
        for image in msg.media or []:
            logger.warning("[feishu] 暂不支持发送图片，已跳过: %s", image)

    async def send(self, chat_id: str, message: str) -> None:
        await self._send_text(chat_id, message)

    async def send_stream(self, chat_id: str, message: str) -> None:
        await self._send_text(chat_id, message)

    async def _send_text(self, chat_id: str, message: str) -> None:
        token = await self._get_access_token()
        receive_id = _strip_channel_prefix(chat_id, self._channel)
        body = {
            "receive_id": receive_id,
            "msg_type": "text",
            "content": json.dumps({"text": message}, ensure_ascii=False),
        }
        await self._api_request(
            "POST",
            f"/im/v1/messages?receive_id_type={self._receive_id_type}",
            body,
            token,
        )

    async def _get_access_token(self) -> str:
        now = time.time()
        if self._token and now < self._token.expires_at - 300:
            return self._token.token
        data = await self._api_request(
            "POST",
            _TOKEN_PATH,
            {
                "app_id": self._app_id,
                "app_secret": self._app_secret,
            },
            token="",
        )
        token = str(data["tenant_access_token"])
        expires_in = int(data.get("expire") or 7200)
        self._token = _TokenCache(token=token, expires_at=now + expires_in)
        return token

    async def _api_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        resp = await self._client.request(
            method,
            f"{_API_BASE}{path}",
            json=body,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        code = int(data.get("code", 0)) if isinstance(data, dict) else 0
        if code != 0:
            raise RuntimeError(f"Feishu API error code={code} msg={data.get('msg')!r}")
        if "data" in data and isinstance(data.get("data"), dict):
            return cast(dict[str, Any], data["data"])
        return cast(dict[str, Any], data)

    def _is_allowed(
        self,
        *,
        open_id: str,
        union_id: str,
        user_id: str,
        chat_id: str,
    ) -> bool:
        if not self._allow_from:
            return True
        return any(
            value and value in self._allow_from
            for value in (open_id, union_id, user_id, chat_id)
        )


def _event_to_dict(data: object) -> dict[str, Any]:
    import lark_oapi as lark

    try:
        raw = lark.JSON.marshal(data)
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        logger.exception("[feishu] 事件序列化失败")
        return {}


def _extract_message_text(message: dict[str, Any]) -> str:
    raw = str(message.get("content") or "")
    if not raw:
        return ""
    try:
        content = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(content, dict):
        return raw
    text = content.get("text")
    if isinstance(text, str):
        return text
    return raw


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _strip_channel_prefix(chat_id: str, channel: str) -> str:
    value = str(chat_id).strip()
    prefix = f"{channel}:"
    if value.startswith(prefix):
        return value[len(prefix):]
    return value
