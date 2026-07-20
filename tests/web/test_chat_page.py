from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from web.api.chat import mount_chat_ui


@pytest.fixture
def chat_static_dir(tmp_path: Path) -> Path:
    static_dir = tmp_path / "chat"
    static_dir.mkdir()
    (static_dir / "index.html").write_text(
        '<link rel="stylesheet" href="/chat/assets/app.css">'
        '<div id="root"></div>'
        '<script src="/chat/assets/app.js"></script>',
        encoding="utf-8",
    )
    (static_dir / "app.css").write_text("body { color: black; }", encoding="utf-8")
    (static_dir / "app.js").write_text("window.chatLoaded = true;", encoding="utf-8")
    return static_dir


@pytest.fixture
def transport(chat_static_dir: Path) -> httpx.ASGITransport:
    app = FastAPI()
    mount_chat_ui(app, chat_static_dir)
    return httpx.ASGITransport(app=app)


@pytest.mark.asyncio
async def test_chat_page_versions_bundled_assets(
    transport: httpx.ASGITransport,
) -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/chat")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert "/chat/assets/app.css?v=" in response.text
    assert "/chat/assets/app.js?v=" in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "content_type", "expected"),
    [
        ("/chat/assets/app.css", "text/css", "color: black"),
        ("/chat/assets/app.js", "text/javascript", "chatLoaded"),
    ],
)
async def test_chat_assets_are_served(
    transport: httpx.ASGITransport,
    path: str,
    content_type: str,
    expected: str,
) -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path)

    assert response.status_code == 200
    assert content_type in response.headers["content-type"]
    assert expected in response.text
