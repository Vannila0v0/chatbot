from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles


def mount_chat_ui(app: FastAPI, static_dir: Path) -> None:
    """Mount the standalone Web chat surface and its versioned assets."""
    app.mount(
        "/chat/assets",
        StaticFiles(directory=static_dir),
        name="chat-assets",
    )

    @app.get("/chat", include_in_schema=False)
    def chat_index() -> Response:
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        for asset_name in ("app.css", "app.js"):
            version = str((static_dir / asset_name).stat().st_mtime_ns)
            asset_path = f"/chat/assets/{asset_name}"
            html = re.sub(
                rf"({re.escape(asset_path)})(\?[^\"]*)?",
                rf"\1?v={version}",
                html,
            )
        return Response(
            content=html,
            media_type="text/html",
            headers={"Cache-Control": "no-cache"},
        )
