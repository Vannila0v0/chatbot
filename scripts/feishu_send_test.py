from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.channels.feishu_channel import FeishuChannel


async def main() -> None:
    app_id = os.environ.get("FEISHU_APP_ID", "").strip()
    app_secret = os.environ.get("FEISHU_APP_SECRET", "").strip()
    chat_id = os.environ.get("FEISHU_CHAT_ID", "").strip()
    if not app_id or not app_secret or not chat_id:
        print("FEISHU_APP_ID, FEISHU_APP_SECRET, and FEISHU_CHAT_ID are required")
        sys.exit(2)
    channel = FeishuChannel(app_id, app_secret, None, None)
    await channel.send(chat_id, "Akashic 飞书通道发送测试成功。")
    await channel.stop()
    print("sent")


if __name__ == "__main__":
    asyncio.run(main())
