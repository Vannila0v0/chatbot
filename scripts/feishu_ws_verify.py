from __future__ import annotations

import os
import sys

import lark_oapi as lark


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(2)
    return value


def on_message(data: lark.im.v1.P2ImMessageReceiveV1) -> None:
    print("[feishu] received im.message.receive_v1 event:")
    print(lark.JSON.marshal(data, indent=2))
    sys.stdout.flush()


def main() -> None:
    app_id = _required_env("FEISHU_APP_ID")
    app_secret = _required_env("FEISHU_APP_SECRET")

    event_handler = (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(on_message)
        .build()
    )
    client = lark.ws.Client(
        app_id,
        app_secret,
        event_handler=event_handler,
        log_level=lark.LogLevel.DEBUG,
    )
    print("[feishu] starting long-connection client...")
    print("[feishu] keep this process running, then click the Feishu console validation button.")
    sys.stdout.flush()
    client.start()


if __name__ == "__main__":
    main()
