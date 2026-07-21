from __future__ import annotations

from pathlib import Path

from starlette.requests import Request

from web.api.identity import WebIdentityService


def _request(cookie: str = "") -> Request:
    headers = []
    if cookie:
        headers.append((b"cookie", f"akashic_web_user={cookie}".encode("ascii")))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "path": "/api/turns",
            "query_string": b"",
            "headers": headers,
            "server": ("test", 80),
            "client": ("test", 1234),
        }
    )


def test_workspace_identity_secret_is_persistent(tmp_path: Path) -> None:
    first = WebIdentityService.from_workspace(tmp_path)
    second = WebIdentityService.from_workspace(tmp_path)

    issued = first.resolve(_request())
    restored = second.resolve(_request(issued.cookie_value))

    assert restored.user_id == issued.user_id
    assert restored.should_set_cookie is False


def test_forged_identity_cookie_is_rejected() -> None:
    identity = WebIdentityService(b"identity-test-secret".ljust(32, b"!"))
    issued = identity.resolve(_request())
    replaced = identity.resolve(_request(f"{issued.cookie_value}forged"))

    assert replaced.user_id != issued.user_id
    assert replaced.should_set_cookie is True
