from __future__ import annotations

import base64
import hashlib
import hmac
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import Request
from starlette.responses import Response

WEB_IDENTITY_COOKIE = "akashic_web_user"
_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365


@dataclass(frozen=True)
class WebPrincipal:
    user_id: str
    cookie_value: str
    should_set_cookie: bool = False


class WebIdentityService:
    """Issue and validate an anonymous, server-signed Web user identity."""

    def __init__(self, secret: bytes) -> None:
        if len(secret) < 32:
            raise ValueError("web identity secret must contain at least 32 bytes")
        self._secret = bytes(secret)

    @classmethod
    def from_workspace(cls, workspace: Path) -> "WebIdentityService":
        secret_path = workspace / "web_identity.key"
        if secret_path.exists():
            secret = secret_path.read_bytes()
        else:
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            secret = os.urandom(32)
            secret_path.write_bytes(secret)
            try:
                secret_path.chmod(0o600)
            except OSError:
                pass
        return cls(secret)

    def resolve(self, request: Request) -> WebPrincipal:
        raw = str(request.cookies.get(WEB_IDENTITY_COOKIE) or "").strip()
        user_id = self._verify(raw)
        if user_id is not None:
            return WebPrincipal(user_id=user_id, cookie_value=raw)

        user_id = str(uuid4())
        return WebPrincipal(
            user_id=user_id,
            cookie_value=self._sign(user_id),
            should_set_cookie=True,
        )

    def apply_cookie(
        self,
        response: Response,
        request: Request,
        principal: WebPrincipal,
    ) -> None:
        if not principal.should_set_cookie:
            return
        response.set_cookie(
            key=WEB_IDENTITY_COOKIE,
            value=principal.cookie_value,
            max_age=_COOKIE_MAX_AGE_SECONDS,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="lax",
            path="/",
        )

    def _sign(self, user_id: str) -> str:
        signature = hmac.new(
            self._secret,
            user_id.encode("ascii"),
            hashlib.sha256,
        ).digest()
        encoded = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
        return f"{user_id}.{encoded}"

    def _verify(self, cookie_value: str) -> str | None:
        try:
            user_id, provided_signature = cookie_value.split(".", 1)
            normalized_user_id = str(UUID(user_id))
        except (ValueError, AttributeError):
            return None
        expected = self._sign(normalized_user_id).split(".", 1)[1]
        if not hmac.compare_digest(provided_signature, expected):
            return None
        return normalized_user_id
