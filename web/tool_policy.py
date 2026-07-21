from __future__ import annotations

from dataclasses import dataclass

DEFAULT_WEB_ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "tool_search",
        "web_search",
    }
)


@dataclass(frozen=True)
class WebToolPolicy:
    """A positive allowlist for tools exposed to untrusted Web turns."""

    allowed_tools: frozenset[str] = DEFAULT_WEB_ALLOWED_TOOLS

    def as_message_metadata(self) -> dict[str, list[str]]:
        return {"allowed_tools": sorted(self.allowed_tools)}
