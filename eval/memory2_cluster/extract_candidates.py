from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)(api[ _-]?key|access[ _-]?token|token|password|secret)"
            r"\s*(?:是|为|[:=])?\s*[A-Za-z0-9_./+=-]{8,}"
        ),
        r"\1=[凭据已脱敏]",
    ),
    (
        re.compile(r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{32,}(?![A-Fa-f0-9])"),
        "[长十六进制凭据已脱敏]",
    ),
    (re.compile(r"(?<!\d)\d{8,}(?!\d)"), "[长数字账号已脱敏]"),
    (re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,}"), "@[账号已脱敏]"),
    (re.compile(r"[A-Za-z]:\\(?:[^\s\\]+\\)*[^\s]+"), "[本地路径已脱敏]"),
    (re.compile(r"https?://[^\s)\]>]+"), "[链接已脱敏]"),
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[邮箱已脱敏]",
    ),
)

def sanitize_content(
    content: str, replacements: dict[str, str] | None = None
) -> str:
    sanitized = content
    for source, replacement in (replacements or {}).items():
        sanitized = sanitized.replace(source, replacement)
    for pattern, replacement in _REDACTIONS:
        sanitized = pattern.sub(replacement, sanitized)
    sanitized = re.sub(
        r"(Steam\s*名叫)\s*\*\*[^*]+\*\*",
        r"\1 **[用户名已脱敏]**",
        sanitized,
        flags=re.IGNORECASE,
    )
    if "正在游戏中" in sanitized or "在线在玩的" in sanitized:
        sanitized = re.sub(
            r"(?m)^(-\s*)\*\*[^*]+\*\*(\s*[—-])",
            r"\1**[好友昵称已脱敏]**\2",
            sanitized,
        )
    return sanitized.strip()


def _source_ref(session_key: str, seq: int) -> str:
    digest = hashlib.sha256(f"{session_key}:{seq}".encode()).hexdigest()[:16]
    return f"source_{digest}"


def _message_kind(role: str, extra: str | None) -> str:
    if role != "assistant" or not extra:
        return role
    try:
        payload = json.loads(extra)
    except (TypeError, json.JSONDecodeError):
        return role
    return "proactive" if payload.get("proactive") else role


def load_windows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    windows = payload.get("windows") or []
    if not windows:
        raise ValueError("windows 配置不能为空")
    previous_end: datetime | None = None
    seen: set[str] = set()
    for window in windows:
        timeline_id = str(window["timeline_id"])
        start = datetime.fromisoformat(str(window["start"]))
        end = datetime.fromisoformat(str(window["end"]))
        if timeline_id in seen:
            raise ValueError(f"重复 timeline_id: {timeline_id}")
        if start >= end:
            raise ValueError(f"无效窗口: {timeline_id}")
        if previous_end is not None and start < previous_end:
            raise ValueError(f"窗口重叠: {timeline_id}")
        seen.add(timeline_id)
        previous_end = end
    return windows


def extract_timelines(
    db_path: Path,
    windows_path: Path,
    output_path: Path,
    replacements_path: Path | None = None,
) -> list[dict[str, Any]]:
    windows = load_windows(windows_path)
    replacements = (
        json.loads(replacements_path.read_text(encoding="utf-8"))
        if replacements_path
        else {}
    )
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    timelines: list[dict[str, Any]] = []
    try:
        for window in windows:
            rows = connection.execute(
                """
                SELECT session_key, seq, role, content, extra, ts
                FROM messages
                WHERE ts >= ? AND ts < ? AND role IN ('user', 'assistant')
                ORDER BY ts, session_key, seq
                """,
                (window["start"], window["end"]),
            ).fetchall()
            messages = []
            for row in rows:
                content = sanitize_content(
                    str(row["content"] or ""), replacements=replacements
                )
                if not content:
                    continue
                messages.append(
                    {
                        "source_ref": _source_ref(str(row["session_key"]), int(row["seq"])),
                        "timestamp": row["ts"],
                        "channel": str(row["session_key"]).split(":", 1)[0],
                        "kind": _message_kind(str(row["role"]), row["extra"]),
                        "content": content,
                    }
                )
            timelines.append(
                {
                    "timeline_id": window["timeline_id"],
                    "start": window["start"],
                    "end": window["end"],
                    "selection_rule": "窗口内完整保留非空用户消息、助手回复和主动推送",
                    "source": "sanitized_akashic_sessions_db",
                    "review_status": "candidate_unlabeled",
                    "messages": messages,
                }
            )
    finally:
        connection.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for timeline in timelines:
            handle.write(json.dumps(timeline, ensure_ascii=False) + "\n")
    return timelines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只读抽取脱敏的自然对话候选时间线")
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replacements", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    timelines = extract_timelines(
        args.db, args.windows, args.output, replacements_path=args.replacements
    )
    summary = [
        {
            "timeline_id": timeline["timeline_id"],
            "messages": len(timeline["messages"]),
            "users": sum(message["kind"] == "user" for message in timeline["messages"]),
            "proactive": sum(
                message["kind"] == "proactive" for message in timeline["messages"]
            ),
        }
        for timeline in timelines
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
