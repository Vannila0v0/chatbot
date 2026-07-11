#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Candidate:
    id: str
    kind: str
    score: int
    reason: str
    session_key: str
    ts: str
    user_input: str
    assistant_output: str = ""
    tool_names: list[str] | None = None
    query: str = ""
    expected: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    context: list[dict[str, Any]] | None = None
    annotation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "kind": self.kind,
            "score": self.score,
            "reason": self.reason,
            "session_key": self.session_key,
            "ts": self.ts,
            "user_input": self.user_input,
            "assistant_output": self.assistant_output,
            "tool_names": self.tool_names or [],
            "query": self.query,
            "expected": self.expected or _blank_expected(),
            "evidence": self.evidence or {},
            "context": self.context or [],
            "annotation": self.annotation or _blank_annotation(),
        }
        return data


def _blank_expected() -> dict[str, Any]:
    return {
        "expected_behavior": "",
        "expected_keywords": [],
        "expected_memory_ids": [],
        "expected_source_refs": [],
        "expected_tools": [],
        "forbidden_tools": [],
    }


def _blank_annotation() -> dict[str, Any]:
    return {
        "is_badcase": None,
        "failure_type": "",
        "severity": "",
        "notes": "",
    }


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    return con


def _json_loads_maybe(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _clip(text: str | None, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def _tool_names(tool_calls: str | None) -> list[str]:
    raw = _json_loads_maybe(tool_calls, [])
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    return names


def _hit_summaries(hits_json: str | None, limit: int = 5) -> list[dict[str, Any]]:
    raw = _json_loads_maybe(hits_json, [])
    if isinstance(raw, dict):
        for key in ("items", "hits", "results"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        return []
    hits: list[dict[str, Any]] = []
    for hit in raw[:limit]:
        if not isinstance(hit, dict):
            continue
        hits.append(
            {
                "id": hit.get("id") or hit.get("item_id") or hit.get("memory_id"),
                "source_ref": hit.get("source_ref"),
                "score": hit.get("score"),
                "summary": _clip(
                    hit.get("summary") or hit.get("text") or hit.get("content"),
                    240,
                ),
            }
        )
    return hits


def _fetch_context(
    sessions_con: sqlite3.Connection | None,
    session_key: str,
    user_input: str,
    ts: str,
    before: int,
    after: int,
) -> list[dict[str, Any]]:
    if sessions_con is None or not session_key:
        return []

    anchor = sessions_con.execute(
        """
        SELECT seq FROM messages
        WHERE session_key = ? AND role = 'user' AND content = ?
        ORDER BY ts DESC
        LIMIT 1
        """,
        (session_key, user_input),
    ).fetchone()

    if anchor is None:
        anchor = sessions_con.execute(
            """
            SELECT seq FROM messages
            WHERE session_key = ? AND ts <= ?
            ORDER BY ts DESC
            LIMIT 1
            """,
            (session_key, ts),
        ).fetchone()

    if anchor is None:
        return []

    seq = int(anchor["seq"])
    rows = sessions_con.execute(
        """
        SELECT id, seq, role, ts, content
        FROM messages
        WHERE session_key = ? AND seq BETWEEN ? AND ?
        ORDER BY seq ASC
        """,
        (session_key, max(0, seq - before), seq + after),
    ).fetchall()

    return [
        {
            "message_id": row["id"],
            "seq": row["seq"],
            "role": row["role"],
            "ts": row["ts"],
            "content": _clip(row["content"], 600),
        }
        for row in rows
    ]


def _turn_candidate(
    row: sqlite3.Row,
    *,
    kind: str,
    score: int,
    reason: str,
    sessions_con: sqlite3.Connection | None,
    context_before: int,
    context_after: int,
) -> Candidate:
    user_input = str(row["user_msg"] or "")
    session_key = str(row["session_key"] or "")
    ts = str(row["ts"] or "")
    tools = _tool_names(row["tool_calls"])
    return Candidate(
        id=f"turn:{row['id']}:{kind}",
        kind=kind,
        score=score,
        reason=reason,
        session_key=session_key,
        ts=ts,
        user_input=user_input,
        assistant_output=_clip(row["llm_output"], 1000),
        tool_names=tools,
        evidence={
            "observe_turn_id": row["id"],
            "source": row["source"],
            "error": row["error"],
            "prompt_tokens": row["prompt_tokens"],
            "history_tokens": row["history_tokens"],
            "react_iterations": row["react_iteration_count"],
            "raw_tool_calls": _json_loads_maybe(row["tool_calls"], row["tool_calls"]),
        },
        context=_fetch_context(
            sessions_con,
            session_key,
            user_input,
            ts,
            context_before,
            context_after,
        ),
    )


def collect_turn_candidates(
    observe_con: sqlite3.Connection,
    sessions_con: sqlite3.Connection | None,
    limit: int,
    context_before: int,
    context_after: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []

    error_rows = observe_con.execute(
        """
        SELECT * FROM turns
        WHERE COALESCE(error, '') <> '' OR TRIM(COALESCE(llm_output, '')) = ''
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    for row in error_rows:
        candidates.append(
            _turn_candidate(
                row,
                kind="turn_error_or_empty_reply",
                score=100,
                reason="Turn has an error or empty assistant output.",
                sessions_con=sessions_con,
                context_before=context_before,
                context_after=context_after,
            )
        )

    tool_rows = observe_con.execute(
        """
        SELECT * FROM turns
        WHERE tool_calls IS NOT NULL AND TRIM(tool_calls) <> ''
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    for row in tool_rows:
        tools = set(_tool_names(row["tool_calls"]))
        score = 70
        reason = "Turn used tools; useful for checking tool choice and tool results."
        if {"recall_memory", "search_messages", "fetch_messages"} & tools:
            score = 85
            reason = "Turn used memory/history tools; useful for recall bad cases."
        candidates.append(
            _turn_candidate(
                row,
                kind="tool_turn",
                score=score,
                reason=reason,
                sessions_con=sessions_con,
                context_before=context_before,
                context_after=context_after,
            )
        )

    recent_rows = observe_con.execute(
        """
        SELECT * FROM turns
        ORDER BY id DESC
        LIMIT ?
        """,
        (max(10, limit // 2),),
    ).fetchall()
    for row in recent_rows:
        candidates.append(
            _turn_candidate(
                row,
                kind="recent_turn_for_manual_review",
                score=30,
                reason="Recent turn included for manual bad-case triage.",
                sessions_con=sessions_con,
                context_before=context_before,
                context_after=context_after,
            )
        )

    return candidates


def collect_rag_candidates(
    observe_con: sqlite3.Connection,
    sessions_con: sqlite3.Connection | None,
    limit: int,
    context_before: int,
    context_after: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    rows = observe_con.execute(
        """
        SELECT * FROM rag_queries
        WHERE COALESCE(error, '') <> '' OR injected_count = 0
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    for row in rows:
        has_error = bool(row["error"])
        kind = "rag_error" if has_error else "rag_zero_injected"
        score = 95 if has_error else 80
        reason = (
            "RAG query errored."
            if has_error
            else "RAG query returned no injected memories; likely recall-miss candidate."
        )
        session_key = str(row["session_key"] or "")
        ts = str(row["ts"] or "")
        query = str(row["query"] or "")
        candidates.append(
            Candidate(
                id=f"rag:{row['id']}:{kind}",
                kind=kind,
                score=score,
                reason=reason,
                session_key=session_key,
                ts=ts,
                user_input="",
                query=query,
                evidence={
                    "rag_query_id": row["id"],
                    "caller": row["caller"],
                    "orig_query": row["orig_query"],
                    "aux_queries": _json_loads_maybe(row["aux_queries"], []),
                    "injected_count": row["injected_count"],
                    "route_decision": row["route_decision"],
                    "error": row["error"],
                    "hits": _hit_summaries(row["hits_json"]),
                },
                context=_fetch_context(
                    sessions_con,
                    session_key,
                    "",
                    ts,
                    context_before,
                    context_after,
                ),
            )
        )
    return candidates


def collect_recall_inspector_candidates(
    recall_path: Path,
    limit: int,
) -> list[Candidate]:
    if not recall_path.exists():
        return []

    candidates: list[Candidate] = []
    lines = recall_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(reversed(lines), start=1):
        if len(candidates) >= limit:
            break
        try:
            item = json.loads(line)
        except Exception:
            continue

        kind = item.get("kind")
        payload = item.get("recall_memory") if kind == "recall_memory" else None
        if not isinstance(payload, dict):
            continue
        count = int(payload.get("count") or 0)
        status = str(payload.get("status") or "")
        if status == "success" and count > 0:
            continue

        args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        query = str(args.get("query") or "")
        candidates.append(
            Candidate(
                id=f"recall_inspector:{len(lines) - idx + 1}",
                kind="recall_memory_zero_or_error",
                score=82,
                reason="recall_memory returned zero items or failed.",
                session_key=str(item.get("session_key") or ""),
                ts=str(item.get("created_at") or item.get("timestamp") or ""),
                user_input=str(item.get("user_text") or ""),
                query=query,
                evidence={
                    "turn_id": item.get("turn_id"),
                    "channel": item.get("channel"),
                    "chat_id": item.get("chat_id"),
                    "recall_status": status,
                    "recall_count": count,
                    "arguments": args,
                    "items": payload.get("items") or [],
                },
            )
        )
    return candidates


def dedupe_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: (c.score, c.ts), reverse=True):
        key = (
            candidate.kind,
            candidate.session_key,
            candidate.user_input or candidate.query,
            candidate.ts[:19],
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def write_jsonl(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for candidate in candidates:
            f.write(json.dumps(candidate.to_dict(), ensure_ascii=False) + "\n")


def write_markdown(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Bad Case Candidates",
        "",
        "Use this file for quick review. Put confirmed cases into bad_cases.jsonl.",
        "",
    ]
    for i, candidate in enumerate(candidates, start=1):
        data = candidate.to_dict()
        lines.extend(
            [
                f"## {i}. {candidate.id}",
                "",
                f"- kind: `{candidate.kind}`",
                f"- score: `{candidate.score}`",
                f"- session: `{candidate.session_key}`",
                f"- ts: `{candidate.ts}`",
                f"- reason: {candidate.reason}",
                "",
            ]
        )
        if candidate.user_input:
            lines.extend(["User:", "", "```text", candidate.user_input, "```", ""])
        if candidate.query:
            lines.extend(["Query:", "", "```text", candidate.query, "```", ""])
        if candidate.assistant_output:
            lines.extend(
                [
                    "Assistant:",
                    "",
                    "```text",
                    candidate.assistant_output,
                    "```",
                    "",
                ]
            )
        if data["tool_names"]:
            lines.append(f"Tools: `{', '.join(data['tool_names'])}`")
            lines.append("")
        if data["context"]:
            lines.append("Context:")
            lines.append("")
            for msg in data["context"]:
                content = str(msg.get("content") or "").replace("\n", " ")
                lines.append(
                    f"- `{msg.get('seq')}` `{msg.get('role')}` "
                    f"{_clip(content, 180)}"
                )
            lines.append("")
        lines.append("Annotation stub:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(data["expected"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export bad-case candidates from Akashic observe/session data."
    )
    parser.add_argument("--workspace", default=".akashic-workspace")
    parser.add_argument("--out", default="bad_cases/candidates.jsonl")
    parser.add_argument("--review-md", default="bad_cases/candidates.md")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--context-before", type=int, default=3)
    parser.add_argument("--context-after", type=int, default=2)
    parser.add_argument("--no-markdown", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace)
    observe_db = workspace / "observe" / "observe.db"
    sessions_db = workspace / "sessions.db"
    recall_path = workspace / "observe" / "recall_inspector.jsonl"

    if not observe_db.exists():
        raise SystemExit(f"observe db not found: {observe_db}")

    observe_con = _connect_readonly(observe_db)
    sessions_con = _connect_readonly(sessions_db) if sessions_db.exists() else None
    try:
        candidates = []
        candidates.extend(
            collect_turn_candidates(
                observe_con,
                sessions_con,
                args.limit,
                args.context_before,
                args.context_after,
            )
        )
        candidates.extend(
            collect_rag_candidates(
                observe_con,
                sessions_con,
                args.limit,
                args.context_before,
                args.context_after,
            )
        )
        candidates.extend(
            collect_recall_inspector_candidates(recall_path, args.limit)
        )
        candidates = dedupe_candidates(candidates)[: args.limit]
    finally:
        observe_con.close()
        if sessions_con is not None:
            sessions_con.close()

    out = Path(args.out)
    write_jsonl(out, candidates)
    if not args.no_markdown:
        write_markdown(Path(args.review_md), candidates)

    print(f"exported {len(candidates)} candidates")
    print(f"jsonl: {out}")
    if not args.no_markdown:
        print(f"review: {args.review_md}")


if __name__ == "__main__":
    main()
