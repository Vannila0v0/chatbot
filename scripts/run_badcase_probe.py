from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL_FIELDS = ("id", "user_input", "expected", "annotation")


def _short(text: Any, limit: int = 100) -> str:
    if text is None:
        return ""
    value = str(text).replace("\r", " ").replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return rows, [f"case file not found: {path}"]

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc}")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {line_no}: expected object, got {type(row).__name__}")
            continue
        row["_line_no"] = line_no
        rows.append(row)
    return rows, errors


def _validate_case(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in row:
            errors.append(f"missing field: {field}")
    expected = row.get("expected")
    if not isinstance(expected, dict):
        errors.append("expected must be an object")
    else:
        for list_field in (
            "expected_keywords",
            "expected_memory_ids",
            "expected_source_refs",
            "expected_tools",
            "forbidden_tools",
        ):
            if list_field in expected and not isinstance(expected[list_field], list):
                errors.append(f"expected.{list_field} must be a list")
    annotation = row.get("annotation")
    if not isinstance(annotation, dict):
        errors.append("annotation must be an object")
    return errors


class WorkspaceIndex:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.sessions_db = workspace / "sessions.db"
        self.observe_db = workspace / "observe" / "observe.db"
        self.sessions_con = self._connect(self.sessions_db)
        self.observe_con = self._connect(self.observe_db)

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection | None:
        if not path.exists():
            return None
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        return con

    def close(self) -> None:
        if self.sessions_con is not None:
            self.sessions_con.close()
        if self.observe_con is not None:
            self.observe_con.close()

    def search_ref(self, ref: str, limit: int = 3) -> dict[str, Any]:
        needle = f"%{ref}%"
        samples: list[dict[str, Any]] = []
        counts: dict[str, int] = {}

        if self.sessions_con is not None:
            counts["sessions"] = self._count(
                self.sessions_con,
                "select count(*) from sessions where key like ?",
                (needle,),
            )
            for row in self.sessions_con.execute(
                "select key, updated_at from sessions where key like ? order by updated_at desc limit ?",
                (needle, limit),
            ):
                samples.append(
                    {
                        "table": "sessions",
                        "session_key": row["key"],
                        "ts": row["updated_at"],
                        "snippet": row["key"],
                    }
                )

            counts["messages"] = self._count(
                self.sessions_con,
                """
                select count(*) from messages
                where id like ? or session_key like ? or content like ?
                """,
                (needle, needle, needle),
            )
            for row in self.sessions_con.execute(
                """
                select id, session_key, role, seq, ts, substr(content, 1, 160) as snippet
                from messages
                where id like ? or session_key like ? or content like ?
                order by ts desc
                limit ?
                """,
                (needle, needle, needle, limit),
            ):
                samples.append(
                    {
                        "table": "messages",
                        "id": row["id"],
                        "session_key": row["session_key"],
                        "role": row["role"],
                        "seq": row["seq"],
                        "ts": row["ts"],
                        "snippet": row["snippet"],
                    }
                )

        if self.observe_con is not None:
            counts["observe_turns"] = self._count(
                self.observe_con,
                """
                select count(*) from turns
                where session_key like ? or user_msg like ? or llm_output like ?
                """,
                (needle, needle, needle),
            )
            for row in self.observe_con.execute(
                """
                select id, session_key, ts, substr(user_msg, 1, 120) as user_snippet
                from turns
                where session_key like ? or user_msg like ? or llm_output like ?
                order by ts desc
                limit ?
                """,
                (needle, needle, needle, limit),
            ):
                samples.append(
                    {
                        "table": "observe.turns",
                        "id": row["id"],
                        "session_key": row["session_key"],
                        "ts": row["ts"],
                        "snippet": row["user_snippet"],
                    }
                )

        return {"ref": ref, "count": sum(counts.values()), "counts": counts, "samples": samples[:limit]}

    def search_keyword(self, keyword: str) -> dict[str, Any]:
        if not keyword:
            return {"keyword": keyword, "count": 0, "counts": {}}
        needle = f"%{keyword}%"
        counts: dict[str, int] = {}
        if self.sessions_con is not None:
            counts["messages"] = self._count(
                self.sessions_con,
                "select count(*) from messages where content like ?",
                (needle,),
            )
        if self.observe_con is not None:
            counts["observe_turns"] = self._count(
                self.observe_con,
                "select count(*) from turns where user_msg like ? or llm_output like ?",
                (needle, needle),
            )
            counts["rag_queries"] = self._count(
                self.observe_con,
                "select count(*) from rag_queries where query like ? or hits_json like ?",
                (needle, needle),
            )
        return {"keyword": keyword, "count": sum(counts.values()), "counts": counts}

    @staticmethod
    def _count(con: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> int:
        row = con.execute(sql, params).fetchone()
        return int(row[0]) if row is not None else 0


def _analyze_case(row: dict[str, Any], index: WorkspaceIndex) -> dict[str, Any]:
    expected = row.get("expected") or {}
    evidence = row.get("evidence") or {}
    annotation = row.get("annotation") or {}

    assistant_output = row.get("assistant_output") or ""
    expected_keywords = [str(item) for item in expected.get("expected_keywords", []) if str(item)]
    expected_tools = [str(item) for item in expected.get("expected_tools", []) if str(item)]
    forbidden_tools = [str(item) for item in expected.get("forbidden_tools", []) if str(item)]
    expected_source_refs = [str(item) for item in expected.get("expected_source_refs", []) if str(item)]
    observed_tools = [str(item) for item in evidence.get("tool_names", []) if str(item)]

    keyword_hits = [item for item in expected_keywords if item in assistant_output]
    keyword_misses = [item for item in expected_keywords if item not in assistant_output]
    missing_expected_tools = [item for item in expected_tools if item not in observed_tools]
    forbidden_tool_hits = [item for item in forbidden_tools if item in observed_tools]

    ref_checks = [index.search_ref(ref) for ref in expected_source_refs]
    missing_refs = [item["ref"] for item in ref_checks if item["count"] == 0]
    keyword_evidence = [index.search_keyword(item) for item in expected_keywords[:8]]

    symptoms: list[str] = []
    if not assistant_output:
        symptoms.append("original assistant output is empty")
    if keyword_misses:
        symptoms.append("original output misses expected keywords")
    if missing_expected_tools:
        symptoms.append("observed run missed expected tools")
    if forbidden_tool_hits:
        symptoms.append("observed run used forbidden tools")
    if missing_refs:
        symptoms.append("expected source refs are not found in workspace")
    if (
        annotation.get("failure_type") == "recall_miss"
        and evidence.get("recall_count") == 0
        and ref_checks
        and all(item["count"] > 0 for item in ref_checks)
    ):
        symptoms.append("recall returned zero although source evidence exists in workspace")

    return {
        "id": row.get("id"),
        "line_no": row.get("_line_no"),
        "source_candidate_id": row.get("source_candidate_id"),
        "session_key": row.get("session_key"),
        "failure_type": annotation.get("failure_type"),
        "severity": annotation.get("severity"),
        "schema_errors": _validate_case(row),
        "user_input": row.get("user_input", ""),
        "assistant_output": assistant_output,
        "expected_behavior": expected.get("expected_behavior", ""),
        "observed_tools": observed_tools,
        "expected_tools": expected_tools,
        "missing_expected_tools": missing_expected_tools,
        "forbidden_tools": forbidden_tools,
        "forbidden_tool_hits": forbidden_tool_hits,
        "expected_keywords": expected_keywords,
        "keyword_hits_in_original_output": keyword_hits,
        "keyword_misses_in_original_output": keyword_misses,
        "expected_source_refs": expected_source_refs,
        "source_ref_checks": ref_checks,
        "missing_source_refs": missing_refs,
        "keyword_evidence_in_workspace": keyword_evidence,
        "symptoms": symptoms,
        "status": "actionable" if not _validate_case(row) and symptoms else "needs_manual_review",
    }


def _next_step_for(failure_type: str | None) -> str:
    mapping = {
        "recall_miss": "下一步接真实 recall/search 探针，验证应该召回的历史消息能否进入上下文。",
        "wrong_channel": "下一步加渠道一致性检查：当入口渠道和 proactive.target 不一致时，回复里要提示实际推送目标。",
        "tool_not_called": "下一步加工具选择成本测试：简单寒暄不应触发文件读取或检索工具。",
        "hallucination": "下一步加回答事实/澄清检查：缺关键条件时先追问或给条件化判断。",
        "no_response": "下一步接消息接收链路探针，检查 polling/长连接是否恢复到可接收状态。",
    }
    return mapping.get(failure_type or "", "下一步补充更明确的 expected 字段，再接自动化回放。")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Bad Case Probe Report")
    lines.append("")
    lines.append(f"- 运行时间：{report['run_at']}")
    lines.append(f"- 用例文件：`{report['cases_path']}`")
    lines.append(f"- Workspace：`{report['workspace']}`")
    lines.append(f"- 用例数：{report['summary']['total_cases']}")
    lines.append(f"- 可行动条目：{report['summary']['actionable_cases']}")
    lines.append(f"- 需要人工复核：{report['summary']['needs_manual_review_cases']}")
    lines.append("")

    if report["parse_errors"]:
        lines.append("## 解析错误")
        lines.append("")
        for item in report["parse_errors"]:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## 分类统计")
    lines.append("")
    for key, value in report["summary"]["by_failure_type"].items():
        lines.append(f"- `{key}`：{value}")
    lines.append("")

    for case in report["cases"]:
        lines.append(f"## {case['id']}")
        lines.append("")
        lines.append(f"- 类型：`{case.get('failure_type')}` / 严重度：`{case.get('severity')}`")
        lines.append(f"- 用户输入：{_short(case.get('user_input'), 140)}")
        lines.append(f"- 原回复：{_short(case.get('assistant_output'), 180) or '(空)'}")
        lines.append(f"- 状态：`{case['status']}`")
        if case["schema_errors"]:
            lines.append(f"- Schema 问题：{'; '.join(case['schema_errors'])}")
        lines.append("")

        lines.append("### 本轮观察")
        lines.append("")
        if case["symptoms"]:
            for symptom in case["symptoms"]:
                lines.append(f"- {symptom}")
        else:
            lines.append("- 未从静态探针里发现明确症状，需要人工继续复核。")
        lines.append("")

        lines.append("### 关键检查")
        lines.append("")
        lines.append(
            f"- 期望工具：{case['expected_tools'] or '[]'}；实际工具：{case['observed_tools'] or '[]'}；缺失：{case['missing_expected_tools'] or '[]'}"
        )
        lines.append(
            f"- 禁用工具：{case['forbidden_tools'] or '[]'}；命中禁用：{case['forbidden_tool_hits'] or '[]'}"
        )
        lines.append(
            f"- 原回复命中的期望关键词：{case['keyword_hits_in_original_output'] or '[]'}"
        )
        lines.append(
            f"- 原回复缺失的期望关键词：{case['keyword_misses_in_original_output'] or '[]'}"
        )
        if case["source_ref_checks"]:
            for ref in case["source_ref_checks"]:
                lines.append(f"- Source ref `{ref['ref']}` 在 workspace 中命中 {ref['count']} 条")
        else:
            lines.append("- Source ref：未配置")
        lines.append("")

        lines.append("### 建议下一步")
        lines.append("")
        lines.append(f"- {_next_step_for(case.get('failure_type'))}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(cases_path: Path, workspace: Path) -> dict[str, Any]:
    rows, parse_errors = _read_jsonl(cases_path)
    index = WorkspaceIndex(workspace)
    try:
        cases = [_analyze_case(row, index) for row in rows]
    finally:
        index.close()

    by_failure_type = Counter(case.get("failure_type") or "unknown" for case in cases)
    by_status = Counter(case.get("status") or "unknown" for case in cases)
    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path),
        "workspace": str(workspace),
        "parse_errors": parse_errors,
        "summary": {
            "total_cases": len(cases),
            "actionable_cases": by_status.get("actionable", 0),
            "needs_manual_review_cases": by_status.get("needs_manual_review", 0),
            "by_failure_type": dict(sorted(by_failure_type.items())),
            "by_status": dict(sorted(by_status.items())),
        },
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight bad-case probe.")
    parser.add_argument("--cases", default="bad_cases/bad_cases.jsonl", help="JSONL bad case file")
    parser.add_argument("--workspace", default=".akashic-workspace", help="Akashic workspace directory")
    parser.add_argument("--out-json", default="bad_cases/probe_report.json", help="JSON report path")
    parser.add_argument("--out-md", default="bad_cases/probe_report.md", help="Markdown report path")
    args = parser.parse_args(argv)

    cases_path = Path(args.cases)
    workspace = Path(args.workspace)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    report = build_report(cases_path, workspace)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(out_md, report)

    summary = report["summary"]
    print(f"cases: {summary['total_cases']}")
    print(f"actionable: {summary['actionable_cases']}")
    print(f"needs_manual_review: {summary['needs_manual_review_cases']}")
    print(f"json_report: {out_json}")
    print(f"md_report: {out_md}")

    if report["parse_errors"]:
        return 2
    if any(case["schema_errors"] for case in report["cases"]):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
