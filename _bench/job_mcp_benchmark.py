from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOB_MCP_ROOT = PROJECT_ROOT / ".akashic-workspace" / "mcp" / "job-mcp"
if str(JOB_MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(JOB_MCP_ROOT))

from src import job_backend


@dataclass
class CaseResult:
    case_id: str
    expected_kind: str
    expected_count_min: int
    actual_count: int
    pass_: bool
    elapsed_ms: float
    event_ids: list[str]


def _write_config(workspace: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "job_mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "db_path": str(workspace / "job_mcp.sqlite3"),
                "followup_after_days": 3,
                "interview_remind_within_hours": 24,
                "event_ack_ttl_hours": 72,
                "high_priority_apply_after_days": 2,
                "max_events": 50,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return config_path


def _reset_workspace(workspace: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    for name in ("job_mcp.sqlite3", "job_mcp.sqlite3-shm", "job_mcp.sqlite3-wal"):
        path = workspace / name
        if path.exists():
            path.unlink()
    config_path = _write_config(workspace)
    job_backend._config_path = lambda: config_path  # type: ignore[method-assign]
    return config_path


def _run_case(case_id: str, kind: str, expected_count_min: int, now_iso: str) -> CaseResult:
    started = time.perf_counter()
    events = job_backend.get_proactive_events(kind=kind, now_iso=now_iso)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return CaseResult(
        case_id=case_id,
        expected_kind=kind,
        expected_count_min=expected_count_min,
        actual_count=len(events),
        pass_=len(events) >= expected_count_min and all(event["kind"] == kind for event in events),
        elapsed_ms=round(elapsed_ms, 3),
        event_ids=[str(event["event_id"]) for event in events],
    )


def _seed(now: datetime) -> dict[str, Any]:
    due_job = job_backend.add_job(
        company="DueAction AI",
        title="Agent Backend Engineer",
        source="manual",
        url="https://example.com/jobs/due-action",
        priority="high",
        tags="agent,llm",
    )["job"]
    job_backend.update_next_action(
        due_job["id"],
        "send follow-up email",
        (now - timedelta(hours=1)).isoformat(),
    )

    stale_job = job_backend.add_job(
        company="StaleApply Inc",
        title="RAG Platform Engineer",
        source="manual",
        url="https://example.com/jobs/stale-apply",
        priority="normal",
        tags="rag,backend",
    )["job"]
    job_backend.update_status(stale_job["id"], "applied")

    high_job = job_backend.add_job(
        company="Priority Runtime",
        title="LLM Agent Engineer",
        source="manual",
        url="https://example.com/jobs/priority-runtime",
        priority="high",
        tags="agent,runtime",
    )["job"]

    return {
        "due_job_id": due_job["id"],
        "stale_job_id": stale_job["id"],
        "high_job_id": high_job["id"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    workspace = Path(args.workspace).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = _reset_workspace(workspace)
    now = datetime.now(timezone(timedelta(hours=8)))
    seeded = _seed(now)
    future = now + timedelta(days=4)

    cases = [
        _run_case("due_next_action_alert", "alert", 1, now.isoformat()),
        _run_case("stale_application_alert", "alert", 2, future.isoformat()),
        _run_case("high_priority_content", "content", 1, future.isoformat()),
        _run_case("status_summary_context", "context", 1, future.isoformat()),
    ]

    before_ack = job_backend.get_proactive_events(kind="alert", now_iso=now.isoformat())
    ack_ids = [str(event["event_id"]) for event in before_ack]
    ack = job_backend.acknowledge_events(ack_ids)
    after_ack = job_backend.get_proactive_events(kind="alert", now_iso=now.isoformat())
    ack_suppression_ok = bool(ack_ids) and len(after_ack) == 0

    elapsed_values = [case.elapsed_ms for case in cases]
    passed = sum(1 for case in cases if case.pass_)
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "benchmark": "job_mcp_benchmark",
        "workspace": str(workspace),
        "config_path": str(config_path),
        "seeded": seeded,
        "case_count": len(cases),
        "passed": passed,
        "accuracy_pct": round(passed / len(cases) * 100, 2),
        "avg_event_generation_ms": round(statistics.mean(elapsed_values), 3),
        "p95_event_generation_ms": round(max(elapsed_values), 3),
        "ack_suppression_ok": ack_suppression_ok,
        "acknowledged_count": len(ack.get("acknowledged", [])),
        "cases": [asdict(case) for case in cases],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"job_mcp_benchmark_{timestamp}.json"
    md_path = output_dir / f"job_mcp_benchmark_{timestamp}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    result["json_path"] = str(json_path)
    result["md_path"] = str(md_path)
    return result


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Job MCP Benchmark",
        "",
        "This benchmark uses an isolated SQLite workspace. It does not touch the normal Job MCP database, memory files, or sessions.",
        "",
        f"- Created at: `{result['created_at']}`",
        f"- Workspace: `{result['workspace']}`",
        f"- Accuracy: `{result['accuracy_pct']}%`",
        f"- Avg event generation: `{result['avg_event_generation_ms']} ms`",
        f"- P95 event generation: `{result['p95_event_generation_ms']} ms`",
        f"- Ack suppression ok: `{result['ack_suppression_ok']}`",
        "",
        "## Cases",
        "",
        "| Case | Expected kind | Min count | Actual count | Pass | Elapsed ms |",
        "|---|---|---:|---:|---|---:|",
    ]
    for case in result["cases"]:
        lines.append(
            "| {case_id} | {kind} | {min_count} | {actual} | {passed} | {elapsed} |".format(
                case_id=case["case_id"],
                kind=case["expected_kind"],
                min_count=case["expected_count_min"],
                actual=case["actual_count"],
                passed=case["pass_"],
                elapsed=case["elapsed_ms"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Job MCP benchmark.")
    parser.add_argument(
        "--workspace",
        default=str(PROJECT_ROOT / "_bench" / "workspaces" / "job_mcp"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "_bench" / "results"),
    )
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
