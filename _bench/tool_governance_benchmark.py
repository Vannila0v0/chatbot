from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class SchemaStats:
    label: str
    tool_count: int
    schema_chars: int
    schema_bytes: int
    rough_tokens: int
    tool_names: list[str]


@dataclass
class SearchCase:
    case_id: str
    query: str
    expected_tool: str
    top_k: int
    hit: bool
    rank: int | None
    returned_tools: list[str]


def _schema_text(schemas: list[dict[str, Any]]) -> str:
    return json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 3)


def _stats(label: str, schemas: list[dict[str, Any]], names: list[str]) -> SchemaStats:
    text = _schema_text(schemas)
    return SchemaStats(
        label=label,
        tool_count=len(names),
        schema_chars=len(text),
        schema_bytes=len(text.encode("utf-8")),
        rough_tokens=_estimate_tokens(text),
        tool_names=names,
    )


def _reduction(base: int, current: int) -> float:
    if base <= 0:
        return 0.0
    return round((1 - current / base) * 100, 2)


def _copy_mcp_config(workspace: Path) -> Path:
    src = PROJECT_ROOT / ".akashic-workspace" / "mcp_servers.json"
    dst = workspace / "mcp_servers.json"
    if src.exists():
        shutil.copyfile(src, dst)
    else:
        dst.write_text('{"servers": {}}', encoding="utf-8")
    return dst


def _default_search_queries(all_names: set[str]) -> list[tuple[str, str]]:
    candidates = [
        ("recall job preference from long-term memory", "recall_memory"),
        ("forget wrong memory item", "forget_memory"),
        ("search prior conversation messages", "search_messages"),
        ("read a local markdown file", "read_file"),
        ("add a job application record", "mcp_job__job_add"),
        ("search job applications by keyword", "mcp_job__job_search"),
        ("list current MCP servers", "mcp_list"),
        ("query rss feed content", "mcp_feed__feed_query"),
    ]
    return [(query, tool) for query, tool in candidates if tool in all_names]


def _run_search_cases(tools: Any, *, top_k: int) -> list[SearchCase]:
    all_names = tools.get_registered_names()
    cases: list[SearchCase] = []
    for idx, (query, expected) in enumerate(_default_search_queries(all_names), start=1):
        rows = tools.search(query=query, top_k=top_k)
        returned = [str(row.get("name", "")) for row in rows]
        rank = returned.index(expected) + 1 if expected in returned else None
        cases.append(
            SearchCase(
                case_id=f"search_{idx:03d}",
                query=query,
                expected_tool=expected,
                top_k=top_k,
                hit=rank is not None,
                rank=rank,
                returned_tools=returned,
            )
        )
    return cases


async def _safe_aclose(obj: object | None) -> None:
    if obj is None:
        return
    method = getattr(obj, "aclose", None)
    if callable(method):
        result = method()
        if hasattr(result, "__await__"):
            await result
        return
    method = getattr(obj, "close", None)
    if callable(method):
        method()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from agent.config_models import Config
    from agent.core.runtime_support import ToolDiscoveryState
    from bootstrap.providers import build_providers, build_vl_provider
    from bootstrap.tools import build_registered_tools
    from bus.event_bus import EventBus
    from bus.queue import MessageBus
    from core.net.http import SharedHttpResources
    from session.store import SessionStore

    config_path = Path(args.config).resolve()
    workspace = Path(args.workspace).resolve()
    output_dir = Path(args.output_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_mcp_config = _copy_mcp_config(workspace)

    config = Config.load(config_path)
    http_resources = SharedHttpResources()
    bus = MessageBus()
    event_bus = EventBus()
    session_store = SessionStore(workspace / "sessions.db")
    mcp_registry = None
    memory_runtime = None
    peer_pm = None
    peer_poller = None
    connected_mcp_error: str | None = None

    try:
        provider, light_provider, _agent_provider = build_providers(config)
        vl_provider = build_vl_provider(config)
        (
            tools,
            _push_tool,
            _scheduler,
            mcp_registry,
            memory_runtime,
            peer_pm,
            peer_poller,
        ) = build_registered_tools(
            config,
            workspace,
            http_resources,
            bus=bus,
            provider=provider,
            light_provider=light_provider,
            vl_provider=vl_provider,
            session_store=session_store,
            event_publisher=event_bus,
        )
        try:
            await mcp_registry.load_and_connect_all()
        except Exception as exc:
            connected_mcp_error = f"{type(exc).__name__}: {exc}"

        all_names = tools.get_registered_order()
        always_names = tools.get_registered_order(tools.get_always_on_names())
        deferred_names = [
            name
            for name in all_names
            if name not in set(always_names) and name != "tool_search"
        ]

        preload_names = [
            name.strip()
            for name in str(args.preload_names or "").split(",")
            if name.strip()
        ]
        if not preload_names:
            preferred = [
                "recall_memory",
                "search_messages",
                "read_file",
                "mcp_job__job_search",
                "mcp_job__job_list",
            ]
            preload_names = [name for name in preferred if name in deferred_names]
            if len(preload_names) < int(args.preload_capacity):
                for name in deferred_names:
                    if name not in preload_names:
                        preload_names.append(name)
                    if len(preload_names) >= int(args.preload_capacity):
                        break
        preload_names = [
            name
            for name in preload_names
            if name in deferred_names
        ][: int(args.preload_capacity)]

        discovery = ToolDiscoveryState(capacity=int(args.preload_capacity))
        session_key = "bench:tool-governance"
        discovery.update(session_key, preload_names, set(always_names))
        preloaded = tools.get_registered_order(discovery.get_preloaded(session_key))

        cold_names = always_names
        warm_names = [
            *always_names,
            *[name for name in preloaded if name not in set(always_names)],
        ]

        full_stats = _stats("Full registry", tools.get_schemas(), all_names)
        cold_stats = _stats("Progressive cold start", tools.get_schemas(names=cold_names), cold_names)
        warm_stats = _stats(
            f"Progressive warm LRU ({len(preloaded)} preloaded)",
            tools.get_schemas(names=warm_names),
            warm_names,
        )

        documents = getattr(tools, "_documents", {})
        risk_counts = Counter(
            getattr(doc, "risk", "unknown") for doc in documents.values()
        )
        source_counts = Counter(
            getattr(doc, "source_type", "unknown") for doc in documents.values()
        )
        mcp_tools_by_server: dict[str, list[str]] = {}
        for name, doc in documents.items():
            if getattr(doc, "source_type", "") == "mcp":
                mcp_tools_by_server.setdefault(getattr(doc, "source_name", ""), []).append(name)
        mcp_tools_by_server = {
            server: sorted(names)
            for server, names in sorted(mcp_tools_by_server.items())
        }

        search_cases = _run_search_cases(tools, top_k=int(args.top_k))
        search_total = len(search_cases)
        search_hit = sum(1 for case in search_cases if case.hit)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = {
            "created_at": datetime.now().astimezone().isoformat(),
            "benchmark": "tool_governance_benchmark",
            "config_path": str(config_path),
            "workspace": str(workspace),
            "copied_mcp_config": str(copied_mcp_config),
            "mcp_connect_error": connected_mcp_error,
            "registered_tool_count": len(all_names),
            "always_on_count": len(always_names),
            "deferred_count": len(deferred_names),
            "mcp_tool_count": sum(len(names) for names in mcp_tools_by_server.values()),
            "risk_counts": dict(sorted(risk_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "mcp_tools_by_server": mcp_tools_by_server,
            "preload_capacity": int(args.preload_capacity),
            "preload_names": preload_names,
            "scenarios": [asdict(full_stats), asdict(cold_stats), asdict(warm_stats)],
            "reductions_vs_full": {
                "cold_tool_count_pct": _reduction(full_stats.tool_count, cold_stats.tool_count),
                "cold_schema_chars_pct": _reduction(full_stats.schema_chars, cold_stats.schema_chars),
                "cold_rough_tokens_pct": _reduction(full_stats.rough_tokens, cold_stats.rough_tokens),
                "warm_tool_count_pct": _reduction(full_stats.tool_count, warm_stats.tool_count),
                "warm_schema_chars_pct": _reduction(full_stats.schema_chars, warm_stats.schema_chars),
                "warm_rough_tokens_pct": _reduction(full_stats.rough_tokens, warm_stats.rough_tokens),
            },
            "search_top_k": int(args.top_k),
            "search_hit_count": search_hit,
            "search_case_count": search_total,
            "search_hit_rate_pct": round(search_hit / search_total * 100, 2)
            if search_total
            else 0.0,
            "search_cases": [asdict(case) for case in search_cases],
        }

        json_path = output_dir / f"tool_governance_benchmark_{timestamp}.json"
        md_path = output_dir / f"tool_governance_benchmark_{timestamp}.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(_render_markdown(result), encoding="utf-8")
        result["json_path"] = str(json_path)
        result["md_path"] = str(md_path)
        return result
    finally:
        if mcp_registry is not None:
            await _safe_aclose(mcp_registry)
        if memory_runtime is not None:
            await _safe_aclose(memory_runtime)
        if peer_poller is not None:
            await _safe_aclose(peer_poller)
        if peer_pm is not None:
            await _safe_aclose(peer_pm)
        await _safe_aclose(http_resources)
        await _safe_aclose(event_bus)


def _format_pct(value: float) -> str:
    return f"{value:.2f}%"


def _render_markdown(result: dict[str, Any]) -> str:
    full, cold, warm = result["scenarios"]
    lines = [
        "# Tool Governance Benchmark",
        "",
        "This benchmark registers the real local tools, connects configured MCP servers, and measures schema exposure plus deterministic tool_search retrieval. It does not call LLMs and does not execute domain tools.",
        "",
        f"- Created at: `{result['created_at']}`",
        f"- Workspace: `{result['workspace']}`",
        f"- Registered tools: `{result['registered_tool_count']}`",
        f"- MCP tools: `{result['mcp_tool_count']}`",
        f"- Deferred tools: `{result['deferred_count']}`",
        f"- Search Hit@{result['search_top_k']}: `{result['search_hit_rate_pct']}%`",
        "",
        "## Schema Exposure",
        "",
        "| Scenario | Tools visible | Schema chars | Rough tokens | Tool count reduction | Token reduction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in (full, cold, warm):
        lines.append(
            "| {label} | {tools} | {chars} | {tokens} | {tool_red} | {tok_red} |".format(
                label=scenario["label"],
                tools=scenario["tool_count"],
                chars=scenario["schema_chars"],
                tokens=scenario["rough_tokens"],
                tool_red=_format_pct(_reduction(full["tool_count"], scenario["tool_count"])),
                tok_red=_format_pct(_reduction(full["rough_tokens"], scenario["rough_tokens"])),
            )
        )
    lines.extend(
        [
            "",
            "## MCP Tools By Server",
            "",
        ]
    )
    if result["mcp_tools_by_server"]:
        for server, names in result["mcp_tools_by_server"].items():
            lines.append(f"- `{server}`: {', '.join(f'`{name}`' for name in names)}")
    else:
        lines.append("(none)")
    lines.extend(
        [
            "",
            "## Search Cases",
            "",
            "| Case | Query | Expected | Hit | Rank | Returned |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for case in result["search_cases"]:
        returned = ", ".join(f"`{name}`" for name in case["returned_tools"])
        lines.append(
            "| {case_id} | {query} | `{expected}` | {hit} | {rank} | {returned} |".format(
                case_id=case["case_id"],
                query=case["query"],
                expected=case["expected_tool"],
                hit=case["hit"],
                rank=case["rank"] if case["rank"] is not None else "",
                returned=returned,
            )
        )
    lines.extend(
        [
            "",
            "## Resume-Safe Metrics",
            "",
            f"- Cold progressive mode reduces visible schema rough tokens by `{result['reductions_vs_full']['cold_rough_tokens_pct']}%` versus full registry.",
            f"- Warm LRU mode reduces visible schema rough tokens by `{result['reductions_vs_full']['warm_rough_tokens_pct']}%` versus full registry.",
            f"- Deterministic tool_search retrieval Hit@{result['search_top_k']} is `{result['search_hit_rate_pct']}%` on the benchmark cases.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tool governance benchmark.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.toml"))
    parser.add_argument(
        "--workspace",
        default=str(PROJECT_ROOT / "_bench" / "workspaces" / "tool_governance"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "_bench" / "results"),
    )
    parser.add_argument("--preload-capacity", type=int, default=5)
    parser.add_argument("--preload-names", default="")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
