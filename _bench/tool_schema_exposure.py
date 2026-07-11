from __future__ import annotations

import argparse
import asyncio
import json
import math
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
    token_estimate: int
    tool_names: list[str]


def _schema_text(schemas: list[dict[str, Any]]) -> str:
    return json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))


def _estimate_tokens(text: str) -> int:
    # A deliberately simple estimate. The target models are provider-specific,
    # so this benchmark reports exact chars/bytes and a rough prompt-token proxy.
    return math.ceil(len(text) / 3)


def _stats(label: str, schemas: list[dict[str, Any]], names: list[str]) -> SchemaStats:
    text = _schema_text(schemas)
    return SchemaStats(
        label=label,
        tool_count=len(names),
        schema_chars=len(text),
        schema_bytes=len(text.encode("utf-8")),
        token_estimate=_estimate_tokens(text),
        tool_names=names,
    )


def _reduction(base: int, current: int) -> float:
    if base <= 0:
        return 0.0
    return round((1.0 - current / base) * 100.0, 2)


def _format_pct(value: float) -> str:
    return f"{value:.2f}%"


def _table(stats: list[SchemaStats], *, base: SchemaStats) -> str:
    lines = [
        "| Scenario | Tool count | Count reduction | Schema chars | Char reduction | Rough tokens | Token reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in stats:
        lines.append(
            "| {label} | {tool_count} | {count_red} | {schema_chars} | {char_red} | {tokens} | {token_red} |".format(
                label=item.label,
                tool_count=item.tool_count,
                count_red=_format_pct(_reduction(base.tool_count, item.tool_count)),
                schema_chars=item.schema_chars,
                char_red=_format_pct(_reduction(base.schema_chars, item.schema_chars)),
                tokens=item.token_estimate,
                token_red=_format_pct(
                    _reduction(base.token_estimate, item.token_estimate)
                ),
            )
        )
    return "\n".join(lines)


def _tool_names_block(title: str, names: list[str]) -> str:
    if not names:
        return f"### {title}\n\n(none)\n"
    values = "\n".join(f"- `{name}`" for name in names)
    return f"### {title}\n\n{values}\n"


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

    config = Config.load(config_path)
    http_resources = SharedHttpResources()
    bus = MessageBus()
    event_bus = EventBus()
    session_store = SessionStore(workspace / "sessions.db")
    memory_runtime = None
    mcp_registry = None
    peer_pm = None
    peer_poller = None
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

        all_names = tools.get_registered_order()
        always_on = tools.get_always_on_names()
        always_order = tools.get_registered_order(always_on)

        deferred_order = [
            name
            for name in all_names
            if name not in always_on and name != "tool_search"
        ]

        preload_names = [
            name.strip()
            for name in str(args.preload_names or "").split(",")
            if name.strip()
        ]
        if not preload_names:
            preload_names = deferred_order[: max(0, int(args.preload_capacity))]
        preload_names = [name for name in preload_names if name in deferred_order]

        discovery = ToolDiscoveryState(capacity=int(args.preload_capacity))
        session_key = "bench:tool-schema"
        discovery.update(session_key, preload_names, always_on)
        preloaded = discovery.get_preloaded(session_key)
        preloaded_order = tools.get_registered_order(preloaded)

        cold_names = always_order
        warm_names = [
            *always_order,
            *[name for name in preloaded_order if name not in always_on],
        ]

        full_stats = _stats("Full registry", tools.get_schemas(), all_names)
        cold_stats = _stats(
            "Tool-search cold start",
            tools.get_schemas(names=cold_names),
            cold_names,
        )
        warm_stats = _stats(
            f"Tool-search warm LRU ({len(preloaded_order)} preloaded)",
            tools.get_schemas(names=warm_names),
            warm_names,
        )

        metadata = getattr(tools, "_metadata", {})
        documents = getattr(tools, "_documents", {})
        risk_counts = Counter(
            getattr(meta, "risk", "unknown") for meta in metadata.values()
        )
        source_counts = Counter(
            getattr(doc, "source_type", "unknown") for doc in documents.values()
        )

        deferred_grouped = tools.get_deferred_names(visible=set(cold_names))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = {
            "created_at": datetime.now().astimezone().isoformat(),
            "project_root": str(PROJECT_ROOT),
            "config_path": str(config_path),
            "benchmark_workspace": str(workspace),
            "tool_search_enabled_in_config": bool(config.tool_search_enabled),
            "preload_capacity": int(args.preload_capacity),
            "preload_names": preload_names,
            "registered_tool_count": len(all_names),
            "always_on_count": len(always_order),
            "deferred_count": len(deferred_order),
            "risk_counts": dict(sorted(risk_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "deferred_grouped": deferred_grouped,
            "scenarios": [asdict(full_stats), asdict(cold_stats), asdict(warm_stats)],
            "reductions_vs_full": {
                "cold_count_pct": _reduction(full_stats.tool_count, cold_stats.tool_count),
                "cold_chars_pct": _reduction(full_stats.schema_chars, cold_stats.schema_chars),
                "cold_tokens_pct": _reduction(
                    full_stats.token_estimate, cold_stats.token_estimate
                ),
                "warm_count_pct": _reduction(full_stats.tool_count, warm_stats.tool_count),
                "warm_chars_pct": _reduction(full_stats.schema_chars, warm_stats.schema_chars),
                "warm_tokens_pct": _reduction(
                    full_stats.token_estimate, warm_stats.token_estimate
                ),
            },
        }

        json_path = output_dir / f"tool_schema_exposure_{timestamp}.json"
        md_path = output_dir / f"tool_schema_exposure_{timestamp}.md"
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report = "\n".join(
            [
                "# Tool Schema Exposure Benchmark",
                "",
                "This benchmark registers the project's real tools in an isolated workspace and measures schema exposure only. It does not execute tools, connect MCP servers, call LLMs, or write to the normal workspace memory.",
                "",
                f"- Created at: `{result['created_at']}`",
                f"- Benchmark workspace: `{workspace}`",
                f"- Config tool_search enabled: `{bool(config.tool_search_enabled)}`",
                f"- Registered tools: `{len(all_names)}`",
                f"- Always-on tools: `{len(always_order)}`",
                f"- Deferred tools: `{len(deferred_order)}`",
                "",
                "## Summary",
                "",
                _table([full_stats, cold_stats, warm_stats], base=full_stats),
                "",
                "## Distribution",
                "",
                f"- Risk counts: `{dict(sorted(risk_counts.items()))}`",
                f"- Source counts: `{dict(sorted(source_counts.items()))}`",
                "",
                _tool_names_block("Always-On Tools", always_order),
                _tool_names_block("Warm LRU Preloaded Tools", preloaded_order),
                _tool_names_block("Deferred Tools", deferred_order),
                "## Notes",
                "",
                "- `Full registry` approximates the old/direct mode where every registered tool schema is supplied to the model.",
                "- `Tool-search cold start` matches the first turn with progressive exposure: only always-on tools are visible.",
                "- `Tool-search warm LRU` simulates cross-turn preloading through `ToolDiscoveryState` with the configured capacity.",
                "- Rough tokens are estimated from schema JSON length and should be treated as a stable comparison proxy, not billing-grade tokenizer output.",
                "- Dynamic MCP server tools are not loaded in this isolated phase because the benchmark does not start MCP connections.",
                "",
            ]
        )
        md_path.write_text(report, encoding="utf-8")

        result["json_path"] = str(json_path)
        result["markdown_path"] = str(md_path)
        return result
    finally:
        await _safe_aclose(peer_poller)
        shutdown_all = getattr(peer_pm, "shutdown_all", None)
        if callable(shutdown_all):
            maybe = shutdown_all()
            if hasattr(maybe, "__await__"):
                await maybe
        await _safe_aclose(mcp_registry)
        await _safe_aclose(memory_runtime)
        await _safe_aclose(event_bus)
        session_store.close()
        await http_resources.aclose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure registered tool schema exposure with and without tool_search."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.toml"))
    parser.add_argument(
        "--workspace",
        default=str(PROJECT_ROOT / "_bench" / "workspaces" / "tool_schema"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "_bench" / "results"),
    )
    parser.add_argument("--preload-capacity", type=int, default=5)
    parser.add_argument(
        "--preload-names",
        default="",
        help="Optional comma-separated deferred tool names to simulate warm LRU.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
