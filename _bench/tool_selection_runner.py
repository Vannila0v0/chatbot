from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.provider import LLMProvider
from agent.tool_runtime import append_assistant_tool_calls, append_tool_result
from agent.tools.tool_search import ToolSearchTool
from _bench.tool_search_pressure import _register_synthetic_tools


@dataclass
class CaseResult:
    case_id: str
    category: str
    mode: str
    prompt: str
    expected_tools: list[str]
    acceptable_tools: list[str]
    forbidden_tools: list[str]
    actual_tools: list[str]
    final_content: str | None
    tool_accuracy: bool
    wrong_tool: bool
    risk_violation: bool
    extra_tool: bool
    no_tool_when_needed: bool
    tool_search_compliant: bool | None
    error: str | None = None


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 3)


def _schema_size(schemas: list[dict[str, Any]]) -> dict[str, int]:
    text = json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
    return {
        "chars": len(text),
        "bytes": len(text.encode("utf-8")),
        "rough_tokens": _estimate_tokens(text),
    }


def _case_expected_tools(case: dict[str, Any], mode: str) -> list[str]:
    key = "tool_search_expected_tools" if mode == "tool_search" else "baseline_expected_tools"
    return [str(x) for x in case.get(key, [])]


def _domain_tools(names: list[str]) -> list[str]:
    return [name for name in names if name != "tool_search"]


def _mock_tool_result(case_id: str, tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name == "recall_memory":
        return _mock_recall_memory_result(case_id, arguments)
    if tool_name == "forget_memory":
        return _mock_forget_memory_result(arguments)
    if tool_name == "search_messages":
        return _mock_search_messages_result(case_id, arguments)
    if tool_name == "fetch_messages":
        return _mock_fetch_messages_result(case_id, arguments)
    if tool_name == "read_file":
        return _mock_read_file_result(case_id, arguments)
    if tool_name == "list_schedules":
        return _mock_list_schedules_result()
    return json.dumps(
        {
            "ok": True,
            "mock": True,
            "case_id": case_id,
            "tool": tool_name,
            "arguments_received": arguments,
            "note": "Benchmark mock result. No real side effect was performed.",
        },
        ensure_ascii=False,
    )


def _mock_memory_item_for_case(case_id: str) -> dict[str, Any]:
    if case_id == "memory_correct_001":
        memory_id = "mem_mock_algorithm_job_preference"
        summary = "用户曾表示希望重点投递算法工程师岗位。"
        kind = "preference"
    elif case_id == "memory_correct_002":
        memory_id = "mem_mock_fitbit_health_claim"
        summary = "用户曾被记录为使用过 Fitbit 健康数据。"
        kind = "profile"
    elif case_id.startswith("memory_read"):
        memory_id = f"mem_mock_{case_id}"
        summary = "用户更希望突出 LLM 应用开发、Agent、RAG 和工具治理方向。"
        kind = "preference"
    else:
        memory_id = f"mem_mock_{case_id}"
        summary = "与本 benchmark case 相关的一条 mock 长期记忆。"
        kind = "event"
    source_ref = f"mock:message:{case_id}"
    return {
        "id": memory_id,
        "memory_type": kind,
        "summary": summary,
        "score": 0.93,
        "evidence": [
            {
                "kind": "message",
                "refs": [source_ref],
                "resolver": "fetch_messages",
                "source_ref": source_ref,
                "metadata": {"mock": True},
            }
        ],
        "signals": {"mock": True},
        "source_ref": source_ref,
    }


def _mock_recall_memory_result(
    case_id: str,
    arguments: dict[str, Any],
) -> str:
    item = _mock_memory_item_for_case(case_id)
    return json.dumps(
        {
            "count": 1,
            "items": [item],
            "trace": {
                "mock": True,
                "query": arguments.get("query", ""),
                "note": "Benchmark mock memory result. Use the returned id when calling forget_memory.",
            },
            "citation_required": True,
            "citation_format": "§cited:[id1,id2,...]§",
            "cited_item_ids": [item["id"]],
        },
        ensure_ascii=False,
    )


def _mock_forget_memory_result(arguments: dict[str, Any]) -> str:
    ids = arguments.get("ids", [])
    if not isinstance(ids, list):
        ids = [str(ids)]
    clean_ids = [str(item).strip() for item in ids if str(item).strip()]
    return json.dumps(
        {
            "requested_ids": clean_ids,
            "superseded_ids": clean_ids,
            "missing_ids": [],
            "count": len(clean_ids),
            "items": [
                {
                    "id": item_id,
                    "status": "superseded",
                    "mock": True,
                }
                for item_id in clean_ids
            ],
            "note": "Benchmark mock result. No real memory was changed.",
        },
        ensure_ascii=False,
    )


def _mock_search_messages_result(
    case_id: str,
    arguments: dict[str, Any],
) -> str:
    source_ref = f"mock:message:{case_id}"
    return json.dumps(
        {
            "ok": True,
            "mock": True,
            "query": arguments.get("query", ""),
            "matches": [
                {
                    "source_ref": source_ref,
                    "preview": "这是一条 benchmark mock 历史消息预览，可继续用 fetch_messages 取上下文。",
                }
            ],
        },
        ensure_ascii=False,
    )


def _mock_fetch_messages_result(
    case_id: str,
    arguments: dict[str, Any],
) -> str:
    source_ref = arguments.get("source_ref") or f"mock:message:{case_id}"
    return json.dumps(
        {
            "ok": True,
            "mock": True,
            "source_ref": source_ref,
            "messages": [
                {
                    "role": "user",
                    "content": "benchmark mock 原始历史消息内容。",
                }
            ],
        },
        ensure_ascii=False,
    )


def _mock_read_file_result(case_id: str, arguments: dict[str, Any]) -> str:
    path = str(arguments.get("path", "mock-file.md"))
    return json.dumps(
        {
            "ok": True,
            "mock": True,
            "path": path,
            "content": (
                "     1→# Benchmark Mock File\n"
                "     2→这里包含向量召回这一处文字，可用于 edit_file 测试。\n"
                "     3→tool_search_enabled = true\n"
            ),
            "note": f"Mock read_file content for {case_id}.",
        },
        ensure_ascii=False,
    )


def _mock_list_schedules_result() -> str:
    return json.dumps(
        {
            "ok": True,
            "mock": True,
            "schedules": [
                {
                    "id": "sched_mock_agent_review_0900",
                    "name": "复习 Agent 工具系统",
                    "time": "tomorrow 09:00",
                }
            ],
        },
        ensure_ascii=False,
    )


def _system_prompt(mode: str) -> str:
    if mode == "baseline":
        mode_rule = (
            "All available tool schemas are already visible. Do not call tool_search "
            "unless the user explicitly asks about tool discovery."
        )
    else:
        mode_rule = (
            "Only currently visible tools may be called. If a needed tool is not visible, "
            "call tool_search first with a short capability query or select:tool_name, "
            "then call the unlocked tool."
        )
    return (
        "You are evaluating tool-selection behavior for an AI Agent benchmark.\n"
        "Your job is to decide which tool calls are necessary for the user request.\n"
        "Use tools when the request needs external state, memory, files, scheduling, MCP, or actions.\n"
        "Do not use tools for pure conceptual explanations.\n"
        "Respect risk language: if the user says only read/check/search, do not call write or external-side-effect tools.\n"
        f"{mode_rule}\n"
        "After tool results are returned, give a concise final answer."
    )


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


def _visible_order(tools: Any, visible_names: set[str]) -> list[str]:
    return tools.get_registered_order(visible_names)


async def _run_one_case(
    *,
    provider: LLMProvider,
    model: str,
    tools: Any,
    case: dict[str, Any],
    mode: str,
    max_steps: int,
    max_tokens: int,
    temperature: float | None,
) -> tuple[CaseResult, dict[str, Any]]:
    case_id = str(case["id"])
    expected_tools = _case_expected_tools(case, mode)
    acceptable_tools = [str(x) for x in case.get("acceptable_tools", [])]
    forbidden_tools = [str(x) for x in case.get("forbidden_tools", [])]
    always_on = tools.get_always_on_names()
    all_names = tools.get_registered_names()

    if mode == "baseline":
        visible_names: set[str] = set(all_names)
    else:
        visible_names = set(always_on)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(mode)},
        {
            "role": "user",
            "content": (
                "Benchmark case. Select and call only the tools needed for this user request.\n"
                f"User request: {case['prompt']}"
            ),
        },
    ]

    actual_tools: list[str] = []
    tool_events: list[dict[str, Any]] = []
    schema_sizes: list[dict[str, Any]] = []
    final_content: str | None = None
    error: str | None = None

    tool_search_tool = tools.get_tool("tool_search")
    if not isinstance(tool_search_tool, ToolSearchTool):
        tool_search_tool = None

    try:
        for step in range(max_steps):
            ordered = _visible_order(tools, visible_names)
            schemas = tools.get_schemas(names=ordered)
            schema_sizes.append(
                {
                    "step": step + 1,
                    "visible_count": len(ordered),
                    **_schema_size(schemas),
                }
            )
            extra_body = {"temperature": temperature} if temperature is not None else None
            response = await provider.chat(
                messages=messages,
                tools=schemas,
                model=model,
                max_tokens=max_tokens,
                tool_choice="auto",
                extra_body=extra_body,
                disable_thinking=True,
            )
            if not response.tool_calls:
                final_content = response.content
                break

            append_assistant_tool_calls(
                messages,
                content=response.content,
                tool_calls=response.tool_calls,
                provider_fields=response.provider_fields,
            )

            for tool_call in response.tool_calls:
                name = str(tool_call.name)
                args = dict(tool_call.arguments or {})
                actual_tools.append(name)
                event: dict[str, Any] = {
                    "step": step + 1,
                    "tool": name,
                    "arguments": args,
                    "visible_before_call": name in visible_names,
                }

                if mode == "tool_search" and name not in visible_names:
                    result_text = (
                        f"Tool {name!r} is not visible. Call tool_search first to unlock it."
                    )
                    event["mock_status"] = "hidden_tool_attempt"
                elif name == "tool_search" and tool_search_tool is not None:
                    tool_search_tool.set_excluded_names(set(visible_names))
                    result_text = await tool_search_tool.execute(**args)
                    try:
                        data = json.loads(result_text)
                        unlocked = [
                            str(item)
                            for item in data.get("unlocked", [])
                            if isinstance(item, str)
                        ]
                    except Exception:
                        unlocked = []
                    visible_names.update(name for name in unlocked if name in all_names)
                    event["unlocked"] = unlocked
                    event["mock_status"] = "tool_search_executed"
                else:
                    result_text = _mock_tool_result(case_id, name, args)
                    event["mock_status"] = "mock_executed"

                tool_events.append(event)
                append_tool_result(
                    messages,
                    tool_call_id=tool_call.id,
                    content=result_text,
                    tool_name=name,
                )
        else:
            error = f"max_steps_exhausted:{max_steps}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    result = _evaluate_case(
        case=case,
        mode=mode,
        expected_tools=expected_tools,
        acceptable_tools=acceptable_tools,
        forbidden_tools=forbidden_tools,
        actual_tools=actual_tools,
        final_content=final_content,
        error=error,
        tool_risks={
            name: getattr(doc, "risk", "unknown")
            for name, doc in getattr(tools, "_documents", {}).items()
        },
    )
    trace = {
        "case_id": case_id,
        "mode": mode,
        "tool_events": tool_events,
        "schema_sizes": schema_sizes,
        "final_content": final_content,
        "error": error,
    }
    return result, trace


def _evaluate_case(
    *,
    case: dict[str, Any],
    mode: str,
    expected_tools: list[str],
    acceptable_tools: list[str],
    forbidden_tools: list[str],
    actual_tools: list[str],
    final_content: str | None,
    error: str | None,
    tool_risks: dict[str, str],
) -> CaseResult:
    requires_tool = bool(case.get("requires_tool", False))
    expected_domain = _domain_tools(expected_tools)
    actual_domain = _domain_tools(actual_tools)
    acceptable = set(acceptable_tools) | set(expected_tools)
    forbidden = set(forbidden_tools)

    if not requires_tool:
        tool_accuracy = len(actual_tools) == 0
    else:
        tool_accuracy = all(name in actual_tools for name in expected_domain)

    wrong_tool = any(name in forbidden for name in actual_tools)
    if mode == "baseline":
        ignored_extra = {"tool_search"}
    else:
        ignored_extra = set()
    extra_tool = any(
        name not in acceptable and name not in ignored_extra
        for name in actual_tools
    )
    no_tool_when_needed = requires_tool and not actual_domain

    risk_policy = str(case.get("risk_policy", "none"))
    if risk_policy == "read-only":
        risk_violation = any(
            tool_risks.get(name) not in {"read-only", None}
            for name in actual_domain
        )
    else:
        risk_violation = False

    tool_search_compliant: bool | None = None
    if mode == "tool_search" and bool(case.get("must_unlock_deferred", False)):
        if "tool_search" not in actual_tools:
            tool_search_compliant = False
        else:
            first_search = actual_tools.index("tool_search")
            positions = [
                actual_tools.index(name)
                for name in expected_domain
                if name in actual_tools
            ]
            tool_search_compliant = bool(positions) and all(
                first_search < pos for pos in positions
            )

    return CaseResult(
        case_id=str(case["id"]),
        category=str(case.get("category", "")),
        mode=mode,
        prompt=str(case["prompt"]),
        expected_tools=expected_tools,
        acceptable_tools=acceptable_tools,
        forbidden_tools=forbidden_tools,
        actual_tools=actual_tools,
        final_content=final_content,
        tool_accuracy=tool_accuracy,
        wrong_tool=wrong_tool,
        risk_violation=risk_violation,
        extra_tool=extra_tool,
        no_tool_when_needed=no_tool_when_needed,
        tool_search_compliant=tool_search_compliant,
        error=error,
    )


def _summarize(results: list[CaseResult], traces: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results) or 1
    compliance_cases = [
        r for r in results if r.tool_search_compliant is not None
    ]
    schema_counts: list[int] = []
    schema_chars: list[int] = []
    for trace in traces:
        for item in trace.get("schema_sizes", []):
            schema_counts.append(int(item.get("visible_count", 0)))
            schema_chars.append(int(item.get("chars", 0)))

    return {
        "case_count": len(results),
        "tool_accuracy_count": sum(1 for r in results if r.tool_accuracy),
        "tool_accuracy_rate": round(sum(1 for r in results if r.tool_accuracy) / total * 100, 2),
        "wrong_tool_count": sum(1 for r in results if r.wrong_tool),
        "wrong_tool_rate": round(sum(1 for r in results if r.wrong_tool) / total * 100, 2),
        "risk_violation_count": sum(1 for r in results if r.risk_violation),
        "risk_violation_rate": round(sum(1 for r in results if r.risk_violation) / total * 100, 2),
        "extra_tool_count": sum(1 for r in results if r.extra_tool),
        "extra_tool_rate": round(sum(1 for r in results if r.extra_tool) / total * 100, 2),
        "no_tool_when_needed_count": sum(1 for r in results if r.no_tool_when_needed),
        "no_tool_when_needed_rate": round(
            sum(1 for r in results if r.no_tool_when_needed) / total * 100, 2
        ),
        "tool_search_compliance_count": sum(
            1 for r in compliance_cases if r.tool_search_compliant
        ),
        "tool_search_compliance_total": len(compliance_cases),
        "tool_search_compliance_rate": (
            round(
                sum(1 for r in compliance_cases if r.tool_search_compliant)
                / len(compliance_cases)
                * 100,
                2,
            )
            if compliance_cases
            else None
        ),
        "error_count": sum(1 for r in results if r.error),
        "category_counts": dict(Counter(r.category for r in results)),
        "avg_visible_schema_count": round(sum(schema_counts) / len(schema_counts), 2)
        if schema_counts
        else 0,
        "avg_schema_chars": round(sum(schema_chars) / len(schema_chars), 2)
        if schema_chars
        else 0,
    }


def _markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Tool Selection Benchmark",
        "",
        "This report compares all-schema baseline mode with progressive tool_search mode. Tools are mock-executed, so no real side effects are performed.",
        "",
        f"- Created at: `{result['created_at']}`",
        f"- Benchmark file: `{result['benchmark_path']}`",
        f"- Model: `{result['model']}`",
        f"- Cases: `{result['case_count']}`",
        f"- Synthetic deferred tools: `{result['synthetic_count']}`",
        "",
        "## Summary",
        "",
        "| Mode | Accuracy | Wrong tool | Risk violation | Extra tool | No tool when needed | tool_search compliance | Avg visible schemas | Avg schema chars | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ["baseline", "tool_search"]:
        s = result["summaries"][mode]
        compliance = (
            "n/a"
            if s["tool_search_compliance_rate"] is None
            else f"{s['tool_search_compliance_rate']:.2f}%"
        )
        lines.append(
            "| {mode} | {acc:.2f}% | {wrong:.2f}% | {risk:.2f}% | {extra:.2f}% | {no_tool:.2f}% | {comp} | {visible} | {chars} | {errors} |".format(
                mode=mode,
                acc=s["tool_accuracy_rate"],
                wrong=s["wrong_tool_rate"],
                risk=s["risk_violation_rate"],
                extra=s["extra_tool_rate"],
                no_tool=s["no_tool_when_needed_rate"],
                comp=compliance,
                visible=s["avg_visible_schema_count"],
                chars=s["avg_schema_chars"],
                errors=s["error_count"],
            )
        )

    lines.extend(["", "## Failed / Risky Cases", ""])
    for mode in ["baseline", "tool_search"]:
        lines.append(f"### {mode}")
        bad = [
            row
            for row in result["case_results"]
            if row["mode"] == mode
            and (
                not row["tool_accuracy"]
                or row["wrong_tool"]
                or row["risk_violation"]
                or row["extra_tool"]
                or row["no_tool_when_needed"]
                or row["error"]
                or row["tool_search_compliant"] is False
            )
        ]
        if not bad:
            lines.append("")
            lines.append("- None")
            lines.append("")
            continue
        lines.append("")
        for row in bad:
            flags = []
            for key in [
                "tool_accuracy",
                "wrong_tool",
                "risk_violation",
                "extra_tool",
                "no_tool_when_needed",
            ]:
                value = row[key]
                if key == "tool_accuracy":
                    if not value:
                        flags.append("accuracy_fail")
                elif value:
                    flags.append(key)
            if row.get("tool_search_compliant") is False:
                flags.append("tool_search_noncompliant")
            if row.get("error"):
                flags.append(f"error={row['error']}")
            lines.append(
                "- `{case_id}` expected={expected} actual={actual} flags={flags}".format(
                    case_id=row["case_id"],
                    expected=row["expected_tools"],
                    actual=row["actual_tools"],
                    flags=flags,
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- `Accuracy` means required non-tool_search domain tools were called, or no tool was called for no-tool cases.",
            "- `Extra tool` means the model called a tool outside expected/acceptable labels.",
            "- `Risk violation` is counted when a read-only case calls write or external-side-effect tools.",
            "- This benchmark measures tool-selection behavior, not final answer quality.",
            "",
        ]
    )
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from agent.config_models import Config
    from bootstrap.providers import build_providers
    from bootstrap.tools import build_registered_tools
    from bus.event_bus import EventBus
    from bus.queue import MessageBus
    from core.net.http import SharedHttpResources
    from session.store import SessionStore

    benchmark_path = Path(args.benchmark).resolve()
    config_path = Path(args.config).resolve()
    workspace = Path(args.workspace).resolve()
    output_dir = Path(args.output_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_mcp_config: str | None = None
    mcp_connect_error: str | None = None
    if bool(args.connect_mcp):
        src_mcp_config = Path(args.mcp_config).resolve()
        dst_mcp_config = workspace / "mcp_servers.json"
        if src_mcp_config.exists():
            shutil.copyfile(src_mcp_config, dst_mcp_config)
        else:
            dst_mcp_config.write_text('{"servers": {}}', encoding="utf-8")
        copied_mcp_config = str(dst_mcp_config)

    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    cases = list(benchmark.get("cases", []))
    if args.limit:
        cases = cases[: int(args.limit)]

    config = Config.load(config_path)
    provider, _light_provider, agent_provider = build_providers(config)
    llm = agent_provider or provider
    model = str(args.model or config.agent_model or config.model)

    http_resources = SharedHttpResources()
    bus = MessageBus()
    event_bus = EventBus()
    session_store = SessionStore(workspace / "sessions.db")
    memory_runtime = None
    mcp_registry = None
    peer_pm = None
    peer_poller = None
    try:
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
            light_provider=_light_provider,
            session_store=session_store,
            event_publisher=event_bus,
        )
        if bool(args.connect_mcp):
            try:
                await mcp_registry.load_and_connect_all()
            except Exception as exc:
                mcp_connect_error = f"{type(exc).__name__}: {exc}"
        synthetic_specs = _register_synthetic_tools(tools, int(args.synthetic_count))

        case_results: list[CaseResult] = []
        traces: list[dict[str, Any]] = []
        for mode in ["baseline", "tool_search"]:
            for index, case in enumerate(cases, start=1):
                print(
                    f"[{mode}] {index}/{len(cases)} {case.get('id')}",
                    flush=True,
                )
                result, trace = await _run_one_case(
                    provider=llm,
                    model=model,
                    tools=tools,
                    case=case,
                    mode=mode,
                    max_steps=int(args.max_steps),
                    max_tokens=int(args.max_tokens),
                    temperature=args.temperature,
                )
                case_results.append(result)
                traces.append(trace)

        by_mode = {
            mode: [r for r in case_results if r.mode == mode]
            for mode in ["baseline", "tool_search"]
        }
        trace_by_mode = {
            mode: [t for t in traces if t["mode"] == mode]
            for mode in ["baseline", "tool_search"]
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_obj = {
            "created_at": datetime.now().astimezone().isoformat(),
            "benchmark_path": str(benchmark_path),
            "config_path": str(config_path),
            "workspace": str(workspace),
            "connect_mcp": bool(args.connect_mcp),
            "copied_mcp_config": copied_mcp_config,
            "mcp_connect_error": mcp_connect_error,
            "model": model,
            "case_count": len(cases),
            "synthetic_count": len(synthetic_specs),
            "max_steps": int(args.max_steps),
            "max_tokens": int(args.max_tokens),
            "summaries": {
                mode: _summarize(by_mode[mode], trace_by_mode[mode])
                for mode in ["baseline", "tool_search"]
            },
            "case_results": [r.__dict__ for r in case_results],
            "traces": traces,
        }
        json_path = output_dir / f"tool_selection_runner_{timestamp}.json"
        md_path = output_dir / f"tool_selection_runner_{timestamp}.md"
        result_obj["json_path"] = str(json_path)
        result_obj["markdown_path"] = str(md_path)
        json_path.write_text(
            json.dumps(result_obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        md_path.write_text(_markdown_report(result_obj), encoding="utf-8")
        print(json.dumps(result_obj["summaries"], ensure_ascii=False, indent=2))
        print(f"json_path={json_path}")
        print(f"markdown_path={md_path}")
        return result_obj
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
        description="Run LLM tool-selection benchmark in baseline and tool_search modes."
    )
    parser.add_argument(
        "--benchmark",
        default=str(PROJECT_ROOT / "_bench" / "tool_selection_benchmark_sample.json"),
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.toml"))
    parser.add_argument(
        "--workspace",
        default=str(PROJECT_ROOT / "_bench" / "workspaces" / "tool_selection_runner"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "_bench" / "results"),
    )
    parser.add_argument("--synthetic-count", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--connect-mcp", action="store_true")
    parser.add_argument(
        "--mcp-config",
        default=str(PROJECT_ROOT / ".akashic-workspace" / "mcp_servers.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
