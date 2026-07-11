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


from agent.tools.base import Tool


@dataclass(frozen=True)
class SyntheticSpec:
    name: str
    description: str
    search_hint: str
    risk: str = "read-only"
    source_name: str = "bench_mcp"


@dataclass(frozen=True)
class QueryCase:
    query: str
    target: str
    allowed_risk: list[str] | None = None


@dataclass
class SchemaStats:
    label: str
    tool_count: int
    schema_chars: int
    schema_bytes: int
    token_estimate: int
    tool_names: list[str]


class SyntheticTool(Tool):
    def __init__(self, spec: SyntheticSpec) -> None:
        self._spec = spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def description(self) -> str:
        return self._spec.description

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search or action text for the synthetic benchmark tool.",
                },
                "payload": {
                    "type": "object",
                    "description": "Optional structured payload for the synthetic benchmark tool.",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return json.dumps(
            {"tool": self.name, "synthetic": True, "arguments": kwargs},
            ensure_ascii=False,
        )


def _base_synthetic_specs() -> list[SyntheticSpec]:
    # Names and hints intentionally look like realistic MCP/business tools. The
    # benchmark measures catalog search quality, not the behavior of these tools.
    return [
        SyntheticSpec("calendar_search_events", "Search calendar events and meeting schedules.", "日程 会议 安排 日历 calendar search event"),
        SyntheticSpec("calendar_create_event", "Create a new calendar event or meeting.", "创建日程 新建会议 安排时间 calendar create event", "write"),
        SyntheticSpec("calendar_cancel_event", "Cancel or delete an existing calendar event.", "取消日程 删除会议 calendar cancel delete", "write"),
        SyntheticSpec("email_search_messages", "Search email messages by sender, subject, and keywords.", "邮件 搜索 收件箱 邮箱 email search inbox"),
        SyntheticSpec("email_send_message", "Send an outbound email message.", "发送邮件 写邮件 email send message", "external-side-effect"),
        SyntheticSpec("email_summarize_thread", "Summarize a long email thread.", "邮件线程 总结 摘要 email summarize thread"),
        SyntheticSpec("todo_list_tasks", "List tasks from a todo system.", "待办 任务 列表 todo task list"),
        SyntheticSpec("todo_create_task", "Create a task in a todo system.", "创建待办 新增任务 todo create task", "write"),
        SyntheticSpec("todo_complete_task", "Mark a todo task as completed.", "完成待办 关闭任务 todo complete", "write"),
        SyntheticSpec("crm_search_contacts", "Search customer and contact records.", "客户 联系人 CRM contact search"),
        SyntheticSpec("crm_update_contact", "Update CRM contact fields.", "更新客户 修改联系人 CRM update contact", "write"),
        SyntheticSpec("crm_create_lead", "Create a sales lead in CRM.", "创建线索 销售线索 CRM lead create", "write"),
        SyntheticSpec("slack_search_messages", "Search Slack messages and channels.", "Slack 搜索 群消息 频道 message search"),
        SyntheticSpec("slack_send_message", "Send a Slack message to a channel or user.", "Slack 发送消息 通知 channel send", "external-side-effect"),
        SyntheticSpec("notion_search_pages", "Search Notion pages and databases.", "Notion 页面 数据库 搜索 page database"),
        SyntheticSpec("notion_create_page", "Create a Notion page.", "Notion 创建页面 新建文档 page create", "write"),
        SyntheticSpec("github_search_issues", "Search GitHub issues and pull requests.", "GitHub issue PR 搜索 缺陷 pull request"),
        SyntheticSpec("github_create_issue", "Create a GitHub issue.", "GitHub 创建 issue 缺陷 bug create", "write"),
        SyntheticSpec("github_comment_pr", "Comment on a GitHub pull request.", "GitHub PR 评论 review comment", "external-side-effect"),
        SyntheticSpec("jira_search_tickets", "Search Jira tickets.", "Jira 工单 需求 缺陷 ticket search"),
        SyntheticSpec("jira_update_ticket", "Update a Jira ticket status or field.", "Jira 更新工单 修改状态 ticket update", "write"),
        SyntheticSpec("rss_list_feeds", "List RSS subscriptions.", "RSS 订阅 源 列表 feed list"),
        SyntheticSpec("rss_add_feed", "Add an RSS subscription.", "RSS 添加订阅 新增 feed add", "write"),
        SyntheticSpec("rss_fetch_latest", "Fetch latest RSS items.", "RSS 最新文章 抓取 feed fetch latest"),
        SyntheticSpec("weather_current", "Get current weather by city.", "天气 当前 weather city"),
        SyntheticSpec("weather_forecast", "Get weather forecast by city and date.", "天气预报 forecast city date"),
        SyntheticSpec("stock_quote", "Get stock price quote.", "股票 股价 行情 quote finance"),
        SyntheticSpec("stock_news", "Search stock market news.", "股票 新闻 财经 market news"),
        SyntheticSpec("fitbit_read_steps", "Read step count from a fitness wearable.", "Fitbit 步数 健康 手环 steps"),
        SyntheticSpec("fitbit_read_sleep", "Read sleep data from a fitness wearable.", "Fitbit 睡眠 健康 手环 sleep"),
        SyntheticSpec("fitbit_read_heart_rate", "Read heart-rate data from a fitness wearable.", "Fitbit 心率 健康 手环 heart rate"),
        SyntheticSpec("browser_open_url", "Open a URL in a controlled browser.", "浏览器 打开网页 browser open url", "external-side-effect"),
        SyntheticSpec("browser_click_element", "Click an element in a controlled browser.", "浏览器 点击 按钮 browser click", "external-side-effect"),
        SyntheticSpec("browser_take_screenshot", "Take a browser screenshot.", "浏览器 截图 screenshot browser"),
        SyntheticSpec("pdf_extract_text", "Extract text from a PDF.", "PDF 提取文本 解析 文档 extract"),
        SyntheticSpec("pdf_summarize", "Summarize a PDF document.", "PDF 总结 摘要 summarize"),
        SyntheticSpec("spreadsheet_read_sheet", "Read spreadsheet rows and cells.", "表格 Excel 读取 spreadsheet sheet"),
        SyntheticSpec("spreadsheet_write_sheet", "Write spreadsheet rows and cells.", "表格 Excel 写入 spreadsheet write", "write"),
        SyntheticSpec("database_query_readonly", "Run a read-only SQL query.", "数据库 SQL 查询 只读 query readonly"),
        SyntheticSpec("database_execute_write", "Run a write SQL command.", "数据库 SQL 写入 更新 execute", "write"),
        SyntheticSpec("k8s_get_pods", "List Kubernetes pods.", "Kubernetes k8s pod 列表 get pods"),
        SyntheticSpec("k8s_restart_deployment", "Restart a Kubernetes deployment.", "Kubernetes k8s 重启 deployment restart", "external-side-effect"),
        SyntheticSpec("cloud_list_buckets", "List cloud storage buckets.", "云存储 bucket 列表 cloud storage"),
        SyntheticSpec("cloud_upload_file", "Upload a file to cloud storage.", "云存储 上传文件 bucket upload", "external-side-effect"),
        SyntheticSpec("image_generate_asset", "Generate a visual asset.", "生成图片 视觉素材 image generate", "external-side-effect"),
        SyntheticSpec("image_edit_asset", "Edit an existing visual asset.", "编辑图片 修改图像 image edit", "external-side-effect"),
        SyntheticSpec("translation_translate_text", "Translate text between languages.", "翻译 文本 translation language"),
        SyntheticSpec("speech_transcribe_audio", "Transcribe audio to text.", "语音转文字 音频 transcription speech"),
        SyntheticSpec("speech_synthesize_voice", "Synthesize voice from text.", "文字转语音 合成 voice speech", "external-side-effect"),
        SyntheticSpec("map_search_places", "Search places on a map.", "地图 地点 搜索 map place"),
        SyntheticSpec("map_route_plan", "Plan a route between locations.", "地图 路线 规划 route plan"),
        SyntheticSpec("expense_scan_receipt", "Extract expense data from a receipt image.", "报销 发票 收据 OCR expense receipt"),
        SyntheticSpec("expense_submit_report", "Submit an expense report.", "提交报销 费用报告 expense submit", "external-side-effect"),
        SyntheticSpec("health_log_mood", "Log a mood check-in.", "心情 情绪 记录 mood health", "write"),
        SyntheticSpec("health_sedentary_alerts", "Read sedentary alert settings.", "久坐提醒 健康 sedentary alert"),
        SyntheticSpec("study_search_notes", "Search study notes.", "学习笔记 搜索 note study"),
        SyntheticSpec("study_create_flashcards", "Create flashcards from text.", "学习 卡片 flashcard create", "write"),
        SyntheticSpec("travel_search_flights", "Search flights.", "旅行 机票 航班 flight search"),
        SyntheticSpec("travel_book_hotel", "Book a hotel.", "旅行 酒店 预订 hotel book", "external-side-effect"),
    ]


def _make_synthetic_specs(count: int) -> list[SyntheticSpec]:
    base = _base_synthetic_specs()
    if count <= len(base):
        return base[:count]
    specs = list(base)
    filler_topics = [
        ("ops", "运维", "operation"),
        ("sales", "销售", "sales"),
        ("hr", "人事", "human resource"),
        ("finance", "财务", "finance"),
        ("legal", "法务", "legal"),
        ("support", "客服", "support"),
    ]
    i = 0
    while len(specs) < count:
        prefix, zh, en = filler_topics[i % len(filler_topics)]
        idx = i // len(filler_topics) + 1
        specs.append(
            SyntheticSpec(
                name=f"{prefix}_bench_tool_{idx:02d}",
                description=f"Synthetic {en} benchmark utility {idx}.",
                search_hint=f"{zh} {en} benchmark synthetic tool {idx}",
                risk=["read-only", "write", "external-side-effect"][i % 3],
                source_name=f"bench_{prefix}",
            )
        )
        i += 1
    return specs


def _query_cases() -> list[QueryCase]:
    return [
        QueryCase("查一下明天有什么会议", "calendar_search_events", ["read-only"]),
        QueryCase("帮我新建一个周会日程", "calendar_create_event"),
        QueryCase("取消下午三点的会议", "calendar_cancel_event"),
        QueryCase("搜索老板上周发来的邮件", "email_search_messages", ["read-only"]),
        QueryCase("把会议纪要发邮件给团队", "email_send_message"),
        QueryCase("总结这串邮件线程", "email_summarize_thread", ["read-only"]),
        QueryCase("列出今天的待办任务", "todo_list_tasks", ["read-only"]),
        QueryCase("创建一个复习 Agent 的待办", "todo_create_task"),
        QueryCase("把这个任务标记完成", "todo_complete_task"),
        QueryCase("查找客户张三的联系方式", "crm_search_contacts", ["read-only"]),
        QueryCase("更新客户电话", "crm_update_contact"),
        QueryCase("创建一个销售线索", "crm_create_lead"),
        QueryCase("搜索 Slack 里的部署消息", "slack_search_messages", ["read-only"]),
        QueryCase("给频道发一条通知", "slack_send_message"),
        QueryCase("搜索 Notion 项目文档", "notion_search_pages", ["read-only"]),
        QueryCase("新建 Notion 页面", "notion_create_page"),
        QueryCase("查 GitHub issue", "github_search_issues", ["read-only"]),
        QueryCase("创建 GitHub bug issue", "github_create_issue"),
        QueryCase("给 PR 留一条评论", "github_comment_pr"),
        QueryCase("查 Jira 工单", "jira_search_tickets", ["read-only"]),
        QueryCase("更新 Jira 状态", "jira_update_ticket"),
        QueryCase("列出 RSS 订阅源", "rss_list_feeds", ["read-only"]),
        QueryCase("添加 RSS 订阅", "rss_add_feed"),
        QueryCase("抓取最新 RSS 文章", "rss_fetch_latest", ["read-only"]),
        QueryCase("查当前天气", "weather_current", ["read-only"]),
        QueryCase("查明天天气预报", "weather_forecast", ["read-only"]),
        QueryCase("查股票行情", "stock_quote", ["read-only"]),
        QueryCase("查股票相关新闻", "stock_news", ["read-only"]),
        QueryCase("读取手环步数", "fitbit_read_steps", ["read-only"]),
        QueryCase("读取昨晚睡眠", "fitbit_read_sleep", ["read-only"]),
        QueryCase("读取心率数据", "fitbit_read_heart_rate", ["read-only"]),
        QueryCase("浏览器打开这个网页", "browser_open_url"),
        QueryCase("浏览器点击按钮", "browser_click_element"),
        QueryCase("截一张网页图", "browser_take_screenshot", ["read-only"]),
        QueryCase("提取 PDF 文字", "pdf_extract_text", ["read-only"]),
        QueryCase("总结 PDF 文档", "pdf_summarize", ["read-only"]),
        QueryCase("读取 Excel 表格", "spreadsheet_read_sheet", ["read-only"]),
        QueryCase("写入 Excel 表格", "spreadsheet_write_sheet"),
        QueryCase("执行只读 SQL 查询", "database_query_readonly", ["read-only"]),
        QueryCase("执行数据库写入", "database_execute_write"),
        QueryCase("查看 k8s pods", "k8s_get_pods", ["read-only"]),
        QueryCase("重启 k8s deployment", "k8s_restart_deployment"),
        QueryCase("列出云存储 bucket", "cloud_list_buckets", ["read-only"]),
        QueryCase("上传文件到云存储", "cloud_upload_file"),
        QueryCase("生成一张视觉素材", "image_generate_asset"),
        QueryCase("编辑图片素材", "image_edit_asset"),
        QueryCase("翻译这段文字", "translation_translate_text", ["read-only"]),
        QueryCase("语音转文字", "speech_transcribe_audio", ["read-only"]),
        QueryCase("文字转语音", "speech_synthesize_voice"),
        QueryCase("地图搜索附近咖啡店", "map_search_places", ["read-only"]),
        QueryCase("规划路线", "map_route_plan", ["read-only"]),
        QueryCase("识别报销收据", "expense_scan_receipt", ["read-only"]),
        QueryCase("提交报销单", "expense_submit_report"),
        QueryCase("记录今天心情", "health_log_mood"),
        QueryCase("查看久坐提醒设置", "health_sedentary_alerts", ["read-only"]),
        QueryCase("搜索学习笔记", "study_search_notes", ["read-only"]),
        QueryCase("生成复习卡片", "study_create_flashcards"),
        QueryCase("搜索航班", "travel_search_flights", ["read-only"]),
        QueryCase("预订酒店", "travel_book_hotel"),
    ]


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
        token_estimate=_estimate_tokens(text),
        tool_names=names,
    )


def _reduction(base: int, current: int) -> float:
    if base <= 0:
        return 0.0
    return round((1.0 - current / base) * 100.0, 2)


def _pct(value: float) -> str:
    return f"{value:.2f}%"


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


def _register_synthetic_tools(tools: Any, count: int) -> list[SyntheticSpec]:
    specs = _make_synthetic_specs(count)
    for spec in specs:
        tools.register(
            SyntheticTool(spec),
            always_on=False,
            risk=spec.risk,
            search_hint=spec.search_hint,
            source_type="mcp",
            source_name=spec.source_name,
        )
    return specs


def _evaluate_search(
    tools: Any,
    cases: list[QueryCase],
    *,
    excluded_names: set[str],
    top_k: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    hit_at_1 = hit_at_3 = hit_at_k = 0
    reciprocal_sum = 0.0
    no_match = 0
    for case in cases:
        results = tools.search(
            query=case.query,
            top_k=top_k,
            allowed_risk=case.allowed_risk,
            excluded_names=excluded_names,
        )
        names = [str(item.get("name", "")) for item in results]
        rank = names.index(case.target) + 1 if case.target in names else None
        if not names:
            no_match += 1
        if rank == 1:
            hit_at_1 += 1
        if rank is not None and rank <= 3:
            hit_at_3 += 1
        if rank is not None and rank <= top_k:
            hit_at_k += 1
            reciprocal_sum += 1.0 / rank
        rows.append(
            {
                "query": case.query,
                "target": case.target,
                "allowed_risk": case.allowed_risk,
                "rank": rank,
                "top_names": names,
            }
        )
    total = len(cases) or 1
    return {
        "case_count": len(cases),
        "top_k": top_k,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        f"hit_at_{top_k}": hit_at_k,
        "hit_at_1_rate": round(hit_at_1 / total * 100, 2),
        "hit_at_3_rate": round(hit_at_3 / total * 100, 2),
        f"hit_at_{top_k}_rate": round(hit_at_k / total * 100, 2),
        "mrr": round(reciprocal_sum / total, 4),
        "no_match_count": no_match,
        "top1_error_count": len(cases) - hit_at_1,
        "rows": rows,
    }


def _evaluate_risk_filter(tools: Any, cases: list[QueryCase], *, excluded_names: set[str]) -> dict[str, Any]:
    blocked_cases = [case for case in cases if case.allowed_risk == ["read-only"]]
    violations: list[dict[str, Any]] = []
    for case in blocked_cases:
        results = tools.search(
            query=case.query,
            top_k=10,
            allowed_risk=case.allowed_risk,
            excluded_names=excluded_names,
        )
        for item in results:
            if item.get("risk") not in case.allowed_risk:
                violations.append(
                    {
                        "query": case.query,
                        "tool": item.get("name"),
                        "risk": item.get("risk"),
                        "allowed_risk": case.allowed_risk,
                    }
                )
    return {
        "filtered_case_count": len(blocked_cases),
        "violation_count": len(violations),
        "violations": violations,
    }


def _scenario_table(stats: list[SchemaStats], *, base: SchemaStats) -> str:
    lines = [
        "| Scenario | Tool count | Count reduction | Schema chars | Char reduction | Rough tokens | Token reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in stats:
        lines.append(
            "| {label} | {count} | {count_red} | {chars} | {char_red} | {tokens} | {token_red} |".format(
                label=item.label,
                count=item.tool_count,
                count_red=_pct(_reduction(base.tool_count, item.tool_count)),
                chars=item.schema_chars,
                char_red=_pct(_reduction(base.schema_chars, item.schema_chars)),
                tokens=item.token_estimate,
                token_red=_pct(_reduction(base.token_estimate, item.token_estimate)),
            )
        )
    return "\n".join(lines)


def _search_table(search_eval: dict[str, Any]) -> str:
    return "\n".join(
        [
            "| Metric | Value |",
            "|---|---:|",
            f"| Cases | {search_eval['case_count']} |",
            f"| Hit@1 | {search_eval['hit_at_1']} ({search_eval['hit_at_1_rate']:.2f}%) |",
            f"| Hit@3 | {search_eval['hit_at_3']} ({search_eval['hit_at_3_rate']:.2f}%) |",
            f"| Hit@{search_eval['top_k']} | {search_eval[f'hit_at_{search_eval['top_k']}']} ({search_eval[f'hit_at_{search_eval['top_k']}_rate']:.2f}%) |",
            f"| MRR | {search_eval['mrr']:.4f} |",
            f"| Top-1 errors | {search_eval['top1_error_count']} |",
            f"| No match | {search_eval['no_match_count']} |",
        ]
    )


def _tool_block(title: str, names: list[str], limit: int = 120) -> str:
    shown = names[:limit]
    lines = [f"### {title}", ""]
    lines.extend(f"- `{name}`" for name in shown)
    if len(names) > limit:
        lines.append(f"- ... ({len(names) - limit} more)")
    lines.append("")
    return "\n".join(lines)


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

        synthetic_specs = _register_synthetic_tools(tools, int(args.synthetic_count))
        all_names = tools.get_registered_order()
        always_on = tools.get_always_on_names()
        always_order = tools.get_registered_order(always_on)
        synthetic_names = [spec.name for spec in synthetic_specs]

        cases = [
            case for case in _query_cases()
            if case.target in set(synthetic_names)
        ]

        discovery = ToolDiscoveryState(capacity=int(args.preload_capacity))
        session_key = "bench:tool-search-pressure"
        warm_seed = [case.target for case in cases[: int(args.preload_capacity)]]
        discovery.update(session_key, warm_seed, always_on)
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

        search_eval = _evaluate_search(
            tools,
            cases,
            excluded_names=always_on,
            top_k=int(args.top_k),
        )
        risk_eval = _evaluate_risk_filter(
            tools,
            cases,
            excluded_names=always_on,
        )

        metadata = getattr(tools, "_metadata", {})
        documents = getattr(tools, "_documents", {})
        risk_counts = Counter(
            getattr(meta, "risk", "unknown") for meta in metadata.values()
        )
        source_counts = Counter(
            getattr(doc, "source_type", "unknown") for doc in documents.values()
        )
        source_name_counts = Counter(
            getattr(doc, "source_name", "unknown") for doc in documents.values()
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = {
            "created_at": datetime.now().astimezone().isoformat(),
            "project_root": str(PROJECT_ROOT),
            "config_path": str(config_path),
            "benchmark_workspace": str(workspace),
            "synthetic_count": len(synthetic_specs),
            "registered_tool_count": len(all_names),
            "always_on_count": len(always_order),
            "deferred_count": len(all_names) - len(always_order),
            "preload_capacity": int(args.preload_capacity),
            "top_k": int(args.top_k),
            "risk_counts": dict(sorted(risk_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "source_name_counts": dict(sorted(source_name_counts.items())),
            "scenarios": [asdict(full_stats), asdict(cold_stats), asdict(warm_stats)],
            "reductions_vs_full": {
                "cold_count_pct": _reduction(full_stats.tool_count, cold_stats.tool_count),
                "cold_chars_pct": _reduction(full_stats.schema_chars, cold_stats.schema_chars),
                "cold_tokens_pct": _reduction(full_stats.token_estimate, cold_stats.token_estimate),
                "warm_count_pct": _reduction(full_stats.tool_count, warm_stats.tool_count),
                "warm_chars_pct": _reduction(full_stats.schema_chars, warm_stats.schema_chars),
                "warm_tokens_pct": _reduction(full_stats.token_estimate, warm_stats.token_estimate),
            },
            "search_eval": search_eval,
            "risk_eval": risk_eval,
            "always_on_names": always_order,
            "preloaded_names": preloaded_order,
            "synthetic_names": synthetic_names,
        }

        json_path = output_dir / f"tool_search_pressure_{timestamp}.json"
        md_path = output_dir / f"tool_search_pressure_{timestamp}.md"
        result["json_path"] = str(json_path)
        result["markdown_path"] = str(md_path)
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        error_rows = [
            row for row in search_eval["rows"]
            if row["rank"] != 1
        ][:20]
        error_lines = [
            f"- `{row['query']}` target=`{row['target']}` rank=`{row['rank']}` top={row['top_names']}"
            for row in error_rows
        ] or ["- None"]

        report = "\n".join(
            [
                "# Tool Search Pressure Benchmark",
                "",
                "This benchmark registers the project's real tools, adds synthetic MCP/business tools, and measures progressive schema exposure plus catalog-search quality. It does not call LLMs or execute tools.",
                "",
                f"- Created at: `{result['created_at']}`",
                f"- Benchmark workspace: `{workspace}`",
                f"- Synthetic deferred tools: `{len(synthetic_specs)}`",
                f"- Registered tools: `{len(all_names)}`",
                f"- Always-on tools: `{len(always_order)}`",
                f"- Deferred tools: `{len(all_names) - len(always_order)}`",
                "",
                "## Schema Exposure",
                "",
                _scenario_table([full_stats, cold_stats, warm_stats], base=full_stats),
                "",
                "## Tool Search Quality",
                "",
                _search_table(search_eval),
                "",
                "## Risk Filtering",
                "",
                f"- Read-only filtered cases: `{risk_eval['filtered_case_count']}`",
                f"- Risk violations: `{risk_eval['violation_count']}`",
                "",
                "## Distribution",
                "",
                f"- Risk counts: `{dict(sorted(risk_counts.items()))}`",
                f"- Source counts: `{dict(sorted(source_counts.items()))}`",
                "",
                "## Top-1 Errors",
                "",
                *error_lines,
                "",
                _tool_block("Always-On Tools", always_order),
                _tool_block("Warm LRU Preloaded Tools", preloaded_order),
                _tool_block("Synthetic Deferred Tools", synthetic_names),
                "## Notes",
                "",
                "- This is an offline catalog-routing benchmark. It can support claims about schema exposure and tool_search retrieval quality.",
                "- It should not be described as LLM tool-selection accuracy unless paired with a separate model-call evaluation.",
                "- Dynamic real MCP servers are not connected here; synthetic tools model a large MCP/business tool catalog without side effects.",
                "",
            ]
        )
        md_path.write_text(report, encoding="utf-8")
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
        description="Pressure-test tool_search schema exposure with synthetic deferred tools."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.example.toml"))
    parser.add_argument(
        "--workspace",
        default=str(PROJECT_ROOT / "_bench" / "workspaces" / "tool_search_pressure"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "_bench" / "results"),
    )
    parser.add_argument("--synthetic-count", type=int, default=60)
    parser.add_argument("--preload-capacity", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
