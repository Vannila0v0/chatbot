from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.config_models import Config
from agent.llm_json import load_json_object_loose
from bootstrap.providers import build_providers


class DraftMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1)
    cluster_id: str = Field(min_length=1)
    memory_type: Literal["event", "profile", "preference", "procedure"]
    summary: str = Field(min_length=1)
    happened_at: datetime
    source_refs: list[str] = Field(min_length=1)
    validity: Literal["current", "stale", "temporary", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    review_notes: str = ""


class DraftCluster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    relation: Literal[
        "stable_fact",
        "state_evolution",
        "preference_reinforcement",
        "procedure",
        "related_events",
        "noise",
    ]
    memory_ids: list[str] = Field(min_length=1)
    review_notes: str = ""


class TimelineDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeline_id: str
    status: Literal["draft_needs_human_review"] = "draft_needs_human_review"
    memories: list[DraftMemory]
    clusters: list[DraftCluster]
    omitted_as_non_memory: list[str] = Field(default_factory=list)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {line_number} 行无效: {exc}") from exc
    return rows


def _render_messages(timeline: dict[str, Any]) -> str:
    blocks = []
    for message in timeline.get("messages") or []:
        blocks.append(
            "\n".join(
                [
                    f"SOURCE_REF: {message['source_ref']}",
                    f"TIME: {message['timestamp']}",
                    f"KIND: {message['kind']}",
                    f"CONTENT:\n{message['content']}",
                ]
            )
        )
    return "\n\n--- MESSAGE ---\n\n".join(blocks)


def _prompt(timeline: dict[str, Any]) -> str:
    return f"""你正在为长期记忆检索 benchmark 生成候选标注草稿。

任务顺序必须保持：现在只从已经冻结的完整时间线中抽取候选记忆和事件簇，不得生成 query、oracle、preferred pair 或评测答案。

要求：
1. 只抽取未来对话中可能有用、能够独立表达的记忆；寒暄、纯工具过程、助手的空泛建议可以省略。
2. 用户明确陈述、纠正或确认的事实优先。助手的猜测不能当成用户事实。
3. 主动推送不要逐条全部记忆化。只保留用户回应过、会影响后续状态，或可作为自然检索噪音的代表性事件，并将纯噪音 cluster 的 relation 标成 noise。
4. 同一事实被重复提及，可以合成一条记忆并累积多个 source_refs；状态发生变化时必须拆成多条记忆，放在同一个 state_evolution cluster 中，并标明 current/stale。
5. profile 是相对稳定的身份/属性；preference 是偏好；procedure 是长期规范；event 是具体发生的事情或临时状态。
6. happened_at 使用支撑该记忆的最新或最关键消息时间。
7. 每条记忆必须引用输入中真实存在的 source_ref。不得编造引用。
8. memory_id、cluster_id 使用 timeline 内唯一、稳定的英文 snake_case。cluster.memory_ids 必须覆盖对应记忆。
9. 这只是供人工审核的草稿。无法确定时降低 confidence，并在 review_notes 中说明。
10. 每条时间线通常抽取 8～25 条候选记忆；质量优先，不为凑数量制造事实。

只输出一个 JSON 对象，不要 Markdown，不要解释。结构必须严格如下：
{{
  "timeline_id": "{timeline['timeline_id']}",
  "status": "draft_needs_human_review",
  "memories": [
    {{
      "memory_id": "...",
      "cluster_id": "...",
      "memory_type": "event|profile|preference|procedure",
      "summary": "...",
      "happened_at": "ISO-8601",
      "source_refs": ["source_..."],
      "validity": "current|stale|temporary|uncertain",
      "confidence": 0.0,
      "review_notes": ""
    }}
  ],
  "clusters": [
    {{
      "cluster_id": "...",
      "title": "...",
      "description": "...",
      "relation": "stable_fact|state_evolution|preference_reinforcement|procedure|related_events|noise",
      "memory_ids": ["..."],
      "review_notes": ""
    }}
  ],
  "omitted_as_non_memory": ["简短说明省略了哪些类型的消息"]
}}

冻结时间线如下：

{_render_messages(timeline)}
"""


def validate_draft_sources(
    timeline: dict[str, Any], draft: TimelineDraft
) -> None:
    if draft.timeline_id != timeline.get("timeline_id"):
        raise ValueError("模型返回的 timeline_id 不匹配")
    valid_refs = {
        str(message["source_ref"]) for message in timeline.get("messages") or []
    }
    memory_ids = [memory.memory_id for memory in draft.memories]
    if len(memory_ids) != len(set(memory_ids)):
        raise ValueError("草稿包含重复 memory_id")
    cluster_ids = [cluster.cluster_id for cluster in draft.clusters]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("草稿包含重复 cluster_id")
    memory_by_id = {memory.memory_id: memory for memory in draft.memories}
    cluster_by_id = {cluster.cluster_id: cluster for cluster in draft.clusters}
    for memory in draft.memories:
        unknown_refs = sorted(set(memory.source_refs) - valid_refs)
        if unknown_refs:
            raise ValueError(
                f"memory {memory.memory_id} 引用了未知 source_ref: {unknown_refs}"
            )
        if memory.cluster_id not in cluster_by_id:
            raise ValueError(
                f"memory {memory.memory_id} 引用了未知 cluster: {memory.cluster_id}"
            )
    for cluster in draft.clusters:
        unknown_memories = sorted(set(cluster.memory_ids) - set(memory_by_id))
        if unknown_memories:
            raise ValueError(
                f"cluster {cluster.cluster_id} 引用了未知 memory: {unknown_memories}"
            )
        expected = {
            memory.memory_id
            for memory in draft.memories
            if memory.cluster_id == cluster.cluster_id
        }
        if set(cluster.memory_ids) != expected:
            raise ValueError(f"cluster {cluster.cluster_id} 的 memory_ids 不完整")


def render_review_markdown(
    timelines: list[dict[str, Any]], drafts: list[TimelineDraft]
) -> str:
    timeline_by_id = {str(row["timeline_id"]): row for row in timelines}
    source_kind = {
        str(message["source_ref"]): str(message["kind"])
        for timeline in timelines
        for message in timeline.get("messages") or []
    }
    all_memories = [memory for draft in drafts for memory in draft.memories]
    low_confidence = [memory for memory in all_memories if memory.confidence < 0.7]
    assistant_only = [
        memory
        for memory in all_memories
        if {source_kind[ref] for ref in memory.source_refs} == {"assistant"}
    ]
    lines = [
        "# Memory2 候选记忆与事件簇人工审核表",
        "",
        "> 本文件由模型自动生成，仅是草稿。请逐项核对 summary、类型、有效性、事件簇和 source_ref；此阶段不要添加 query 或 oracle。",
        "",
        "## 自动审计摘要",
        "",
        f"- 时间线：{len(drafts)}",
        f"- 候选记忆：{len(all_memories)}",
        f"- 事件簇：{sum(len(draft.clusters) for draft in drafts)}",
        f"- 低置信度记忆（< 0.70）：{len(low_confidence)}",
        f"- 仅由普通 assistant 消息支撑：{len(assistant_only)}",
        "- 建议优先审核低置信度、assistant-only、profile、procedure 和 stale 条目。",
        "",
        "### 优先审核 ID",
        "",
        "- 低置信度："
        + (", ".join(f"`{memory.memory_id}`" for memory in low_confidence) or "无"),
        "- assistant-only："
        + (", ".join(f"`{memory.memory_id}`" for memory in assistant_only) or "无"),
        "",
    ]
    for draft in drafts:
        timeline = timeline_by_id[draft.timeline_id]
        source_by_ref = {
            str(message["source_ref"]): message
            for message in timeline.get("messages") or []
        }
        lines.extend(
            [
                f"## {draft.timeline_id}",
                "",
                f"- 候选记忆：{len(draft.memories)}",
                f"- 事件簇：{len(draft.clusters)}",
                "- 人工结论：`待审核`",
                "",
            ]
        )
        memory_by_id = {memory.memory_id: memory for memory in draft.memories}
        for cluster in draft.clusters:
            lines.extend(
                [
                    f"### {cluster.title} (`{cluster.cluster_id}`)",
                    "",
                    f"- 关系：`{cluster.relation}`",
                    f"- 描述：{cluster.description}",
                    f"- 草稿备注：{cluster.review_notes or '无'}",
                    "- 人工结论：[ ] 保留　[ ] 修改　[ ] 删除　[ ] 拆分　[ ] 合并",
                    "",
                ]
            )
            for memory_id in cluster.memory_ids:
                memory = memory_by_id[memory_id]
                lines.extend(
                    [
                        f"#### `{memory.memory_id}`",
                        "",
                        f"- 类型：`{memory.memory_type}`",
                        f"- 有效性：`{memory.validity}`",
                        f"- 置信度：{memory.confidence:.2f}",
                        f"- 摘要：{memory.summary}",
                        f"- 发生时间：{memory.happened_at.isoformat()}",
                        f"- source_ref：{', '.join(f'`{ref}`' for ref in memory.source_refs)}",
                        f"- 草稿备注：{memory.review_notes or '无'}",
                        "- 人工结论：[ ] 保留　[ ] 修改　[ ] 删除",
                        "- 来源摘录：",
                        "",
                    ]
                )
                for ref in memory.source_refs:
                    source = source_by_ref[ref]
                    excerpt = " ".join(str(source["content"]).split())
                    if len(excerpt) > 240:
                        excerpt = excerpt[:237] + "..."
                    lines.append(
                        f"  - `{ref}` [{source['kind']}] {source['timestamp']}：{excerpt}"
                    )
                lines.append("")
        if draft.omitted_as_non_memory:
            lines.extend(
                [
                    "### 模型认为不应记忆的内容",
                    "",
                    *(f"- {item}" for item in draft.omitted_as_non_memory),
                    "",
                ]
            )
    return "\n".join(lines)


async def generate_draft(
    provider: Any,
    model: str,
    timeline: dict[str, Any],
    *,
    max_tokens: int,
    attempts: int = 2,
) -> TimelineDraft:
    last_error: Exception | None = None
    prompt = _prompt(timeline)
    for attempt in range(attempts):
        suffix = ""
        if attempt and last_error is not None:
            suffix = (
                "\n\n上一版未通过结构校验。具体错误如下：\n"
                f"{last_error}\n"
                "请重新输出完整 JSON，并严格修复该错误，保证所有引用和关联一致。"
            )
        try:
            response = await provider.chat(
                messages=[{"role": "user", "content": prompt + suffix}],
                tools=[],
                model=model,
                max_tokens=max_tokens,
                disable_thinking=True,
            )
            payload = load_json_object_loose(response.content or "")
            if payload is None:
                raise ValueError("模型没有返回 JSON 对象")
            draft = TimelineDraft.model_validate(payload)
            validate_draft_sources(timeline, draft)
            return draft
        except (ValidationError, ValueError, RuntimeError) as exc:
            last_error = exc
    raise RuntimeError(f"时间线 {timeline['timeline_id']} 生成失败: {last_error}")


async def run(args: argparse.Namespace) -> None:
    timelines = _read_jsonl(args.input)
    config = Config.load(args.config)
    provider, _, _ = build_providers(config)
    semaphore = asyncio.Semaphore(max(1, args.workers))
    existing_drafts = (
        [TimelineDraft.model_validate(row) for row in _read_jsonl(args.output)]
        if args.output.exists()
        else []
    )
    drafts_by_id = {draft.timeline_id: draft for draft in existing_drafts}
    pending = [
        timeline
        for timeline in timelines
        if str(timeline["timeline_id"]) not in drafts_by_id
    ]

    async def process(timeline: dict[str, Any]) -> TimelineDraft:
        async with semaphore:
            draft = await generate_draft(
                provider,
                config.model,
                timeline,
                max_tokens=args.max_tokens,
                attempts=args.attempts,
            )
            print(
                f"{draft.timeline_id}: memories={len(draft.memories)} "
                f"clusters={len(draft.clusters)}"
            )
            return draft

    results = await asyncio.gather(
        *(process(timeline) for timeline in pending), return_exceptions=True
    )
    errors = []
    for timeline, result in zip(pending, results):
        if isinstance(result, BaseException):
            errors.append(
                {
                    "timeline_id": timeline["timeline_id"],
                    "error": str(result),
                }
            )
        else:
            drafts_by_id[result.timeline_id] = result
    order = {str(timeline["timeline_id"]): index for index, timeline in enumerate(timelines)}
    drafts = sorted(drafts_by_id.values(), key=lambda draft: order[draft.timeline_id])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for draft in drafts:
            handle.write(draft.model_dump_json() + "\n")
    if args.review_output:
        args.review_output.parent.mkdir(parents=True, exist_ok=True)
        args.review_output.write_text(
            render_review_markdown(timelines, drafts), encoding="utf-8"
        )
    error_path = args.output.with_suffix(".errors.json")
    if errors:
        error_path.write_text(
            json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            f"{len(errors)} 条时间线失败，{len(drafts)} 条成功结果已保存；"
            f"详情见 {error_path}"
        )
    if error_path.exists():
        error_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从冻结时间线生成记忆和事件簇草稿")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--attempts", type=int, default=2)
    return parser


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
