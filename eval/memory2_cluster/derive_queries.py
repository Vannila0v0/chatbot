from __future__ import annotations

import argparse
import asyncio
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.config_models import Config
from agent.llm_json import load_json_object_loose
from bootstrap.providers import build_providers

from .models import ClusterProbe, ClusterRole, EventTimeline


class QuerySuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    cluster_oracle: dict[str, ClusterRole] = Field(min_length=1)
    preferred_pairs: list[tuple[str, str]] = Field(default_factory=list)
    memory_oracle: dict[str, ClusterRole] = Field(min_length=1)
    preferred_memory_pairs: list[tuple[str, str]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class QueryBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeline_id: str
    queries: list[QuerySuggestion] = Field(min_length=4, max_length=4)


def _read_timelines(path: Path) -> list[EventTimeline]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(EventTimeline.model_validate_json(line))
            except (ValidationError, ValueError) as exc:
                raise ValueError(f"时间线第 {line_number} 行无效: {exc}") from exc
    return rows


def _render_timeline(timeline: EventTimeline) -> str:
    cluster_by_id = {cluster.cluster_id: cluster for cluster in timeline.clusters}
    blocks = []
    for cluster_id in sorted({memory.cluster_id for memory in timeline.memories}):
        cluster = cluster_by_id.get(cluster_id)
        header = f"CLUSTER: {cluster_id}"
        if cluster:
            header += (
                f"\nTITLE: {cluster.title}\nRELATION: {cluster.relation}"
                f"\nDESCRIPTION: {cluster.description}"
            )
        memories = [
            memory for memory in timeline.memories if memory.cluster_id == cluster_id
        ]
        memory_lines = [
            (
                f"- {memory.local_id} | type={memory.memory_type} | "
                f"reinforcement={memory.reinforcement} | "
                f"last_used_days_ago={memory.last_used_days_ago:.3f} | "
                f"summary={memory.summary}"
            )
            for memory in memories
        ]
        blocks.append(header + "\nMEMORIES:\n" + "\n".join(memory_lines))
    return "\n\n---\n\n".join(blocks)


def _prompt(timeline: EventTimeline) -> str:
    cluster_ids = sorted({memory.cluster_id for memory in timeline.memories})
    memory_ids = sorted(memory.local_id for memory in timeline.memories)
    return f"""你正在从已经冻结、并在 query 生成之前完成的自然对话记忆时间线中派生检索评测问题。

必须生成恰好 4 个自然、互不重复的 query。不要修改记忆，不要增加时间线里没有的事实。

四个 query 的覆盖原则：
1. 一个普通事实或偏好查询，不要求时间/强化机制一定获益。
2. 一个当前状态或时间变化查询；若时间线没有合理的状态变化，则改为第二个普通查询。
3. 一个 guardrail 查询，检查旧但稳定的 profile/procedure/preference 是否会被近期高频噪音压过；没有合适场景时使用稳定事实查询。
4. 一个需要多个相关簇共同支持、或需要区分相关与无关信息的自然查询。

标注规则：
- cluster_oracle 必须包含下面列出的全部 cluster_id，不能漏标。
- memory_oracle 必须包含下面列出的全部 memory_id，不能漏标。它用于区分同一事件簇中的新旧状态。
- core：回答问题必须召回；supporting：明显有帮助但非必需；weak：弱相关。
- irrelevant：不相关；forbidden：会直接导致错误答案的冲突事实或过期状态。不要把普通噪音滥标为 forbidden。
- preferred_pairs 只在“前者必须排在后者之前”具有明确事实依据时填写，否则留空。
- preferred_pairs 的前者必须具有比后者更高的 oracle 相关性，不能比较同一个簇。
- preferred_memory_pairs 使用相同规则；状态演化时应明确标出“当前记忆 > 旧记忆”。
- query 必须像用户真的会问的话，不能出现“召回、记忆簇、benchmark、热度”等评测术语。
- 不要为了让 reinforcement 或时间衰减获胜而故意造问题。普通、获益和护栏场景要同时存在。
- rationale 只供人工审核，简要说明标注依据，不会注入被测系统。
- tags 从 neutral、temporal、guardrail、multi_cluster、preference、profile、procedure、noise 中选择。

全部 cluster_id：
{json.dumps(cluster_ids, ensure_ascii=False)}

全部 memory_id：
{json.dumps(memory_ids, ensure_ascii=False)}

只输出 JSON，不要 Markdown：
{{
  "timeline_id": "{timeline.timeline_id}",
  "queries": [
    {{
      "query": "...",
      "cluster_oracle": {{"每个cluster_id": "core|supporting|weak|irrelevant|forbidden"}},
      "preferred_pairs": [["更应优先的cluster", "应排后的cluster"]],
      "memory_oracle": {{"每个memory_id": "core|supporting|weak|irrelevant|forbidden"}},
      "preferred_memory_pairs": [["更应优先的memory", "应排后的memory"]],
      "tags": ["neutral"],
      "rationale": "..."
    }}
  ]
}}

冻结记忆时间线：

{_render_timeline(timeline)}
"""


def validate_query_batch(timeline: EventTimeline, batch: QueryBatch) -> None:
    if batch.timeline_id != timeline.timeline_id:
        raise ValueError("模型返回的 timeline_id 不匹配")
    cluster_ids = {memory.cluster_id for memory in timeline.memories}
    seen_queries: set[str] = set()
    for index, query in enumerate(batch.queries, 1):
        normalized = "".join(query.query.lower().split())
        if normalized in seen_queries:
            raise ValueError(f"第 {index} 个 query 与其他 query 重复")
        seen_queries.add(normalized)
        oracle_ids = set(query.cluster_oracle)
        if oracle_ids != cluster_ids:
            missing = sorted(cluster_ids - oracle_ids)
            extra = sorted(oracle_ids - cluster_ids)
            raise ValueError(
                f"第 {index} 个 query 的 oracle 不完整: missing={missing}, extra={extra}"
            )
        if "core" not in query.cluster_oracle.values():
            raise ValueError(f"第 {index} 个 query 没有 core cluster")
        memory_ids = {memory.local_id for memory in timeline.memories}
        oracle_memory_ids = set(query.memory_oracle)
        if oracle_memory_ids != memory_ids:
            missing = sorted(memory_ids - oracle_memory_ids)
            extra = sorted(oracle_memory_ids - memory_ids)
            raise ValueError(
                f"第 {index} 个 query 的 memory_oracle 不完整: "
                f"missing={missing}, extra={extra}"
            )
        if "core" not in query.memory_oracle.values():
            raise ValueError(f"第 {index} 个 query 没有 core memory")
        pair_ids = {cluster_id for pair in query.preferred_pairs for cluster_id in pair}
        if not pair_ids <= cluster_ids:
            raise ValueError(f"第 {index} 个 query 的 preferred_pairs 含未知 cluster")
        role_priority = {
            "core": 4,
            "supporting": 3,
            "weak": 2,
            "irrelevant": 1,
            "forbidden": 0,
        }
        for better, worse in query.preferred_pairs:
            if better == worse:
                raise ValueError(f"第 {index} 个 query 的 preferred_pair 不能引用同一簇")
            if role_priority[query.cluster_oracle[better]] <= role_priority[
                query.cluster_oracle[worse]
            ]:
                raise ValueError(
                    f"第 {index} 个 query 的 preferred_pair 方向与 oracle 相关性冲突: "
                    f"{better} -> {worse}"
                )
        pair_memory_ids = {
            memory_id
            for pair in query.preferred_memory_pairs
            for memory_id in pair
        }
        if not pair_memory_ids <= memory_ids:
            raise ValueError(
                f"第 {index} 个 query 的 preferred_memory_pairs 含未知 memory"
            )
        for better, worse in query.preferred_memory_pairs:
            if better == worse:
                raise ValueError(
                    f"第 {index} 个 query 的 preferred_memory_pair 不能引用同一记忆"
                )
            if role_priority[query.memory_oracle[better]] <= role_priority[
                query.memory_oracle[worse]
            ]:
                raise ValueError(
                    f"第 {index} 个 query 的 preferred_memory_pair 方向与 oracle 冲突: "
                    f"{better} -> {worse}"
                )


def normalize_preferred_pairs(batch: QueryBatch) -> None:
    role_priority = {
        "core": 4,
        "supporting": 3,
        "weak": 2,
        "irrelevant": 1,
        "forbidden": 0,
    }
    for query in batch.queries:
        normalized: list[tuple[str, str]] = []
        known = set(query.cluster_oracle)
        for better, worse in query.preferred_pairs:
            pair = (better, worse)
            if pair in normalized or better == worse or not {better, worse} <= known:
                continue
            if role_priority[query.cluster_oracle[better]] <= role_priority[
                query.cluster_oracle[worse]
            ]:
                continue
            normalized.append(pair)
        query.preferred_pairs = normalized
        normalized_memories: list[tuple[str, str]] = []
        known_memories = set(query.memory_oracle)
        for better, worse in query.preferred_memory_pairs:
            pair = (better, worse)
            if (
                pair in normalized_memories
                or better == worse
                or not {better, worse} <= known_memories
            ):
                continue
            if role_priority[query.memory_oracle[better]] <= role_priority[
                query.memory_oracle[worse]
            ]:
                continue
            normalized_memories.append(pair)
        query.preferred_memory_pairs = normalized_memories


async def generate_batch(
    provider: Any,
    model: str,
    timeline: EventTimeline,
    *,
    max_tokens: int,
    attempts: int,
) -> QueryBatch:
    prompt = _prompt(timeline)
    last_error: Exception | None = None
    for attempt in range(attempts):
        suffix = ""
        if attempt and last_error is not None:
            suffix = (
                "\n\n上一版未通过校验，错误如下：\n"
                f"{last_error}\n请重新输出完整 JSON 并修复。"
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
            batch = QueryBatch.model_validate(payload)
            normalize_preferred_pairs(batch)
            validate_query_batch(timeline, batch)
            return batch
        except (ValidationError, ValueError, RuntimeError) as exc:
            last_error = exc
    raise RuntimeError(f"时间线 {timeline.timeline_id} 生成失败: {last_error}")


def _dataset_split(index: int, total: int) -> Literal["dev", "validation", "test"]:
    dev_end = max(1, round(total * 0.6))
    validation_end = max(dev_end + 1, round(total * 0.8))
    if index < dev_end:
        return "dev"
    if index < validation_end:
        return "validation"
    return "test"


def materialize_probes(
    timelines: list[EventTimeline], batches: dict[str, QueryBatch]
) -> list[ClusterProbe]:
    probes = []
    for timeline_index, timeline in enumerate(timelines):
        if timeline.window_end is None:
            raise ValueError(f"时间线 {timeline.timeline_id} 缺少 window_end")
        split = _dataset_split(timeline_index, len(timelines))
        batch = batches[timeline.timeline_id]
        for query_index, suggestion in enumerate(batch.queries, 1):
            probes.append(
                ClusterProbe(
                    case_id=f"{timeline.timeline_id}_q{query_index:02d}",
                    timeline_id=timeline.timeline_id,
                    query=suggestion.query,
                    query_time=timeline.window_end + timedelta(hours=1),
                    top_k=5,
                    cluster_oracle=suggestion.cluster_oracle,
                    preferred_pairs=suggestion.preferred_pairs,
                    memory_oracle=suggestion.memory_oracle,
                    preferred_memory_pairs=suggestion.preferred_memory_pairs,
                    split="natural",
                    dataset_split=split,
                    review_status="candidate",
                    rationale=suggestion.rationale,
                    tags=["expanded", split, *suggestion.tags],
                )
            )
    return probes


def render_query_review(probes: list[ClusterProbe]) -> str:
    lines = [
        "# Memory2 派生 Query 人工审核表",
        "",
        "> 记忆与事件簇已经冻结。本文件只审核 query、cluster oracle 和 preferred pair；不得反向修改冻结记忆来迎合 query。",
        "",
        f"- Query 总数：{len(probes)}",
        f"- Dev：{sum(probe.dataset_split == 'dev' for probe in probes)}",
        f"- Validation：{sum(probe.dataset_split == 'validation' for probe in probes)}",
        f"- Test：{sum(probe.dataset_split == 'test' for probe in probes)}",
        "",
    ]
    for probe in probes:
        lines.extend(
            [
                f"## `{probe.case_id}`",
                "",
                f"- 数据划分：`{probe.dataset_split}`",
                f"- Query：{probe.query}",
                f"- Tags：{', '.join(f'`{tag}`' for tag in probe.tags)}",
                f"- Preferred pairs：{probe.preferred_pairs or '无'}",
                f"- Preferred memory pairs：{probe.preferred_memory_pairs or '无'}",
                f"- 标注理由：{probe.rationale}",
                "- 人工结论：[ ] 通过　[ ] 修改　[ ] 删除",
                "",
                "| Cluster | Role |",
                "|---|---|",
            ]
        )
        lines.extend(
            f"| `{cluster_id}` | `{role}` |"
            for cluster_id, role in probe.cluster_oracle.items()
        )
        lines.extend(["", "| Memory | Role |", "|---|---|"])
        lines.extend(
            f"| `{memory_id}` | `{role}` |"
            for memory_id, role in probe.memory_oracle.items()
        )
        lines.append("")
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> None:
    timelines = _read_timelines(args.timelines)
    config = Config.load(args.config)
    provider, _, _ = build_providers(config)
    semaphore = asyncio.Semaphore(max(1, args.workers))

    async def process(timeline: EventTimeline) -> tuple[str, QueryBatch | BaseException]:
        async with semaphore:
            try:
                batch = await generate_batch(
                    provider,
                    config.model,
                    timeline,
                    max_tokens=args.max_tokens,
                    attempts=args.attempts,
                )
                print(f"{timeline.timeline_id}: queries={len(batch.queries)}")
                return timeline.timeline_id, batch
            except BaseException as exc:
                return timeline.timeline_id, exc

    results = await asyncio.gather(*(process(timeline) for timeline in timelines))
    errors = {timeline_id: result for timeline_id, result in results if isinstance(result, BaseException)}
    if errors:
        details = "; ".join(f"{key}: {value}" for key, value in errors.items())
        raise RuntimeError(f"query 生成失败: {details}")
    batches = {
        timeline_id: result
        for timeline_id, result in results
        if isinstance(result, QueryBatch)
    }
    probes = materialize_probes(timelines, batches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for probe in probes:
            handle.write(probe.model_dump_json() + "\n")
    if args.review_output:
        args.review_output.write_text(render_query_review(probes), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从冻结记忆时间线派生 query 草稿")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--timelines", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--attempts", type=int, default=3)
    return parser


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
