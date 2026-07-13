from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .draft_clusters import TimelineDraft, validate_draft_sources
from .models import ClusterDefinition, ClusterMemory, EventTimeline


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


def freeze_timeline(
    source_timeline: dict[str, Any], draft: TimelineDraft
) -> EventTimeline:
    validate_draft_sources(source_timeline, draft)
    source_by_ref = {
        str(message["source_ref"]): message
        for message in source_timeline.get("messages") or []
    }
    window_end = datetime.fromisoformat(str(source_timeline["end"]))
    memories = []
    for memory in draft.memories:
        timestamps = {
            datetime.fromisoformat(str(source_by_ref[ref]["timestamp"]))
            for ref in memory.source_refs
        }
        last_used_at = max(timestamps)
        last_used_days_ago = max(
            0.0, (window_end - last_used_at).total_seconds() / 86400.0
        )
        memories.append(
            ClusterMemory(
                local_id=memory.memory_id,
                cluster_id=memory.cluster_id,
                memory_type=memory.memory_type,
                summary=memory.summary,
                happened_at=memory.happened_at,
                reinforcement=max(1, len(timestamps)),
                last_used_days_ago=last_used_days_ago,
                emotional_weight=0,
                source_refs=memory.source_refs,
                confidence=memory.confidence,
            )
        )
    return EventTimeline(
        timeline_id=draft.timeline_id,
        description="从完整脱敏对话时间线抽取并经人工整批确认的冻结记忆",
        source="human_approved_sanitized_akashic_conversation",
        window_start=datetime.fromisoformat(str(source_timeline["start"])),
        window_end=window_end,
        memories=memories,
        clusters=[
            ClusterDefinition.model_validate(cluster.model_dump())
            for cluster in draft.clusters
        ],
    )


def freeze_all(source_path: Path, draft_path: Path) -> list[EventTimeline]:
    source_rows = _read_jsonl(source_path)
    source_by_id = {str(row["timeline_id"]): row for row in source_rows}
    drafts = [
        TimelineDraft.model_validate(row) for row in _read_jsonl(draft_path)
    ]
    draft_ids = {draft.timeline_id for draft in drafts}
    if draft_ids != set(source_by_id):
        raise ValueError("时间线草稿与来源时间线集合不一致，不能冻结")
    return [freeze_timeline(source_by_id[draft.timeline_id], draft) for draft in drafts]


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结人工确认的记忆与事件簇")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--drafts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    timelines = freeze_all(args.source, args.drafts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for timeline in timelines:
            handle.write(timeline.model_dump_json() + "\n")
    print(
        f"frozen timelines={len(timelines)} "
        f"memories={sum(len(timeline.memories) for timeline in timelines)}"
    )


if __name__ == "__main__":
    main()
