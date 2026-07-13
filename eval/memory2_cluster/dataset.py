from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .models import ClusterProbe, EventTimeline


def _read_jsonl(path: Path | str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {line_number} 行无效: {exc}") from exc
    return rows


def load_cluster_dataset(
    timelines_path: Path | str, probes_path: Path | str
) -> tuple[dict[str, EventTimeline], list[ClusterProbe]]:
    timelines: dict[str, EventTimeline] = {}
    for payload in _read_jsonl(timelines_path):
        try:
            timeline = EventTimeline.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"时间线无效: {exc}") from exc
        if timeline.timeline_id in timelines:
            raise ValueError(f"重复 timeline_id: {timeline.timeline_id}")
        local_ids = [memory.local_id for memory in timeline.memories]
        if len(local_ids) != len(set(local_ids)):
            raise ValueError(f"时间线 {timeline.timeline_id} 有重复 local_id")
        timelines[timeline.timeline_id] = timeline

    probes: list[ClusterProbe] = []
    case_ids: set[str] = set()
    for payload in _read_jsonl(probes_path):
        try:
            probe = ClusterProbe.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"查询无效: {exc}") from exc
        if probe.case_id in case_ids:
            raise ValueError(f"重复 case_id: {probe.case_id}")
        timeline = timelines.get(probe.timeline_id)
        if timeline is None:
            raise ValueError(f"case {probe.case_id} 引用了未知 timeline")
        cluster_ids = {memory.cluster_id for memory in timeline.memories}
        unknown_clusters = sorted(set(probe.cluster_oracle) - cluster_ids)
        if unknown_clusters:
            raise ValueError(f"case {probe.case_id} 引用了未知 cluster: {unknown_clusters}")
        if "core" not in probe.cluster_oracle.values():
            raise ValueError(f"case {probe.case_id} 至少需要一个 core cluster")
        pair_clusters = {cluster_id for pair in probe.preferred_pairs for cluster_id in pair}
        unknown_pair_clusters = sorted(pair_clusters - cluster_ids)
        if unknown_pair_clusters:
            raise ValueError(
                f"case {probe.case_id} 的 preferred_pairs 引用了未知 cluster: "
                f"{unknown_pair_clusters}"
            )
        case_ids.add(probe.case_id)
        probes.append(probe)
    return timelines, probes
