from datetime import datetime

from eval.memory2_cluster.derive_queries import (
    QueryBatch,
    materialize_probes,
    normalize_preferred_pairs,
    validate_query_batch,
)
from eval.memory2_cluster.models import ClusterMemory, EventTimeline


def _timeline(timeline_id: str = "timeline_1") -> EventTimeline:
    return EventTimeline(
        timeline_id=timeline_id,
        window_end=datetime.fromisoformat("2026-01-10T00:00:00+08:00"),
        memories=[
            ClusterMemory(
                local_id="memory_a",
                cluster_id="cluster_a",
                memory_type="event",
                summary="当前状态。",
                happened_at=datetime.fromisoformat("2026-01-09T00:00:00+08:00"),
            ),
            ClusterMemory(
                local_id="memory_b",
                cluster_id="cluster_b",
                memory_type="event",
                summary="旧状态。",
                happened_at=datetime.fromisoformat("2026-01-01T00:00:00+08:00"),
            ),
        ],
    )


def _batch() -> QueryBatch:
    query = {
        "query": "现在是什么状态？",
        "cluster_oracle": {"cluster_a": "core", "cluster_b": "forbidden"},
        "preferred_pairs": [["cluster_a", "cluster_b"]],
        "memory_oracle": {"memory_a": "core", "memory_b": "forbidden"},
        "preferred_memory_pairs": [["memory_a", "memory_b"]],
        "tags": ["temporal"],
        "rationale": "当前状态应优先于旧状态。",
    }
    return QueryBatch.model_validate(
        {"timeline_id": "timeline_1", "queries": [query] * 4}
    )


def test_validate_query_batch_requires_distinct_queries() -> None:
    batch = _batch()

    try:
        validate_query_batch(_timeline(), batch)
    except ValueError as exc:
        assert "重复" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_materialize_probes_splits_by_whole_timeline() -> None:
    timelines = [_timeline(f"timeline_{index}") for index in range(1, 11)]
    batches = {}
    for timeline in timelines:
        queries = []
        for index in range(4):
            queries.append(
                {
                    "query": f"问题 {index}",
                    "cluster_oracle": {"cluster_a": "core", "cluster_b": "irrelevant"},
                    "preferred_pairs": [],
                    "memory_oracle": {"memory_a": "core", "memory_b": "irrelevant"},
                    "preferred_memory_pairs": [],
                    "tags": ["neutral"],
                    "rationale": "普通事实问题。",
                }
            )
        batches[timeline.timeline_id] = QueryBatch(
            timeline_id=timeline.timeline_id, queries=queries
        )

    probes = materialize_probes(timelines, batches)

    assert len(probes) == 40
    assert {probe.timeline_id for probe in probes if probe.dataset_split == "dev"} == {
        f"timeline_{index}" for index in range(1, 7)
    }
    assert {probe.timeline_id for probe in probes if probe.dataset_split == "validation"} == {
        "timeline_7",
        "timeline_8",
    }
    assert {probe.timeline_id for probe in probes if probe.dataset_split == "test"} == {
        "timeline_9",
        "timeline_10",
    }


def test_validate_query_batch_rejects_reversed_preferred_pair() -> None:
    queries = []
    for index in range(4):
        queries.append(
            {
                "query": f"问题 {index}",
                "cluster_oracle": {"cluster_a": "core", "cluster_b": "irrelevant"},
                "preferred_pairs": [["cluster_b", "cluster_a"]],
                "memory_oracle": {"memory_a": "core", "memory_b": "irrelevant"},
                "preferred_memory_pairs": [],
                "tags": ["neutral"],
                "rationale": "反向排序。",
            }
        )
    batch = QueryBatch(timeline_id="timeline_1", queries=queries)

    try:
        validate_query_batch(_timeline(), batch)
    except ValueError as exc:
        assert "方向与 oracle" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_normalize_preferred_pairs_removes_invalid_auxiliary_pairs() -> None:
    queries = []
    for index in range(4):
        queries.append(
            {
                "query": f"问题 {index}",
                "cluster_oracle": {"cluster_a": "core", "cluster_b": "irrelevant"},
                "preferred_pairs": [
                    ["cluster_b", "cluster_a"],
                    ["cluster_a", "cluster_a"],
                    ["cluster_a", "cluster_b"],
                ],
                "memory_oracle": {"memory_a": "core", "memory_b": "irrelevant"},
                "preferred_memory_pairs": [
                    ["memory_b", "memory_a"],
                    ["memory_a", "memory_a"],
                    ["memory_a", "memory_b"],
                ],
                "tags": ["neutral"],
                "rationale": "混合排序。",
            }
        )
    batch = QueryBatch(timeline_id="timeline_1", queries=queries)

    normalize_preferred_pairs(batch)

    assert all(query.preferred_pairs == [("cluster_a", "cluster_b")] for query in batch.queries)
    assert all(
        query.preferred_memory_pairs == [("memory_a", "memory_b")]
        for query in batch.queries
    )
