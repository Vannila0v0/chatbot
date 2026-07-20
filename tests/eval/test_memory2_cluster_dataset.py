from pathlib import Path

from eval.memory2_cluster.dataset import load_cluster_dataset


ROOT = Path("eval/memory2_cluster/datasets")


def test_pilot_cluster_datasets_are_valid_and_timeline_isolated() -> None:
    timelines, dev = load_cluster_dataset(ROOT / "timelines.jsonl", ROOT / "dev.jsonl")
    _, test = load_cluster_dataset(ROOT / "timelines.jsonl", ROOT / "test.jsonl")

    assert len(timelines) == 3
    assert len(dev) == 6
    assert len(test) == 3
    assert {case.timeline_id for case in dev}.isdisjoint(
        {case.timeline_id for case in test}
    )
    for timeline in timelines.values():
        assert len(timeline.memories) >= 18
        assert len({memory.cluster_id for memory in timeline.memories}) >= 6
