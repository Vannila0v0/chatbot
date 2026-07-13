from eval.memory2_cluster.draft_clusters import TimelineDraft
from eval.memory2_cluster.freeze_drafts import freeze_timeline


def test_freeze_timeline_derives_reinforcement_and_age_mechanically() -> None:
    source = {
        "timeline_id": "timeline_1",
        "start": "2026-01-01T00:00:00+08:00",
        "end": "2026-01-11T00:00:00+08:00",
        "messages": [
            {
                "source_ref": "source_a",
                "timestamp": "2026-01-01T10:00:00+08:00",
                "kind": "user",
                "content": "第一次",
            },
            {
                "source_ref": "source_b",
                "timestamp": "2026-01-09T00:00:00+08:00",
                "kind": "user",
                "content": "第二次",
            },
        ],
    }
    draft = TimelineDraft.model_validate(
        {
            "timeline_id": "timeline_1",
            "memories": [
                {
                    "memory_id": "memory_1",
                    "cluster_id": "cluster_1",
                    "memory_type": "preference",
                    "summary": "用户重复表达某偏好。",
                    "happened_at": "2026-01-09T00:00:00+08:00",
                    "source_refs": ["source_a", "source_b"],
                    "validity": "current",
                    "confidence": 0.9,
                }
            ],
            "clusters": [
                {
                    "cluster_id": "cluster_1",
                    "title": "偏好",
                    "description": "重复偏好",
                    "relation": "preference_reinforcement",
                    "memory_ids": ["memory_1"],
                }
            ],
        }
    )

    frozen = freeze_timeline(source, draft)

    assert frozen.memories[0].reinforcement == 2
    assert frozen.memories[0].last_used_days_ago == 2.0
    assert frozen.memories[0].source_refs == ["source_a", "source_b"]
    assert frozen.clusters[0].memory_ids == ["memory_1"]
