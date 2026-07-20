from eval.memory2_cluster.draft_clusters import (
    TimelineDraft,
    render_review_markdown,
    validate_draft_sources,
)


def _timeline() -> dict:
    return {
        "timeline_id": "timeline_1",
        "messages": [
            {
                "source_ref": "source_a",
                "timestamp": "2026-01-01T10:00:00+08:00",
                "kind": "user",
                "content": "我喜欢火锅",
            }
        ],
    }


def _draft() -> TimelineDraft:
    return TimelineDraft.model_validate(
        {
            "timeline_id": "timeline_1",
            "status": "draft_needs_human_review",
            "memories": [
                {
                    "memory_id": "food_preference_1",
                    "cluster_id": "food_preference",
                    "memory_type": "preference",
                    "summary": "用户喜欢火锅。",
                    "happened_at": "2026-01-01T10:00:00+08:00",
                    "source_refs": ["source_a"],
                    "validity": "current",
                    "confidence": 0.95,
                    "review_notes": "",
                }
            ],
            "clusters": [
                {
                    "cluster_id": "food_preference",
                    "title": "食物偏好",
                    "description": "用户长期食物偏好。",
                    "relation": "preference_reinforcement",
                    "memory_ids": ["food_preference_1"],
                    "review_notes": "",
                }
            ],
            "omitted_as_non_memory": [],
        }
    )


def test_validate_draft_sources_accepts_consistent_draft() -> None:
    validate_draft_sources(_timeline(), _draft())


def test_validate_draft_sources_rejects_unknown_source() -> None:
    draft = _draft()
    draft.memories[0].source_refs = ["source_missing"]

    try:
        validate_draft_sources(_timeline(), draft)
    except ValueError as exc:
        assert "未知 source_ref" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_draft_sources_rejects_incomplete_cluster_membership() -> None:
    draft = _draft()
    draft.clusters[0].memory_ids = []

    try:
        validate_draft_sources(_timeline(), draft)
    except ValueError as exc:
        assert "memory_ids 不完整" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_render_review_markdown_contains_sources_and_review_slots() -> None:
    review = render_review_markdown([_timeline()], [_draft()])

    assert "food_preference_1" in review
    assert "source_a" in review
    assert "我喜欢火锅" in review
    assert "人工结论" in review
