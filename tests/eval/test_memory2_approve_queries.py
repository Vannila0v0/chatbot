import json

from eval.memory2_cluster.approve_queries import approve_queries, update_manifest


def test_approve_queries_preserves_content_and_updates_status(tmp_path) -> None:
    source = tmp_path / "candidate.jsonl"
    output = tmp_path / "approved.jsonl"
    manifest = tmp_path / "manifest.json"
    source.write_text(
        json.dumps(
            {
                "case_id": "case_1",
                "timeline_id": "timeline_1",
                "query": "用户喜欢什么？",
                "query_time": "2026-01-01T00:00:00+08:00",
                "cluster_oracle": {"preference": "core"},
                "review_status": "candidate",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    probes = approve_queries(source, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    updated = update_manifest(
        manifest,
        approved_path=output,
        probes=probes,
        approval_note="人工确认全部通过",
    )

    assert payload["query"] == "用户喜欢什么？"
    assert payload["review_status"] == "approved"
    assert updated["query_status"] == "frozen_human_approved"
    assert len(updated["approved_query_sha256"]) == 64
    assert updated["benchmark_executed"] is False
