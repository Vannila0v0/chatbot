from argparse import Namespace

from eval.memory2_cluster.compare import _probe_payload, _select_probes, build_parser
from eval.memory2_cluster.models import ClusterProbe


def _probe(case_id: str, dataset_split: str, review_status: str) -> ClusterProbe:
    return ClusterProbe(
        case_id=case_id,
        timeline_id=f"timeline_{dataset_split}",
        query="测试问题",
        query_time="2026-01-01T00:00:00+08:00",
        cluster_oracle={"cluster": "core"},
        memory_oracle={"memory": "core"},
        dataset_split=dataset_split,
        review_status=review_status,
    )


def test_parser_defaults_to_approved_gate_and_supports_dry_run() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "config.toml",
            "--timelines",
            "timelines.jsonl",
            "--dataset",
            "queries.jsonl",
            "--dataset-split",
            "dev",
            "--dry-run",
        ]
    )

    assert args.dataset_split == "dev"
    assert args.require_approved is True
    assert args.dry_run is True


def test_select_probes_filters_dev_and_rejects_unapproved() -> None:
    args = Namespace(
        dataset_split="dev",
        case_id=None,
        limit=0,
        require_approved=True,
    )
    probes = [_probe("dev_ok", "dev", "approved"), _probe("test_ok", "test", "approved")]

    assert [probe.case_id for probe in _select_probes(probes, args)] == ["dev_ok"]

    probes.append(_probe("dev_bad", "dev", "candidate"))
    try:
        _select_probes(probes, args)
    except ValueError as exc:
        assert "未批准" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_probe_payload_contains_memory_oracle_and_hashes() -> None:
    probe = _probe("dev_ok", "dev", "approved")

    payload = _probe_payload(
        probe, timelines_sha256="timeline-hash", dataset_sha256="dataset-hash"
    )

    assert payload["timeline_id"] == "timeline_dev"
    assert payload["dataset_split"] == "dev"
    assert payload["memory_oracle"] == {"memory": "core"}
    assert payload["timelines_sha256"] == "timeline-hash"
    assert payload["dataset_sha256"] == "dataset-hash"
