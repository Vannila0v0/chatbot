from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import ClusterProbe


def approve_queries(input_path: Path, output_path: Path) -> list[ClusterProbe]:
    probes: list[ClusterProbe] = []
    case_ids: set[str] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                probe = ClusterProbe.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"Query 第 {line_number} 行无效: {exc}") from exc
            if probe.case_id in case_ids:
                raise ValueError(f"重复 case_id: {probe.case_id}")
            probe.review_status = "approved"
            case_ids.add(probe.case_id)
            probes.append(probe)
    if not probes:
        raise ValueError("Query 文件不能为空")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for probe in probes:
            handle.write(probe.model_dump_json() + "\n")
    return probes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def update_manifest(
    manifest_path: Path,
    *,
    approved_path: Path,
    probes: list[ClusterProbe],
    approval_note: str,
) -> dict[str, Any]:
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    split_counts = {
        split: sum(probe.dataset_split == split for probe in probes)
        for split in ("dev", "validation", "test")
    }
    manifest.update(
        {
            "query_status": "frozen_human_approved",
            "query_approval_note": approval_note,
            "queries": len(probes),
            "dev_queries": split_counts["dev"],
            "validation_queries": split_counts["validation"],
            "test_queries": split_counts["test"],
            "approved_query_file": str(approved_path),
            "approved_query_sha256": sha256_file(approved_path),
            "query_approved_at": datetime.now().astimezone().isoformat(),
            "benchmark_executed": False,
            "test_split_executed": False,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="批准并冻结人工审核后的 query")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--approval-note", required=True)
    args = parser.parse_args()
    probes = approve_queries(args.input, args.output)
    manifest = update_manifest(
        args.manifest,
        approved_path=args.output,
        probes=probes,
        approval_note=args.approval_note,
    )
    print(
        f"approved queries={len(probes)} "
        f"sha256={manifest['approved_query_sha256']}"
    )


if __name__ == "__main__":
    main()
