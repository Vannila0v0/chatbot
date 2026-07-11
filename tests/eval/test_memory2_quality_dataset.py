from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.memory2_quality.dataset import load_cases


def _base_case() -> dict:
    return {
        "case_id": "case_001",
        "category": "type_identification",
        "description": "event type",
        "reference_time": "2026-07-01T10:00:00+08:00",
        "initial_memories": [],
        "recall_fixture_memories": [],
        "sessions": [
            {
                "session_id": "s1",
                "timestamp": "2026-07-01T10:00:00+08:00",
                "messages": [{"role": "user", "content": "我昨天完成了答辩"}],
                "consolidate_after": True,
            }
        ],
        "expected_write": {
            "required": [
                {
                    "label": "defense_event",
                    "memory_type": "event",
                    "facts": ["用户完成了答辩"],
                }
            ]
        },
        "recall_probes": [
            {
                "probe_id": "p1",
                "query": "我最近完成了什么？",
                "required_memory_labels": ["defense_event"],
            }
        ],
        "tags": ["event"],
    }


def _write_jsonl(path: Path, *items: dict) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )


def test_load_cases_parses_valid_case(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    _write_jsonl(path, _base_case())

    cases = load_cases(path)

    assert len(cases) == 1
    assert cases[0].case_id == "case_001"
    assert cases[0].expected_write.required[0].memory_type == "event"


def test_load_cases_rejects_duplicate_case_id(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    _write_jsonl(path, _base_case(), _base_case())

    with pytest.raises(ValueError, match="重复 case_id"):
        load_cases(path)


def test_load_cases_rejects_unknown_action_target(tmp_path: Path) -> None:
    case = _base_case()
    case["expected_write"]["expected_actions"] = [
        {"target_local_id": "missing", "action": "supersede"}
    ]
    path = tmp_path / "cases.jsonl"
    _write_jsonl(path, case)

    with pytest.raises(ValueError, match="missing"):
        load_cases(path)


def test_load_cases_rejects_unknown_required_memory_label(tmp_path: Path) -> None:
    case = _base_case()
    case["recall_probes"][0]["required_memory_labels"] = ["missing_label"]
    path = tmp_path / "cases.jsonl"
    _write_jsonl(path, case)

    with pytest.raises(ValueError, match="missing_label"):
        load_cases(path)


def test_load_cases_rejects_duplicate_local_id(tmp_path: Path) -> None:
    case = _base_case()
    memory = {
        "local_id": "same",
        "memory_type": "profile",
        "summary": "用户住在桂林",
    }
    case["initial_memories"] = [memory, memory]
    path = tmp_path / "cases.jsonl"
    _write_jsonl(path, case)

    with pytest.raises(ValueError, match="重复 local_id"):
        load_cases(path)

