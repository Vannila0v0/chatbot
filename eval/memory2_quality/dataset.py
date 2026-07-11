from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .models import MemoryEvalCase


def load_cases(path: Path | str) -> list[MemoryEvalCase]:
    dataset_path = Path(path)
    cases: list[MemoryEvalCase] = []
    seen_case_ids: set[str] = set()

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                case = MemoryEvalCase.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"数据集第 {line_number} 行无效: {exc}") from exc
            if case.case_id in seen_case_ids:
                raise ValueError(f"重复 case_id: {case.case_id}")
            _validate_case_references(case)
            seen_case_ids.add(case.case_id)
            cases.append(case)
    return cases


def _validate_case_references(case: MemoryEvalCase) -> None:
    fixtures = [*case.initial_memories, *case.recall_fixture_memories]
    local_ids = [item.local_id for item in fixtures]
    duplicate_ids = sorted({item for item in local_ids if local_ids.count(item) > 1})
    if duplicate_ids:
        raise ValueError(f"case {case.case_id} 存在重复 local_id: {duplicate_ids}")

    known_local_ids = set(local_ids)
    required_labels = {
        fact.label for fact in case.expected_write.required if fact.label is not None
    }
    for action in case.expected_write.expected_actions:
        if action.target_local_id and action.target_local_id not in known_local_ids:
            raise ValueError(
                f"case {case.case_id} action 引用了未知 local_id: "
                f"{action.target_local_id}"
            )
        if action.target_label and action.target_label not in required_labels:
            raise ValueError(
                f"case {case.case_id} action 引用了未知 label: {action.target_label}"
            )

    for probe in case.recall_probes:
        for local_id in [*probe.required_local_ids, *probe.forbidden_local_ids]:
            if local_id not in known_local_ids:
                raise ValueError(
                    f"case {case.case_id} probe {probe.probe_id} 引用了未知 local_id: "
                    f"{local_id}"
                )
        for label in probe.required_memory_labels:
            if label not in required_labels:
                raise ValueError(
                    f"case {case.case_id} probe {probe.probe_id} 引用了未知 label: {label}"
                )
