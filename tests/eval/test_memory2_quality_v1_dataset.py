from collections import Counter
from pathlib import Path

from eval.memory2_quality.dataset import load_cases


def test_v1_dataset_has_expected_distribution() -> None:
    cases = load_cases(Path("eval/memory2_quality/datasets/v1.jsonl"))
    assert len(cases) == 60
    assert Counter(case.category for case in cases) == {
        "type_identification": 16,
        "temporary_state": 10,
        "history_current_conflict": 10,
        "entity_attribute_conflict": 8,
        "noise_extraction": 10,
        "negative": 6,
    }
    assert all(case.source for case in cases)
    assert all(
        case.expected_write.required
        or case.expected_write.forbidden
        or case.expected_write.allowed_new_count is not None
        for case in cases
    )

