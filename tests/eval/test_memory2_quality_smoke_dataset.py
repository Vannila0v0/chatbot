from pathlib import Path

from eval.memory2_quality.dataset import load_cases


def test_smoke_dataset_has_ten_valid_cases() -> None:
    path = Path("eval/memory2_quality/datasets/smoke.jsonl")
    cases = load_cases(path)
    assert len(cases) == 10
    assert len({case.case_id for case in cases}) == 10
    assert all(case.sessions for case in cases)
    assert {case.category for case in cases} == {
        "type_identification",
        "temporary_state",
        "history_current_conflict",
        "entity_attribute_conflict",
        "noise_extraction",
    }

