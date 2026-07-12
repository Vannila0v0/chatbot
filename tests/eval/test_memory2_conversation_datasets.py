from __future__ import annotations

import re
from pathlib import Path

from eval.memory2_quality.dataset import load_cases


DATASET_DIR = Path("eval/memory2_quality/datasets")
DEV_PATH = DATASET_DIR / "conversation_dev.jsonl"
TEST_PATH = DATASET_DIR / "conversation_test.jsonl"


def test_conversation_dataset_sizes_and_topic_isolation() -> None:
    dev = load_cases(DEV_PATH)
    test = load_cases(TEST_PATH)

    assert len(dev) == 8
    assert len(test) == 4
    assert {case.case_id for case in dev}.isdisjoint(
        {case.case_id for case in test}
    )
    dev_topics = {tag for case in dev for tag in case.tags if tag.startswith("topic:")}
    test_topics = {tag for case in test for tag in case.tags if tag.startswith("topic:")}
    assert dev_topics.isdisjoint(test_topics)


def test_conversation_cases_are_multisession_and_multirole() -> None:
    cases = [*load_cases(DEV_PATH), *load_cases(TEST_PATH)]

    for case in cases:
        assert case.source == "sanitized_daily_conversation"
        assert 2 <= len(case.sessions) <= 4
        roles = {
            message.role
            for session in case.sessions
            for message in session.messages
        }
        assert {"user", "assistant"} <= roles
        assert case.recall_probes


def test_conversation_datasets_exclude_sensitive_source_material() -> None:
    public_text = DEV_PATH.read_text(encoding="utf-8") + TEST_PATH.read_text(
        encoding="utf-8"
    )
    forbidden_patterns = {
        "platform session": r"(?:telegram|feishu):",
        "http link": r"https?://",
        "long hexadecimal secret": r"(?i)\b[0-9a-f]{24,}\b",
        "long numeric account id": r"\b\d{12,}\b",
        "credential label": r"(?i)api[_ -]?key|access[_ -]?token|secret[_ -]?key",
    }
    for label, pattern in forbidden_patterns.items():
        assert re.search(pattern, public_text) is None, label


def test_temporary_cases_forbid_long_term_writes() -> None:
    cases = [*load_cases(DEV_PATH), *load_cases(TEST_PATH)]
    temporary = [case for case in cases if "temporary_state" in case.tags]

    assert temporary
    for case in temporary:
        count = case.expected_write.allowed_new_count
        assert count is not None
        assert count.max == 0
        assert case.expected_write.forbidden
