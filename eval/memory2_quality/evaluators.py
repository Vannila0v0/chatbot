from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from .models import MemoryEvalCase, WriteRunResult

_PUNCTUATION_RE = re.compile(r"[\s，。！？；：、,.!?;:'\"“”‘’（）()【】\[\]{}<>《》]+")


def normalize_fact(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    return _PUNCTUATION_RE.sub("", normalized)


def fact_is_entailed(expected: str, actual: str) -> bool:
    expected_norm = normalize_fact(expected)
    actual_norm = normalize_fact(actual)
    return bool(expected_norm) and expected_norm in actual_norm


def evaluate_recall_ranking(
    *,
    required_ids: set[str],
    forbidden_ids: set[str],
    ranked_ids: list[str],
) -> dict[str, float | bool | list[str]]:
    ranked_set = set(ranked_ids)
    if required_ids:
        recalled = required_ids & ranked_set
        recall_at_k = len(recalled) / len(required_ids)
        ranks = [ranked_ids.index(item_id) + 1 for item_id in required_ids if item_id in ranked_set]
        mrr = (1.0 / min(ranks)) if ranks else 0.0
        missing = sorted(required_ids - ranked_set)
    else:
        recall_at_k = 1.0
        mrr = 1.0
        missing = []
    forbidden_hits = [item_id for item_id in ranked_ids if item_id in forbidden_ids]
    forbidden_rate = (
        len(set(forbidden_hits)) / len(forbidden_ids) if forbidden_ids else 0.0
    )
    return {
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "forbidden_recall_rate": forbidden_rate,
        "missing_required_ids": missing,
        "forbidden_hits": forbidden_hits,
        "passed": recall_at_k == 1.0 and not forbidden_hits,
    }


def evaluate_write_result(
    case: MemoryEvalCase,
    result: WriteRunResult,
    local_id_to_item_id: dict[str, str],
) -> dict[str, Any]:
    active = result.state_diff.active_after
    required_labels = {
        item.label for item in case.expected_write.required if item.label
    }
    missing_labels = sorted(required_labels - set(result.label_to_item_id))
    forbidden_hits: list[str] = []
    for forbidden in case.expected_write.forbidden:
        for fact in forbidden.facts:
            if any(fact_is_entailed(fact, item.summary) for item in active):
                forbidden_hits.append(fact)
    action_failures: list[str] = []
    superseded_ids = {item.id for item in result.state_diff.superseded}
    for action in case.expected_write.expected_actions:
        if action.target_local_id and action.action == "supersede":
            target_id = local_id_to_item_id.get(action.target_local_id)
            if not target_id or target_id not in superseded_ids:
                action_failures.append(f"supersede:{action.target_local_id}")
    count = len(result.state_diff.created)
    count_ok = True
    allowed_count = case.expected_write.allowed_new_count
    if allowed_count is not None:
        count_ok = count >= allowed_count.min and (
            allowed_count.max is None or count <= allowed_count.max
        )
    passed = not result.error and not missing_labels and not forbidden_hits and not action_failures and count_ok
    components = [not missing_labels, not forbidden_hits, not action_failures, count_ok]
    return {
        "passed": passed,
        "score": sum(components) / len(components),
        "missing_required_labels": missing_labels,
        "forbidden_fact_hits": forbidden_hits,
        "action_failures": action_failures,
        "new_count_ok": count_ok,
    }


def aggregate_scores(case_results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(case_results)

    def aggregate(group: list[dict[str, Any]]) -> dict[str, int | float]:
        count = len(group)
        if count == 0:
            return {"n": 0, "passed": 0, "pass_rate": 0.0, "average_score": 0.0, "errors": 0}
        passed = sum(bool(item.get("passed")) for item in group)
        return {
            "n": count,
            "passed": passed,
            "pass_rate": passed / count,
            "average_score": sum(float(item.get("score") or 0.0) for item in group) / count,
            "errors": sum(bool(item.get("error")) for item in group),
        }

    categories: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        categories.setdefault(str(item.get("category") or "unknown"), []).append(item)
    return {
        "overall": aggregate(items),
        "by_category": {
            name: aggregate(group) for name, group in sorted(categories.items())
        },
    }
