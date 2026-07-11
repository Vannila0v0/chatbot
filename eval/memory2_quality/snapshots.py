from __future__ import annotations

import json
from typing import Any

from memory2.store import MemoryStore2

from .models import MemorySnapshotItem, MemoryStateDiff


def take_snapshot(store: MemoryStore2) -> list[MemorySnapshotItem]:
    rows = store._db.execute(
        "SELECT id, memory_type, summary, source_ref, happened_at, status, "
        "reinforcement, emotional_weight, extra_json FROM memory_items ORDER BY id"
    ).fetchall()
    return [
        MemorySnapshotItem(
            id=str(row[0]),
            memory_type=row[1],
            summary=str(row[2]),
            source_ref=str(row[3] or ""),
            happened_at=str(row[4]) if row[4] else None,
            status=str(row[5] or "active"),
            reinforcement=int(row[6] or 0),
            emotional_weight=int(row[7] or 0),
            extra_json=_parse_extra(row[8]),
        )
        for row in rows
    ]


def diff_snapshots(
    before: list[MemorySnapshotItem], after: list[MemorySnapshotItem]
) -> MemoryStateDiff:
    before_by_id = {item.id: item for item in before}
    after_by_id = {item.id: item for item in after}
    result = MemoryStateDiff(
        created=[item for item in after if item.id not in before_by_id],
        active_after=[item for item in after if item.status == "active"],
    )
    for item_id, old in before_by_id.items():
        new = after_by_id.get(item_id)
        if new is None:
            continue
        changes = _field_changes(old, new)
        if changes:
            result.field_changes[item_id] = changes
        if old.status == "active" and new.status == "superseded":
            result.superseded.append(new)
        elif new.summary != old.summary or new.extra_json != old.extra_json:
            result.merged.append(new)
        elif new.reinforcement > old.reinforcement:
            result.reinforced.append(new)
        else:
            result.unchanged.append(new)
    return result


def _parse_extra(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _field_changes(
    old: MemorySnapshotItem, new: MemorySnapshotItem
) -> dict[str, tuple[Any, Any]]:
    changes: dict[str, tuple[Any, Any]] = {}
    for field in (
        "memory_type",
        "summary",
        "source_ref",
        "happened_at",
        "status",
        "reinforcement",
        "emotional_weight",
        "extra_json",
    ):
        old_value = getattr(old, field)
        new_value = getattr(new, field)
        if old_value != new_value:
            changes[field] = (old_value, new_value)
    return changes
