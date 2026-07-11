from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MemoryType = Literal["event", "profile", "preference", "procedure"]
MemoryAction = Literal["create", "reinforce", "merge", "supersede", "keep", "ignore"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MemoryFixture(StrictModel):
    local_id: str = Field(min_length=1)
    memory_type: MemoryType
    summary: str = Field(min_length=1)
    status: Literal["active", "superseded"] = "active"
    happened_at: datetime | None = None
    reinforcement: int = Field(default=1, ge=0)
    emotional_weight: int = Field(default=0, ge=0, le=10)
    extra: dict[str, Any] = Field(default_factory=dict)


class EvalMessage(StrictModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)
    timestamp: datetime | None = None


class EvalSession(StrictModel):
    session_id: str = Field(min_length=1)
    timestamp: datetime
    messages: list[EvalMessage] = Field(min_length=1)
    consolidate_after: bool = True


class ExpectedFact(StrictModel):
    label: str | None = None
    memory_type: MemoryType | None = None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    facts: list[str] = Field(default_factory=list)


class ExpectedAction(StrictModel):
    target_local_id: str | None = None
    target_label: str | None = None
    action: MemoryAction


class CountRange(StrictModel):
    min: int = Field(default=0, ge=0)
    max: int | None = Field(default=None, ge=0)


class ExpectedWrite(StrictModel):
    required: list[ExpectedFact] = Field(default_factory=list)
    forbidden: list[ExpectedFact] = Field(default_factory=list)
    expected_actions: list[ExpectedAction] = Field(default_factory=list)
    allowed_types: list[MemoryType] = Field(default_factory=list)
    forbidden_types: list[MemoryType] = Field(default_factory=list)
    allowed_new_count: CountRange | None = None


class RecallProbe(StrictModel):
    probe_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    intent: str = "answer"
    top_k: int = Field(default=5, gt=0)
    required_memory_labels: list[str] = Field(default_factory=list)
    required_local_ids: list[str] = Field(default_factory=list)
    required_facts: list[str] = Field(default_factory=list)
    forbidden_local_ids: list[str] = Field(default_factory=list)
    forbidden_facts: list[str] = Field(default_factory=list)
    required_types: list[MemoryType] = Field(default_factory=list)


class MemoryEvalCase(StrictModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    category: str = Field(min_length=1)
    description: str = ""
    source: str = "manual"
    reference_time: datetime
    initial_memories: list[MemoryFixture] = Field(default_factory=list)
    recall_fixture_memories: list[MemoryFixture] = Field(default_factory=list)
    sessions: list[EvalSession] = Field(default_factory=list)
    expected_write: ExpectedWrite = Field(default_factory=ExpectedWrite)
    recall_probes: list[RecallProbe] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class MemorySnapshotItem(StrictModel):
    id: str
    memory_type: MemoryType
    summary: str
    source_ref: str = ""
    happened_at: str | None = None
    status: str = "active"
    reinforcement: int = 1
    emotional_weight: int = 0
    extra_json: dict[str, Any] = Field(default_factory=dict)


class MemoryStateDiff(StrictModel):
    created: list[MemorySnapshotItem] = Field(default_factory=list)
    reinforced: list[MemorySnapshotItem] = Field(default_factory=list)
    merged: list[MemorySnapshotItem] = Field(default_factory=list)
    superseded: list[MemorySnapshotItem] = Field(default_factory=list)
    unchanged: list[MemorySnapshotItem] = Field(default_factory=list)
    active_after: list[MemorySnapshotItem] = Field(default_factory=list)
    field_changes: dict[str, dict[str, tuple[Any, Any]]] = Field(default_factory=dict)


class WriteRunResult(StrictModel):
    before: list[MemorySnapshotItem]
    after: list[MemorySnapshotItem]
    state_diff: MemoryStateDiff
    label_to_item_id: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class RecallHit(StrictModel):
    id: str
    memory_type: str
    summary: str
    score: float = 0.0


class RecallProbeResult(StrictModel):
    probe_id: str
    query: str
    hits: list[RecallHit] = Field(default_factory=list)
    ranked_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    trace: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
