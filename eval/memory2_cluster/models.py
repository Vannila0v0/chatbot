from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


MemoryType = Literal["event", "profile", "preference", "procedure"]
ClusterRole = Literal["core", "supporting", "weak", "irrelevant", "forbidden"]
ClusterRelation = Literal[
    "stable_fact",
    "state_evolution",
    "preference_reinforcement",
    "procedure",
    "related_events",
    "noise",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClusterMemory(StrictModel):
    local_id: str = Field(min_length=1)
    cluster_id: str = Field(min_length=1)
    memory_type: MemoryType
    summary: str = Field(min_length=1)
    happened_at: datetime
    reinforcement: int = Field(default=1, ge=0)
    last_used_days_ago: float = Field(default=0.0, ge=0.0)
    emotional_weight: int = Field(default=0, ge=0, le=10)
    source_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ClusterDefinition(StrictModel):
    cluster_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    relation: ClusterRelation
    memory_ids: list[str] = Field(min_length=1)
    review_notes: str = ""


class EventTimeline(StrictModel):
    timeline_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    description: str = ""
    source: str = "sanitized_daily_conversation"
    window_start: datetime | None = None
    window_end: datetime | None = None
    memories: list[ClusterMemory] = Field(min_length=1)
    clusters: list[ClusterDefinition] = Field(default_factory=list)


class ClusterProbe(StrictModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    timeline_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    query: str = Field(min_length=1)
    query_time: datetime
    top_k: int = Field(default=8, gt=0)
    cluster_oracle: dict[str, ClusterRole] = Field(min_length=1)
    preferred_pairs: list[tuple[str, str]] = Field(default_factory=list)
    memory_oracle: dict[str, ClusterRole] = Field(default_factory=dict)
    preferred_memory_pairs: list[tuple[str, str]] = Field(default_factory=list)
    split: Literal["natural", "challenge"] = "natural"
    dataset_split: Literal["dev", "validation", "test"] = "dev"
    review_status: Literal["candidate", "approved"] = "approved"
    rationale: str = ""
    tags: list[str] = Field(default_factory=list)
