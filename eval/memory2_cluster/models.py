from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


MemoryType = Literal["event", "profile", "preference", "procedure"]
ClusterRole = Literal["core", "supporting", "weak", "irrelevant", "forbidden"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClusterMemory(StrictModel):
    local_id: str = Field(min_length=1)
    cluster_id: str = Field(min_length=1)
    memory_type: MemoryType
    summary: str = Field(min_length=1)
    happened_at: datetime
    reinforcement: int = Field(default=1, ge=0)
    emotional_weight: int = Field(default=0, ge=0, le=10)


class EventTimeline(StrictModel):
    timeline_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    description: str = ""
    source: str = "sanitized_daily_conversation"
    memories: list[ClusterMemory] = Field(min_length=1)


class ClusterProbe(StrictModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    timeline_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    query: str = Field(min_length=1)
    query_time: datetime
    top_k: int = Field(default=8, gt=0)
    cluster_oracle: dict[str, ClusterRole] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
