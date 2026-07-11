from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class LangSmithSink:
    client: Any | None = None
    project_name: str = "memory2-quality"
    enabled: bool = True
    errors: list[str] = field(default_factory=list)

    @classmethod
    def disabled(cls) -> "LangSmithSink":
        return cls(client=None, enabled=False)

    @classmethod
    def from_environment(cls, project_name: str) -> "LangSmithSink":
        if not os.getenv("LANGSMITH_API_KEY"):
            return cls.disabled()
        try:
            from langsmith import Client
        except ImportError:
            sink = cls.disabled()
            sink.errors.append("langsmith SDK 未安装")
            return sink
        return cls(client=Client(), project_name=project_name)

    async def record_case(self, case: dict[str, Any], result: dict[str, Any]) -> None:
        if not self.enabled or self.client is None:
            return
        run_id = str(result.get("run_id") or uuid.uuid4())
        result["run_id"] = run_id
        try:
            await asyncio.to_thread(
                self.client.create_run,
                name="memory2_eval_case",
                inputs={"case": case},
                run_type="chain",
                id=run_id,
                project_name=self.project_name,
                tags=[str(case.get("category") or "unknown")],
                extra={"metadata": {"case_id": case.get("case_id")}},
            )
            await asyncio.to_thread(
                self.client.update_run,
                run_id,
                outputs={"result": result},
                error=str(result.get("error")) if result.get("error") else None,
                end_time=datetime.now(timezone.utc),
            )
            await asyncio.to_thread(
                self.client.create_feedback,
                run_id,
                key="memory2_quality_score",
                score=float(result.get("score") or 0.0),
                comment=f"case_id={case.get('case_id')}",
            )
        except Exception as exc:
            self.errors.append(str(exc))

    async def sync_dataset(
        self, dataset_name: str, cases: list[dict[str, Any]]
    ) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            try:
                dataset = await asyncio.to_thread(
                    self.client.read_dataset, dataset_name=dataset_name
                )
            except Exception:
                dataset = await asyncio.to_thread(
                    self.client.create_dataset,
                    dataset_name=dataset_name,
                    description="Memory2 write and retrieval quality cases",
                )
            dataset_id = str(dataset.id)
            for case in cases:
                await asyncio.to_thread(
                    self.client.create_example,
                    inputs={
                        "case_id": case.get("case_id"),
                        "sessions": case.get("sessions", []),
                        "initial_memories": case.get("initial_memories", []),
                        "recall_probes": case.get("recall_probes", []),
                    },
                    outputs={"expected_write": case.get("expected_write", {})},
                    metadata={
                        "category": case.get("category"),
                        "source": case.get("source"),
                    },
                    dataset_id=dataset_id,
                )
        except Exception as exc:
            self.errors.append(str(exc))

    async def finalize(self, summary: dict[str, Any]) -> None:
        _ = summary
