from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _trace(**kwargs: Any):
    from langsmith import trace

    return trace(**kwargs)


async def _aevaluate(*args: Any, **kwargs: Any):
    from langsmith import aevaluate

    return await aevaluate(*args, **kwargs)


class _Stage:
    def __init__(self, run: Any | None = None) -> None:
        self.run = run
        self.outputs: dict[str, Any] | None = None

    def set_outputs(self, outputs: dict[str, Any]) -> None:
        self.outputs = outputs


@contextmanager
def traced_stage(enabled: bool, name: str, inputs: dict[str, Any] | None = None):
    """Create a child Run when executing inside a LangSmith experiment."""
    if not enabled:
        yield _Stage()
        return
    with _trace(name=name, run_type="chain", inputs=inputs or {}) as run:
        stage = _Stage(run)
        yield stage
        if stage.outputs is not None:
            run.end(outputs=stage.outputs)


def feedback_scores_from_result(result: dict[str, Any]) -> dict[str, float]:
    write = result.get("write_metrics") or {}
    recalls = [item.get("metrics") or {} for item in result.get("recall") or []]

    def average(key: str, default: float) -> float:
        values = [float(item[key]) for item in recalls if key in item]
        return sum(values) / len(values) if values else default

    scores = {
        "memory2_quality_score": float(result.get("score") or 0.0),
        "case_pass": float(bool(result.get("passed"))),
    }
    optional = {
        "write_required_fact_recall": write.get("required_fact_recall"),
        "write_action_accuracy": write.get("action_accuracy"),
        "write_forbidden_fact_rate": write.get("forbidden_fact_rate"),
    }
    scores.update({key: float(value) for key, value in optional.items() if value is not None})
    scores.update(
        {
            "recall_at_k": average("recall_at_k", 1.0),
            "mrr": average("mrr", 1.0),
            "forbidden_recall_rate": average("forbidden_recall_rate", 0.0),
        }
    )
    return scores


def _metric_evaluator(run: Any, example: Any):
    _ = example
    outputs = getattr(run, "outputs", None) or {}
    return [{"key": key, "score": score} for key, score in feedback_scores_from_result(outputs).items()]


async def run_experiment(
    *,
    target: Any,
    data: Any,
    experiment_prefix: str,
    max_concurrency: int,
    evaluators: list[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    return await _aevaluate(
        target,
        data=data,
        evaluators=evaluators or [_metric_evaluator],
        experiment_prefix=experiment_prefix,
        max_concurrency=max(1, max_concurrency),
        blocking=True,
        metadata=metadata or {"benchmark": "memory2-quality"},
    )


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
                example_id = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"langsmith:{dataset_id}:{case.get('case_id')}",
                )
                fields = {
                    "inputs": {
                        "case_id": case.get("case_id"),
                        "timeline_id": case.get("timeline_id"),
                        "dataset_split": case.get("dataset_split"),
                        "sessions": case.get("sessions", []),
                        "initial_memories": case.get("initial_memories", []),
                        "recall_probes": case.get("recall_probes", []),
                        "query": case.get("query"),
                        "query_time": case.get("query_time"),
                        "top_k": case.get("top_k"),
                        "tags": case.get("tags", []),
                        "preferred_pairs": case.get("preferred_pairs", []),
                        "preferred_memory_pairs": case.get(
                            "preferred_memory_pairs", []
                        ),
                    },
                    "outputs": {
                        "expected_write": case.get("expected_write", {}),
                        "cluster_oracle": case.get("cluster_oracle", {}),
                        "memory_oracle": case.get("memory_oracle", {}),
                    },
                    "metadata": {
                        "category": case.get("category"),
                        "source": case.get("source"),
                        "review_status": case.get("review_status"),
                        "timelines_sha256": case.get("timelines_sha256"),
                        "dataset_sha256": case.get("dataset_sha256"),
                    },
                }
                try:
                    await asyncio.to_thread(
                        self.client.create_example,
                        dataset_id=dataset_id,
                        example_id=example_id,
                        **fields,
                    )
                except Exception:
                    await asyncio.to_thread(
                        self.client.update_example,
                        example_id,
                        dataset_id=dataset_id,
                        **fields,
                    )
        except Exception as exc:
            self.errors.append(str(exc))

    async def selected_examples(
        self, dataset_name: str, case_ids: set[str]
    ) -> list[Any]:
        if not self.enabled or self.client is None:
            return []
        examples = await asyncio.to_thread(
            lambda: list(self.client.list_examples(dataset_name=dataset_name))
        )
        return [
            example
            for example in examples
            if str((example.inputs or {}).get("case_id")) in case_ids
        ]

    async def finalize(self, summary: dict[str, Any]) -> None:
        _ = summary
