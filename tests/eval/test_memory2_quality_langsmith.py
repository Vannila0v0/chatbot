import asyncio

from eval.memory2_quality.langsmith_sync import (
    LangSmithSink,
    feedback_scores_from_result,
    run_experiment,
    traced_stage,
)


def test_disabled_sink_is_noop() -> None:
    sink = LangSmithSink.disabled()
    asyncio.run(sink.record_case({"case_id": "c1"}, {"passed": True}))
    asyncio.run(sink.finalize({"pass_rate": 1.0}))
    assert sink.enabled is False
    assert sink.errors == []


def test_sink_records_client_errors_without_raising() -> None:
    class BrokenClient:
        def create_run(self, **kwargs):
            return None

        def update_run(self, *args, **kwargs):
            return None

        def create_feedback(self, *args, **kwargs):
            raise RuntimeError("offline")

    sink = LangSmithSink(client=BrokenClient(), project_name="test")
    asyncio.run(sink.record_case({"case_id": "c1"}, {"run_id": "r1", "score": 0.5}))
    assert sink.errors
    assert "offline" in sink.errors[0]


def test_sink_creates_dataset_example_and_case_trace() -> None:
    class Client:
        def __init__(self):
            self.runs = []
            self.examples = []

        def read_dataset(self, *, dataset_name):
            return type("Dataset", (), {"id": "dataset-1"})()

        def create_example(self, **kwargs):
            self.examples.append(kwargs)

        def create_run(self, **kwargs):
            self.runs.append(("create", kwargs))

        def update_run(self, *args, **kwargs):
            self.runs.append(("update", args, kwargs))

        def create_feedback(self, *args, **kwargs):
            self.runs.append(("feedback", args, kwargs))

    client = Client()
    sink = LangSmithSink(client=client, project_name="experiment")
    case = {
        "case_id": "c1",
        "timeline_id": "timeline-1",
        "dataset_split": "dev",
        "top_k": 5,
        "tags": ["temporal"],
        "preferred_pairs": [["new", "old"]],
        "preferred_memory_pairs": [["new-memory", "old-memory"]],
        "cluster_oracle": {"new": "core", "old": "forbidden"},
        "memory_oracle": {"new-memory": "core", "old-memory": "forbidden"},
        "review_status": "approved",
        "timelines_sha256": "timeline-hash",
        "dataset_sha256": "dataset-hash",
        "expected_write": {"required": []},
    }
    asyncio.run(sink.sync_dataset("memory2-quality-v1", [case]))
    result = {"passed": True, "score": 1.0}
    asyncio.run(sink.record_case(case, result))
    assert client.examples[0]["dataset_id"] == "dataset-1"
    assert client.examples[0]["inputs"]["timeline_id"] == "timeline-1"
    assert client.examples[0]["inputs"]["dataset_split"] == "dev"
    assert client.examples[0]["outputs"]["memory_oracle"]["new-memory"] == "core"
    assert client.examples[0]["metadata"]["review_status"] == "approved"
    assert result["run_id"]
    assert [entry[0] for entry in client.runs] == ["create", "update", "feedback"]


def test_dataset_sync_updates_existing_deterministic_example() -> None:
    class Client:
        def __init__(self):
            self.ids = set()
            self.updated = []

        def read_dataset(self, *, dataset_name):
            return type("Dataset", (), {"id": "dataset-1"})()

        def create_example(self, **kwargs):
            example_id = kwargs["example_id"]
            if example_id in self.ids:
                raise RuntimeError("already exists")
            self.ids.add(example_id)

        def update_example(self, example_id, **kwargs):
            self.updated.append(example_id)

    client = Client()
    sink = LangSmithSink(client=client, project_name="experiment")
    case = {"case_id": "stable-case", "expected_write": {"required": []}}
    asyncio.run(sink.sync_dataset("memory2-quality-v1", [case]))
    asyncio.run(sink.sync_dataset("memory2-quality-v1", [case]))
    assert len(client.ids) == 1
    assert client.updated == list(client.ids)


def test_feedback_scores_expose_individual_quality_metrics() -> None:
    result = {
        "passed": False,
        "score": 0.6,
        "write_metrics": {
            "required_fact_recall": 0.5,
            "action_accuracy": 1.0,
            "forbidden_fact_rate": 0.25,
        },
        "recall": [
            {"metrics": {"recall_at_k": 1.0, "mrr": 0.5, "forbidden_recall_rate": 0.0}},
            {"metrics": {"recall_at_k": 0.0, "mrr": 0.0, "forbidden_recall_rate": 0.5}},
        ],
    }

    assert feedback_scores_from_result(result) == {
        "memory2_quality_score": 0.6,
        "case_pass": 0.0,
        "write_required_fact_recall": 0.5,
        "write_action_accuracy": 1.0,
        "write_forbidden_fact_rate": 0.25,
        "recall_at_k": 0.5,
        "mrr": 0.25,
        "forbidden_recall_rate": 0.25,
    }


def test_traced_stage_records_child_outputs(monkeypatch) -> None:
    recorded = []

    class Run:
        def end(self, *, outputs):
            recorded.append(outputs)

    class Context:
        def __enter__(self):
            return Run()

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "eval.memory2_quality.langsmith_sync._trace",
        lambda **kwargs: Context(),
    )
    with traced_stage(True, "seed_memories", {"count": 2}) as stage:
        stage.set_outputs({"seeded": 2})
    assert recorded == [{"seeded": 2}]


def test_run_experiment_uses_standard_aevaluate_and_metric_evaluator(monkeypatch) -> None:
    captured = {}

    async def fake_aevaluate(target, **kwargs):
        captured.update(kwargs)
        output = await target({"case_id": "c1"})
        evaluator_result = kwargs["evaluators"][0](
            type("Run", (), {"outputs": output})(), type("Example", (), {})()
        )
        captured["output"] = output
        captured["feedback"] = evaluator_result
        return type("Results", (), {"wait": lambda self: None})()

    monkeypatch.setattr("eval.memory2_quality.langsmith_sync._aevaluate", fake_aevaluate)

    async def target(inputs):
        return {"case_id": inputs["case_id"], "passed": True, "score": 1.0, "recall": []}

    asyncio.run(
        run_experiment(
            target=target,
            data=[{"case_id": "c1"}],
            experiment_prefix="memory2-test",
            max_concurrency=2,
        )
    )
    assert captured["experiment_prefix"] == "memory2-test"
    assert captured["max_concurrency"] == 2
    assert captured["feedback"][0] == {"key": "memory2_quality_score", "score": 1.0}
