import asyncio

from eval.memory2_quality.langsmith_sync import LangSmithSink


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
    case = {"case_id": "c1", "expected_write": {"required": []}}
    asyncio.run(sink.sync_dataset("memory2-quality-v1", [case]))
    result = {"passed": True, "score": 1.0}
    asyncio.run(sink.record_case(case, result))
    assert client.examples[0]["dataset_id"] == "dataset-1"
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
