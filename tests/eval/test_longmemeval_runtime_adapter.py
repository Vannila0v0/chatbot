from types import SimpleNamespace

import pytest

from eval.longmemeval.runtime import BenchmarkConsolidationAdapter


@pytest.mark.asyncio
async def test_consolidation_adapter_uses_current_markdown_maintenance() -> None:
    calls = []

    class Maintenance:
        async def consolidate(self, request):
            calls.append(request)
            return SimpleNamespace(trace={"mode": "committed"})

    session = SimpleNamespace(key="case:1")
    adapter = BenchmarkConsolidationAdapter(Maintenance())

    result = await adapter.consolidate(session, archive_all=True)

    assert result.trace["mode"] == "committed"
    assert calls[0].session is session
    assert calls[0].archive_all is True
    assert calls[0].force is True
