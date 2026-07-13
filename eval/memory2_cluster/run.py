from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.memory2_quality.report import write_report
from eval.memory2_quality.runtime import safe_case_workspace
from eval.memory2_quality.langsmith_sync import LangSmithSink

from .dataset import load_cluster_dataset
from .langsmith import run_cluster_experiment
from .runner import run_cluster_probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Memory2 事件簇召回评测")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--timelines", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--langsmith", action="store_true")
    parser.add_argument("--experiment-prefix", default="memory2-cluster")
    return parser


async def _run_one(
    args: argparse.Namespace,
    timeline: Any,
    probe: Any,
    workspace: Path,
    *,
    tracing: bool = False,
) -> dict[str, Any]:
    from eval.longmemeval.runtime import close_runtime, create_runtime

    runtime = await create_runtime(args.config, workspace)
    try:
        return await run_cluster_probe(runtime, timeline, probe, tracing=tracing)
    except Exception as exc:
        return {
            "case_id": probe.case_id,
            "timeline_id": timeline.timeline_id,
            "category": "event_cluster_retrieval",
            "passed": False,
            "score": 0.0,
            "error": str(exc),
        }
    finally:
        await close_runtime(runtime)


async def run_evaluation(args: argparse.Namespace) -> Path:
    timelines, probes = load_cluster_dataset(args.timelines, args.dataset)
    if args.case_id:
        probes = [probe for probe in probes if probe.case_id == args.case_id]
    if args.limit:
        probes = probes[: args.limit]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = args.output or Path("eval/memory2_cluster/results") / timestamp
    run_root.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, args.workers))
    sink = (
        LangSmithSink.from_environment(f"{args.experiment_prefix}-{timestamp}")
        if args.langsmith
        else LangSmithSink.disabled()
    )
    dataset_name = f"memory2-cluster-{args.dataset.stem}"
    payloads = [
        {
            "case_id": probe.case_id,
            "category": "event_cluster_retrieval",
            "source": "sanitized_daily_conversation",
            "query": probe.query,
            "query_time": probe.query_time.isoformat(),
            "cluster_oracle": probe.cluster_oracle,
        }
        for probe in probes
    ]
    await sink.sync_dataset(dataset_name, payloads)

    async def process(probe: Any, *, tracing: bool = False) -> dict[str, Any]:
        async with semaphore:
            workspace = safe_case_workspace(run_root, probe.case_id)
            result = await _run_one(
                args, timelines[probe.timeline_id], probe, workspace, tracing=tracing
            )
            (workspace / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            return result

    if sink.enabled:
        probes_by_id = {probe.case_id: probe for probe in probes}
        results: list[dict[str, Any]] = []

        async def target(inputs: dict[str, Any]) -> dict[str, Any]:
            result = await process(probes_by_id[str(inputs["case_id"])], tracing=True)
            results.append(result)
            return result

        examples = await sink.selected_examples(dataset_name, set(probes_by_id))
        await run_cluster_experiment(
            target=target,
            data=examples,
            experiment_prefix=f"{args.experiment_prefix}-{timestamp}",
            max_concurrency=args.workers,
        )
        results.sort(key=lambda item: str(item["case_id"]))
    else:
        results = await asyncio.gather(*(process(probe) for probe in probes))
    write_report(
        run_root,
        {
            "dataset": str(args.dataset),
            "timelines": str(args.timelines),
            "case_count": len(probes),
            "created_at": datetime.now().isoformat(),
            "langsmith_enabled": sink.enabled,
            "langsmith_errors": sink.errors,
        },
        results,
    )
    return run_root


def main() -> None:
    output = asyncio.run(run_evaluation(build_parser().parse_args()))
    print(f"Memory2 cluster evaluation results: {output}")


if __name__ == "__main__":
    main()
