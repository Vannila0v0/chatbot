from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from .dataset import load_cases
from .evaluators import evaluate_write_result
from .langsmith_sync import LangSmithSink, run_experiment, traced_stage
from .recall_runner import run_recall_probes
from .report import write_report
from .runtime import safe_case_workspace
from .seed import seed_memories
from .write_runner import run_write_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评测 Memory2 写入与召回质量")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--mode", choices=("write", "recall", "all"), default="all")
    parser.add_argument("--case-id")
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--langsmith", action="store_true")
    parser.add_argument("--experiment-prefix", default="memory2-quality")
    return parser


async def run_evaluation(args: argparse.Namespace) -> Path:
    cases = load_cases(args.dataset)
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
    if args.category:
        cases = [case for case in cases if case.category == args.category]
    if args.limit > 0:
        cases = cases[: args.limit]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = args.output or Path("eval/memory2_quality/results") / timestamp
    run_root.mkdir(parents=True, exist_ok=True)
    sink = (
        LangSmithSink.from_environment(f"{args.experiment_prefix}-{timestamp}")
        if args.langsmith
        else LangSmithSink.disabled()
    )
    dataset_name = f"memory2-quality-{args.dataset.stem}"
    await sink.sync_dataset(
        dataset_name,
        [case.model_dump(mode="json") for case in cases],
    )
    semaphore = asyncio.Semaphore(max(1, args.workers))

    async def process(case, *, tracing: bool = False):
        async with semaphore:
            case_dir = safe_case_workspace(run_root, case.case_id)
            cached = case_dir / "result.json"
            if args.resume and cached.exists():
                import json
                return json.loads(cached.read_text(encoding="utf-8"))
            result = await _run_case(args, case, case_dir, tracing=tracing)
            import json
            cached.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            if not tracing:
                await sink.record_case(case.model_dump(mode="json"), result)
            return result

    if sink.enabled:
        case_by_id = {case.case_id: case for case in cases}
        results = []

        async def target(inputs: dict[str, Any]) -> dict[str, Any]:
            case = case_by_id[str(inputs["case_id"])]
            result = await process(case, tracing=True)
            results.append(result)
            return result

        examples = await sink.selected_examples(dataset_name, set(case_by_id))
        await run_experiment(
            target=target,
            data=examples,
            experiment_prefix=f"{args.experiment_prefix}-{timestamp}",
            max_concurrency=args.workers,
        )
        results.sort(key=lambda item: str(item.get("case_id")))
    else:
        results = await asyncio.gather(*(process(case) for case in cases))
    manifest = {
        "dataset": str(args.dataset),
        "mode": args.mode,
        "case_count": len(cases),
        "created_at": datetime.now().isoformat(),
        "langsmith_enabled": sink.enabled,
        "langsmith_errors": sink.errors,
    }
    write_report(run_root, manifest, results)
    await sink.finalize(manifest)
    return run_root


async def _run_case(
    args: argparse.Namespace,
    case: Any,
    workspace: Path,
    *,
    tracing: bool = False,
) -> dict[str, Any]:
    from eval.longmemeval.runtime import close_runtime, create_runtime

    runtime = await create_runtime(args.config, workspace)
    try:
        engine = runtime.core.memory_runtime.engine
        store = engine._v2_store
        embedder = engine._embedder
        fixtures = list(case.initial_memories)
        if args.mode == "recall":
            fixtures.extend(case.recall_fixture_memories)
        with traced_stage(tracing, "seed_memories", {"count": len(fixtures)}) as stage:
            local_map = await seed_memories(store, embedder, fixtures)
            stage.set_outputs({"seeded_count": len(local_map)})
        write_result = None
        label_map: dict[str, str] = {}
        write_metrics: dict[str, Any] = {}
        if args.mode in {"write", "all"}:
            with traced_stage(tracing, "write_memory", {"case_id": case.case_id}) as stage:
                write_result = await run_write_case(runtime, case)
                stage.set_outputs(write_result.model_dump(mode="json"))
            label_map = write_result.label_to_item_id
            with traced_stage(tracing, "evaluate_state_diff") as stage:
                write_metrics = evaluate_write_result(case, write_result, local_map)
                stage.set_outputs(write_metrics)
        recall_results = []
        if args.mode in {"recall", "all"}:
            with traced_stage(
                tracing, "recall_memories", {"probe_count": len(case.recall_probes)}
            ) as stage:
                recall_results = await run_recall_probes(runtime, case, label_map, local_map)
                stage.set_outputs(
                    {"probes": [item.model_dump(mode="json") for item in recall_results]}
                )
        recall_passed = all(bool(item.metrics.get("passed")) and not item.error for item in recall_results)
        passed = bool(write_metrics.get("passed", True)) and recall_passed
        scores = [float(write_metrics.get("score", 1.0))]
        scores.extend(1.0 if item.metrics.get("passed") else 0.0 for item in recall_results)
        result = {
            "case_id": case.case_id,
            "category": case.category,
            "passed": passed,
            "score": sum(scores) / len(scores),
            "error": write_result.error if write_result else None,
            "write": write_result.model_dump(mode="json") if write_result else None,
            "write_metrics": write_metrics,
            "recall": [item.model_dump(mode="json") for item in recall_results],
        }
        with traced_stage(tracing, "evaluate_case") as stage:
            stage.set_outputs({
                "passed": result["passed"],
                "score": result["score"],
            })
        return result
    except Exception as exc:
        return {"case_id": case.case_id, "category": case.category, "passed": False, "score": 0.0, "error": str(exc)}
    finally:
        await close_runtime(runtime)


def main() -> None:
    args = build_parser().parse_args()
    output = asyncio.run(run_evaluation(args))
    print(f"Memory2 evaluation results: {output}")


if __name__ == "__main__":
    main()
