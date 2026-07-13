from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.memory2_quality.langsmith_sync import LangSmithSink
from eval.memory2_quality.runtime import safe_case_workspace

from .ablation import run_paired_ablation, summarize_ablation
from .dataset import load_cluster_dataset
from .langsmith import run_ablation_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Memory2 热度排序配对 A/B 评测")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--timelines", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--treatment-alpha", type=float, default=0.2)
    parser.add_argument("--half-life-days", type=float, default=14.0)
    parser.add_argument("--langsmith", action="store_true")
    parser.add_argument("--experiment-prefix", default="memory2-ranking-ablation")
    return parser


async def _run_one(
    args: argparse.Namespace,
    timeline: Any,
    probe: Any,
    workspace: Path,
    *,
    tracing: bool,
) -> dict[str, Any]:
    from eval.longmemeval.runtime import close_runtime, create_runtime

    runtime = await create_runtime(args.config, workspace)
    try:
        return await run_paired_ablation(
            runtime,
            timeline,
            probe,
            tracing=tracing,
            treatment_alpha=args.treatment_alpha,
            half_life_days=args.half_life_days,
        )
    except Exception as exc:
        return {
            "case_id": probe.case_id,
            "timeline_id": probe.timeline_id,
            "split": probe.split,
            "tags": probe.tags,
            "passed": False,
            "score": 0.0,
            "error": str(exc),
            "baseline": {"metrics": {}},
            "treatment": {"metrics": {}},
            "comparison": {"treatment_improved": False, "regression": False},
        }
    finally:
        await close_runtime(runtime)


def _write_report(root: Path, manifest: dict[str, Any], results: list[dict[str, Any]]) -> None:
    summary = summarize_ablation([result for result in results if not result.get("error")])
    payload = {"manifest": manifest, "summary": summary, "cases": results}
    (root / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    lines = ["# Memory2 热度排序配对 A/B 报告", ""]
    for split, values in summary.items():
        lines.extend(
            [
                f"## {split}",
                "",
                f"- Case 数：{values['n']}",
                f"- Treatment 改善：{values['treatment_wins']}",
                f"- Treatment 退化：{values['regressions']}",
                f"- 混合变化：{values['mixed']}",
                "",
                "| 指标 | Baseline | Treatment |",
                "|---|---:|---:|",
            ]
        )
        for key, baseline_value in values["baseline"].items():
            lines.append(
                f"| {key} | {baseline_value:.3f} | {values['treatment'][key]:.3f} |"
            )
        lines.append("")
        for cohort, cohort_values in values.get("cohorts", {}).items():
            lines.extend(
                [
                    f"### {cohort}",
                    "",
                    f"- Case 数：{cohort_values['n']}",
                    f"- 改善 / 退化 / 混合：{cohort_values['treatment_wins']} / "
                    f"{cohort_values['regressions']} / {cohort_values['mixed']}",
                    "",
                    "| 指标 | Baseline | Treatment |",
                    "|---|---:|---:|",
                ]
            )
            for key, baseline_value in cohort_values["baseline"].items():
                lines.append(
                    f"| {key} | {baseline_value:.3f} | "
                    f"{cohort_values['treatment'][key]:.3f} |"
                )
            lines.append("")
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


async def run_evaluation(args: argparse.Namespace) -> Path:
    timelines, probes = load_cluster_dataset(args.timelines, args.dataset)
    if args.case_id:
        probes = [probe for probe in probes if probe.case_id == args.case_id]
    if args.limit:
        probes = probes[: args.limit]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = args.output or Path("eval/memory2_cluster/results") / f"ablation-{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, args.workers))
    sink = (
        LangSmithSink.from_environment(f"{args.experiment_prefix}-{timestamp}")
        if args.langsmith
        else LangSmithSink.disabled()
    )
    dataset_name = f"memory2-ablation-{args.dataset.stem}"
    payloads = [
        {
            "case_id": probe.case_id,
            "category": probe.split,
            "source": "sanitized_daily_conversation",
            "query": probe.query,
            "query_time": probe.query_time.isoformat(),
            "cluster_oracle": probe.cluster_oracle,
        }
        for probe in probes
    ]
    await sink.sync_dataset(dataset_name, payloads)

    async def process(probe: Any, *, tracing: bool) -> dict[str, Any]:
        async with semaphore:
            workspace = safe_case_workspace(root, probe.case_id)
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
        await run_ablation_experiment(
            target=target,
            data=examples,
            experiment_prefix=f"{args.experiment_prefix}-{timestamp}",
            max_concurrency=args.workers,
        )
        results.sort(key=lambda item: str(item["case_id"]))
    else:
        results = await asyncio.gather(*(process(probe, tracing=False) for probe in probes))

    manifest = {
        "dataset": str(args.dataset),
        "timelines": str(args.timelines),
        "case_count": len(probes),
        "baseline_hotness_alpha": 0.0,
        "treatment_hotness_alpha": args.treatment_alpha,
        "half_life_days": args.half_life_days,
        "created_at": datetime.now().isoformat(),
        "langsmith_enabled": sink.enabled,
        "langsmith_errors": sink.errors,
    }
    _write_report(root, manifest, results)
    return root


def main() -> None:
    output = asyncio.run(run_evaluation(build_parser().parse_args()))
    print(f"Memory2 ranking ablation results: {output}")


if __name__ == "__main__":
    main()
