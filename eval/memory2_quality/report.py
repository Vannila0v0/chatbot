from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluators import aggregate_scores


def write_report(
    output_dir: Path | str,
    manifest: dict[str, Any],
    case_results: list[dict[str, Any]],
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary = aggregate_scores(case_results)
    payload = {"manifest": manifest, "summary": summary, "cases": case_results}
    json_path = root / "results.json"
    markdown_path = root / "report.md"
    manifest_path = root / "manifest.json"
    _write_json_atomic(json_path, payload)
    _write_json_atomic(manifest_path, manifest)
    _write_text_atomic(markdown_path, _render_markdown(summary, case_results))
    return {"json": json_path, "markdown": markdown_path, "manifest": manifest_path}


def _render_markdown(summary: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    overall = summary["overall"]
    lines = [
        "# Memory2 质量评测报告",
        "",
        "## 总体结果",
        "",
        "| Case 数 | 通过 | 通过率 | 平均分 | 错误 |",
        "|---:|---:|---:|---:|---:|",
        f"| {overall['n']} | {overall['passed']} | {overall['pass_rate']:.1%} | "
        f"{overall['average_score']:.3f} | {overall['errors']} |",
        "",
        "## 按类别",
        "",
        "| 类别 | Case 数 | 通过率 | 平均分 |",
        "|---|---:|---:|---:|",
    ]
    for category, values in summary["by_category"].items():
        lines.append(
            f"| {category} | {values['n']} | {values['pass_rate']:.1%} | "
            f"{values['average_score']:.3f} |"
        )
    failed = [item for item in cases if not item.get("passed")]
    lines.extend(["", "## 失败 Case", "", "| Case | 类别 | 错误 |", "|---|---|---|"])
    for item in failed:
        error = str(item.get("error") or "指标未通过").replace("|", "\\|")
        lines.append(f"| {item.get('case_id')} | {item.get('category')} | {error} |")
    if not failed:
        lines.append("| - | - | 无 |")
    return "\n".join(lines) + "\n"


def _write_json_atomic(path: Path, payload: object) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
