from pathlib import Path

from eval.memory2_quality.report import write_report
from eval.memory2_quality.run import build_parser


def test_write_report_creates_json_and_markdown(tmp_path: Path) -> None:
    results = [
        {"case_id": "c1", "category": "conflict", "passed": True, "score": 1.0, "error": None},
        {"case_id": "c2", "category": "noise", "passed": False, "score": 0.0, "error": "boom"},
    ]
    paths = write_report(tmp_path, {"mode": "all"}, results)
    assert paths["json"].is_file()
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "总体结果" in markdown
    assert "按类别" in markdown
    assert "c2" in markdown


def test_cli_parser_supports_modes_and_filters() -> None:
    args = build_parser().parse_args(
        [
            "--config", "config.toml",
            "--dataset", "cases.jsonl",
            "--mode", "recall",
            "--case-id", "c1",
            "--workers", "2",
        ]
    )
    assert args.mode == "recall"
    assert args.case_id == "c1"
    assert args.workers == 2
