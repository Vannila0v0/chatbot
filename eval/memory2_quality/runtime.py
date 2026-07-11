from __future__ import annotations

import re
from pathlib import Path

_SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def safe_case_workspace(run_root: Path | str, case_id: str) -> Path:
    if not _SAFE_CASE_ID.fullmatch(case_id) or case_id in {".", ".."}:
        raise ValueError(f"不安全的 case_id: {case_id!r}")
    root = Path(run_root).resolve()
    cases_root = (root / "cases").resolve()
    workspace = (cases_root / case_id).resolve()
    if workspace.parent != cases_root:
        raise ValueError(f"case_id 逃逸运行目录: {case_id!r}")
    real_workspace = (Path.home() / ".akashic" / "workspace").resolve()
    if workspace == real_workspace or real_workspace in workspace.parents:
        raise ValueError("评测 workspace 不能使用真实 Akashic workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace
