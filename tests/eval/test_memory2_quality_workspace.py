from pathlib import Path

import pytest

from eval.memory2_quality.runtime import safe_case_workspace


def test_safe_case_workspace_creates_case_directory(tmp_path: Path) -> None:
    workspace = safe_case_workspace(tmp_path, "case_001")
    assert workspace == (tmp_path / "cases" / "case_001").resolve()
    assert workspace.is_dir()


@pytest.mark.parametrize("case_id", ["../escape", "a/b", "a\\b", "", "."])
def test_safe_case_workspace_rejects_unsafe_case_id(
    tmp_path: Path, case_id: str
) -> None:
    with pytest.raises(ValueError, match="case_id"):
        safe_case_workspace(tmp_path, case_id)

