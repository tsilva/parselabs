from pathlib import Path

import pytest

from parselabs.exceptions import PipelineError
from parselabs.pipeline import _discover_pdf_files


def test_discover_pdf_files_matches_pattern_case_insensitively(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    matching = input_dir / "Report.PDF"
    matching.write_text("pdf", encoding="utf-8")
    (input_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    assert _discover_pdf_files(input_dir, "*.pdf") == [matching]


def test_discover_pdf_files_surfaces_permission_errors(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    original_iterdir = Path.iterdir

    def fake_iterdir(self):
        if self == input_dir:
            raise PermissionError(1, "Operation not permitted", str(self))
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)

    with pytest.raises(PipelineError, match="Cannot access input directory"):
        _discover_pdf_files(input_dir, "*.pdf")
