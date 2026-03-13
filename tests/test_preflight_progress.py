from pathlib import Path

from parselabs import pipeline as main


def test_prepare_pdf_run_shows_hash_progress_for_multiple_pdfs(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"a")
    pdf_b.write_bytes(b"b")
    calls: list[str] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs["desc"])
        return iterable

    monkeypatch.setattr(main, "tqdm", fake_tqdm)
    monkeypatch.setattr(main, "_compute_file_hash", lambda path: path.stem)

    preflight = main._prepare_pdf_run([pdf_a, pdf_b], output_dir)

    assert [task.file_hash for task in preflight.pdfs_to_process] == ["a", "b"]
    assert calls == ["Hashing PDFs"]


def test_prepare_pdf_run_skips_hash_progress_for_single_pdf(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pdf_a = tmp_path / "a.pdf"
    pdf_a.write_bytes(b"a")
    calls: list[str] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs["desc"])
        return iterable

    monkeypatch.setattr(main, "tqdm", fake_tqdm)
    monkeypatch.setattr(main, "_compute_file_hash", lambda path: "a")

    preflight = main._prepare_pdf_run([pdf_a], output_dir)

    assert [task.file_hash for task in preflight.pdfs_to_process] == ["a"]
    assert calls == []
