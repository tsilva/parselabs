from pathlib import Path

import main


def test_deduplicate_pdf_files_shows_hashing_progress_for_multiple_pdfs(tmp_path, monkeypatch):
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"a")
    pdf_b.write_bytes(b"b")
    calls: list[str] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs["desc"])
        return iterable

    monkeypatch.setattr(main, "tqdm", fake_tqdm)

    unique_files, duplicates = main._deduplicate_pdf_files([pdf_a, pdf_b])

    assert unique_files == [pdf_a, pdf_b]
    assert duplicates == []
    assert calls == ["Hashing PDFs"]


def test_filter_pdfs_to_process_shows_cache_progress_for_multiple_pdfs(tmp_path, monkeypatch):
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

    pdfs_to_process, skipped_count = main._filter_pdfs_to_process([pdf_a, pdf_b], output_dir)

    assert pdfs_to_process == [pdf_a, pdf_b]
    assert skipped_count == 0
    assert calls == ["Checking cached outputs"]
