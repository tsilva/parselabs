from types import SimpleNamespace

from parselabs import pipeline as main
from parselabs.store import build_document_ref


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
    monkeypatch.setattr(
        main,
        "plan_pdf_run",
        lambda pdf_files, output_path: SimpleNamespace(
            documents_to_process=[
                build_document_ref(pdf_a, output_path, "a"),
                build_document_ref(pdf_b, output_path, "b"),
            ],
            duplicates=[],
        ),
    )

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
    monkeypatch.setattr(
        main,
        "plan_pdf_run",
        lambda pdf_files, output_path: SimpleNamespace(
            documents_to_process=[build_document_ref(pdf_a, output_path, "a")],
            duplicates=[],
        ),
    )

    preflight = main._prepare_pdf_run([pdf_a], output_dir)

    assert [task.file_hash for task in preflight.pdfs_to_process] == ["a"]
    assert calls == []
