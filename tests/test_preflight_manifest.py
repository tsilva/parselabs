from pathlib import Path

import pandas as pd

from parselabs import pipeline as main
from parselabs.config import ExtractionConfig


def _write_valid_csv(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=main.REQUIRED_CSV_COLS).to_csv(csv_path, index=False)


def _build_config(tmp_path: Path) -> ExtractionConfig:
    output_path = tmp_path / "output"
    output_path.mkdir(exist_ok=True)
    return ExtractionConfig(
        input_path=tmp_path,
        output_path=output_path,
        openrouter_api_key="test-key",
        extract_model_id="test-model",
        max_workers=1,
    )


def test_prepare_pdf_run_hashes_each_pdf_and_builds_hashed_targets(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"a")
    pdf_b.write_bytes(b"b")
    calls: list[Path] = []

    def fake_hash(path: Path) -> str:
        calls.append(path)
        return path.stem

    monkeypatch.setattr(main, "_compute_file_hash", fake_hash)

    preflight = main._prepare_pdf_run([pdf_a, pdf_b], output_dir)

    assert calls == [pdf_a.resolve(), pdf_b.resolve()]
    assert [task.file_hash for task in preflight.pdfs_to_process] == ["a", "b"]
    assert [task.csv_path for task in preflight.pdfs_to_process] == [
        output_dir / "a_a" / "a.csv",
        output_dir / "b_b" / "b.csv",
    ]


def test_prepare_pdf_run_deduplicates_exact_content(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"same")
    pdf_b.write_bytes(b"same")

    monkeypatch.setattr(main, "_compute_file_hash", lambda path: "same-hash")

    preflight = main._prepare_pdf_run([pdf_a, pdf_b], output_dir)

    assert [task.file_hash for task in preflight.pdfs_to_process] == ["same-hash"]
    assert preflight.duplicates == [(pdf_b, pdf_a)]


def test_process_single_pdf_uses_precomputed_hash(tmp_path, monkeypatch):
    config = _build_config(tmp_path)
    pdf_path = tmp_path / "worker.pdf"
    pdf_path.write_bytes(b"worker")

    monkeypatch.setattr(main, "_compute_file_hash", lambda path: (_ for _ in ()).throw(AssertionError("hash not expected")))
    monkeypatch.setattr(main, "_copy_pdf_to_output", lambda pdf, doc_out_dir: doc_out_dir / pdf.name)
    monkeypatch.setattr(main, "_extract_data_from_pdf", lambda *args: ([{"raw_lab_name": "Lab A"}], "2024-01-01"))
    monkeypatch.setattr(main, "rebuild_document_csv", lambda doc_out_dir, lab_specs: doc_out_dir / "worker.csv")

    csv_path, failed_pages = main.process_single_pdf(pdf_path, "knownhash", config.output_path, config, object())

    assert failed_pages == []
    assert csv_path == config.output_path / "worker_knownhash" / "worker.csv"


def test_process_pdfs_or_use_cache_returns_injected_cached_csvs(tmp_path):
    config = _build_config(tmp_path)
    cached_csv = config.output_path / "cached_hash" / "cached.csv"
    _write_valid_csv(cached_csv)
    preflight = main.PdfPreflightResult(
        pdfs_to_process=[],
        duplicates=[],
        skipped_count=0,
        cached_csv_paths=[cached_csv],
    )

    csv_paths, failed_pages, pdfs_failed = main._process_pdfs_or_use_cache(
        preflight,
        config,
        object(),
        config.output_path / "logs",
    )

    assert csv_paths == [cached_csv]
    assert failed_pages == []
    assert pdfs_failed == 0
