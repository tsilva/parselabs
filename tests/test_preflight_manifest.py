import json
from pathlib import Path

import pandas as pd

from parselabs import pipeline as main
from parselabs.config import ExtractionConfig
from parselabs.store import build_document_ref


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
    monkeypatch.setattr(main, "compute_file_hash", lambda pdf_path: pdf_path.stem)

    preflight = main._prepare_pdf_run([pdf_a, pdf_b], output_dir)

    assert [task.file_hash for task in preflight.pdfs_to_process] == ["a", "b"]
    assert [task.doc_dir / f"{task.stem}.csv" for task in preflight.pdfs_to_process] == [
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

    monkeypatch.setattr(main, "compute_file_hash", lambda pdf_path: "same-hash")

    preflight = main._prepare_pdf_run([pdf_a, pdf_b], output_dir)

    assert [task.file_hash for task in preflight.pdfs_to_process] == ["same-hash"]
    assert preflight.duplicates == [(pdf_b, pdf_a)]


def test_prepare_pdf_run_classifies_fully_cached_documents(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pdf_a = tmp_path / "a.pdf"
    pdf_a.write_bytes(b"a")
    doc_ref = build_document_ref(pdf_a, output_dir, "deadbeef")
    _write_valid_csv(doc_ref.doc_dir / "a.csv")
    (doc_ref.doc_dir / "a.001.json").write_text(
        json.dumps({"page_has_lab_data": True, "lab_results": [{"raw_lab_name": "Glucose"}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(main, "_get_pdf_page_count", lambda pdf_path: 1)
    monkeypatch.setattr(main, "compute_file_hash", lambda pdf_path: "deadbeef")

    preflight = main._prepare_pdf_run([pdf_a], output_dir)

    assert preflight.pdfs_to_process == []
    assert preflight.cached_csv_paths == [doc_ref.doc_dir / "a.csv"]


def test_process_single_pdf_uses_precomputed_hash(tmp_path, monkeypatch):
    config = _build_config(tmp_path)
    pdf_path = tmp_path / "worker.pdf"
    pdf_path.write_bytes(b"worker")

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


def test_process_pdfs_or_use_cache_keeps_cached_csvs_with_new_processing_results(tmp_path, monkeypatch):
    config = _build_config(tmp_path)
    cached_csv = config.output_path / "cached_hash" / "cached.csv"
    processed_csv = config.output_path / "processed_hash" / "processed.csv"
    _write_valid_csv(cached_csv)
    _write_valid_csv(processed_csv)
    pdf_path = tmp_path / "processed.pdf"
    pdf_path.write_bytes(b"processed")
    preflight = main.PdfPreflightResult(
        pdfs_to_process=[build_document_ref(pdf_path, config.output_path, "processed_hash")],
        duplicates=[],
        cached_csv_paths=[cached_csv],
    )

    monkeypatch.setattr(main, "_process_pdfs_in_parallel", lambda *args, **kwargs: ([processed_csv], [], 0))

    csv_paths, failed_pages, pdfs_failed = main._process_pdfs_or_use_cache(
        preflight,
        config,
        object(),
        config.output_path / "logs",
    )

    assert csv_paths == [cached_csv, processed_csv]
    assert failed_pages == []
    assert pdfs_failed == 0
