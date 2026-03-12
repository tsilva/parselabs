import json
from pathlib import Path

import pandas as pd

import main
from parselabs.config import ExtractionConfig


def _write_valid_csv(csv_path: Path) -> None:
    """Create the smallest CSV that still counts as valid pipeline output."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=main.REQUIRED_CSV_COLS).to_csv(csv_path, index=False)


def _write_manifest(output_dir: Path, entries: dict[str, main.PdfInventoryEntry]) -> None:
    """Write a raw manifest payload for the test output directory."""

    manifest_path = main._get_pdf_inventory_path(output_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({source_path: entry.to_dict() for source_path, entry in entries.items()}, indent=2),
        encoding="utf-8",
    )


def _build_manifest_entry(pdf_path: Path, file_hash: str, csv_path: Path) -> tuple[str, main.PdfInventoryEntry]:
    """Build one manifest entry matching the PDF's current stat data."""

    pdf_stat = main._stat_pdf_file(pdf_path)
    entry = main._build_inventory_entry(pdf_stat, file_hash, csv_path)
    return str(pdf_stat.resolved_path), entry


def _build_config(tmp_path: Path) -> ExtractionConfig:
    """Create the smallest extraction config needed by unit tests."""

    output_path = tmp_path / "output"
    output_path.mkdir(exist_ok=True)
    return ExtractionConfig(
        input_path=tmp_path,
        output_path=output_path,
        openrouter_api_key="test-key",
        extract_model_id="test-model",
        max_workers=1,
    )


def test_prepare_pdf_run_uses_manifest_cache_without_hashing(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pdf_path = tmp_path / "cached.pdf"
    pdf_path.write_bytes(b"cached")
    csv_path = main._build_hashed_csv_path(pdf_path, output_dir, "hash-cached")
    _write_valid_csv(csv_path)
    source_path, entry = _build_manifest_entry(pdf_path, "hash-cached", csv_path)
    _write_manifest(output_dir, {source_path: entry})

    monkeypatch.setattr(main, "_compute_file_hash", lambda path: (_ for _ in ()).throw(AssertionError("hash not expected")))

    preflight = main._prepare_pdf_run([pdf_path], output_dir)

    assert preflight.cached_csv_paths == [csv_path]
    assert preflight.skipped_count == 1
    assert preflight.pdfs_to_process == []


def test_prepare_pdf_run_rehashes_when_manifest_entry_is_stale(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pdf_path = tmp_path / "stale.pdf"
    pdf_path.write_bytes(b"stale")
    csv_path = main._build_hashed_csv_path(pdf_path, output_dir, "old-hash")
    _write_valid_csv(csv_path)
    source_path, entry = _build_manifest_entry(pdf_path, "old-hash", csv_path)
    stale_entry = main.PdfInventoryEntry(
        source_path=entry.source_path,
        size_bytes=entry.size_bytes,
        mtime_ns=entry.mtime_ns - 1,
        file_hash=entry.file_hash,
        csv_path=entry.csv_path,
    )
    _write_manifest(output_dir, {source_path: stale_entry})

    calls: list[Path] = []

    def fake_hash(path: Path) -> str:
        calls.append(path)
        return "fresh-hash"

    monkeypatch.setattr(main, "_compute_file_hash", fake_hash)

    preflight = main._prepare_pdf_run([pdf_path], output_dir)

    assert calls == [pdf_path.resolve()]
    assert preflight.skipped_count == 0
    assert [task.file_hash for task in preflight.pdfs_to_process] == ["fresh-hash"]


def test_prepare_pdf_run_deduplicates_against_cached_manifest_entry(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"same")
    pdf_b.write_bytes(b"same")
    csv_a = main._build_hashed_csv_path(pdf_a, output_dir, "same-hash")
    _write_valid_csv(csv_a)
    source_path, entry = _build_manifest_entry(pdf_a, "same-hash", csv_a)
    _write_manifest(output_dir, {source_path: entry})

    monkeypatch.setattr(main, "_compute_file_hash", lambda path: "same-hash")

    preflight = main._prepare_pdf_run([pdf_a, pdf_b], output_dir)

    assert preflight.cached_csv_paths == [csv_a]
    assert preflight.pdfs_to_process == []
    assert preflight.duplicates == [(pdf_b, pdf_a)]
    assert preflight.inventory_candidates[str(pdf_b.resolve())].csv_path == str(csv_a)


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


def test_process_pdfs_or_use_cache_merges_paths_without_rehashing(tmp_path, monkeypatch):
    config = _build_config(tmp_path)
    cached_csv = config.output_path / "cached_hash" / "cached.csv"
    processed_csv = config.output_path / "processed_hash" / "processed.csv"
    _write_valid_csv(cached_csv)
    _write_valid_csv(processed_csv)
    task = main.PreflightPdfTask(
        pdf_path=tmp_path / "processed.pdf",
        resolved_path=(tmp_path / "processed.pdf").resolve(),
        size_bytes=1,
        mtime_ns=1,
        file_hash="processed-hash",
        csv_path=processed_csv,
    )
    preflight = main.PdfPreflightResult(
        cached_csv_paths=[cached_csv],
        pdfs_to_process=[task],
        duplicates=[],
        skipped_count=1,
        inventory={},
        inventory_candidates={},
    )

    monkeypatch.setattr(main, "_process_pdfs_in_parallel", lambda *args: ([processed_csv], [], 0))

    csv_paths, failed_pages, pdfs_failed = main._process_pdfs_or_use_cache(
        preflight,
        config,
        object(),
        config.output_path / "logs",
    )

    assert csv_paths == [cached_csv, processed_csv]
    assert failed_pages == []
    assert pdfs_failed == 0


def test_prepare_pdf_run_recovers_from_corrupt_manifest(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"broken")
    manifest_path = main._get_pdf_inventory_path(output_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{not valid json", encoding="utf-8")

    monkeypatch.setattr(main, "_compute_file_hash", lambda path: "fresh-hash")

    preflight = main._prepare_pdf_run([pdf_path], output_dir)

    assert preflight.skipped_count == 0
    assert [task.file_hash for task in preflight.pdfs_to_process] == ["fresh-hash"]
