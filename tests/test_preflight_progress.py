import json
from pathlib import Path

import pandas as pd

import main


def _write_valid_csv(csv_path: Path) -> None:
    """Create the smallest CSV that still counts as valid pipeline output."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=main.REQUIRED_CSV_COLS).to_csv(csv_path, index=False)


def _write_manifest_entry(output_dir: Path, pdf_path: Path, file_hash: str, csv_path: Path) -> None:
    """Persist one manifest entry matching the current PDF stat data."""

    pdf_stat = main._stat_pdf_file(pdf_path)
    entry = main._build_inventory_entry(pdf_stat, file_hash, csv_path)
    manifest_path = main._get_pdf_inventory_path(output_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({str(pdf_stat.resolved_path): entry.to_dict()}, indent=2),
        encoding="utf-8",
    )


def test_prepare_pdf_run_shows_cache_and_hash_progress_for_uncached_pdfs(tmp_path, monkeypatch):
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
    assert calls == ["Checking cached outputs", "Hashing PDFs"]


def test_prepare_pdf_run_skips_hash_progress_for_cached_pdfs(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"a")
    pdf_b.write_bytes(b"b")
    csv_a = main._build_hashed_csv_path(pdf_a, output_dir, "hash-a")
    csv_b = main._build_hashed_csv_path(pdf_b, output_dir, "hash-b")
    _write_valid_csv(csv_a)
    _write_valid_csv(csv_b)

    manifest_path = main._get_pdf_inventory_path(output_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                str(main._stat_pdf_file(pdf_a).resolved_path): main._build_inventory_entry(
                    main._stat_pdf_file(pdf_a),
                    "hash-a",
                    csv_a,
                ).to_dict(),
                str(main._stat_pdf_file(pdf_b).resolved_path): main._build_inventory_entry(
                    main._stat_pdf_file(pdf_b),
                    "hash-b",
                    csv_b,
                ).to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs["desc"])
        return iterable

    monkeypatch.setattr(main, "tqdm", fake_tqdm)
    monkeypatch.setattr(main, "_compute_file_hash", lambda path: (_ for _ in ()).throw(AssertionError("hash not expected")))

    preflight = main._prepare_pdf_run([pdf_a, pdf_b], output_dir)

    assert preflight.cached_csv_paths == [csv_a, csv_b]
    assert preflight.pdfs_to_process == []
    assert calls == ["Checking cached outputs"]
