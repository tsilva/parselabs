from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from parselabs.config import LabSpecsConfig, ProfileConfig
from parselabs.review_sync import (
    ReviewStateError,
    build_document_expected_dataframe_from_reviewed_json,
    build_review_corpus_report,
    build_document_review_dataframe,
    get_document_review_summary,
    get_review_summary,
    rebuild_document_csv,
    save_missing_row_marker,
    save_review_status,
)
from utils import regression_cases


def _write_processed_document(
    doc_dir: Path,
    statuses: list[str | None],
    stem: str = "glucose",
    raw_names: list[str] | None = None,
    raw_units: list[str] | None = None,
    missing_markers: list[dict] | None = None,
) -> None:
    """Create a minimal processed-document directory for review tests."""

    doc_dir.mkdir(parents=True)
    (doc_dir / f"{stem}.pdf").write_bytes(b"%PDF-1.4")
    lab_results = []

    for idx, status in enumerate(statuses):
        result = {
            "raw_lab_name": raw_names[idx] if raw_names is not None else "Glucose",
            "raw_value": str(90 + idx),
            "raw_lab_unit": raw_units[idx] if raw_units is not None else "mg/dL",
            "raw_reference_min": 70,
            "raw_reference_max": 100,
        }

        # Persist explicit review status only when the test requests it.
        if status is not None:
            result["review_status"] = status

        lab_results.append(result)

    payload = {
        "collection_date": "2024-01-05",
        "lab_results": lab_results,
    }

    # Persist unresolved missing-row markers when the test requests them.
    if missing_markers is not None:
        payload["review_missing_rows"] = missing_markers

    (doc_dir / f"{stem}.001.json").write_text(json.dumps(payload), encoding="utf-8")


def _make_lab_specs(tmp_path: Path) -> LabSpecsConfig:
    """Create a minimal lab specs file for review-sync tests."""

    config_path = tmp_path / "lab_specs.json"
    config_path.write_text(
        json.dumps(
            {
                "Blood - Glucose": {
                    "primary_unit": "mg/dL",
                    "lab_type": "blood",
                    "loinc_code": "2345-7",
                    "ranges": {"default": [70, 100]},
                    "biological_min": 0,
                    "biological_max": 1000,
                }
            }
        ),
        encoding="utf-8",
    )
    return LabSpecsConfig(config_path=config_path)


def _stub_standardization(monkeypatch) -> None:
    """Stub cache-backed standardization so tests stay self-contained."""

    monkeypatch.setattr(
        "parselabs.review_sync.standardize_lab_names",
        lambda raw_names, _: {name: "Blood - Glucose" for name in raw_names},
    )
    monkeypatch.setattr(
        "parselabs.review_sync.standardize_lab_units",
        lambda unit_contexts, *_: {context: "mg/dL" for context in unit_contexts},
    )


def test_rebuild_document_csv_preserves_row_identity_and_review_status(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _stub_standardization(monkeypatch)
    doc_dir = tmp_path / "processed" / "glucose_deadbeef"
    _write_processed_document(doc_dir, [None, "accepted"])

    csv_path = rebuild_document_csv(doc_dir, lab_specs)
    review_df = pd.read_csv(csv_path, keep_default_na=False)

    assert review_df["source_file"].tolist() == ["glucose.csv", "glucose.csv"]
    assert review_df["page_number"].tolist() == [1, 1]
    assert review_df["result_index"].tolist() == [0, 1]
    assert review_df["review_status"].tolist() == ["", "accepted"]
    assert review_df["lab_name_standardized"].tolist() == ["Blood - Glucose", "Blood - Glucose"]
    assert review_df["lab_name"].tolist() == ["Blood - Glucose", "Blood - Glucose"]


def test_save_review_status_updates_json_and_summary(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _stub_standardization(monkeypatch)
    doc_dir = tmp_path / "processed" / "glucose_deadbeef"
    _write_processed_document(doc_dir, [None, None])

    success, error = save_review_status(doc_dir, page_number=1, result_index=0, status="accepted")

    assert success is True
    assert error == ""

    rebuild_document_csv(doc_dir, lab_specs)
    review_df = build_document_review_dataframe(doc_dir, lab_specs)
    summary = get_review_summary(review_df)

    assert review_df.loc[0, "review_status"] == "accepted"
    assert summary.accepted == 1
    assert summary.pending == 1


def test_save_missing_row_marker_preserves_current_row_status(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _stub_standardization(monkeypatch)
    doc_dir = tmp_path / "processed" / "glucose_deadbeef"
    _write_processed_document(doc_dir, ["accepted", None])

    success, error = save_missing_row_marker(doc_dir, page_number=1, anchor_result_index=1)

    assert success is True
    assert error == ""

    rebuild_document_csv(doc_dir, lab_specs)
    review_df = build_document_review_dataframe(doc_dir, lab_specs)
    summary = get_document_review_summary(doc_dir, review_df)

    assert review_df["review_status"].fillna("").tolist() == ["accepted", ""]
    assert summary.missing_row_markers == 1
    assert summary.pending == 1


def test_build_document_expected_dataframe_requires_fixture_ready_review(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _stub_standardization(monkeypatch)
    doc_dir = tmp_path / "processed" / "glucose_deadbeef"
    _write_processed_document(doc_dir, ["accepted", None])

    with pytest.raises(ReviewStateError, match="not fixture-ready"):
        build_document_expected_dataframe_from_reviewed_json(doc_dir, lab_specs)

    expected_df = build_document_expected_dataframe_from_reviewed_json(
        doc_dir,
        lab_specs,
        allow_pending=True,
    )

    assert expected_df["raw_value"].tolist() == ["90"]


def test_build_document_expected_dataframe_blocks_unresolved_missing_markers(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _stub_standardization(monkeypatch)
    doc_dir = tmp_path / "processed" / "glucose_deadbeef"
    _write_processed_document(
        doc_dir,
        ["accepted"],
        missing_markers=[{"anchor_result_index": 0, "created_at": "2024-01-05T00:00:00Z"}],
    )

    with pytest.raises(ReviewStateError, match="missing-row marker"):
        build_document_expected_dataframe_from_reviewed_json(doc_dir, lab_specs)


def test_regression_case_sync_copies_fixture_ready_documents_with_rejections(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _stub_standardization(monkeypatch)
    output_path = tmp_path / "processed"
    approved_dir = tmp_path / "fixtures"

    _write_processed_document(output_path / "valid_deadbeef", ["accepted", "rejected"], stem="valid")
    _write_processed_document(output_path / "pending_deadbeef", [None], stem="pending")
    _write_processed_document(
        output_path / "missing_deadbeef",
        ["accepted"],
        stem="missing",
        missing_markers=[{"anchor_result_index": 0, "created_at": "2024-01-05T00:00:00Z"}],
    )

    monkeypatch.setattr(regression_cases, "APPROVED_FIXTURES_DIR", approved_dir)
    monkeypatch.setattr(regression_cases, "discover_approved_cases", lambda: [])
    monkeypatch.setattr(regression_cases, "LabSpecsConfig", lambda: lab_specs)
    monkeypatch.setattr(
        regression_cases,
        "load_profile",
        lambda _: ProfileConfig(name="test", output_path=output_path),
    )

    regression_cases.approve_cases(argparse.Namespace(command="approve", profile="test"))

    case_dirs = sorted(path.name for path in approved_dir.iterdir())
    expected_case_id = f"valid_{regression_cases.compute_file_hash(output_path / 'valid_deadbeef' / 'valid.pdf')}"
    assert case_dirs == [expected_case_id]
    assert (approved_dir / expected_case_id / "expected.csv").exists()
    assert not any("pending" in case_dir for case_dir in case_dirs)
    assert not any("missing" in case_dir for case_dir in case_dirs)


def test_review_corpus_report_counts_rejections_missing_rows_and_unknowns(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    output_path = tmp_path / "processed"

    _write_processed_document(
        output_path / "mixed_deadbeef",
        ["accepted", "rejected"],
        stem="mixed",
        raw_names=["Glucose", "Mystery Lab"],
        raw_units=["mg/dL", "odd-unit"],
        missing_markers=[{"anchor_result_index": 1, "created_at": "2024-01-05T00:00:00Z"}],
    )

    monkeypatch.setattr(
        "parselabs.review_sync.standardize_lab_names",
        lambda raw_names, _: {
            name: ("Blood - Glucose" if name == "Glucose" else "$UNKNOWN$")
            for name in raw_names
        },
    )
    monkeypatch.setattr(
        "parselabs.review_sync.standardize_lab_units",
        lambda unit_contexts, *_: {
            context: ("mg/dL" if context[0] == "mg/dL" else "$UNKNOWN$")
            for context in unit_contexts
        },
    )

    report = build_review_corpus_report(output_path, lab_specs)

    assert report.document_count == 1
    assert report.fixture_ready_document_count == 0
    assert report.rejected_rows == 1
    assert report.unresolved_missing_rows == 1
    assert report.unknown_standardized_names == 1
    assert report.unknown_standardized_units == 0
    assert report.rejected_raw_name_counts["Mystery Lab"] == 1
    assert report.rejected_raw_unit_counts["odd-unit"] == 1
