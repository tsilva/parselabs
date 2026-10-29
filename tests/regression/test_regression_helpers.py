from __future__ import annotations

import json
from io import StringIO

import pandas as pd
import pytest

from parselabs.regression import (
    build_case_diff,
    canonical_csv_text,
    canonicalize_export_df,
    discover_approved_cases,
    export_fingerprint,
    get_approved_fixtures_dir,
    require_source_review,
    source_fingerprints,
)


def test_canonicalize_export_df_sorts_rows_and_normalizes_values():
    df = pd.DataFrame(
        [
            {
                "date": "2024-01-03 00:00:00",
                "lab_name": "Blood - Z",
                "value": 1.0,
                "lab_unit": " mg/dL ",
                "source_file": "doc.csv",
                "page_number": 2.0,
                "reference_min": None,
                "reference_max": 5.0,
                "raw_lab_name": " Z ",
                "raw_value": " 1.0 ",
                "raw_unit": " mg/dL ",
                "review_needed": True,
                "review_reason": " FLAG ",
                "is_below_limit": None,
                "is_above_limit": False,
                "lab_type": "blood",
                "result_index": 3.0,
            },
            {
                "date": "2024-01-02",
                "lab_name": "Blood - A",
                "value": 0.000123456789,
                "lab_unit": "g/L",
                "source_file": "doc.csv",
                "page_number": 1,
                "reference_min": 0,
                "reference_max": 1,
                "raw_lab_name": "A",
                "raw_value": "0.000123456789",
                "raw_unit": "g/L",
                "review_needed": False,
                "review_reason": None,
                "is_below_limit": False,
                "is_above_limit": True,
                "lab_type": "blood",
                "result_index": 1,
            },
        ]
    )

    canonical = canonicalize_export_df(df)

    assert canonical["lab_name"].tolist() == ["Blood - A", "Blood - Z"]
    assert canonical.loc[0, "date"] == "2024-01-02"
    assert canonical.loc[0, "value"] == "0.000123456789"
    assert canonical.loc[0, "page_number"] == "1"
    assert canonical.loc[0, "review_needed"] == "false"
    assert canonical.loc[0, "is_above_limit"] == "true"
    assert canonical.loc[1, "lab_unit"] == "mg/dL"
    assert canonical.loc[1, "review_reason"] == "FLAG"
    assert canonical.loc[1, "is_below_limit"] == ""


def test_discover_approved_cases_keeps_distinct_hashes_with_the_same_stem(tmp_path):
    fixtures_dir = tmp_path / "approved"
    for case_id, filename in (("doc_a1b2c3d4", "one.pdf"), ("doc_e5f6g7h8", "two.pdf")):
        case_dir = fixtures_dir / case_id
        case_dir.mkdir(parents=True)
        (case_dir / "document.pdf").write_bytes(b"pdf")
        (case_dir / "expected.csv").write_text("date,lab_name\n", encoding="utf-8")
        (case_dir / "case.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "original_filename": filename,
                    "stem": "doc",
                    "file_hash": case_id.split("_", 1)[1],
                    "profile": "test",
                }
            ),
            encoding="utf-8",
        )

    assert len(discover_approved_cases(fixtures_dir)) == 2


def test_build_case_diff_returns_unified_diff():
    expected_df = pd.DataFrame([{"date": "2024-01-01", "lab_name": "Blood - Glucose", "value": 10}])
    actual_df = pd.DataFrame([{"date": "2024-01-01", "lab_name": "Blood - Glucose", "value": 12}])

    diff = build_case_diff(expected_df, actual_df, "glucose_deadbeef")

    assert "glucose_deadbeef/expected.csv" in diff
    assert "---" in diff
    assert "+2024-01-01,Blood - Glucose,12" in diff


def test_expected_csv_round_trip_preserves_raw_text_and_numeric_row_order():
    original = pd.DataFrame([
        {"date": "2024-01-01", "lab_name": "Blood - Synthetic", "page_number": 1, "result_index": 10, "raw_value": "001.00", "value": 1},
        {"date": "2024-01-01", "lab_name": "Blood - Synthetic", "page_number": 1, "result_index": 2, "raw_value": "288", "value": 288},
    ])
    text = canonical_csv_text(original)
    restored = pd.read_csv(StringIO(text), keep_default_na=False, dtype=str)

    assert restored.result_index.tolist() == ["2", "10"]
    assert restored.raw_value.tolist() == ["288", "001.00"]
    assert canonical_csv_text(restored) == text
    assert export_fingerprint(restored) == export_fingerprint(original)


def test_private_corpus_rejects_a_git_checkout_even_if_gitignored(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    monkeypatch.setenv("PARSELABS_APPROVED_FIXTURES_DIR", str(checkout / "ignored" / "goldens"))

    with pytest.raises(ValueError, match="outside every Git checkout"):
        get_approved_fixtures_dir()


def test_private_corpus_rejects_a_symlink_into_a_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    alias = tmp_path / "private-looking-alias"
    alias.symlink_to(checkout, target_is_directory=True)
    monkeypatch.setenv("PARSELABS_APPROVED_FIXTURES_DIR", str(alias / "goldens"))

    with pytest.raises(ValueError, match="outside every Git checkout"):
        get_approved_fixtures_dir()


def test_source_review_rejects_source_edits_and_normalization_drift(tmp_path):
    pdf = tmp_path / "synthetic.pdf"
    pdf.write_bytes(b"synthetic PDF content")
    page = tmp_path / "synthetic.001.json"
    page.write_text('{"lab_results": []}', encoding="utf-8")
    expected = pd.DataFrame([{"date": "2024-01-01", "lab_name": "Blood - Synthetic", "value": 1}])
    review = {
        "reviewed_at": "2024-01-01T00:00:00Z",
        "note": "Synthetic source and export checked.",
        "checks": dict.fromkeys(("source_fidelity", "page_completeness", "patient_identity", "dates", "normalized_export"), True),
        "fingerprints": source_fingerprints(tmp_path, pdf),
        "expected_export_sha256": export_fingerprint(expected),
    }
    (tmp_path / "source_review.json").write_text(json.dumps(review), encoding="utf-8")

    assert require_source_review(tmp_path, pdf, expected) == review
    changed = expected.assign(value=2)
    with pytest.raises(ValueError, match="normalized-export review"):
        require_source_review(tmp_path, pdf, changed)

    page.write_text('{"lab_results": [], "report_date": "2024-01-02"}', encoding="utf-8")
    with pytest.raises(ValueError, match="PDF or page JSON changed"):
        require_source_review(tmp_path, pdf, expected)


def test_source_review_requires_explicit_fidelity_and_completeness_checks(tmp_path):
    pdf = tmp_path / "synthetic.pdf"
    pdf.write_bytes(b"synthetic PDF content")

    with pytest.raises(ValueError, match="Missing source review"):
        require_source_review(tmp_path, pdf)

    (tmp_path / "source_review.json").write_text(json.dumps({"note": "reviewed", "reviewed_at": "2024-01-01"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete source-review"):
        require_source_review(tmp_path, pdf)


def test_same_document_stem_can_exist_in_different_profiles(tmp_path):
    for profile in ("alpha", "beta"):
        case_dir = tmp_path / profile / "synthetic_deadbeef"
        case_dir.mkdir(parents=True)
        (case_dir / "document.pdf").write_bytes(b"synthetic PDF content")
        (case_dir / "expected.csv").write_text("date,lab_name\n", encoding="utf-8")
        (case_dir / "case.json").write_text(json.dumps({
            "stem": "synthetic", "file_hash": "deadbeef", "original_filename": "synthetic.pdf", "profile": profile,
        }), encoding="utf-8")

    assert len(discover_approved_cases(tmp_path)) == 2


def test_missing_case_metadata_is_not_silently_omitted(tmp_path):
    case_dir = tmp_path / "synthetic-profile" / "synthetic_deadbeef"
    case_dir.mkdir(parents=True)
    (case_dir / "document.pdf").write_bytes(b"synthetic PDF content")
    (case_dir / "expected.csv").write_text("date,lab_name\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required files: case.json"):
        discover_approved_cases(tmp_path)


def test_changed_existing_baseline_requires_explicit_replacement(tmp_path):
    from types import SimpleNamespace

    from parselabs.admin.regression_cases import _check_replacements

    previous_dir = tmp_path / "old"
    current_dir = tmp_path / "new"
    previous_dir.mkdir()
    current_dir.mkdir()
    previous_review = {"fingerprints": {"synthetic.pdf": "old"}, "expected_export_sha256": "old"}
    current_review = {"fingerprints": {"synthetic.pdf": "new"}}
    (previous_dir / "source_review.json").write_text(json.dumps(previous_review), encoding="utf-8")
    (current_dir / "source_review.json").write_text(json.dumps(current_review), encoding="utf-8")
    previous = {"synthetic": SimpleNamespace(case_dir=previous_dir)}
    current = {"synthetic": (current_dir, current_dir / "synthetic.pdf", pd.DataFrame())}

    with pytest.raises(ValueError, match="use --replace"):
        _check_replacements(previous, current, False)

    _check_replacements(previous, current, True)
    assert json.loads((previous_dir / "source_review.json").read_text()) == previous_review
