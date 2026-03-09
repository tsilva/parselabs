from __future__ import annotations

import json

import pandas as pd
import pytest

from parselabs.regression import build_case_diff, canonicalize_export_df, discover_approved_cases


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


def test_discover_approved_cases_rejects_duplicate_stems(tmp_path):
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

    with pytest.raises(ValueError, match="unique stems"):
        discover_approved_cases(fixtures_dir)


def test_build_case_diff_returns_unified_diff():
    expected_df = pd.DataFrame([{"date": "2024-01-01", "lab_name": "Blood - Glucose", "value": 10}])
    actual_df = pd.DataFrame([{"date": "2024-01-01", "lab_name": "Blood - Glucose", "value": 12}])

    diff = build_case_diff(expected_df, actual_df, "glucose_deadbeef")

    assert "glucose_deadbeef/expected.csv" in diff
    assert "---" in diff
    assert "+2024-01-01,Blood - Glucose,12" in diff
