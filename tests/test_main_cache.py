import json

import pandas as pd

from main import REQUIRED_CSV_COLS, _build_merged_review_dataframe_from_csv_paths, _is_csv_valid


def test_is_csv_valid_rejects_empty_csv_when_page_extraction_failed(tmp_path):
    doc_dir = tmp_path / "document_12345678"
    doc_dir.mkdir()
    csv_path = doc_dir / "document.csv"

    pd.DataFrame(columns=REQUIRED_CSV_COLS).to_csv(csv_path, index=False)
    (doc_dir / "document.001.json").write_text(
        json.dumps(
            {
                "page_has_lab_data": True,
                "lab_results": [],
                "_extraction_failed": True,
            }
        ),
        encoding="utf-8",
    )

    assert _is_csv_valid(csv_path) is False


def test_is_csv_valid_accepts_empty_csv_for_confirmed_blank_document(tmp_path):
    doc_dir = tmp_path / "document_12345678"
    doc_dir.mkdir()
    csv_path = doc_dir / "document.csv"

    pd.DataFrame(columns=REQUIRED_CSV_COLS).to_csv(csv_path, index=False)
    (doc_dir / "document.001.json").write_text(
        json.dumps(
            {
                "page_has_lab_data": False,
                "lab_results": [],
            }
        ),
        encoding="utf-8",
    )

    assert _is_csv_valid(csv_path) is True


def test_is_csv_valid_accepts_non_empty_csv_with_required_columns(tmp_path):
    doc_dir = tmp_path / "document_12345678"
    doc_dir.mkdir()
    csv_path = doc_dir / "document.csv"

    pd.DataFrame(
        [
            {
                "page_number": 1,
                "result_index": 0,
                "raw_lab_name": "Glicose",
                "raw_value": "92",
                "raw_lab_unit": "mg/dL",
                "review_status": "",
            }
        ],
        columns=REQUIRED_CSV_COLS,
    ).to_csv(csv_path, index=False)

    assert _is_csv_valid(csv_path) is True


def test_build_merged_review_dataframe_from_csv_paths_keeps_review_rows(tmp_path):
    first_csv = tmp_path / "doc1.csv"
    second_csv = tmp_path / "doc2.csv"

    pd.DataFrame(
        [
            {
                "page_number": 1,
                "result_index": 0,
                "source_file": "a.pdf",
                "raw_lab_name": "Glicose",
                "raw_value": "92",
                "raw_lab_unit": "mg/dL",
                "lab_name": "Blood - Glucose",
                "value": 92.0,
                "lab_unit": "mg/dL",
                "review_status": "",
            }
        ]
    ).to_csv(first_csv, index=False)
    pd.DataFrame(
        [
            {
                "page_number": 2,
                "result_index": 1,
                "source_file": "b.pdf",
                "raw_lab_name": "Colesterol",
                "raw_value": "180",
                "raw_lab_unit": "mg/dL",
                "lab_name": "Blood - Total Cholesterol",
                "value": 180.0,
                "lab_unit": "mg/dL",
                "review_status": "accepted",
            }
        ]
    ).to_csv(second_csv, index=False)

    merged_df = _build_merged_review_dataframe_from_csv_paths([first_csv, second_csv])

    assert merged_df["source_file"].tolist() == ["a.pdf", "b.pdf"]
    assert merged_df["review_status"].fillna("").tolist() == ["", "accepted"]
