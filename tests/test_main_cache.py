import json
from pathlib import Path

import pandas as pd

import main
from parselabs.config import ExtractionConfig


def _build_config(tmp_path: Path) -> ExtractionConfig:
    return ExtractionConfig(
        input_path=tmp_path,
        output_path=tmp_path / "output",
        openrouter_api_key="test-key",
        extract_model_id="test-model",
        max_workers=1,
    )


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

    merged_df = main._build_merged_review_dataframe_from_csv_paths([first_csv, second_csv])

    assert merged_df["source_file"].tolist() == ["a.pdf", "b.pdf"]
    assert merged_df["review_status"].fillna("").tolist() == ["", "accepted"]


def test_extract_or_load_page_data_reuses_valid_cached_json(tmp_path, monkeypatch):
    config = _build_config(tmp_path)
    json_path = tmp_path / "glucose.001.json"
    json_path.write_text(
        json.dumps(
            {
                "page_has_lab_data": True,
                "lab_results": [
                    {
                        "raw_lab_name": "Glucose",
                        "raw_value": "92",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(main, "_extract_page_data_from_text", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("text extraction not expected")))
    monkeypatch.setattr(main, "_extract_page_data_from_image", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("image extraction not expected")))

    page_data = main._extract_or_load_page_data(
        {"primary": tmp_path / "primary.jpg", "fallback": tmp_path / "fallback.jpg"},
        json_path,
        "glucose.001",
        config,
        "glucose",
        0,
        [],
        "",
    )

    assert page_data["lab_results"][0]["raw_lab_name"] == "Glucose"


def test_extract_or_load_page_data_retries_failed_cached_json(tmp_path, monkeypatch):
    config = _build_config(tmp_path)
    json_path = tmp_path / "glucose.001.json"
    json_path.write_text(
        json.dumps(
            {
                "_extraction_failed": True,
                "_failure_reason": "boom",
                "lab_results": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        main,
        "_extract_page_data_from_image",
        lambda *args, **kwargs: {
            "page_has_lab_data": True,
            "lab_results": [
                {
                    "raw_lab_name": "Glucose",
                    "raw_value": "92",
                }
            ],
        },
    )

    page_data = main._extract_or_load_page_data(
        {"primary": tmp_path / "primary.jpg", "fallback": tmp_path / "fallback.jpg"},
        json_path,
        "glucose.001",
        config,
        "glucose",
        0,
        [],
        "",
    )

    assert page_data.get("_extraction_failed") is not True
    assert page_data["lab_results"][0]["raw_lab_name"] == "Glucose"
