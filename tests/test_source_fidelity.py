"""Source fidelity regressions using synthetic page payloads."""

import json
from pathlib import Path

import pandas as pd
import pytest

from parselabs.config import LabSpecsConfig
from parselabs.exceptions import PipelineError
from parselabs.normalization import apply_normalizations
from parselabs.pipeline import validate_patient_identity
from parselabs.rows import (
    _flatten_page_payloads,
    _infer_explicit_conversion_unit_from_raw_unit,
    load_document_review_rows,
)


def pages():
    return [{"collection_date": date, "lab_results": [{"raw_lab_name": "Synthetic", "raw_value": "2"}]} for date in ("2024-06-09", "2024-04-03")]


def test_patient_header_rejects_another_person_and_allows_continuation_pages():
    with pytest.raises(PipelineError, match="Patient header"):
        validate_patient_identity({"patient_name": "John Sample"}, "Jane Example")
    validate_patient_identity({"patient_name": "Jane Maria Example"}, "Jane Example")
    validate_patient_identity({}, "Jane Example")


def test_page_payloads_preserve_distinct_collection_dates():
    assert _flatten_page_payloads(pages())["date"].tolist() == ["2024-06-09", "2024-04-03"]


def test_cached_json_rebuild_preserves_each_page_date(tmp_path):
    directory = tmp_path / "2024-06-12 - synthetic_12345678"
    directory.mkdir()
    for index, payload in enumerate(pages(), 1):
        (directory / f"synthetic.{index:03}.json").write_text(json.dumps(payload))
    assert load_document_review_rows(directory)["date"].tolist() == ["2024-06-09", "2024-04-03"]


@pytest.mark.parametrize("unit", ["/mmc", "/mm3", "/mm³", "cells/µL"])
def test_reticulocyte_explicit_unit_converts_value_and_ranges(unit):
    specs = LabSpecsConfig(Path(__file__).parents[1] / "config/lab_specs.json")
    name = "Blood - Reticulocyte Count"
    source_unit = _infer_explicit_conversion_unit_from_raw_unit(unit, name, specs)
    assert source_unit is not None
    df = pd.DataFrame(
        [
            {
                "lab_name_standardized": name,
                "lab_unit_standardized": source_unit,
                "raw_value": "20850",
                "raw_reference_min": 25000,
                "raw_reference_max": 75000,
                "raw_lab_unit": unit,
                "raw_comments": "",
            }
        ]
    )
    normalized = apply_normalizations(df, specs).iloc[0]
    assert normalized.value_primary == pytest.approx(20.85)
    assert normalized.reference_min_primary == 25
    assert normalized.reference_max_primary == 75
    assert normalized.lab_unit_primary == "10⁹/L"


def test_later_page_date_precedes_filename_fallback(tmp_path):
    directory = tmp_path / "2024-06-12 - synthetic_12345678"
    directory.mkdir()
    data = [{"lab_results": [{"raw_lab_name": "Synthetic", "raw_value": "2"}]}, pages()[1]]
    for index, payload in enumerate(data, 1):
        (directory / f"synthetic.{index:03}.json").write_text(json.dumps(payload))
    assert load_document_review_rows(directory)["date"].tolist() == ["2024-04-03", "2024-04-03"]


def test_recursive_discovery_excludes_orders_but_finds_nested_labs(tmp_path):
    from parselabs.store import discover_pdf_files

    nested = tmp_path / "misc"
    nested.mkdir()
    lab = nested / "2024-01-15 - Analises.PDF"
    lab.touch()
    (tmp_path / "2024-01-15 - receita - Analises.pdf").touch()
    assert discover_pdf_files(tmp_path, "*analises.pdf") == [lab]


@pytest.mark.parametrize("bounds", [(1.5, 6.0), (None, 0.1)])
def test_percentage_reference_repeated_from_absolute_sibling_is_reviewable(bounds):
    from parselabs.rows import _flag_suspicious_reference_ranges

    specs = LabSpecsConfig(Path(__file__).parents[1] / "config/lab_specs.json")
    name = next(n for n in specs._specs if specs.get_non_percentage_variant(n))
    absolute = specs.get_non_percentage_variant(name)
    df = pd.DataFrame(
        [
            {"lab_name_standardized": name, "lab_unit_primary": "%"},
            {"lab_name_standardized": absolute, "lab_unit_primary": "10⁹/L"},
        ]
    )
    df["reference_min_primary"] = bounds[0]
    df["reference_max_primary"] = bounds[1]
    df["source_file"] = "synthetic.pdf"
    df["page_number"] = 1
    df["raw_value"] = "2"
    df["review_needed"] = False
    df["review_reason"] = ""
    reviewed = _flag_suspicious_reference_ranges(df, specs)
    assert reviewed.iloc[0].review_needed
    assert reviewed.iloc[0].reference_max_primary == bounds[1]


@pytest.mark.parametrize("name,bounds,review", [
    ("Blood - Lymphocytes (%)", (1.5, 4.0), True),
    ("Blood - Lymphocytes (%)", (20.0, 40.0), False),
    ("Blood - Basophils (%)", (None, 0.1), True),
])
def test_absolute_reference_on_percentage_with_missing_sibling_interval(name, bounds, review):
    from parselabs.rows import _flag_suspicious_reference_ranges

    specs = LabSpecsConfig(Path(__file__).parents[1] / "config/lab_specs.json")
    df = pd.DataFrame([
        {"lab_name_standardized": name, "lab_unit_primary": "%", "reference_min_primary": bounds[0], "reference_max_primary": bounds[1]},
        {"lab_name_standardized": specs.get_non_percentage_variant(name), "lab_unit_primary": "10⁹/L", "reference_min_primary": None, "reference_max_primary": None},
    ])
    df["source_file"] = "synthetic.pdf"
    df["page_number"] = 1
    df["raw_value"] = "2"
    df["review_needed"] = False
    df["review_reason"] = ""
    reviewed = _flag_suspicious_reference_ranges(df, specs)
    assert bool(reviewed.iloc[0].review_needed) is review
    assert pd.isna(reviewed.iloc[1].reference_max_primary)


def test_corrupt_cached_page_image_is_recreated(tmp_path):
    from PIL import Image

    from parselabs.pipeline import _prepare_page_images

    (tmp_path / "synthetic.001.jpg").write_bytes(b"truncated")
    paths = _prepare_page_images(Image.new("RGB", (100, 200)), "synthetic.001", tmp_path)
    for path in paths.values():
        with Image.open(path) as image:
            image.load()
            assert image.width > 0


def test_continuation_report_date_does_not_replace_collection_date():
    payloads = pages()
    payloads[1].pop("collection_date")
    payloads[1]["report_date"] = "2024-06-12"
    assert _flatten_page_payloads(payloads)["date"].tolist() == ["2024-06-09", "2024-06-09"]


def test_registration_date_is_separate_from_collection_and_report_date():
    payloads = pages()
    for payload in payloads:
        payload.pop("collection_date")
        payload["report_date"] = "2024-06-12"
    payloads[0]["registration_date"] = "2024-06-09"
    assert _flatten_page_payloads(payloads)["date"].tolist() == ["2024-06-09", "2024-06-09"]
