"""Synthetic pathology observations exercise the shared review/export pipeline."""

import json

import pandas as pd
import pytest

from parselabs import standardization
from parselabs.config import LabSpecsConfig
from parselabs.lab_specs_validation import LabSpecsValidator
from parselabs.rows import build_document_review_dataframe, build_review_rows, prepare_rows, rebuild_document_csv


@pytest.fixture
def tissue_cache(tmp_path, monkeypatch):
    """Keep synthetic aliases isolated from the user's standardization overrides."""

    names = {
        "Margins": "Tissue - Excision Margins",
        "Regression": "Tissue - Tumor Regression",
        "Breslow": "Tissue - Breslow Thickness",
        "Mitoses": "Tissue - Mitotic Rate",
        "PRAME": "Tissue - PRAME Immunostain",
        "SOX10": "Tissue - SOX10 Immunostain",
        "p16": "Tissue - p16 Immunostain",
        "Diagnosis": "Tissue - Pathology Diagnosis",
        "H. pylori histology": "Tissue - Helicobacter pylori (Histology)",
    }
    monkeypatch.setattr(standardization, "CACHE_DIR", tmp_path / "cache")
    standardization.save_cache("name_standardization", {
        standardization.build_name_cache_key(raw, "Pathology"): name for raw, name in names.items()
    })
    standardization.save_cache("unit_standardization", {})


def _payload(observations):
    """Build an anonymous report with objective source fields and pending decisions."""

    return {
        "report_date": "2025-01-01",
        "page_number": 1,
        "lab_results": [
            {"raw_lab_name": name, "raw_section_name": "Pathology", "raw_value": value, "raw_lab_unit": unit}
            for name, value, unit in observations
        ],
    }


def test_tissue_rows_keep_grades_and_narratives_separate_from_measurements(tissue_cache):
    payload = _payload([
        ("Margins", "1.7", "mm"),
        ("Margins", "Not assessed", None),
        ("Regression", "Ca. 25", "%"),
        ("Regression", "Not identified", None),
        ("PRAME", "3+", None),
        ("SOX10", "expression", None),
        ("p16", "expression", None),
        ("Diagnosis", "Synthetic descriptive finding", None),
        ("H. pylori histology", "negative", None),
        ("Mitoses", "2", "/mm²"),
    ])
    result = build_review_rows([payload], LabSpecsConfig())

    assert len(result) == 10
    assert result.lab_type.eq("tissue").all()
    assert not result.review_reason.str.contains("UNKNOWN_", na=False).any()
    assert result.loc[0, "lab_name"] == "Tissue - Excision Margin Distance"
    assert result.loc[0, "value"] == 1.7
    assert result.loc[0, "lab_unit"] == "mm"
    assert result.loc[2, "lab_name"] == "Tissue - Tumor Regression (%)"
    assert result.loc[2, "value"] == 25
    assert result.loc[2, "value_text"] == "Ca. 25"
    assert result.loc[2, "lab_unit"] == "%"
    assert result.loc[[1, 3, 4, 5, 6, 7, 8], "value"].isna().all()
    assert result.loc[[1, 3, 4, 5, 6, 7, 8], "lab_unit"].eq("text").all()
    assert result.loc[4, "value_text"] == "3+"
    assert result.loc[8, "lab_name"] == "Tissue - Helicobacter pylori (Histology)"
    assert result.loc[9, "value"] == 2
    assert result.loc[9, "lab_unit"] == "/mm2"
    assert result.raw_value.tolist() == [r["raw_value"] for r in payload["lab_results"]]
    assert result.review_status.fillna("").eq("").all()


def test_tissue_measurements_work_without_companions_and_do_not_infer_missing_units(tissue_cache):
    result = build_review_rows([_payload([
        ("Margins", "0.3", "cm"),
        ("Breslow", "0.07", "cm"),
        ("Regression", "8", None),
        ("Margins", "3", None),
    ])], LabSpecsConfig())

    assert result.loc[0, "value"] == 3
    assert result.loc[0, "lab_unit"] == "mm"
    assert result.loc[1, "value"] == pytest.approx(0.7)
    assert result.loc[1, "lab_unit"] == "mm"
    assert result.loc[2:, "value"].isna().all()
    assert result.loc[2:, "value_text"].tolist() == ["8", "3"]
    assert result.raw_lab_unit.tolist()[:2] == ["cm", "cm"]


@pytest.mark.parametrize("deep_value", ["6.5", "2.5"])
def test_explicit_margin_sites_survive_duplicate_detection_and_export(tissue_cache, deep_value):
    payload = _payload([("Margins", "2.5", "mm"), ("Margins", deep_value, "mm")])
    payload["lab_results"][0]["raw_comments"] = "superficial margin"
    payload["lab_results"][1]["raw_comments"] = "margem profunda"
    result = build_review_rows([payload], LabSpecsConfig())

    assert result.date.notna().all()
    assert result.lab_name.tolist() == [
        "Tissue - Excision Margin Distance (Superficial)",
        "Tissue - Excision Margin Distance (Deep)",
    ]
    assert not result.review_reason.str.contains("DUPLICATE_ENTRY", na=False).any()
    assert result.raw_comments.tolist() == ["superficial margin", "margem profunda"]
    exported, _ = prepare_rows(result, LabSpecsConfig(), mode="export", apply_standardization=False)
    assert len(exported) == 2


def test_conflicting_same_margin_results_still_need_review_and_true_repeats_deduplicate(tissue_cache):
    payload = _payload([("Margins", "2.5", "mm"), ("Margins", "6.5", "mm")])
    for row in payload["lab_results"]:
        row["raw_comments"] = "deep margin"
    result = build_review_rows([payload], LabSpecsConfig())

    assert result.lab_name.eq("Tissue - Excision Margin Distance (Deep)").all()
    assert result.review_reason.str.contains("DUPLICATE_ENTRY").all()

    # An actual repeat of the same location and measurement still collapses in final export.
    payload["lab_results"][1]["raw_value"] = "2.5"
    result = build_review_rows([payload], LabSpecsConfig())
    exported, _ = prepare_rows(result, LabSpecsConfig(), mode="export", apply_standardization=False)
    assert len(exported) == 1


def test_ambiguous_margin_location_does_not_infer_a_site_from_value_or_row_order(tissue_cache):
    payload = _payload([("Margins", "2.5", "mm"), ("Margins", "6.5", "mm")])
    for row in payload["lab_results"]:
        row["raw_comments"] = "superficial and deep margins"
    result = build_review_rows([payload], LabSpecsConfig())

    assert result.lab_name.eq("Tissue - Excision Margin Distance").all()
    assert result.review_reason.str.contains("DUPLICATE_ENTRY").all()


def test_rebuilding_tissue_csv_preserves_page_json_and_review_decisions(tmp_path, tissue_cache):
    document = tmp_path / "synthetic_deadbeef"
    document.mkdir()
    payload = _payload([("Margins", "2.5", "mm"), ("PRAME", "2+", None)])
    payload["lab_results"][0]["review_status"] = "accepted"
    payload["lab_results"][1]["review_status"] = "rejected"
    page = document / "synthetic.001.json"
    page.write_text(json.dumps(payload), encoding="utf-8")
    original = page.read_bytes()

    csv_path = rebuild_document_csv(document, LabSpecsConfig())
    result = pd.read_csv(csv_path)

    assert page.read_bytes() == original
    assert result.review_status.tolist() == ["accepted", "rejected"]
    assert result.lab_name.tolist() == ["Tissue - Excision Margin Distance", "Tissue - PRAME Immunostain"]
    assert len(build_document_review_dataframe(document, LabSpecsConfig())) == 2


def test_tissue_loinc_exceptions_are_explicit_and_documented(tmp_path):
    specs = LabSpecsConfig().specs
    tissue = {name: spec for name, spec in specs.items() if isinstance(spec, dict) and spec.get("lab_type") == "tissue"}

    assert tissue
    assert all(name.startswith("Tissue - ") for name in tissue)
    assert all(not spec.get("ranges") for spec in tissue.values())
    assert tissue["Tissue - Breslow Thickness"]["loinc_code"] == "92839-0"
    assert tissue["Tissue - Mitotic Rate"]["loinc_code"] == "86669-9"
    assert tissue["Tissue - Helicobacter pylori (Histology)"]["loinc_code"] is None
    assert all(name in LabSpecsValidator.LOINC_EXCEPTIONS and spec.get("loinc_note") for name, spec in tissue.items() if not spec.get("loinc_code"))

    # New tissue terms cannot bypass the catalog's explicit LOINC allowlist.
    path = tmp_path / "specs.json"
    path.write_text(json.dumps({"Tissue - Unverified": {"lab_type": "tissue", "primary_unit": "text", "loinc_code": None}}))
    validator = LabSpecsValidator(path)
    assert validator.validate() is False
    assert any("Missing LOINC code" in error for error in validator.errors)

    # An allowlisted exception must still retain its terminology explanation.
    path.write_text(json.dumps({"Tissue - PRAME Immunostain": {"lab_type": "tissue", "primary_unit": "text", "loinc_code": None}}))
    validator = LabSpecsValidator(path)
    assert validator.validate() is False
    assert any("loinc_note" in error for error in validator.errors)
