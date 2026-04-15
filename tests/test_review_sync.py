from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from parselabs import pipeline as main
from parselabs import rows as rows_module
from parselabs.config import ExtractionConfig, LabSpecsConfig, ProfileConfig
from parselabs.normalization import apply_normalizations
from parselabs.rows import (
    ReviewStateError,
    apply_cached_standardization,
    build_document_expected_dataframe_from_reviewed_json,
    build_document_review_dataframe,
    build_review_corpus_report,
    get_document_review_summary,
    get_review_summary,
    iter_processed_documents,
    load_document_review_rows,
    rebuild_document_csv,
    save_missing_row_marker,
    save_review_status,
)
from parselabs.types import ReviewMissingRowMarker
from utils import regression_cases


def _write_processed_document(
    doc_dir: Path,
    statuses: list[str | None],
    stem: str = "glucose",
    raw_names: list[str] | None = None,
    raw_sections: list[str | None] | None = None,
    raw_values: list[str] | None = None,
    raw_units: list[str] | None = None,
    raw_reference_mins: list[float | None] | None = None,
    raw_reference_maxs: list[float | None] | None = None,
    bboxes: list[dict[str, float] | None] | None = None,
    missing_markers: list[ReviewMissingRowMarker] | None = None,
) -> None:
    """Create a minimal processed-document directory for review tests."""

    doc_dir.mkdir(parents=True)
    (doc_dir / f"{stem}.pdf").write_bytes(b"%PDF-1.4")
    lab_results = []

    for idx, status in enumerate(statuses):
        result = {
            "raw_lab_name": raw_names[idx] if raw_names is not None else "Glucose",
            "raw_value": raw_values[idx] if raw_values is not None else str(90 + idx),
            "raw_lab_unit": raw_units[idx] if raw_units is not None else "mg/dL",
            "raw_reference_min": raw_reference_mins[idx] if raw_reference_mins is not None else 70,
            "raw_reference_max": raw_reference_maxs[idx] if raw_reference_maxs is not None else 100,
        }

        # Persist section metadata when the test needs context-aware standardization.
        if raw_sections is not None and raw_sections[idx] is not None:
            result["raw_section_name"] = raw_sections[idx]

        # Persist bbox metadata when the test needs review-image highlighting.
        if bboxes is not None and bboxes[idx] is not None:
            result.update(bboxes[idx])

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


def _make_mixed_section_lab_specs(tmp_path: Path) -> LabSpecsConfig:
    """Create lab specs that support both blood and urine disambiguation."""

    config_path = tmp_path / "lab_specs_mixed.json"
    config_path.write_text(
        json.dumps(
            {
                "Blood - Glucose (Fasting)": {
                    "primary_unit": "mg/dL",
                    "lab_type": "blood",
                    "loinc_code": "1558-6",
                    "ranges": {"default": [70, 100]},
                },
                "Blood - Leukocytes": {
                    "primary_unit": "10⁹/L",
                    "lab_type": "blood",
                    "loinc_code": "6690-2",
                    "ranges": {"default": [4.0, 11.0]},
                },
                "Urine Type II - Glucose": {
                    "primary_unit": "boolean",
                    "lab_type": "urine",
                    "loinc_code": "5792-7",
                    "ranges": {"default": [0, 0]},
                },
                "Urine Type II - Glucose (Qualitative)": {
                    "primary_unit": "boolean",
                    "lab_type": "urine",
                    "loinc_code": "5792-7-qual",
                    "ranges": {"default": [0, 0]},
                },
                "Urine Type II - Color": {
                    "primary_unit": "boolean",
                    "lab_type": "urine",
                    "loinc_code": "5778-1",
                    "ranges": {"default": [0, 0]},
                },
                "Urine Type II - pH": {
                    "primary_unit": "pH",
                    "lab_type": "urine",
                    "loinc_code": "5803-7",
                    "ranges": {"default": [5.0, 8.0]},
                },
                "Urine Type II - Sediment - Leukocytes": {
                    "primary_unit": "/field",
                    "lab_type": "urine",
                    "loinc_code": "20408-1",
                    "alternatives": [{"unit": "/campo", "factor": 1.0}],
                    "ranges": {"default": [0, 5]},
                },
            }
        ),
        encoding="utf-8",
    )
    return LabSpecsConfig(config_path=config_path)


def _make_percentage_variant_lab_specs(tmp_path: Path) -> LabSpecsConfig:
    """Create lab specs that include percentage-vs-absolute sibling analytes."""

    config_path = tmp_path / "lab_specs_percentage.json"
    config_path.write_text(
        json.dumps(
            {
                "Blood - Monocytes": {
                    "primary_unit": "10⁹/L",
                    "lab_type": "blood",
                    "loinc_code": "742-7",
                    "ranges": {"default": [0.1, 0.8]},
                },
                "Blood - Monocytes (%)": {
                    "primary_unit": "%",
                    "lab_type": "blood",
                    "loinc_code": "5905-5",
                    "ranges": {"default": [2.0, 10.0]},
                },
                "Blood - Lymphocytes": {
                    "primary_unit": "10⁹/L",
                    "lab_type": "blood",
                    "loinc_code": "731-0",
                    "ranges": {"default": [1.0, 4.2]},
                },
                "Blood - Lymphocytes (%)": {
                    "primary_unit": "%",
                    "lab_type": "blood",
                    "loinc_code": "736-9",
                    "ranges": {"default": [19.0, 48.0]},
                },
                "Blood - Neutrophils": {
                    "primary_unit": "10⁹/L",
                    "lab_type": "blood",
                    "loinc_code": "751-8",
                    "ranges": {"default": [2.1, 7.6]},
                },
                "Blood - Neutrophils (%)": {
                    "primary_unit": "%",
                    "lab_type": "blood",
                    "loinc_code": "770-8",
                    "ranges": {"default": [40.0, 75.0]},
                },
                "Blood - Mystery Cells": {
                    "primary_unit": "10⁹/L",
                    "lab_type": "blood",
                    "loinc_code": "753-4",
                    "ranges": {"default": [0.0, 1.0]},
                },
            }
        ),
        encoding="utf-8",
    )
    return LabSpecsConfig(config_path=config_path)


def _make_serology_variant_lab_specs(tmp_path: Path) -> LabSpecsConfig:
    """Create lab specs for assays that expose quantitative and qualitative companions."""

    config_path = tmp_path / "lab_specs_serology.json"
    config_path.write_text(
        json.dumps(
            {
                "Blood - Anti-Cytomegalovirus IgG": {
                    "primary_unit": "IU/mL",
                    "lab_type": "blood",
                    "loinc_code": "13949-3",
                    "ranges": {"default": [0.0, 15.0]},
                },
                "Blood - Anti-Cytomegalovirus IgG (Qualitative)": {
                    "primary_unit": "boolean",
                    "lab_type": "blood",
                    "loinc_code": "5124-3",
                    "ranges": {"default": [0, 0]},
                },
            }
        ),
        encoding="utf-8",
    )
    return LabSpecsConfig(config_path=config_path)


def _stub_standardization(monkeypatch) -> None:
    """Stub cache-backed standardization so tests stay self-contained."""

    def fake_standardize_lab_names(name_contexts):
        return {context: "Blood - Glucose" for context in name_contexts}

    monkeypatch.setattr(
        "parselabs.rows.standardize_lab_names",
        fake_standardize_lab_names,
    )
    monkeypatch.setattr(
        "parselabs.rows.standardize_lab_units",
        lambda unit_contexts: {context: "mg/dL" for context in unit_contexts},
    )


def _stub_unknown_unit_standardization(monkeypatch) -> None:
    """Stub standardization with known lab names but unresolved units."""

    def fake_standardize_lab_names(name_contexts):
        return {context: "Blood - Glucose" for context in name_contexts}

    monkeypatch.setattr(
        "parselabs.rows.standardize_lab_names",
        fake_standardize_lab_names,
    )
    monkeypatch.setattr(
        "parselabs.rows.standardize_lab_units",
        lambda unit_contexts: {context: "$UNKNOWN$" for context in unit_contexts},
    )


def _stub_standardization_maps(
    monkeypatch,
    *,
    name_map: dict[str, str],
    unit_map: dict[tuple[str, str], str],
) -> None:
    """Stub cache-backed standardization with explicit row-level mappings."""

    def fake_standardize_lab_names(name_contexts):
        results = {}

        # Resolve context-aware keys first so mixed-section tests can disambiguate duplicate raw names.
        for raw_name, raw_section_name in name_contexts:
            if (raw_name, raw_section_name) in name_map:
                results[(raw_name, raw_section_name)] = name_map[(raw_name, raw_section_name)]
                continue

            results[(raw_name, raw_section_name)] = name_map.get(raw_name, "$UNKNOWN$")

        return results

    monkeypatch.setattr(
        "parselabs.rows.standardize_lab_names",
        fake_standardize_lab_names,
    )
    monkeypatch.setattr(
        "parselabs.rows.standardize_lab_units",
        lambda unit_contexts: {context: unit_map.get(context, "$UNKNOWN$") for context in unit_contexts},
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


def test_build_document_review_dataframe_preserves_bbox_fields(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _stub_standardization(monkeypatch)
    doc_dir = tmp_path / "processed" / "glucose_deadbeef"
    _write_processed_document(
        doc_dir,
        [None],
        bboxes=[
            {
                "bbox_left": 100.0,
                "bbox_top": 200.0,
                "bbox_right": 450.0,
                "bbox_bottom": 320.0,
            }
        ],
    )

    review_df = build_document_review_dataframe(doc_dir, lab_specs)

    assert review_df.loc[0, "bbox_left"] == 100.0
    assert review_df.loc[0, "bbox_top"] == 200.0
    assert review_df.loc[0, "bbox_right"] == 450.0
    assert review_df.loc[0, "bbox_bottom"] == 320.0


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


def test_save_review_status_can_clear_existing_decision(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _stub_standardization(monkeypatch)
    doc_dir = tmp_path / "processed" / "glucose_deadbeef"
    _write_processed_document(doc_dir, ["accepted", None])

    success, error = save_review_status(doc_dir, page_number=1, result_index=0, status=None)

    assert success is True
    assert error == ""

    rebuild_document_csv(doc_dir, lab_specs)
    review_df = build_document_review_dataframe(doc_dir, lab_specs)
    summary = get_review_summary(review_df)

    assert review_df["review_status"].fillna("").tolist() == ["", ""]
    assert summary.accepted == 0
    assert summary.pending == 2


def test_load_document_review_rows_normalizes_whitespace_and_invalid_statuses(tmp_path):
    doc_dir = tmp_path / "processed" / "glucose_deadbeef"
    _write_processed_document(doc_dir, [" ACCEPTED ", "nonsense"])

    review_df = load_document_review_rows(doc_dir)

    assert review_df["review_status"].fillna("").tolist() == ["accepted", ""]


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


def test_iter_processed_documents_skips_legacy_output_dirs(tmp_path):
    output_path = tmp_path / "processed"
    _write_processed_document(output_path / "legacy", ["accepted"], stem="legacy")
    _write_processed_document(output_path / "hashed_deadbeef", ["accepted"], stem="hashed")

    documents = iter_processed_documents(output_path)

    assert [document.doc_dir.name for document in documents] == ["hashed_deadbeef"]


def test_run_pipeline_and_reviewed_rebuild_produce_matching_exports(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _stub_standardization(monkeypatch)
    output_path = tmp_path / "processed"
    doc_dir = output_path / "glucose_deadbeef"
    _write_processed_document(doc_dir, ["accepted", "accepted"])
    csv_path = rebuild_document_csv(doc_dir, lab_specs)

    pdf_path = tmp_path / "input" / "glucose.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4")

    config = ExtractionConfig(
        input_path=pdf_path.parent,
        output_path=output_path,
        openrouter_api_key="test-key",
        extract_model_id="test-model",
        max_workers=1,
    )
    preflight = main.PdfPreflightResult(
        cached_csv_paths=[csv_path],
        pdfs_to_process=[],
        duplicates=[],
    )
    monkeypatch.setattr(main, "_prepare_pdf_run", lambda pdf_files, _: preflight)

    pipeline_result = main.run_pipeline_for_pdf_files([pdf_path], config, lab_specs)
    reviewed_df, _ = main.build_final_output_dataframe_from_reviewed_json(output_path, lab_specs)

    pd.testing.assert_frame_equal(pipeline_result.final_df, reviewed_df)


def test_run_pipeline_with_only_pending_rows_publishes_empty_final_export(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _stub_standardization(monkeypatch)
    output_path = tmp_path / "processed"
    doc_dir = output_path / "glucose_deadbeef"
    _write_processed_document(doc_dir, [None, None])
    csv_path = rebuild_document_csv(doc_dir, lab_specs)

    pdf_path = tmp_path / "input" / "glucose.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4")

    config = ExtractionConfig(
        input_path=pdf_path.parent,
        output_path=output_path,
        openrouter_api_key="test-key",
        extract_model_id="test-model",
        max_workers=1,
    )
    preflight = main.PdfPreflightResult(
        cached_csv_paths=[csv_path],
        pdfs_to_process=[],
        duplicates=[],
    )
    monkeypatch.setattr(main, "_prepare_pdf_run", lambda pdf_files, _: preflight)

    pipeline_result = main.run_pipeline_for_pdf_files([pdf_path], config, lab_specs)

    assert pipeline_result.csv_paths == [csv_path]
    assert pipeline_result.final_df.empty


def test_build_document_review_dataframe_flags_unknown_units_for_review(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _stub_unknown_unit_standardization(monkeypatch)
    doc_dir = tmp_path / "processed" / "glucose_deadbeef"
    _write_processed_document(doc_dir, [None], raw_units=["odd-unit"])

    review_df = build_document_review_dataframe(doc_dir, lab_specs)

    assert bool(review_df.loc[0, "review_needed"]) is True
    assert "UNKNOWN_UNIT_MAPPING" in str(review_df.loc[0, "review_reason"])


def test_apply_cached_standardization_infers_safe_missing_primary_units(tmp_path, monkeypatch):
    config_path = tmp_path / "lab_specs.json"
    config_path.write_text(
        json.dumps(
            {
                "Urine Type II - pH": {
                    "primary_unit": "pH",
                    "lab_type": "urine",
                    "loinc_code": "5803-2",
                    "ranges": {"default": [5.0, 8.0]},
                }
            }
        ),
        encoding="utf-8",
    )
    lab_specs = LabSpecsConfig(config_path=config_path)

    monkeypatch.setattr(
        "parselabs.rows.standardize_lab_names",
        lambda name_contexts: {context: "Urine Type II - pH" for context in name_contexts},
    )
    monkeypatch.setattr(
        "parselabs.rows.standardize_lab_units",
        lambda unit_contexts: {context: "$UNKNOWN$" for context in unit_contexts},
    )

    review_df = pd.DataFrame(
        [
            {
                "raw_lab_name": "pH",
                "raw_lab_unit": "",
            }
        ]
    )

    standardized_df = apply_cached_standardization(review_df, lab_specs)

    assert standardized_df.loc[0, "lab_name_standardized"] == "Urine Type II - pH"
    assert standardized_df.loc[0, "lab_unit_standardized"] == "pH"


def test_apply_cached_standardization_uses_raw_section_name_to_disambiguate_mixed_sections(tmp_path, monkeypatch):
    lab_specs = _make_mixed_section_lab_specs(tmp_path)
    _stub_standardization_maps(
        monkeypatch,
        name_map={
            ("Glicose", "Bioquímica"): "Blood - Glucose (Fasting)",
            ("Glicose", "Elementos anormais"): "Urine Type II - Glucose",
            ("LEUCOCITOS", "Hemograma"): "Blood - Leukocytes",
            ("LEUCOCITOS", "Sedimento urinário"): "Urine Type II - Sediment - Leukocytes",
        },
        unit_map={
            ("mg/dL", "Blood - Glucose (Fasting)"): "mg/dL",
            ("", "Urine Type II - Glucose"): "boolean",
            ("/campo", "Urine Type II - Sediment - Leukocytes"): "/field",
            ("10^9/L", "Blood - Leukocytes"): "10⁹/L",
        },
    )

    review_df = pd.DataFrame(
        [
            {
                "raw_lab_name": "Glicose",
                "raw_section_name": "Elementos anormais",
                "raw_lab_unit": "",
            },
            {
                "raw_lab_name": "LEUCOCITOS",
                "raw_section_name": "Sedimento urinário",
                "raw_lab_unit": "/campo",
            },
            {
                "raw_lab_name": "Glicose",
                "raw_section_name": "Bioquímica",
                "raw_lab_unit": "mg/dL",
            },
            {
                "raw_lab_name": "LEUCOCITOS",
                "raw_section_name": "Hemograma",
                "raw_lab_unit": "10^9/L",
            },
        ]
    )

    standardized_df = apply_cached_standardization(review_df, lab_specs)

    assert standardized_df["lab_name_standardized"].tolist() == [
        "Urine Type II - Glucose",
        "Urine Type II - Sediment - Leukocytes",
        "Blood - Glucose (Fasting)",
        "Blood - Leukocytes",
    ]


def test_apply_cached_standardization_prefers_variant_specific_companion_keys(tmp_path, monkeypatch):
    lab_specs = _make_serology_variant_lab_specs(tmp_path)
    _stub_standardization_maps(
        monkeypatch,
        name_map={
            ("AC. ANTI-CITOMEGALOVIRUS IgG", "SEROLOGIA INFECCIOSA"): "Blood - Anti-Cytomegalovirus IgG",
            ("AC. ANTI-CITOMEGALOVIRUS IgG (quantitative)", "SEROLOGIA INFECCIOSA"): "Blood - Anti-Cytomegalovirus IgG",
            ("AC. ANTI-CITOMEGALOVIRUS IgG (qualitative)", "SEROLOGIA INFECCIOSA"): "Blood - Anti-Cytomegalovirus IgG (Qualitative)",
        },
        unit_map={
            ("UI/ml", "Blood - Anti-Cytomegalovirus IgG"): "IU/mL",
            ("", "Blood - Anti-Cytomegalovirus IgG (Qualitative)"): "boolean",
        },
    )

    review_df = pd.DataFrame(
        [
            {
                "page_number": 3,
                "raw_lab_name": "AC. ANTI-CITOMEGALOVIRUS IgG",
                "raw_section_name": "SEROLOGIA INFECCIOSA",
                "raw_value": "0,4",
                "raw_lab_unit": "UI/ml",
            },
            {
                "page_number": 3,
                "raw_lab_name": "AC. ANTI-CITOMEGALOVIRUS IgG",
                "raw_section_name": "SEROLOGIA INFECCIOSA",
                "raw_value": "Negativo",
                "raw_lab_unit": None,
            },
        ]
    )

    standardized_df = apply_cached_standardization(review_df, lab_specs)

    assert standardized_df["lab_name_standardized"].tolist() == [
        "Blood - Anti-Cytomegalovirus IgG",
        "Blood - Anti-Cytomegalovirus IgG (Qualitative)",
    ]
    assert standardized_df["lab_unit_standardized"].tolist() == ["IU/mL", "boolean"]


def test_backfill_missing_raw_sections_infers_urine_page_context():
    review_df = pd.DataFrame(
        [
            {"page_number": 3, "raw_lab_name": "Cor", "raw_section_name": None},
            {"page_number": 3, "raw_lab_name": "pH", "raw_section_name": None},
            {"page_number": 3, "raw_lab_name": "Glicose", "raw_section_name": None},
            {"page_number": 3, "raw_lab_name": "LEUCOCITOS", "raw_section_name": None},
            {"page_number": 3, "raw_lab_name": "Antigenio HBs", "raw_section_name": None},
            {"page_number": 1, "raw_lab_name": "Leucocitos", "raw_section_name": None},
        ]
    )

    enriched_df = rows_module._backfill_missing_raw_sections(review_df)

    assert enriched_df.loc[0, "raw_section_name"] == "< TIPO II >"
    assert enriched_df.loc[1, "raw_section_name"] == "< TIPO II >"
    assert enriched_df.loc[2, "raw_section_name"] == "Elementos anormais"
    assert enriched_df.loc[3, "raw_section_name"] == "EXAME MICROSCOPICO DO SEDIMENTO"
    assert pd.isna(enriched_df.loc[4, "raw_section_name"])
    assert pd.isna(enriched_df.loc[5, "raw_section_name"])


def test_build_document_review_dataframe_infers_missing_sections_for_urine_disambiguation(tmp_path, monkeypatch):
    lab_specs = _make_mixed_section_lab_specs(tmp_path)
    _stub_standardization_maps(
        monkeypatch,
        name_map={
            ("Cor", "< TIPO II >"): "Urine Type II - Color",
            ("pH", "< TIPO II >"): "Urine Type II - pH",
            ("Glicose", "Elementos anormais"): "Urine Type II - Glucose",
            ("LEUCOCITOS", "EXAME MICROSCOPICO DO SEDIMENTO"): "Urine Type II - Sediment - Leukocytes",
        },
        unit_map={
            ("", "Urine Type II - Color"): "boolean",
            ("", "Urine Type II - pH"): "pH",
            ("", "Urine Type II - Glucose"): "boolean",
            ("/campo", "Urine Type II - Sediment - Leukocytes"): "/field",
        },
    )

    doc_dir = tmp_path / "processed" / "urine_deadbeef"
    _write_processed_document(
        doc_dir,
        [None, None, None, None],
        stem="urine",
        raw_names=["Cor", "pH", "Glicose", "LEUCOCITOS"],
        raw_values=["AMARELA", "6.0", "NAO CONTEM", "1 a 2"],
        raw_units=[None, None, None, "/campo"],
    )

    review_df = build_document_review_dataframe(doc_dir, lab_specs)

    assert review_df.loc[0, "raw_section_name"] == "< TIPO II >"
    assert review_df.loc[0, "lab_name_standardized"] == "Urine Type II - Color (Qualitative)"
    assert review_df.loc[2, "raw_section_name"] == "Elementos anormais"
    assert review_df.loc[2, "lab_name_standardized"] == "Urine Type II - Glucose (Qualitative)"
    assert review_df.loc[3, "raw_section_name"] == "EXAME MICROSCOPICO DO SEDIMENTO"
    assert review_df.loc[3, "lab_name_standardized"] == "Urine Type II - Sediment - Leukocytes"


def test_build_document_review_dataframe_preserves_raw_section_name(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _stub_standardization(monkeypatch)
    doc_dir = tmp_path / "processed" / "glucose_deadbeef"
    _write_processed_document(
        doc_dir,
        [None],
        raw_sections=["Bioquímica"],
    )

    review_df = build_document_review_dataframe(doc_dir, lab_specs)

    assert review_df.loc[0, "raw_section_name"] == "Bioquímica"


def test_apply_cached_standardization_remaps_percent_units_to_percentage_variants(tmp_path, monkeypatch):
    lab_specs = _make_percentage_variant_lab_specs(tmp_path)
    _stub_standardization_maps(
        monkeypatch,
        name_map={
            "Monocitos": "Blood - Monocytes",
            "Linfocitos": "Blood - Lymphocytes",
            "Neutrofilos": "Blood - Neutrophils",
        },
        unit_map={
            ("%", "Blood - Monocytes"): "%",
            ("%", "Blood - Lymphocytes"): "%",
            ("%", "Blood - Neutrophils"): "%",
        },
    )

    review_df = pd.DataFrame(
        [
            {"raw_lab_name": "Monocitos", "raw_lab_unit": "%"},
            {"raw_lab_name": "Linfocitos", "raw_lab_unit": "%"},
            {"raw_lab_name": "Neutrofilos", "raw_lab_unit": "%"},
        ]
    )

    standardized_df = apply_cached_standardization(review_df, lab_specs)

    assert standardized_df["lab_name_standardized"].tolist() == [
        "Blood - Monocytes (%)",
        "Blood - Lymphocytes (%)",
        "Blood - Neutrophils (%)",
    ]
    assert standardized_df["lab_unit_standardized"].tolist() == ["%", "%", "%"]


def test_apply_cached_standardization_remaps_absolute_units_to_non_percentage_variants(tmp_path, monkeypatch):
    lab_specs = _make_percentage_variant_lab_specs(tmp_path)
    _stub_standardization_maps(
        monkeypatch,
        name_map={
            "Monocitos (%)": "Blood - Monocytes (%)",
            "Linfocitos (%)": "Blood - Lymphocytes (%)",
        },
        unit_map={
            ("10^9/L", "Blood - Monocytes (%)"): "10⁹/L",
            ("/mm3", "Blood - Lymphocytes (%)"): "/mm3",
        },
    )

    review_df = pd.DataFrame(
        [
            {"raw_lab_name": "Monocitos (%)", "raw_lab_unit": "10^9/L"},
            {"raw_lab_name": "Linfocitos (%)", "raw_lab_unit": "/mm3"},
        ]
    )

    standardized_df = apply_cached_standardization(review_df, lab_specs)

    assert standardized_df["lab_name_standardized"].tolist() == [
        "Blood - Monocytes",
        "Blood - Lymphocytes",
    ]
    assert standardized_df["lab_unit_standardized"].tolist() == ["10⁹/L", "10⁹/L"]


def test_apply_cached_standardization_infers_missing_absolute_units_from_reference_ranges(tmp_path, monkeypatch):
    lab_specs = _make_percentage_variant_lab_specs(tmp_path)
    _stub_standardization_maps(
        monkeypatch,
        name_map={
            ("Neutrófilos", "Leucograma"): "Blood - Neutrophils (%)",
        },
        unit_map={
            ("%", "Blood - Neutrophils (%)"): "%",
            ("", "Blood - Neutrophils (%)"): "%",
        },
    )

    review_df = pd.DataFrame(
        [
            {
                "raw_lab_name": "Neutrófilos",
                "raw_section_name": "Leucograma",
                "raw_lab_unit": "",
                "raw_reference_min": 1.5,
                "raw_reference_max": 6.9,
                "raw_value": "3.6",
            },
            {
                "raw_lab_name": "Neutrófilos",
                "raw_section_name": "Leucograma",
                "raw_lab_unit": "%",
                "raw_reference_min": 40.0,
                "raw_reference_max": 75.0,
                "raw_value": "66.9",
            },
        ]
    )

    standardized_df = apply_cached_standardization(review_df, lab_specs)

    assert standardized_df["lab_name_standardized"].tolist() == [
        "Blood - Neutrophils",
        "Blood - Neutrophils (%)",
    ]
    assert standardized_df["lab_unit_standardized"].tolist() == ["10⁹/L", "%"]


def test_apply_cached_standardization_overrides_stale_percentage_unit_cache_for_absolute_units(tmp_path, monkeypatch):
    lab_specs = _make_percentage_variant_lab_specs(tmp_path)
    _stub_standardization_maps(
        monkeypatch,
        name_map={
            ("Neutrófilos", "Fórmula Leucocitária"): "Blood - Neutrophils (%)",
        },
        unit_map={
            ("x10E3/µl", "Blood - Neutrophils (%)"): "%",
        },
    )

    review_df = pd.DataFrame(
        [
            {
                "raw_lab_name": "Neutrófilos",
                "raw_section_name": "Fórmula Leucocitária",
                "raw_lab_unit": "x10E3/µl",
            }
        ]
    )

    standardized_df = apply_cached_standardization(review_df, lab_specs)

    assert standardized_df["lab_name_standardized"].tolist() == ["Blood - Neutrophils"]
    assert standardized_df["lab_unit_standardized"].tolist() == ["10⁹/L"]


def test_apply_cached_standardization_preserves_mm3_units_for_conversion(tmp_path, monkeypatch):
    config_path = tmp_path / "lab_specs_counts.json"
    config_path.write_text(
        json.dumps(
            {
                "Blood - Lymphocytes": {
                    "primary_unit": "10⁹/L",
                    "lab_type": "blood",
                    "loinc_code": "731-0",
                    "alternatives": [
                        {"unit": "/mm3", "factor": 0.001},
                    ],
                    "ranges": {"default": [1.0, 3.5]},
                }
            }
        ),
        encoding="utf-8",
    )
    lab_specs = LabSpecsConfig(config_path=config_path)
    _stub_standardization_maps(
        monkeypatch,
        name_map={"Linfócitos": "Blood - Lymphocytes"},
        unit_map={
            ("/mm3", "Blood - Lymphocytes"): "10⁹/L",
        },
    )

    review_df = pd.DataFrame(
        [
            {
                "raw_lab_name": "Linfócitos",
                "raw_comments": None,
                "raw_lab_unit": "/mm3",
                "raw_value": "1350.0",
                "raw_reference_min": 1000.0,
                "raw_reference_max": 3500.0,
            }
        ]
    )

    standardized_df = apply_cached_standardization(review_df, lab_specs)
    normalized_df = apply_normalizations(standardized_df, lab_specs)

    assert standardized_df["lab_unit_standardized"].tolist() == ["/mm3"]
    assert normalized_df["lab_unit_primary"].tolist() == ["10⁹/L"]
    assert normalized_df["value_primary"].tolist() == [1.35]


def test_flag_percentage_variant_ambiguity_ignores_single_variant_labs(tmp_path):
    config_path = tmp_path / "lab_specs.json"
    config_path.write_text(
        json.dumps(
            {
                "Blood - Hematocrit (HCT) (%)": {
                    "primary_unit": "%",
                    "lab_type": "blood",
                    "loinc_code": "4544-3",
                    "alternatives": [{"unit": "L/L", "factor": 100.0}],
                    "ranges": {"default": [40.0, 50.0]},
                }
            }
        ),
        encoding="utf-8",
    )
    lab_specs = LabSpecsConfig(config_path=config_path)

    review_df = pd.DataFrame(
        [
            {
                "lab_name_standardized": "Blood - Hematocrit (HCT) (%)",
                "lab_unit_standardized": "L/L",
            }
        ]
    )

    flagged_df = rows_module._flag_percentage_variant_ambiguity(review_df, lab_specs)

    assert "review_reason" not in flagged_df.columns or flagged_df.loc[0, "review_reason"] in ("", None)


def test_apply_cached_standardization_leaves_correct_unknown_and_orphan_rows_unchanged(tmp_path, monkeypatch):
    lab_specs = _make_percentage_variant_lab_specs(tmp_path)
    _stub_standardization_maps(
        monkeypatch,
        name_map={
            "Monocitos (%)": "Blood - Monocytes (%)",
            "Monocitos": "Blood - Monocytes",
            "Mystery Cells": "Blood - Mystery Cells",
        },
        unit_map={
            ("%", "Blood - Monocytes (%)"): "%",
            ("", "Blood - Monocytes"): "$UNKNOWN$",
            ("%", "Blood - Mystery Cells"): "%",
        },
    )

    review_df = pd.DataFrame(
        [
            {"raw_lab_name": "Monocitos (%)", "raw_lab_unit": "%"},
            {"raw_lab_name": "Monocitos", "raw_lab_unit": ""},
            {"raw_lab_name": "Mystery Cells", "raw_lab_unit": "%"},
        ]
    )

    standardized_df = apply_cached_standardization(review_df, lab_specs)

    assert standardized_df["lab_name_standardized"].tolist() == [
        "Blood - Monocytes (%)",
        "Blood - Monocytes",
        "Blood - Mystery Cells",
    ]
    assert standardized_df["lab_unit_standardized"].tolist() == ["%", "$UNKNOWN$", "%"]


def test_build_document_review_dataframe_separates_percentage_and_absolute_variants(tmp_path, monkeypatch):
    lab_specs = _make_percentage_variant_lab_specs(tmp_path)
    _stub_standardization_maps(
        monkeypatch,
        name_map={"Monocitos": "Blood - Monocytes"},
        unit_map={
            ("%", "Blood - Monocytes"): "%",
            ("10^9/L", "Blood - Monocytes"): "10⁹/L",
        },
    )
    doc_dir = tmp_path / "processed" / "monocytes_deadbeef"
    _write_processed_document(
        doc_dir,
        [None, None],
        stem="monocytes",
        raw_names=["Monocitos", "Monocitos"],
        raw_values=["6.0", "0.354"],
        raw_units=["%", "10^9/L"],
        raw_reference_mins=[2.0, 0.1],
        raw_reference_maxs=[10.0, 0.8],
    )

    review_df = build_document_review_dataframe(doc_dir, lab_specs)

    assert review_df["lab_name_standardized"].tolist() == [
        "Blood - Monocytes (%)",
        "Blood - Monocytes",
    ]
    assert review_df["lab_name"].tolist() == [
        "Blood - Monocytes (%)",
        "Blood - Monocytes",
    ]
    assert review_df["review_reason"].fillna("").tolist() == ["", ""]


def test_build_document_review_dataframe_keeps_extraction_failures_visible(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _stub_standardization(monkeypatch)
    doc_dir = tmp_path / "processed" / "glucose_deadbeef"
    doc_dir.mkdir(parents=True)
    (doc_dir / "glucose.pdf").write_bytes(b"%PDF-1.4")
    payload = {
        "collection_date": "2024-01-05",
        "_extraction_failed": True,
        "_failure_reason": "Malformed output after 2 attempts",
        "lab_results": [
            {
                "raw_lab_name": "[EXTRACTION FAILED]",
                "raw_value": None,
                "raw_lab_unit": None,
                "raw_reference_min": None,
                "raw_reference_max": None,
                "raw_comments": "Malformed output after 2 attempts",
            }
        ],
    }
    (doc_dir / "glucose.001.json").write_text(json.dumps(payload), encoding="utf-8")

    review_df = build_document_review_dataframe(doc_dir, lab_specs)

    assert bool(review_df.loc[0, "review_needed"]) is True
    assert "EXTRACTION_FAILED" in str(review_df.loc[0, "review_reason"])


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

    regression_cases.approve_cases(argparse.Namespace(command="sync-reviewed", profile="test"))

    case_dirs = sorted(path.name for path in approved_dir.iterdir())
    expected_case_id = f"valid_{regression_cases.compute_file_hash(output_path / 'valid_deadbeef' / 'valid.pdf')}"
    assert case_dirs == [expected_case_id]
    assert (approved_dir / expected_case_id / "expected.csv").exists()
    review_state_path = approved_dir / expected_case_id / "review_state.json"
    assert review_state_path.exists()
    review_state = json.loads(review_state_path.read_text(encoding="utf-8"))
    assert review_state["rows"] == [
        {"page_number": 1, "result_index": 0, "review_status": "accepted"},
        {"page_number": 1, "result_index": 1, "review_status": "rejected"},
    ]
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
        "parselabs.rows.standardize_lab_names",
        lambda name_contexts: {
            context: ("Blood - Glucose" if context[0] == "Glucose" else "$UNKNOWN$")
            for context in name_contexts
        },
    )
    monkeypatch.setattr(
        "parselabs.rows.standardize_lab_units",
        lambda unit_contexts: {context: ("mg/dL" if context[0] == "mg/dL" else "$UNKNOWN$") for context in unit_contexts},
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
