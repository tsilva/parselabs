import json

import pandas as pd

from parselabs.config import LabSpecsConfig
from parselabs.normalization import apply_normalizations


def _make_lab_specs(tmp_path):
    config_path = tmp_path / "lab_specs.json"
    config_path.write_text(
        json.dumps(
            {
                "Urine Type II - Proteins": {
                    "lab_type": "urine",
                    "primary_unit": "mg/dL",
                    "alternatives": [{"unit": "mg/dl", "factor": 1.0}],
                    "ranges": {"default": [0, 15]},
                    "loinc_code": "2888-6",
                },
                "Urine Type II - Proteins, Qualitative": {
                    "lab_type": "urine",
                    "primary_unit": "boolean",
                    "alternatives": [],
                    "ranges": {"default": [0, 0]},
                    "loinc_code": "2888-6-qual",
                },
                "Urine Type II - Nitrites": {
                    "lab_type": "urine",
                    "primary_unit": "boolean",
                    "alternatives": [],
                    "ranges": {"default": [0, 0]},
                    "loinc_code": "5802-9",
                },
                "Urine Type II - Blood": {
                    "lab_type": "urine",
                    "primary_unit": "boolean",
                    "alternatives": [],
                    "ranges": {"default": [0, 0]},
                    "loinc_code": "5794-8",
                },
                "Urine Type II - Color": {
                    "lab_type": "urine",
                    "primary_unit": "boolean",
                    "alternatives": [],
                    "ranges": {"default": [0, 0]},
                    "loinc_code": "5778-1",
                },
                "Urine Type II - Ketones": {
                    "lab_type": "urine",
                    "primary_unit": "mg/dL",
                    "alternatives": [{"unit": "mg/dl", "factor": 1.0}],
                    "ranges": {"default": [0, 0]},
                    "loinc_code": "2514-8",
                },
                "Urine Type II - Ketones, Qualitative": {
                    "lab_type": "urine",
                    "primary_unit": "boolean",
                    "alternatives": [],
                    "ranges": {"default": [0, 0]},
                    "loinc_code": "2514-8-qual",
                },
                "Urine Type II - Sediment - Epithelial Cells": {
                    "lab_type": "urine",
                    "primary_unit": "/field",
                    "alternatives": [{"unit": "/campo", "factor": 1.0}],
                    "ranges": {"default": [0, 5]},
                    "loinc_code": "5787-2",
                },
                "Urine Type II - Sediment - Erythrocytes": {
                    "lab_type": "urine",
                    "primary_unit": "/field",
                    "alternatives": [{"unit": "/campo", "factor": 1.0}],
                    "ranges": {"default": [0, 2]},
                    "loinc_code": "5807-8",
                },
                "Urine Type II - Sediment - Leukocytes": {
                    "lab_type": "urine",
                    "primary_unit": "/field",
                    "alternatives": [{"unit": "/campo", "factor": 1.0}],
                    "ranges": {"default": [0, 5]},
                    "loinc_code": "5821-9",
                },
            }
        ),
        encoding="utf-8",
    )
    return LabSpecsConfig(config_path=config_path)


def test_apply_normalizations_remaps_text_urine_result_to_boolean_variant(tmp_path):
    lab_specs = _make_lab_specs(tmp_path)
    df = pd.DataFrame(
        [
            {
                "raw_lab_name": "Proteinas",
                "raw_value": "NEGATIVO",
                "raw_comments": None,
                "raw_lab_unit": "",
                "raw_reference_min": None,
                "raw_reference_max": None,
                "lab_name_standardized": "Urine Type II - Proteins",
                "lab_unit_standardized": None,
            }
        ]
    )

    normalized_df = apply_normalizations(df, lab_specs)

    assert normalized_df.loc[0, "lab_name_standardized"] == "Urine Type II - Proteins (Qualitative)"
    assert normalized_df.loc[0, "lab_unit_standardized"] == "boolean"
    assert normalized_df.loc[0, "lab_unit_primary"] == "boolean"
    assert normalized_df.loc[0, "value_primary"] == 0


def test_apply_normalizations_canonicalizes_boolean_nitrites_name(tmp_path):
    lab_specs = _make_lab_specs(tmp_path)
    df = pd.DataFrame(
        [
            {
                "raw_lab_name": "Nitritos",
                "raw_value": "NEGATIVO",
                "raw_comments": None,
                "raw_lab_unit": "",
                "raw_reference_min": None,
                "raw_reference_max": None,
                "lab_name_standardized": "Urine Type II - Nitrites",
                "lab_unit_standardized": "boolean",
            }
        ]
    )

    normalized_df = apply_normalizations(df, lab_specs)

    assert normalized_df.loc[0, "lab_name_standardized"] == "Urine Type II - Nitrites (Qualitative)"
    assert normalized_df.loc[0, "lab_unit_primary"] == "boolean"
    assert normalized_df.loc[0, "value_primary"] == 0


def test_lab_specs_resolves_legacy_and_canonical_qualitative_names(tmp_path):
    lab_specs = _make_lab_specs(tmp_path)

    assert lab_specs.get_primary_unit("Urine Type II - Proteins, Qualitative") == "boolean"
    assert lab_specs.get_primary_unit("Urine Type II - Proteins (Qualitative)") == "boolean"
    assert lab_specs.get_canonical_lab_name("Urine Type II - Proteins, Qualitative") == "Urine Type II - Proteins (Qualitative)"


def test_apply_normalizations_keeps_numeric_urine_result_on_numeric_lab(tmp_path):
    lab_specs = _make_lab_specs(tmp_path)
    df = pd.DataFrame(
        [
            {
                "raw_lab_name": "Proteinas",
                "raw_value": "30",
                "raw_comments": None,
                "raw_section_name": "Bioquímica",
                "raw_lab_unit": "mg/dl",
                "raw_reference_min": 0,
                "raw_reference_max": 15,
                "lab_name_standardized": "Urine Type II - Proteins",
                "lab_unit_standardized": "mg/dl",
            }
        ]
    )

    normalized_df = apply_normalizations(df, lab_specs)

    assert normalized_df.loc[0, "lab_name_standardized"] == "Urine Type II - Proteins"
    assert normalized_df.loc[0, "lab_unit_primary"] == "mg/dL"
    assert normalized_df.loc[0, "value_primary"] == 30


def test_apply_normalizations_remaps_numeric_urine_strip_ketones_to_qualitative_variant(tmp_path):
    lab_specs = _make_lab_specs(tmp_path)
    df = pd.DataFrame(
        [
            {
                "raw_lab_name": "Corpos Cetónicos",
                "raw_value": "10",
                "raw_comments": None,
                "raw_section_name": "ANÁLISE SUMÁRIA DA URINA",
                "raw_lab_unit": "mg/dl",
                "raw_reference_min": None,
                "raw_reference_max": None,
                "lab_name_standardized": "Urine Type II - Ketones",
                "lab_unit_standardized": "mg/dL",
            }
        ]
    )

    normalized_df = apply_normalizations(df, lab_specs)

    assert normalized_df.loc[0, "lab_name_standardized"] == "Urine Type II - Ketones (Qualitative)"
    assert normalized_df.loc[0, "lab_unit_standardized"] == "boolean"
    assert normalized_df.loc[0, "lab_unit_primary"] == "boolean"
    assert normalized_df.loc[0, "value_primary"] == 1


def test_apply_normalizations_remaps_numeric_urine_strip_blood_to_qualitative_variant(tmp_path):
    lab_specs = _make_lab_specs(tmp_path)
    df = pd.DataFrame(
        [
            {
                "raw_lab_name": "Hemoglobina",
                "raw_value": "0,00",
                "raw_comments": None,
                "raw_section_name": "Exame Sumário",
                "raw_lab_unit": "mg/dL",
                "raw_reference_min": None,
                "raw_reference_max": None,
                "lab_name_standardized": "Urine Type II - Blood",
                "lab_unit_standardized": "mg/dL",
            }
        ]
    )

    normalized_df = apply_normalizations(df, lab_specs)

    assert normalized_df.loc[0, "lab_name_standardized"] == "Urine Type II - Blood (Qualitative)"
    assert normalized_df.loc[0, "lab_unit_standardized"] == "boolean"
    assert normalized_df.loc[0, "lab_unit_primary"] == "boolean"
    assert normalized_df.loc[0, "value_primary"] == 0


def test_apply_normalizations_classifies_urine_color_as_boolean(tmp_path):
    lab_specs = _make_lab_specs(tmp_path)
    df = pd.DataFrame(
        [
            {
                "raw_lab_name": "Cor",
                "raw_value": "AMARELA",
                "raw_comments": None,
                "raw_lab_unit": "",
                "raw_reference_min": None,
                "raw_reference_max": None,
                "lab_name_standardized": "Urine Type II - Color",
                "lab_unit_standardized": None,
            }
        ]
    )

    normalized_df = apply_normalizations(df, lab_specs)

    assert normalized_df.loc[0, "lab_name_standardized"] == "Urine Type II - Color (Qualitative)"
    assert normalized_df.loc[0, "lab_unit_standardized"] == "boolean"
    assert normalized_df.loc[0, "lab_unit_primary"] == "boolean"
    assert normalized_df.loc[0, "value_primary"] == 0


def test_apply_normalizations_uses_midpoint_for_interval_style_microscopy_values(tmp_path):
    lab_specs = _make_lab_specs(tmp_path)
    df = pd.DataFrame(
        [
            {
                "raw_lab_name": "Células epiteliais",
                "raw_value": "1 a 2/campo",
                "raw_comments": None,
                "raw_lab_unit": "/campo",
                "raw_reference_min": 0,
                "raw_reference_max": 5,
                "lab_name_standardized": "Urine Type II - Sediment - Epithelial Cells",
                "lab_unit_standardized": "/campo",
            }
        ]
    )

    normalized_df = apply_normalizations(df, lab_specs)

    assert normalized_df.loc[0, "raw_value"] == "1 a 2/campo"
    assert normalized_df.loc[0, "value_primary"] == 1.5
    assert normalized_df.loc[0, "lab_unit_primary"] == "/field"


def test_apply_normalizations_uses_midpoint_for_plain_interval_values(tmp_path):
    lab_specs = _make_lab_specs(tmp_path)
    df = pd.DataFrame(
        [
            {
                "raw_lab_name": "Células epiteliais",
                "raw_value": "0 - 2",
                "raw_comments": None,
                "raw_lab_unit": "/campo",
                "raw_reference_min": 0,
                "raw_reference_max": 5,
                "lab_name_standardized": "Urine Type II - Sediment - Epithelial Cells",
                "lab_unit_standardized": "/campo",
            }
        ]
    )

    normalized_df = apply_normalizations(df, lab_specs)

    assert normalized_df.loc[0, "value_primary"] == 1.0


def test_apply_normalizations_handles_interval_values_in_string_dtype_columns(tmp_path):
    lab_specs = _make_lab_specs(tmp_path)
    df = pd.DataFrame(
        [
            {
                "raw_lab_name": "Células epiteliais",
                "raw_value": "1 a 2/campo",
                "raw_comments": None,
                "raw_lab_unit": "/campo",
                "raw_reference_min": 0,
                "raw_reference_max": 5,
                "lab_name_standardized": "Urine Type II - Sediment - Epithelial Cells",
                "lab_unit_standardized": "/campo",
            }
        ]
    ).astype(
        {
            "raw_lab_name": "string",
            "raw_value": "string",
            "raw_lab_unit": "string",
            "lab_name_standardized": "string",
            "lab_unit_standardized": "string",
        }
    )

    normalized_df = apply_normalizations(df, lab_specs)

    assert normalized_df.loc[0, "value_primary"] == 1.5


def test_apply_normalizations_remaps_field_style_sediment_rows_out_of_mislabeled_ul_units(tmp_path):
    lab_specs = _make_lab_specs(tmp_path)
    df = pd.DataFrame(
        [
            {
                "raw_lab_name": "Leucócitos",
                "raw_value": "1-5 / campo",
                "raw_comments": None,
                "raw_section_name": "Sedimento Urinário",
                "raw_lab_unit": "/µl",
                "raw_reference_min": None,
                "raw_reference_max": None,
                "lab_name_standardized": "Urine Type II - Sediment - Leukocytes",
                "lab_unit_standardized": "/ul",
            },
            {
                "raw_lab_name": "Células Epiteliais de Descamação",
                "raw_value": "Raras",
                "raw_comments": None,
                "raw_section_name": "Sedimento Urinário",
                "raw_lab_unit": "/µl",
                "raw_reference_min": None,
                "raw_reference_max": None,
                "lab_name_standardized": "Urine Type II - Sediment - Epithelial Cells",
                "lab_unit_standardized": "/ul",
            },
        ]
    )

    normalized_df = apply_normalizations(df, lab_specs)

    assert normalized_df.loc[0, "lab_unit_standardized"] == "/campo"
    assert normalized_df.loc[0, "lab_unit_primary"] == "/field"
    assert normalized_df.loc[0, "value_primary"] == 3.0

    assert normalized_df.loc[1, "lab_unit_standardized"] == "/campo"
    assert normalized_df.loc[1, "lab_unit_primary"] == "/field"
    assert pd.isna(normalized_df.loc[1, "value_primary"])
