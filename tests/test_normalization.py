import pandas as pd

from parselabs.config import LabSpecsConfig
from parselabs.normalization import apply_normalizations, flag_duplicate_entries, preprocess_numeric_value


def test_preprocess_numeric_value_collapses_decimal_spacing():
    assert preprocess_numeric_value("13 .0") == "13.0"
    assert preprocess_numeric_value("4 ,04") == "4.04"
    assert preprocess_numeric_value(" 0 , 60 ") == "0.60"
    assert preprocess_numeric_value("Ca. 15") == "15"


def test_classify_qualitative_value_handles_accented_negative_tokens():
    from parselabs.normalization import classify_qualitative_value

    assert classify_qualitative_value("Límpido") == 0
    assert classify_qualitative_value("Não detetado") == 0
    assert classify_qualitative_value("Cultura estéril após incubação") == 0
    assert classify_qualitative_value("Amarela ouro") == 0


def test_classify_qualitative_value_handles_positive_marker_tokens():
    from parselabs.normalization import classify_qualitative_value

    assert classify_qualitative_value("(*)") == 1
    assert classify_qualitative_value("1+") == 1
    assert classify_qualitative_value("Raras células") == 1


def test_flag_duplicate_entries_ignores_equivalent_dual_unit_rows():
    review_df = pd.DataFrame(
        [
            {
                "date": "2024-01-05",
                "lab_name_standardized": "Blood - 25-OH Vitamin D",
                "value_primary": 35.0,
                "lab_unit_primary": "ng/mL",
                "raw_value": "35.0",
                "is_below_limit": False,
                "is_above_limit": False,
                "review_needed": False,
                "review_reason": "",
            },
            {
                "date": "2024-01-05",
                "lab_name_standardized": "Blood - 25-OH Vitamin D",
                "value_primary": 35.04,
                "lab_unit_primary": "ng/mL",
                "raw_value": "87.6",
                "is_below_limit": False,
                "is_above_limit": False,
                "review_needed": False,
                "review_reason": "",
            },
        ]
    )

    flagged_df = flag_duplicate_entries(review_df)

    assert flagged_df["review_reason"].fillna("").tolist() == ["", ""]
    assert flagged_df["review_needed"].tolist() == [False, False]


def test_flag_duplicate_entries_keeps_conflicting_rows_flagged():
    review_df = pd.DataFrame(
        [
            {
                "date": "2024-01-05",
                "lab_name_standardized": "Blood - Creatinine",
                "value_primary": 0.98,
                "lab_unit_primary": "mg/dL",
                "raw_value": "0.98",
                "is_below_limit": False,
                "is_above_limit": False,
                "review_needed": False,
                "review_reason": "",
            },
            {
                "date": "2024-01-05",
                "lab_name_standardized": "Blood - Creatinine",
                "value_primary": 1.35,
                "lab_unit_primary": "mg/dL",
                "raw_value": "1.35",
                "is_below_limit": False,
                "is_above_limit": False,
                "review_needed": False,
                "review_reason": "",
            },
        ]
    )

    flagged_df = flag_duplicate_entries(review_df)

    assert all("DUPLICATE_ENTRY" in reason for reason in flagged_df["review_reason"].fillna(""))
    assert flagged_df["review_needed"].tolist() == [True, True]


def test_apply_normalizations_recovers_field_style_units_for_more_sediment_labs(tmp_path):
    config_path = tmp_path / "lab_specs.json"
    config_path.write_text(
        """
        {
          "Urine Type II - Sediment - Crystals": {
            "primary_unit": "/field",
            "lab_type": "urine",
            "loinc_code": "5776-5",
            "alternatives": [{"unit": "/campo", "factor": 1.0}],
            "ranges": {"default": [0, 0]}
          },
          "Urine Type II - Sediment - Small Round Cells": {
            "primary_unit": "/field",
            "lab_type": "urine",
            "loinc_code": "20409-9",
            "alternatives": [{"unit": "/campo", "factor": 1.0}],
            "ranges": {"default": [0, 0]}
          }
        }
        """,
        encoding="utf-8",
    )
    lab_specs = LabSpecsConfig(config_path=config_path)
    review_df = pd.DataFrame(
        [
            {
                "lab_name_standardized": "Urine Type II - Sediment - Crystals",
                "lab_unit_standardized": "/ul",
                "raw_section_name": "Exame Microscópico do Sedimento",
                "raw_value": "Raros (0 - 2 / campo)",
                "raw_reference_min": None,
                "raw_reference_max": None,
                "raw_comments": None,
            },
            {
                "lab_name_standardized": "Urine Type II - Sediment - Small Round Cells",
                "lab_unit_standardized": "/µl",
                "raw_section_name": "Exame Microscópico do Sedimento",
                "raw_value": "Raras",
                "raw_reference_min": None,
                "raw_reference_max": None,
                "raw_comments": None,
            },
        ]
    )

    normalized_df = apply_normalizations(review_df, lab_specs)

    assert normalized_df["lab_unit_standardized"].tolist() == ["/campo", "/campo"]


def test_apply_normalizations_remaps_absolute_sediment_rows_to_absolute_siblings(tmp_path):
    config_path = tmp_path / "lab_specs.json"
    config_path.write_text(
        """
        {
          "Urine Type II - Sediment - Leukocytes": {
            "primary_unit": "/field",
            "lab_type": "urine",
            "loinc_code": "5821-9",
            "alternatives": [{"unit": "/campo", "factor": 1.0}],
            "ranges": {"default": [0, 5]}
          },
          "Urine Type II - Sediment - Leukocytes (Absolute)": {
            "primary_unit": "/µL",
            "lab_type": "urine",
            "loinc_code": "5821-9-ABS",
            "alternatives": [{"unit": "/ul", "factor": 1.0}, {"unit": "/µl", "factor": 1.0}],
            "ranges": {"default": [0, 28]}
          }
        }
        """,
        encoding="utf-8",
    )
    lab_specs = LabSpecsConfig(config_path=config_path)
    review_df = pd.DataFrame(
        [
            {
                "lab_name_standardized": "Urine Type II - Sediment - Leukocytes",
                "lab_unit_standardized": "/ul",
                "raw_section_name": "Sedimento urinário",
                "raw_value": "17",
                "raw_reference_min": None,
                "raw_reference_max": None,
                "raw_comments": None,
            },
            {
                "lab_name_standardized": "Urine Type II - Sediment - Leukocytes",
                "lab_unit_standardized": "/ul",
                "raw_section_name": "Sedimento urinário",
                "raw_value": "Raros (1 a 4 / campo)",
                "raw_reference_min": None,
                "raw_reference_max": None,
                "raw_comments": None,
            },
        ]
    )

    normalized_df = apply_normalizations(review_df, lab_specs)

    assert normalized_df["lab_name_standardized"].tolist() == [
        "Urine Type II - Sediment - Leukocytes (Absolute)",
        "Urine Type II - Sediment - Leukocytes",
    ]
    assert normalized_df["lab_unit_standardized"].tolist() == ["/µL", "/campo"]
