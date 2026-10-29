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
    assert classify_qualitative_value("Não reativo") == 0
    assert classify_qualitative_value("Cultura estéril após incubação") == 0
    assert classify_qualitative_value("Amarela ouro") == 0


def test_classify_qualitative_value_handles_positive_marker_tokens():
    from parselabs.normalization import classify_qualitative_value

    assert classify_qualitative_value("(*)") == 1
    assert classify_qualitative_value("1+") == 1
    assert classify_qualitative_value("Raras células") == 1
    assert classify_qualitative_value("IMUNE") == 1


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


def test_sediment_rarity_does_not_invent_a_per_field_unit():
    frame = pd.DataFrame([{
        "lab_name_standardized": "Urine Type II - Sediment - Epithelial Cells",
        "lab_unit_standardized": "/µl", "raw_lab_unit": "/µl",
        "raw_section_name": "Sedimento Urinário", "raw_value": "Raras",
        "raw_reference_min": None, "raw_reference_max": None, "raw_comments": None,
    }])
    result = apply_normalizations(frame, LabSpecsConfig())
    assert result.lab_name_standardized.tolist() == ["Urine Type II - Sediment - Epithelial Cells (Absolute)"]
    assert result.lab_unit_primary.tolist() == ["/µL"]
    assert result.value_primary.isna().all()
    assert result.value_text.tolist() == ["Raras"]


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


def _source_result(name, value, unit, reference=None, lower=None, upper=None):
    return {
        'lab_name_standardized': name, 'lab_unit_standardized': unit,
        'raw_lab_name': 'Synthetic assay', 'raw_value': value, 'raw_lab_unit': None,
        'raw_reference_range': reference, 'raw_reference_min': lower, 'raw_reference_max': upper,
        'raw_comments': None,
    }


def test_textual_results_preserve_color_and_sediment_without_invented_counts():
    specs = LabSpecsConfig()
    frame = pd.DataFrame([
        _source_result('Urine Type II - Color', 'Amarela clara', 'boolean'),
        _source_result('Urine Type II - Sediment - Leukocytes', 'Raros (< 5 /campo)', '/field'),
    ])
    normalized = apply_normalizations(frame, specs)
    assert normalized.value_text.tolist() == ['Amarela clara', 'Raros (< 5 /campo)']
    assert pd.isna(normalized.iloc[0].value_primary)
    assert normalized.iloc[0].lab_unit_primary == 'text'
    assert pd.isna(normalized.iloc[1].value_primary)


def test_negative_quantitative_assay_does_not_become_zero_concentration():
    frame = pd.DataFrame([_source_result('Blood - Glucose (Fasting)', 'NEGATIVO', 'mg/dL')])
    result = apply_normalizations(frame, LabSpecsConfig()).iloc[0]
    assert pd.isna(result.value_primary)
    assert result.value_text == 'NEGATIVO'
    assert result.lab_unit_primary == 'mg/dL'


def test_qualitative_crp_uses_its_own_boolean_assay():
    frame = pd.DataFrame([_source_result('Blood - C-Reactive Protein (CRP)', 'NEGATIVA', 'mg/L')])
    result = apply_normalizations(frame, LabSpecsConfig()).iloc[0]
    assert result.lab_name_standardized == 'Blood - C-Reactive Protein (CRP) (Qualitative)'
    assert result.lab_unit_primary == 'boolean'
    assert result.value_primary == 0
    assert result.value_text == 'NEGATIVA'


def test_semquantitative_urine_grade_is_preserved_and_classified_as_presence():
    frame = pd.DataFrame([_source_result('Urine Type II - Urobilinogen', 'Contém(+--)', 'mg/dL')])
    result = apply_normalizations(frame, LabSpecsConfig()).iloc[0]
    assert result.lab_name_standardized == 'Urine Type II - Urobilinogen (Qualitative)'
    assert result.value_primary == 1
    assert result.value_text == 'Contém(+--)'
    assert result.lab_unit_primary == 'boolean'


def test_qualitative_ana_does_not_reuse_titer_scale_or_reference_denominator():
    frame = pd.DataFrame([_source_result(
        'Blood - Antinuclear Antibodies (ANA)', 'Negativo [AC 0]', 'index',
        'Título: <1/160', upper=160,
    )])
    result = apply_normalizations(frame, LabSpecsConfig()).iloc[0]
    assert result.lab_name_standardized == 'Blood - Antinuclear Antibodies (ANA) (Qualitative)'
    assert result.lab_unit_primary == 'boolean'
    assert result.value_primary == 0
    assert result.value_text == 'Negativo [AC 0]'
    assert pd.isna(result.reference_max_primary)


def test_blood_groups_are_nominal_text_even_when_rh_is_positive():
    frame = pd.DataFrame([
        _source_result('Blood - ABO Group', 'AB', 'boolean'),
        _source_result('Blood - Rh Type', 'Positivo', 'boolean'),
    ])
    normalized = apply_normalizations(frame, LabSpecsConfig())
    assert normalized.value_primary.isna().all()
    assert normalized.value_text.tolist() == ['AB', 'Positivo']
    assert normalized.lab_unit_primary.tolist() == ['text', 'text']


def test_two_hour_sedimentation_distinguishes_distance_from_explicit_hourly_rate():
    distance = _source_result(
        'Blood - Erythrocyte Sedimentation Rate (ESR) - 2h', '27', 'mm/h', upper=17,
    )
    distance['raw_lab_unit'] = 'mm'
    rate = _source_result(
        'Blood - Erythrocyte Sedimentation Rate (ESR) - 2h', '50', 'mm/h', upper=30,
    )
    rate['raw_lab_unit'] = 'mm/h'
    normalized = apply_normalizations(pd.DataFrame([distance, rate]), LabSpecsConfig())
    assert normalized.value_primary.tolist() == [27, 100]
    assert normalized.reference_max_primary.tolist() == [17, 60]
    assert normalized.lab_unit_primary.tolist() == ['mm/(2.h)', 'mm/(2.h)']
    assert LabSpecsConfig().get_loinc_code(distance['lab_name_standardized']) == '18184-2'


def test_normal_urobilinogen_does_not_assert_absence():
    result = apply_normalizations(pd.DataFrame([
        _source_result('Urine Type II - Urobilinogen (Qualitative)', 'Normal', 'boolean'),
    ]), LabSpecsConfig()).iloc[0]
    assert pd.isna(result.value_primary)
    assert result.value_text == 'Normal'
    assert result.lab_unit_primary == 'text'


def test_specific_gravity_four_digit_notation_preserves_decimal_ratios():
    frame = pd.DataFrame([
        _source_result('Urine Type II - Specific Gravity', '1012', 'unitless', lower=1003, upper=1030),
        _source_result('Urine Type II - Specific Gravity', '1.025', 'unitless', lower=1.003, upper=1.030),
    ])
    normalized = apply_normalizations(frame, LabSpecsConfig())
    assert normalized.value_primary.tolist() == [1.012, 1.025]
    assert normalized.reference_min_primary.tolist() == [1.003, 1.003]
    assert normalized.reference_max_primary.tolist() == [1.030, 1.030]
    assert normalized.raw_value.tolist() == ['1012', '1.025']


def test_numeric_interpretations_and_approximations_remain_visible():
    frame = pd.DataFrame([
        _source_result('Blood - Anti-Scleroderma-70 Antibody (Anti-Scl-70 Ab)', '9= NR', 'U'),
        _source_result('Urine Type II - Sediment - Leukocytes (Absolute)', 'Ca. 125', '/µL'),
        _source_result('Blood - Glucose', '1e2', 'mg/dL'),
        _source_result('Urine - Bacteriological - Colony Count', '< 100 000', 'CFU/mL'),
    ])
    normalized = apply_normalizations(frame, LabSpecsConfig())
    assert normalized.iloc[0].value_primary == 9
    assert normalized.iloc[0].value_text == '9= NR'
    assert normalized.iloc[0].lab_unit_primary == 'U'
    assert normalized.iloc[1].value_primary == 125
    assert normalized.iloc[1].value_text == 'Ca. 125'
    assert pd.isna(normalized.iloc[2].value_text)
    assert normalized.iloc[3].value_primary == 100000
    assert pd.isna(normalized.iloc[3].value_text)


def test_titer_limits_use_dilution_denominators_and_standalone_reference_bounds():
    frame = pd.DataFrame([
        _source_result('Blood - Antinuclear Antibodies (ANA)', '<1/80', 'index', '< 1/160'),
        _source_result('Blood - Widal (S. Typhi O)', 'TITULO 1/40', 'boolean', '(Titulo com significado > 1/80)', lower=80),
    ])
    normalized = apply_normalizations(frame, LabSpecsConfig())
    assert normalized.value_primary.tolist() == [80, 40]
    assert normalized.lab_unit_primary.tolist() == ['titer', 'titer']
    assert normalized.value_text.tolist() == ['<1/80', 'TITULO 1/40']
    assert normalized.iloc[0].is_below_limit
    assert normalized.iloc[0].reference_max_primary == 160
    assert pd.isna(normalized.iloc[1].reference_min_primary)
    assert pd.isna(normalized.iloc[1].reference_max_primary)


def test_arbitrary_numeric_fractions_are_not_assumed_to_be_titers():
    frame = pd.DataFrame([_source_result('Blood - Glucose (Fasting)', '1/80', 'mg/dL')])
    result = apply_normalizations(frame, LabSpecsConfig()).iloc[0]
    assert pd.isna(result.value_primary)
    assert result.value_text == '1/80'


def test_antibody_qualitative_siblings_do_not_reuse_quantitative_identifiers():
    names = [
        'Blood - Myeloperoxidase Anti-Neutrophil Cytoplasmic Antibodies (MPO-ANCA)',
        'Blood - Anti-Proteinase 3 Antibody IgG (Anti-PR3 Ab)',
    ]
    frame = pd.DataFrame([_source_result(name, 'Negativo', 'unitless') for name in names])
    normalized = apply_normalizations(frame, LabSpecsConfig())
    assert normalized.lab_name_standardized.tolist() == [name + ' (Qualitative)' for name in names]
    assert normalized.value_primary.tolist() == [0, 0]
    assert normalized.lab_unit_primary.tolist() == ['boolean', 'boolean']


def test_distinct_textual_findings_survive_deduplication():
    from parselabs.normalization import deduplicate_results

    frame = pd.DataFrame([
        {'date': '2024-01-01', 'lab_name_standardized': 'Urine - Synthetic', 'value_text': 'Finding A'},
        {'date': '2024-01-01', 'lab_name_standardized': 'Urine - Synthetic', 'value_text': 'Finding B'},
        {'date': '2024-01-01', 'lab_name_standardized': 'Urine - Synthetic', 'value_text': 'Finding A'},
    ])
    assert deduplicate_results(frame, LabSpecsConfig()).value_text.tolist() == ['Finding A', 'Finding B']


def test_separately_numbered_samples_survive_equal_result_deduplication():
    from parselabs.normalization import deduplicate_results

    frame = pd.DataFrame([
        {"date": "2000-01-01", "lab_name_standardized": "Feces - Fecal Occult Blood (FOBT)",
         "raw_lab_name": name, "lab_unit_standardized": "boolean", "value_text": "Negativo"}
        for name in ["1ª Amostra", "2ª Amostra", "3ª Amostra", "1ª Amostra"]
    ])
    actual = deduplicate_results(frame, LabSpecsConfig())
    assert len(actual) == 3
    assert set(actual.raw_lab_name) == {"1ª Amostra", "2ª Amostra", "3ª Amostra"}


def test_reported_intervals_remain_text_without_invented_numeric_midpoints():
    raw = ["0-4", "1 a 3/campo", "2 - 8 / campo"]
    frame = pd.DataFrame([{
        "lab_name_standardized": "Urine Type II - Sediment - Leukocytes",
        "lab_unit_standardized": "/campo", "raw_lab_unit": "/campo",
        "raw_section_name": "Sedimento", "raw_value": value,
        "raw_reference_min": None, "raw_reference_max": None, "raw_comments": None,
    } for value in raw])
    actual = apply_normalizations(frame, LabSpecsConfig())
    assert actual.value_primary.isna().all()
    assert actual.value_text.tolist() == raw


def test_literal_sediment_volume_unit_overrides_cached_field_unit():
    frame = pd.DataFrame([{
        "lab_name_standardized": "Urine Type II - Sediment - Epithelial Cells",
        "lab_unit_standardized": "/field", "raw_lab_unit": "/µl",
        "raw_section_name": "Sedimento", "raw_value": "Raras",
        "raw_reference_min": None, "raw_reference_max": None, "raw_comments": None,
    }])
    actual = apply_normalizations(frame, LabSpecsConfig()).iloc[0]
    assert actual.lab_unit_primary == "/µL"
    assert pd.isna(actual.value_primary)
    assert actual.value_text == "Raras"
