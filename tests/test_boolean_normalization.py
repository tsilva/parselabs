import pandas as pd

from parselabs.normalization import apply_normalizations

from tests.test_urine_qualitative_variants import _make_lab_specs


def test_apply_normalizations_leaves_unmapped_boolean_values_empty(tmp_path):
    lab_specs = _make_lab_specs(tmp_path)
    df = pd.DataFrame(
        [
            {
                "raw_lab_name": "Nitritos",
                "raw_value": "DUVIDOSO",
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

    assert pd.isna(normalized_df.loc[0, "value_primary"])
    assert normalized_df.loc[0, "lab_unit_primary"] == "boolean"
