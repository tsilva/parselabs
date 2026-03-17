import pandas as pd

from parselabs.normalization import flag_duplicate_entries, preprocess_numeric_value


def test_preprocess_numeric_value_collapses_decimal_spacing():
    assert preprocess_numeric_value("13 .0") == "13.0"
    assert preprocess_numeric_value("4 ,04") == "4.04"
    assert preprocess_numeric_value(" 0 , 60 ") == "0.60"


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
