import pandas as pd

from parselabs.dataset import _check_lab_unit_percent_value_range


def test_percent_value_range_allows_prothrombin_time_percent_above_hundred():
    report: dict[str, list[str]] = {}
    df = pd.DataFrame(
        [
            {
                "source_file": "pt.csv",
                "lab_name": "Blood - Prothrombin Time (PT) (%)",
                "lab_unit": "%",
                "value": 109.0,
            },
            {
                "source_file": "other.csv",
                "lab_name": "Blood - Hematocrit (HCT) (%)",
                "lab_unit": "%",
                "value": 109.0,
            },
        ]
    )

    _check_lab_unit_percent_value_range(df, report)

    assert "pt.csv" not in report
    assert report["other.csv"] == [
        'Row at index 1 (lab_name="Blood - Hematocrit (HCT) (%)") has unit="%" but value=109.0 (should be between 0 and 100)'
    ]


def test_distinct_textual_assay_findings_are_not_duplicate_observations():
    from parselabs.dataset import _check_unique_date_lab_name

    frame = pd.DataFrame([
        {'date': '2024-01-01', 'lab_name': 'Urine - Synthetic', 'value_text': 'Finding A'},
        {'date': '2024-01-01', 'lab_name': 'Urine - Synthetic', 'value_text': 'Finding B'},
    ])
    report = {}
    _check_unique_date_lab_name(frame, report)
    assert not report
    _check_unique_date_lab_name(pd.concat([frame, frame.iloc[[0]]], ignore_index=True), report)
    assert report
