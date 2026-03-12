from parselabs.extraction import _fix_lab_results_format


def test_fix_lab_results_format_recovers_stringified_json_items():
    payload = {
        "collection_date": "27/12/2001",
        "lab_results": [
            '{"raw_lab_name": "Glicose", "raw_value": "92", "raw_lab_unit": "mg/dL"}',
            '{"raw_lab_name": "Colesterol Total", "raw_value": "180", "raw_lab_unit": "mg/dL"}',
        ],
    }

    fixed_payload = _fix_lab_results_format(payload)

    assert fixed_payload["collection_date"] == "2001-12-27"
    assert fixed_payload["lab_results"] == [
        {"raw_lab_name": "Glicose", "raw_value": "92", "raw_lab_unit": "mg/dL"},
        {"raw_lab_name": "Colesterol Total", "raw_value": "180", "raw_lab_unit": "mg/dL"},
    ]


def test_fix_lab_results_format_recovers_label_packed_string_items():
    payload = {
        "lab_results": [
            "raw_lab_name: Hemoglobina, raw_value: 14.2, raw_lab_unit: g/dL, raw_reference_range: 12.0 - 16.0, raw_reference_min: 12.0, raw_reference_max: 16.0"
        ]
    }

    fixed_payload = _fix_lab_results_format(payload)

    assert fixed_payload["lab_results"] == [
        {
            "raw_lab_name": "Hemoglobina",
            "raw_value": "14.2",
            "raw_lab_unit": "g/dL",
            "raw_reference_range": "12.0 - 16.0",
            "raw_reference_min": 12.0,
            "raw_reference_max": 16.0,
        }
    ]


def test_fix_lab_results_format_recovers_stringified_top_level_lab_results():
    payload = {
        "lab_results": '[{"raw_lab_name": "Trigliceridos", "raw_value": "120", "raw_unit": "mg/dL"}]'
    }

    fixed_payload = _fix_lab_results_format(payload)

    assert fixed_payload["lab_results"] == [
        {"raw_lab_name": "Trigliceridos", "raw_value": "120", "raw_lab_unit": "mg/dL"}
    ]


def test_fix_lab_results_format_cleans_partial_bbox_fields():
    payload = {
        "lab_results": [
            {
                "raw_lab_name": "Glucose",
                "raw_value": "92",
                "raw_lab_unit": "mg/dL",
                "bbox_left": "120",
                "bbox_top": "240",
                "bbox_right": "480",
            },
            {
                "raw_lab_name": "Hemoglobin",
                "raw_value": "14.2",
                "raw_lab_unit": "g/dL",
                "bbox_left": "100",
                "bbox_top": "200",
                "bbox_right": "400",
                "bbox_bottom": "320",
            },
        ]
    }

    fixed_payload = _fix_lab_results_format(payload)

    assert fixed_payload["lab_results"][0]["bbox_left"] is None
    assert fixed_payload["lab_results"][0]["bbox_top"] is None
    assert fixed_payload["lab_results"][0]["bbox_right"] is None
    assert fixed_payload["lab_results"][0]["bbox_bottom"] is None
    assert fixed_payload["lab_results"][1]["bbox_left"] == 100.0
    assert fixed_payload["lab_results"][1]["bbox_top"] == 200.0
    assert fixed_payload["lab_results"][1]["bbox_right"] == 400.0
    assert fixed_payload["lab_results"][1]["bbox_bottom"] == 320.0
