import warnings
from pathlib import Path

from PIL import Image
from pydantic.warnings import PydanticDeprecatedSince211

from parselabs.extraction import (
    TOOLS,
    HealthLabReport,
    HealthLabReportExtraction,
    LabResult,
    _fix_lab_results_format,
    _validate_extraction_result,
)
from parselabs.utils import create_page_image_variants


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
            "raw_lab_name: Hemoglobina, raw_section_name: Hemograma, raw_value: 14.2, raw_lab_unit: g/dL, raw_reference_range: 12.0 - 16.0, raw_reference_min: 12.0, raw_reference_max: 16.0"
        ]
    }

    fixed_payload = _fix_lab_results_format(payload)

    assert fixed_payload["lab_results"] == [
        {
            "raw_lab_name": "Hemoglobina",
            "raw_section_name": "Hemograma",
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


def test_normalize_empty_optionals_clears_empty_strings_without_pydantic_deprecation():
    report = HealthLabReport(
        collection_date="2024-01-01",
        source_file="report.pdf",
        lab_results=[
            LabResult(
                raw_lab_name="Glucose",
                raw_section_name="",
                raw_value="",
                raw_lab_unit="",
                raw_reference_range="",
                raw_comments="",
            )
        ],
    )

    # Escalate the old instance-level field access warning so this regression stays fixed.
    with warnings.catch_warnings():
        warnings.simplefilter("error", PydanticDeprecatedSince211)
        report.normalize_empty_optionals()

    assert report.lab_results[0].raw_value is None
    assert report.lab_results[0].raw_section_name is None
    assert report.lab_results[0].raw_lab_unit is None
    assert report.lab_results[0].raw_reference_range is None
    assert report.lab_results[0].raw_comments is None


def test_raw_section_name_survives_model_round_trip():
    report = HealthLabReport(
        collection_date="2024-01-01",
        lab_results=[
            LabResult(
                raw_lab_name="Glicose",
                raw_section_name="Elementos anormais",
                raw_value="NAO CONTEM",
                raw_lab_unit=None,
            )
        ],
    )

    payload = report.model_dump(mode="json")

    assert payload["lab_results"][0]["raw_section_name"] == "Elementos anormais"


def test_llm_tool_schema_field_shape_remains_stable():
    schema = HealthLabReportExtraction.model_json_schema()
    tool_schema = TOOLS[0]["function"]["parameters"]
    lab_result_schema = schema["$defs"]["LabResultExtraction"]["properties"]

    assert tool_schema == schema
    assert list(schema["properties"]) == [
        "collection_date",
        "report_date",
        "lab_facility",
        "page_has_lab_data",
        "lab_results",
    ]
    assert list(lab_result_schema) == [
        "raw_lab_name",
        "raw_section_name",
        "raw_value",
        "raw_lab_unit",
        "raw_reference_range",
        "raw_reference_min",
        "raw_reference_max",
        "raw_comments",
        "bbox_left",
        "bbox_top",
        "bbox_right",
        "bbox_bottom",
    ]
    assert "pattern" not in schema["properties"]["collection_date"]


def test_validate_extraction_result_retries_when_any_bbox_coordinate_is_missing():
    payload = {
        "collection_date": "2024-01-01",
        "lab_results": [
            {
                "raw_lab_name": "Glucose",
                "raw_value": "92",
                "raw_lab_unit": "mg/dL",
                "bbox_left": 100,
                "bbox_top": 200,
                "bbox_right": 400,
                "bbox_bottom": None,
            }
        ],
    }

    result = _validate_extraction_result(
        tool_result_dict=payload,
        image_path=Path("page.jpg"),
        attempt=0,
        current_temp=0.0,
    )

    assert result["success"] is False
    assert result["should_retry"] is True
    assert "bounding boxes" in result["error_msg"]


def test_validate_extraction_result_accepts_rows_with_complete_bboxes():
    payload = {
        "collection_date": "2024-01-01",
        "lab_results": [
            {
                "raw_lab_name": "Glucose",
                "raw_value": "92",
                "raw_lab_unit": "mg/dL",
                "bbox_left": 100,
                "bbox_top": 200,
                "bbox_right": 400,
                "bbox_bottom": 320,
            }
        ],
    }

    result = _validate_extraction_result(
        tool_result_dict=payload,
        image_path=Path("page.jpg"),
        attempt=0,
        current_temp=0.0,
    )

    assert result["success"] is True
    assert result["should_retry"] is False
    assert result["data"]["lab_results"][0]["bbox_bottom"] == 320.0


def test_validate_extraction_result_retries_for_column_like_bbox():
    payload = {
        "collection_date": "2024-01-01",
        "lab_results": [
            {
                "raw_lab_name": "Eritrócitos",
                "raw_value": "3.5",
                "raw_lab_unit": "10¹²/L",
                "bbox_left": 390,
                "bbox_top": 90,
                "bbox_right": 405,
                "bbox_bottom": 755,
            }
        ],
    }

    result = _validate_extraction_result(
        tool_result_dict=payload,
        image_path=Path("page.jpg"),
        attempt=0,
        current_temp=0.0,
    )

    assert result["success"] is False
    assert result["should_retry"] is True
    assert "column-like bounding box" in result["error_msg"]


def test_validate_extraction_result_retries_for_clustered_column_like_bboxes():
    payload = {
        "collection_date": "2024-01-01",
        "lab_results": [
            {
                "raw_lab_name": "Eritrócitos",
                "raw_value": "3.5",
                "raw_lab_unit": "10¹²/L",
                "bbox_left": 390,
                "bbox_top": 90,
                "bbox_right": 405,
                "bbox_bottom": 755,
            },
            {
                "raw_lab_name": "Hemoglobina",
                "raw_value": "10.6",
                "raw_lab_unit": "g/dL",
                "bbox_left": 405,
                "bbox_top": 90,
                "bbox_right": 420,
                "bbox_bottom": 755,
            },
            {
                "raw_lab_name": "Hematócrito",
                "raw_value": "31.8",
                "raw_lab_unit": "%",
                "bbox_left": 420,
                "bbox_top": 90,
                "bbox_right": 435,
                "bbox_bottom": 755,
            },
        ],
    }

    result = _validate_extraction_result(
        tool_result_dict=payload,
        image_path=Path("page.jpg"),
        attempt=0,
        current_temp=0.0,
    )

    assert result["success"] is False
    assert result["should_retry"] is True
    assert "clustered column-like bounding boxes" in result["error_msg"]


def test_create_page_image_variants_adds_padding_before_processing():
    image = Image.new("RGB", (100, 50), "black")

    variants = create_page_image_variants(image)

    assert variants["primary"].size == (228, 178)
    assert variants["fallback"].size == (228, 178)
    assert variants["primary"].getpixel((0, 0)) == (255, 255, 255)
    assert variants["fallback"].getpixel((0, 0)) == 255
