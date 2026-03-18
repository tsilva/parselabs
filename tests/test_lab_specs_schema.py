import json

from parselabs.config import LabSpecsConfig
from utils.validate_lab_specs_schema import LabSpecsValidator


def test_lab_specs_validator_allows_one_sided_ranges(tmp_path):
    config_path = tmp_path / "lab_specs.json"
    config_path.write_text(
        json.dumps(
            {
                "Blood - Example": {
                    "lab_type": "blood",
                    "primary_unit": "mg/dL",
                    "alternatives": [],
                    "ranges": {"default": [None, 99.0]},
                    "loinc_code": "1234-5",
                },
                "_relationships": [],
            }
        ),
        encoding="utf-8",
    )

    validator = LabSpecsValidator(config_path)

    assert validator.validate() is True
    assert validator.errors == []


def test_lab_specs_validator_rejects_ranges_with_both_bounds_missing(tmp_path):
    config_path = tmp_path / "lab_specs.json"
    config_path.write_text(
        json.dumps(
            {
                "Blood - Example": {
                    "lab_type": "blood",
                    "primary_unit": "mg/dL",
                    "alternatives": [],
                    "ranges": {"default": [None, None]},
                    "loinc_code": "1234-5",
                },
                "_relationships": [],
            }
        ),
        encoding="utf-8",
    )

    validator = LabSpecsValidator(config_path)

    assert validator.validate() is False
    assert any("must define at least one bound" in error for error in validator.errors)


def test_lab_specs_config_returns_one_sided_optimal_range(tmp_path):
    config_path = tmp_path / "lab_specs.json"
    config_path.write_text(
        json.dumps(
            {
                "Blood - Example": {
                    "lab_type": "blood",
                    "primary_unit": "mg/dL",
                    "alternatives": [],
                    "ranges": {"default": [60.0, None]},
                    "loinc_code": "1234-5",
                },
                "_relationships": [],
            }
        ),
        encoding="utf-8",
    )

    lab_specs = LabSpecsConfig(config_path)

    assert lab_specs.get_optimal_range_for_demographics("Blood - Example") == (60.0, None)
