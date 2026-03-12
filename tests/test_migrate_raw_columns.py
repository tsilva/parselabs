import json

import pandas as pd

from utils import migrate_raw_columns


def test_migrate_json_file_renames_legacy_raw_keys(tmp_path):
    json_path = tmp_path / "page.json"
    json_path.write_text(
        json.dumps(
            {
                "lab_results": [
                    {
                        "lab_name_raw": "Glucose",
                        "value_raw": "90",
                        "lab_unit_raw": "mg/dL",
                        "reference_min_raw": 70,
                        "reference_max_raw": 100,
                        "comments": "fasting",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    changed = migrate_raw_columns._migrate_json_file(json_path, dry_run=False)
    migrated = json.loads(json_path.read_text(encoding="utf-8"))

    assert changed is True
    assert migrated["lab_results"][0] == {
        "raw_lab_name": "Glucose",
        "raw_value": "90",
        "raw_lab_unit": "mg/dL",
        "raw_reference_min": 70,
        "raw_reference_max": 100,
        "raw_comments": "fasting",
    }


def test_migrate_csv_file_renames_legacy_export_columns(tmp_path):
    csv_path = tmp_path / "all.csv"
    pd.DataFrame(
        [
            {
                "lab_name_raw": "Glucose",
                "value_raw": "90",
                "unit_raw": "mg/dL",
            }
        ]
    ).to_csv(csv_path, index=False, encoding="utf-8")

    changed = migrate_raw_columns._migrate_csv_file(
        csv_path,
        migrate_raw_columns._CSV_EXPORT_RENAMES,
        dry_run=False,
    )
    migrated = pd.read_csv(csv_path, keep_default_na=False)

    assert changed is True
    assert migrated.columns.tolist() == ["raw_lab_name", "raw_value", "raw_unit"]
