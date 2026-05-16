from __future__ import annotations

import json

from parselabs.store import apply_review_action
from parselabs.types import PagePayload


def test_apply_review_action_persists_accept_and_missing_row(tmp_path):
    doc_dir = tmp_path / "glucose_deadbeef"
    doc_dir.mkdir(parents=True)
    payload: PagePayload = {
        "lab_results": [
            {
                "raw_lab_name": "Glucose",
                "raw_value": "92",
            }
        ]
    }
    json_path = doc_dir / "glucose.001.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    success, error = apply_review_action(doc_dir, 1, 0, "accept")
    assert success is True
    assert error == ""

    success, error = apply_review_action(doc_dir, 1, 0, "missing_row")
    assert success is True
    assert error == ""

    updated_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert updated_payload["lab_results"][0]["review_status"] == "accepted"
    assert "review_completed_at" in updated_payload["lab_results"][0]
    assert updated_payload["review_missing_rows"][0]["anchor_result_index"] == 0


def test_apply_review_action_deduplicates_missing_row_marker_for_same_anchor(tmp_path):
    doc_dir = tmp_path / "glucose_deadbeef"
    doc_dir.mkdir(parents=True)
    payload: PagePayload = {
        "lab_results": [
            {
                "raw_lab_name": "Glucose",
                "raw_value": "92",
            }
        ],
        "review_missing_rows": [
            {
                "anchor_result_index": 0,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ],
    }
    json_path = doc_dir / "glucose.001.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    success, error = apply_review_action(doc_dir, 1, 0, "missing_row")

    updated_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert success is True
    assert error == ""
    assert updated_payload["review_missing_rows"] == payload["review_missing_rows"]
