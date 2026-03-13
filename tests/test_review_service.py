from __future__ import annotations

import json

from parselabs.store import apply_review_action


def test_apply_review_action_persists_accept_and_missing_row(tmp_path):
    doc_dir = tmp_path / "glucose_deadbeef"
    doc_dir.mkdir(parents=True)
    payload = {
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
