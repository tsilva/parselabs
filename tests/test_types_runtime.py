from __future__ import annotations

import json
from pathlib import Path

from parselabs import review_artifacts_backend
from parselabs.store import read_page_payload, write_page_payload
from parselabs.types import (
    ExtractionFailureRecord,
    PagePayload,
    ReviewDecisionResult,
    ReviewMissingRowMarker,
    ReviewRow,
    RowIdentity,
    build_row_identity_token,
    parse_row_identity_token,
)


def test_page_payload_round_trips_through_store(tmp_path):
    missing_marker: ReviewMissingRowMarker = {
        "anchor_result_index": 0,
        "created_at": "2024-01-02T03:04:05Z",
    }
    payload: PagePayload = {
        "collection_date": "2024-01-02",
        "source_file": "labs.pdf",
        "page_number": 1,
        "review_missing_rows": [missing_marker],
        "lab_results": [
            {
                "raw_lab_name": "Glucose",
                "raw_value": "95",
                "bbox_left": 100.0,
                "bbox_top": 200.0,
                "bbox_right": 450.0,
                "bbox_bottom": 320.0,
                "review_status": "accepted",
            }
        ],
    }
    json_path = tmp_path / "page.json"

    write_page_payload(json_path, payload)

    assert read_page_payload(json_path) == payload


def test_row_identity_shape_stays_stable():
    identity: RowIdentity = {
        "source_file": "labs.pdf",
        "page_number": 3,
        "result_index": 7,
    }
    review_row: ReviewRow = {
        **identity,
        "raw_lab_name": "Glucose",
        "review_status": "accepted",
    }

    assert review_row["source_file"] == "labs.pdf"
    assert review_row["page_number"] == 3
    assert review_row["result_index"] == 7


def test_extraction_failure_record_shape_stays_stable():
    failure: ExtractionFailureRecord = {
        "page": "labs page 3",
        "reason": "Malformed output after 2 attempts",
    }

    assert failure == {
        "page": "labs page 3",
        "reason": "Malformed output after 2 attempts",
    }


def test_row_identity_token_round_trip():
    identity: RowIdentity = {
        "source_file": "labs.pdf",
        "page_number": 3,
        "result_index": 7,
    }

    token = build_row_identity_token(identity)

    assert token == '["labs.pdf",3,7]'
    assert parse_row_identity_token(token) == identity
    assert parse_row_identity_token(json.dumps(token)) == identity


def test_review_decision_result_shape_from_backend(tmp_path):
    output_path = tmp_path / "output"
    doc_dir = output_path / "doc_12345678"
    doc_dir.mkdir(parents=True)
    row = review_artifacts_backend.PendingReviewRow(
        doc_dir=doc_dir,
        doc_dir_name=doc_dir.name,
        doc_stem="doc",
        source_pdf_path=doc_dir / "doc.pdf",
        page_json_path=doc_dir / "doc.001.json",
        page_image_path=None,
        page_number=1,
        result_index=2,
        stored_result={"raw_lab_name": "Glucose"},
    )

    row_id = review_artifacts_backend.encode_row_id(row)
    target = review_artifacts_backend.decode_row_id(row_id, output_path)
    payload: ReviewDecisionResult = {
        "ok": True,
        "profile": "test",
        "row_id": row_id,
        "decision": "accept",
        "doc_dir": str(target.doc_dir),
        "page_number": target.page_number,
        "result_index": target.result_index,
    }

    assert payload["ok"] is True
    assert Path(payload["doc_dir"]) == doc_dir
    assert payload["page_number"] == 1
    assert payload["result_index"] == 2


def test_format_bbox_payload_round_trip():
    bbox = review_artifacts_backend.format_bbox_payload((1.0, 2.0, 3.0, 4.0))
    assert bbox == {"left": 1.0, "top": 2.0, "right": 3.0, "bottom": 4.0}
