from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from parselabs import review_artifacts_backend
from utils import review_artifacts


def _write_processed_document(
    doc_dir: Path,
    *,
    statuses: list[str | None],
    stem: str = "cbc",
) -> None:
    """Create one minimal processed document with one page image and one JSON payload."""

    doc_dir.mkdir(parents=True)
    (doc_dir / f"{stem}.pdf").write_bytes(b"%PDF-1.4")
    Image.new("RGB", (240, 120), "white").save(doc_dir / f"{stem}.001.jpg")

    lab_results = []

    for index, status in enumerate(statuses):
        result = {
            "raw_lab_name": f"Lab {index}",
            "raw_value": str(90 + index),
            "raw_lab_unit": "mg/dL",
            "raw_reference_min": 70,
            "raw_reference_max": 100,
            "bbox_left": 100,
            "bbox_top": 100 + (index * 50),
            "bbox_right": 600,
            "bbox_bottom": 220 + (index * 50),
        }

        # Persist explicit review state only when the test asks for it.
        if status is not None:
            result["review_status"] = status

        lab_results.append(result)

    payload = {
        "collection_date": "2024-01-05",
        "lab_results": lab_results,
    }
    (doc_dir / f"{stem}.001.json").write_text(json.dumps(payload), encoding="utf-8")


def test_next_returns_pending_row_and_crop(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "output"
    doc_dir = output_path / "cbc_12345678"
    artifacts_dir = tmp_path / "artifacts"
    _write_processed_document(doc_dir, statuses=["accepted", None])

    # Keep the CLI focused on the synthetic processed output tree.
    monkeypatch.setattr(
        review_artifacts_backend,
        "load_review_context",
        lambda profile_name: SimpleNamespace(output_path=output_path),
    )

    exit_code = review_artifacts.main(
        [
            "next",
            "--profile",
            "demo",
            "--artifacts-dir",
            str(artifacts_dir),
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["done"] is False
    assert payload["result_index"] == 1
    assert Path(payload["page_image_path"]).exists()
    assert Path(payload["bbox_clip_path"]).exists()
    assert payload["stored_result"]["raw_lab_name"] == "Lab 1"


def test_decide_persists_review_status_for_row_id(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "output"
    doc_dir = output_path / "cbc_12345678"
    _write_processed_document(doc_dir, statuses=[None])

    # Keep the CLI focused on the synthetic processed output tree.
    monkeypatch.setattr(
        review_artifacts_backend,
        "load_review_context",
        lambda profile_name: SimpleNamespace(output_path=output_path),
    )

    review_artifacts.main(
        [
            "next",
            "--profile",
            "demo",
        ]
    )
    next_payload = json.loads(capsys.readouterr().out)

    exit_code = review_artifacts.main(
        [
            "decide",
            "--profile",
            "demo",
            "--row-id",
            next_payload["row_id"],
            "--decision",
            "reject",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    stored_page = json.loads((doc_dir / "cbc.001.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["ok"] is True
    assert stored_page["lab_results"][0]["review_status"] == "rejected"


def test_next_returns_done_when_no_pending_rows(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "output"
    doc_dir = output_path / "cbc_12345678"
    _write_processed_document(doc_dir, statuses=["accepted"])

    # Keep the CLI focused on the synthetic processed output tree.
    monkeypatch.setattr(
        review_artifacts_backend,
        "load_review_context",
        lambda profile_name: SimpleNamespace(output_path=output_path),
    )

    exit_code = review_artifacts.main(
        [
            "next",
            "--profile",
            "demo",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["done"] is True
