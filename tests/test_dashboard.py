"""Synthetic end-to-end checks for the Svelte dashboard's Python API."""

from __future__ import annotations

import json
import logging
import shutil

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from parselabs.config import LabSpecsConfig, ProfileConfig
from parselabs.dashboard import create_dashboard_app
from parselabs.runtime import RuntimeContext


@pytest.fixture
def dashboard(tmp_path):
    """Use real page JSON and the shared loader; never read a private profile."""

    output = tmp_path / "output"
    directory = output / "demo_deadbeef"
    directory.mkdir(parents=True)
    (directory / "demo.pdf").write_bytes(b"%PDF-1.4\n% synthetic fixture\n")
    result = {
        "raw_lab_name": "Glucose", "raw_value": "92", "raw_lab_unit": "mg/dL",
        "raw_reference_min": 70, "raw_reference_max": 100,
        "bbox_left": 100, "bbox_top": 200, "bbox_right": 900, "bbox_bottom": 240,
    }
    payload = {"collection_date": "2026-10-08", "lab_results": [result]}
    json_path = directory / "demo.001.json"
    json_path.write_text(json.dumps(payload))
    (directory / "demo.002.json").write_text(json.dumps({"collection_date": "2026-10-08", "lab_results": []}))
    Image.new("RGB", (800, 1000), "white").save(directory / "demo.001.jpg")
    context = RuntimeContext(
        profile_name="synthetic", profile=ProfileConfig(name="Synthetic", output_path=output),
        extraction_config=None, input_path=None, output_path=output, demographics=None,
        lab_specs=LabSpecsConfig(), logger=logging.getLogger(__name__),
    )
    with TestClient(create_dashboard_app(context, "review")) as client:
        token = client.get("/api/bootstrap").json()["token"]
        rows = client.get("/api/results").json()["rows"]
        yield client, rows[0], {"X-Parselabs-Token": token}, json_path


def test_dashboard_persists_decisions_and_reloads_json(dashboard):
    client, row, headers, json_path = dashboard
    assert client.get("/api/bootstrap").json()["tab"] == "review"
    assert row["raw_value"] == "92"
    assert row["date"].startswith("2026-10-08")
    endpoint = f"/api/results/{row['id']}/decision"

    response = client.post(endpoint, headers=headers, json={"action": "accept", "expected_status": ""})
    assert response.status_code == 200
    assert response.json()["review_status"] == "accepted"
    persisted = json.loads(json_path.read_text())["lab_results"][0]
    assert persisted["review_status"] == "accepted"
    assert persisted["review_completed_at"]
    assert client.get("/api/results").json()["rows"][0]["review_status"] == "accepted"

    # A second reviewer cannot silently overwrite the newly accepted decision.
    stale = client.post(endpoint, headers=headers, json={"action": "reject", "expected_status": ""})
    assert stale.status_code == 409
    assert json.loads(json_path.read_text())["lab_results"][0]["review_status"] == "accepted"

    response = client.post(endpoint, headers=headers, json={"action": "clear", "expected_status": "accepted"})
    assert response.status_code == 200
    assert response.json()["review_status"] == ""
    assert "review_completed_at" not in json.loads(json_path.read_text())["lab_results"][0]


def test_missing_row_markers_are_idempotent(dashboard):
    client, row, headers, json_path = dashboard
    endpoint = f"/api/results/{row['id']}/decision"
    for _ in range(2):
        response = client.post(endpoint, headers=headers, json={"action": "missing_row", "expected_status": ""})
        assert response.status_code == 200
    assert len(json.loads(json_path.read_text())["review_missing_rows"]) == 1
    assert response.json()["review_status"] == ""


def test_same_filename_reports_keep_distinct_source_and_decision_identity(dashboard):
    client, _, headers, original = dashboard
    second = original.parent.parent / "demo_aabbccdd"
    shutil.copytree(original.parent, second)
    rows = client.get("/api/results").json()["rows"]
    assert len(rows) == 2
    assert rows[0]["source_file"] == rows[1]["source_file"]
    assert rows[0]["id"] != rows[1]["id"]
    selected = next(row for row in rows if row["document_key"] == second.name)

    response = client.post(f"/api/results/{selected['id']}/decision", headers=headers, json={"action": "reject", "expected_status": ""})
    assert response.status_code == 200
    assert json.loads((second / "demo.001.json").read_text())["lab_results"][0]["review_status"] == "rejected"
    assert "review_status" not in json.loads(original.read_text())["lab_results"][0]


def test_dashboard_contains_sources_and_protects_writes(dashboard, tmp_path):
    client, row, headers, json_path = dashboard
    endpoint = f"/api/results/{row['id']}"
    source = client.get(f"{endpoint}/source")
    assert source.status_code == 200
    assert source.headers["cache-control"] == "no-store"
    assert source.json()["pages"] == [1, 2]
    assert source.json()["bbox"] == {"left": 10, "top": 20, "width": 80, "height": 4}
    image = client.get(source.json()["image_url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert client.get(f"{endpoint}/source?page=2").json()["image_url"] is None
    assert client.get(f"{endpoint}/image?page=99").status_code == 404
    assert client.get("/api/results/nonexistent/source").status_code == 404

    original = json_path.read_text()
    assert client.post(f"{endpoint}/decision", json={"action": "accept", "expected_status": ""}).status_code == 403
    assert client.post(f"{endpoint}/decision", headers=headers, json={"action": "delete", "expected_status": ""}).status_code == 422
    assert json_path.read_text() == original
    assert client.get("/api/results", headers={"Host": "outside.example"}).status_code == 400

    # A malicious image symlink must never expose another directory's file.
    image_path = json_path.with_suffix(".jpg")
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (800, 1000), "red").save(outside)
    image_path.unlink()
    image_path.symlink_to(outside)
    assert client.get(f"{endpoint}/image").status_code == 403


def test_dashboard_serves_bundled_svelte_and_empty_profiles(tmp_path):
    context = RuntimeContext(
        profile_name="empty", profile=ProfileConfig(name="Empty", output_path=tmp_path),
        extraction_config=None, input_path=None, output_path=tmp_path, demographics=None,
        lab_specs=LabSpecsConfig(), logger=logging.getLogger(__name__),
    )
    with TestClient(create_dashboard_app(context)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert 'type="module"' in response.text
        assert client.get("/api/results").json() == {"rows": []}
        assert client.get("/brand/icon.png").status_code == 200
