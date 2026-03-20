from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image

from parselabs import review_artifacts_backend, review_mcp


def _write_processed_document(
    doc_dir: Path,
    *,
    statuses: list[str | None] | None = None,
    page_statuses: list[list[str | None]] | None = None,
    stem: str = "cbc",
) -> None:
    """Create one minimal processed document with one page image and one JSON payload."""

    doc_dir.mkdir(parents=True)
    (doc_dir / f"{stem}.pdf").write_bytes(b"%PDF-1.4")

    resolved_page_statuses = page_statuses if page_statuses is not None else [statuses or []]

    for page_index, page_result_statuses in enumerate(resolved_page_statuses, start=1):
        Image.new("RGB", (240, 120), "white").save(doc_dir / f"{stem}.{page_index:03d}.jpg")
        payload = {
            "collection_date": "2024-01-05",
            "lab_results": [
                {
                    "raw_lab_name": f"Lab {page_index}-{result_index}",
                    "raw_value": str(90 + result_index),
                    "raw_lab_unit": "mg/dL",
                    "raw_reference_min": 70,
                    "raw_reference_max": 100,
                    "bbox_left": 100,
                    "bbox_top": 100 + (result_index * 50),
                    "bbox_right": 600,
                    "bbox_bottom": 220 + (result_index * 50),
                    **({"review_status": status} if status is not None else {}),
                }
                for result_index, status in enumerate(page_result_statuses)
            ],
        }
        (doc_dir / f"{stem}.{page_index:03d}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_next_pending_row_returns_structured_payload_and_images(tmp_path, monkeypatch):
    output_path = tmp_path / "output"
    doc_dir = output_path / "cbc_12345678"
    artifacts_dir = tmp_path / "artifacts"
    _write_processed_document(doc_dir, statuses=[None])

    # Keep the MCP tool focused on the synthetic processed output tree.
    monkeypatch.setattr(
        review_artifacts_backend,
        "load_review_context",
        lambda profile_name: SimpleNamespace(output_path=output_path),
    )

    server = review_mcp.build_server()
    result = asyncio.run(
        server.call_tool(
            "next_pending_row",
            {
                "profile": "demo",
                "artifacts_dir": str(artifacts_dir),
            },
        )
    )

    assert isinstance(result, CallToolResult)
    assert result.structuredContent["done"] is False
    assert result.structuredContent["result_index"] == 0
    assert any(isinstance(item, TextContent) for item in result.content)
    assert sum(isinstance(item, ImageContent) for item in result.content) == 2


def test_decide_row_persists_status(tmp_path, monkeypatch):
    output_path = tmp_path / "output"
    doc_dir = output_path / "cbc_12345678"
    _write_processed_document(doc_dir, statuses=[None])

    # Keep the MCP tool focused on the synthetic processed output tree.
    monkeypatch.setattr(
        review_artifacts_backend,
        "load_review_context",
        lambda profile_name: SimpleNamespace(output_path=output_path),
    )

    server = review_mcp.build_server()
    next_result = asyncio.run(
        server.call_tool(
            "next_pending_row",
            {
                "profile": "demo",
            },
        )
    )
    row_id = next_result.structuredContent["row_id"]

    decide_result = asyncio.run(
        server.call_tool(
            "decide_row",
            {
                "profile": "demo",
                "row_id": row_id,
                "decision": "accept",
            },
        )
    )

    stored_page = json.loads((doc_dir / "cbc.001.json").read_text(encoding="utf-8"))

    assert isinstance(decide_result, CallToolResult)
    assert decide_result.structuredContent["ok"] is True
    assert stored_page["lab_results"][0]["review_status"] == "accepted"
