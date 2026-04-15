from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pandas as pd
from PIL import Image

from parselabs import review, review_data, runtime
from parselabs.config import ExtractionConfig, LabSpecsConfig


def _build_config(tmp_path: Path, *, base_url: str = "https://openrouter.ai/api/v1", api_key: str = "test-key") -> ExtractionConfig:
    return ExtractionConfig(
        input_path=tmp_path,
        output_path=tmp_path / "output",
        openrouter_api_key=api_key,
        openrouter_base_url=base_url,
        extract_model_id="test-model",
        max_workers=1,
    )


def test_runtime_get_openai_client_caches_by_endpoint_and_key(monkeypatch, tmp_path):
    calls: list[tuple[str, str]] = []

    class FakeOpenAI:
        def __init__(self, *, base_url: str, api_key: str):
            calls.append((base_url, api_key))
            self.base_url = base_url
            self.api_key = api_key

    monkeypatch.setattr(runtime, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(runtime, "_CLIENT_CACHE", {})

    first = runtime.get_openai_client(_build_config(tmp_path))
    second = runtime.get_openai_client(_build_config(tmp_path))
    third = runtime.get_openai_client(_build_config(tmp_path, api_key="other-key"))

    assert first is second
    assert first is not third
    assert calls == [
        ("https://openrouter.ai/api/v1", "test-key"),
        ("https://openrouter.ai/api/v1", "other-key"),
    ]


def test_review_data_ignores_root_all_csv_without_processed_documents(tmp_path):
    output_path = tmp_path / "output"
    output_path.mkdir()
    pd.DataFrame(
        [
            {
                "source_file": "legacy.csv",
                "page_number": 1,
                "result_index": 0,
                "lab_name": "Blood - Glucose",
            }
        ]
    ).to_csv(output_path / "all.csv", index=False)

    result = review_data.load_results_dataframe(output_path, LabSpecsConfig(), demographics=None)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_review_build_page_image_value_for_entry_adds_overlay(tmp_path):
    output_path = tmp_path / "output"
    doc_dir = output_path / "glucose_deadbeef"
    doc_dir.mkdir(parents=True)
    (doc_dir / "glucose.pdf").write_bytes(b"%PDF-1.4")
    Image.new("RGB", (200, 100), "white").save(doc_dir / "glucose.001.jpg")

    image_value = review.build_page_image_value_for_entry(
        {
            "source_file": "glucose.pdf",
            "page_number": 1,
            "bbox_left": 100,
            "bbox_top": 200,
            "bbox_right": 600,
            "bbox_bottom": 800,
        },
        output_path,
    )

    assert image_value is not None
    image_path, annotations = image_value
    assert image_path.endswith("glucose.001.jpg")
    assert annotations == [((20, 20, 120, 80), review.SOURCE_BBOX_LABEL)]


def test_root_test_wrapper_delegates_to_dataset_helpers(monkeypatch):
    test_module = importlib.import_module("test")
    calls: dict[str, object] = {}

    def _fake_build_integrity_report(*, profile_names=None):
        calls["profile_names"] = profile_names
        return {"ok": ["done"]}

    monkeypatch.setattr(test_module, "parse_args", lambda: argparse.Namespace(profile="alpha"))
    monkeypatch.setattr(test_module, "build_integrity_report", _fake_build_integrity_report)
    monkeypatch.setattr(test_module, "print_integrity_report", lambda report: calls.setdefault("report", report))

    test_module.main()

    assert calls["profile_names"] == ["alpha"]
    assert calls["report"] == {"ok": ["done"]}
