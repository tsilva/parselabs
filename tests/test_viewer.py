from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pandas as pd
from PIL import Image

plotly_module = types.ModuleType("plotly")
graph_objects_module = types.ModuleType("plotly.graph_objects")
subplots_module = types.ModuleType("plotly.subplots")
subplots_module.make_subplots = lambda *args, **kwargs: None


class _FakeFigure:
    def add_trace(self, *args, **kwargs):
        return None

    def add_hrect(self, *args, **kwargs):
        return None

    def add_hline(self, *args, **kwargs):
        return None

    def add_annotation(self, *args, **kwargs):
        return None

    def update_layout(self, *args, **kwargs):
        return None

    def update_xaxes(self, *args, **kwargs):
        return None

    def update_yaxes(self, *args, **kwargs):
        return None


graph_objects_module.Figure = _FakeFigure
graph_objects_module.Scatter = lambda *args, **kwargs: None

sys.modules.setdefault("plotly", plotly_module)
sys.modules.setdefault("plotly.graph_objects", graph_objects_module)
sys.modules.setdefault("plotly.subplots", subplots_module)

viewer = importlib.import_module("parselabs.results_view")
review_helpers = importlib.import_module("parselabs.review")


def _write_viewer_page(output_path: Path, stem: str = "glucose") -> Path:
    """Create a minimal processed page image for viewer tests."""

    doc_dir = output_path / f"{stem}_deadbeef"
    doc_dir.mkdir(parents=True)
    Image.new("RGB", (200, 100), "white").save(doc_dir / f"{stem}.001.jpg")
    return doc_dir


def test_build_page_image_value_for_entry_scales_normalized_bbox(tmp_path):
    _write_viewer_page(tmp_path)

    entry = {
        "source_file": "glucose.csv",
        "page_number": 1,
        "bbox_left": 100,
        "bbox_top": 200,
        "bbox_right": 600,
        "bbox_bottom": 800,
    }

    value = review_helpers.build_page_image_value_for_entry(entry, tmp_path)

    assert value is not None
    image_path, annotations = value
    assert image_path.endswith("glucose.001.jpg")
    assert annotations == [((20, 20, 120, 80), review_helpers.SOURCE_BBOX_LABEL)]


def test_build_page_image_value_for_entry_returns_plain_image_when_bbox_missing(tmp_path):
    _write_viewer_page(tmp_path)

    entry = {
        "source_file": "glucose.csv",
        "page_number": 1,
    }

    value = review_helpers.build_page_image_value_for_entry(entry, tmp_path)

    assert value is not None
    image_path, annotations = value
    assert image_path.endswith("glucose.001.jpg")
    assert annotations == []


def test_handle_navigation_wraps_between_first_and_last_rows(tmp_path):
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "lab_name": "Blood - Glucose",
                "value": 91.0,
                "lab_unit": "mg/dL",
                "reference_min": 70.0,
                "reference_max": 99.0,
                "review_status": "",
                "review_reason": "",
                "source_file": "glucose.csv",
                "page_number": 1,
                "result_index": 0,
            },
            {
                "date": pd.Timestamp("2024-01-01"),
                "lab_name": "Blood - Glucose",
                "value": 92.0,
                "lab_unit": "mg/dL",
                "reference_min": 70.0,
                "reference_max": 99.0,
                "review_status": "",
                "review_reason": "",
                "source_file": "glucose.csv",
                "page_number": 1,
                "result_index": 1,
            },
        ]
    )

    result = viewer.handle_navigation(0, df, df, None, -1, tmp_path)

    assert result[1] == 1
    assert result[2] == "**Row 2 of 2**"


def test_handle_review_action_uses_shared_entry_persistence(monkeypatch, tmp_path):
    full_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "lab_name": "Blood - Glucose",
                "value": 91.0,
                "lab_unit": "mg/dL",
                "reference_min": 70.0,
                "reference_max": 99.0,
                "review_status": "",
                "review_reason": "",
                "source_file": "glucose.csv",
                "page_number": 1,
                "result_index": 0,
            }
        ]
    )
    calls: list[str] = []

    monkeypatch.setattr(
        viewer,
        "apply_review_action_for_entry",
        lambda entry, output_path, action: (calls.append(action) or True, ""),
    )

    result = viewer.handle_review_action(0, full_df.copy(), full_df, None, False, "All", "accepted", tmp_path)

    assert calls == ["accept"]
    assert result[0].loc[0, "review_status"] == "accepted"
