from __future__ import annotations

import importlib
import sys
import types
import warnings
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

    def to_json(self):
        return "{}"


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


def test_dispatch_row_select_preserves_gradio_select_argument_order(monkeypatch, tmp_path):
    filtered_df = pd.DataFrame([{"lab_name": "Blood - Glucose"}])
    full_df = filtered_df.copy()
    evt = types.SimpleNamespace(index=(0, 0))
    calls: list[tuple[object, object, object, object, object]] = []

    monkeypatch.setattr(
        viewer,
        "handle_row_select",
        lambda evt_arg, filtered_arg, full_arg, lab_arg, output_arg: (
            calls.append((evt_arg, filtered_arg, full_arg, lab_arg, output_arg)) or ("ok",)
        ),
    )

    result = viewer._dispatch_row_select(filtered_df, full_df, "Blood - Glucose", evt, tmp_path)

    assert result == ("ok",)
    assert calls[0][0] is evt
    assert calls[0][1].equals(filtered_df)
    assert calls[0][2].equals(full_df)
    assert calls[0][3] == "Blood - Glucose"
    assert calls[0][4] == tmp_path


def test_create_app_does_not_render_profile_selector(monkeypatch, tmp_path):
    empty_df = pd.DataFrame()
    empty_state = viewer.ViewerRenderState(
        display_df=empty_df,
        summary_html="",
        plot=viewer.go.Figure(),
        filtered_df=empty_df,
        current_idx=0,
        position_text="",
        source_image_value=None,
        details_html="",
        status_html="",
        banner_html="",
    )

    monkeypatch.setattr(viewer, "_load_output_data", lambda output_path, lab_specs, demographics: (empty_df, []))
    monkeypatch.setattr(
        viewer,
        "_render_viewer_state",
        lambda full_df, filtered_df, output_path, idx, summary_df=None: empty_state,
    )

    context = types.SimpleNamespace(
        output_path=tmp_path,
        lab_specs=None,
        demographics=None,
    )

    with warnings.catch_warnings(record=True) as caught:
        demo = viewer.create_app(context)

    labels = [getattr(component, "label", None) for component in demo.blocks.values()]
    values = [value for value in (getattr(component, "value", None) for component in demo.blocks.values()) if isinstance(value, str)]
    warning_messages = [str(item.message) for item in caught]

    assert "Profile" not in labels
    assert "# Lab Results Viewer" not in values
    assert "Browse, analyze, and review extracted lab results." not in values
    assert not any("Expected 4 arguments for function" in message for message in warning_messages)
    assert not any("Expected at least 4 arguments for function" in message for message in warning_messages)
