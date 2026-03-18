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


def test_get_document_choices_prioritizes_bbox_docs_for_review_mode():
    df = pd.DataFrame(
        [
            {
                "source_file": "beta.csv",
                "review_status": "",
                "review_needed": True,
                "bbox_left": None,
                "bbox_top": None,
                "bbox_right": None,
                "bbox_bottom": None,
            },
            {
                "source_file": "alpha.csv",
                "review_status": "",
                "review_needed": False,
                "bbox_left": 1,
                "bbox_top": 2,
                "bbox_right": 3,
                "bbox_bottom": 4,
            },
        ]
    )

    choices = viewer.get_document_choices(df, prioritize_review_sources=True)

    assert choices == [
        ("alpha (1)", "alpha.csv"),
        ("beta (1)", "beta.csv"),
    ]


def test_get_initial_document_prefers_bbox_backed_review_document():
    df = pd.DataFrame(
        [
            {
                "source_file": "beta.csv",
                "review_status": "",
                "review_needed": True,
                "bbox_left": None,
                "bbox_top": None,
                "bbox_right": None,
                "bbox_bottom": None,
            },
            {
                "source_file": "alpha.csv",
                "review_status": "",
                "review_needed": False,
                "bbox_left": 1,
                "bbox_top": 2,
                "bbox_right": 3,
                "bbox_bottom": 4,
            },
        ]
    )

    assert viewer.get_initial_document(df, prioritize_review_sources=True) == "alpha.csv"
    assert viewer.get_initial_document(df, prioritize_review_sources=False) is None


def test_apply_filters_sorts_oldest_first_in_document_page_order():
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-03"),
                "lab_name": "Blood - Potassium",
                "source_file": "newer.csv",
                "page_number": 1,
                "result_index": 0,
                "review_status": "",
            },
            {
                "date": pd.Timestamp("2024-01-01"),
                "lab_name": "Blood - Zeta",
                "source_file": "older.csv",
                "page_number": 2,
                "result_index": 0,
                "review_status": "",
            },
            {
                "date": pd.Timestamp("2024-01-01"),
                "lab_name": "Blood - Alpha",
                "source_file": "older.csv",
                "page_number": 1,
                "result_index": 1,
                "review_status": "",
            },
            {
                "date": pd.Timestamp("2024-01-01"),
                "lab_name": "Blood - Beta",
                "source_file": "older.csv",
                "page_number": 1,
                "result_index": 0,
                "review_status": "",
            },
        ]
    )

    filtered = viewer.apply_filters(
        df,
        None,
        False,
        "All",
        sort_order=viewer.SORT_DATE_ASC,
    )

    assert filtered[["source_file", "page_number", "result_index"]].to_dict("records") == [
        {"source_file": "older.csv", "page_number": 1, "result_index": 0},
        {"source_file": "older.csv", "page_number": 1, "result_index": 1},
        {"source_file": "older.csv", "page_number": 2, "result_index": 0},
        {"source_file": "newer.csv", "page_number": 1, "result_index": 0},
    ]


def test_apply_filters_keeps_single_document_in_page_order_for_all_sort_modes():
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "lab_name": "Blood - Zeta",
                "source_file": "older.csv",
                "page_number": 2,
                "result_index": 0,
                "review_status": "",
            },
            {
                "date": pd.Timestamp("2024-01-01"),
                "lab_name": "Blood - Alpha",
                "source_file": "older.csv",
                "page_number": 1,
                "result_index": 1,
                "review_status": "",
            },
            {
                "date": pd.Timestamp("2024-01-01"),
                "lab_name": "Blood - Beta",
                "source_file": "older.csv",
                "page_number": 1,
                "result_index": 0,
                "review_status": "",
            },
        ]
    )

    filtered = viewer.apply_filters(
        df,
        None,
        False,
        "All",
        document_name="older.csv",
        sort_order=viewer.SORT_DATE_ASC,
    )

    assert filtered[["page_number", "result_index"]].to_dict("records") == [
        {"page_number": 1, "result_index": 0},
        {"page_number": 1, "result_index": 1},
        {"page_number": 2, "result_index": 0},
    ]


def test_handle_navigation_stays_on_first_row_when_moving_backward(tmp_path):
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

    assert result[0] == 0
    assert result[1] == "**Result 1 of 2**"
    assert 'data-selected-row="0"' in result[5]
    assert 'data-row-count="2"' in result[5]
    assert result[6]["interactive"] is False
    assert result[7]["interactive"] is True


def test_handle_navigation_stays_on_last_row_when_moving_forward(tmp_path):
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

    result = viewer.handle_navigation(1, df, df, None, 1, tmp_path)

    assert result[0] == 1
    assert result[1] == "**Result 2 of 2**"
    assert 'data-selected-row="1"' in result[5]
    assert result[6]["interactive"] is True
    assert result[7]["interactive"] is False


def test_build_selection_state_html_tracks_selected_row_and_row_count():
    html = viewer._build_selection_state_html(3, 12)

    assert 'id="viewer-selection-state"' in html
    assert 'data-selected-row="3"' in html
    assert 'data-row-count="12"' in html


def test_resolve_plot_point_row_index_matches_filtered_row():
    df = pd.DataFrame(
        [
            {
                "source_file": "alpha.csv",
                "page_number": 1,
                "result_index": 0,
            },
            {
                "source_file": "beta.csv",
                "page_number": 2,
                "result_index": 3,
            },
        ]
    )

    resolved_idx = viewer._resolve_plot_point_row_index(df, '["beta.csv", 2, 3]')

    assert resolved_idx == 1


def test_handle_plot_point_select_updates_selection_from_token(tmp_path):
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

    result = viewer.handle_plot_point_select('["glucose.csv", 1, 1]', 0, df, df, None, tmp_path)

    assert result[0] == 1
    assert result[1] == "**Result 2 of 2**"
    assert 'data-selected-row="1"' in result[5]
    assert result[6]["interactive"] is True
    assert result[7]["interactive"] is False


def test_handle_plot_point_select_keeps_current_selection_when_token_missing(tmp_path):
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

    result = viewer.handle_plot_point_select('["missing.csv", 9, 9]', 1, df, df, None, tmp_path)

    assert result[0] == 1
    assert result[1] == "**Result 2 of 2**"
    assert 'data-selected-row="1"' in result[5]


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
    assert result[3] == 0
    assert result[4] == "**Result 1 of 1**"
    assert result[10]["interactive"] is False
    assert result[11]["interactive"] is False


def test_handle_review_action_keeps_last_row_selected(monkeypatch, tmp_path):
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
    calls: list[str] = []

    monkeypatch.setattr(
        viewer,
        "apply_review_action_for_entry",
        lambda entry, output_path, action: (calls.append(action) or True, ""),
    )

    result = viewer.handle_review_action(1, full_df.copy(), full_df, None, False, "All", "accepted", tmp_path)

    assert calls == ["accept"]
    assert result[3] == 1
    assert result[4] == "**Result 2 of 2**"
    assert 'data-selected-row="1"' in result[9]
    assert result[10]["interactive"] is True
    assert result[11]["interactive"] is False


def test_dispatch_row_select_preserves_gradio_select_argument_order(monkeypatch, tmp_path):
    filtered_df = pd.DataFrame([{"lab_name": "Blood - Glucose"}])
    full_df = filtered_df.copy()
    evt = types.SimpleNamespace(index=(0, 0))
    calls: list[tuple[object, object, object, object, object]] = []

    monkeypatch.setattr(
        viewer,
        "handle_row_select",
        lambda evt_arg, filtered_arg, full_arg, lab_arg, output_arg, document_name=None: (
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
        inspector_html="",
        selection_html=viewer._build_selection_state_html(None, 0),
        prev_button_props=viewer.gr.update(interactive=False),
        next_button_props=viewer.gr.update(interactive=False),
    )

    monkeypatch.setattr(viewer, "_load_output_data", lambda output_path, lab_specs, demographics: (empty_df, []))
    monkeypatch.setattr(
        viewer,
        "_render_viewer_state",
        lambda full_df, filtered_df, output_path, idx, summary_df=None, document_name=None, **kwargs: empty_state,
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


def test_build_selection_inspector_html_surfaces_missing_source_box():
    html = viewer.build_selection_inspector_html(
        {
            "lab_name": "Blood - Glucose",
            "review_status": "",
            "source_file": "glucose.csv",
            "page_number": 1,
            "result_index": 0,
        }
    )

    assert "Source Box" in html
    assert "Not stored for this row" in html
