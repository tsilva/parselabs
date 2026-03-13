from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from PIL import Image

plotly_module = types.ModuleType("plotly")
graph_objects_module = types.ModuleType("plotly.graph_objects")
subplots_module = types.ModuleType("plotly.subplots")
subplots_module.make_subplots = lambda *args, **kwargs: None

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
