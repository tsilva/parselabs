"""Shared review helpers for the results explorer and document reviewer."""

from __future__ import annotations

import html
import logging
from pathlib import Path

import pandas as pd
from PIL import Image

from parselabs.config import Demographics, LabSpecsConfig
from parselabs.rows import build_corpus_review_rows
from parselabs.store import (
    get_page_image_path as get_page_image_path_from_store,
)
from parselabs.store import (
    load_legacy_merged_review_dataframe,
    read_page_payload,
    resolve_page_path,
)

logger = logging.getLogger(__name__)

SOURCE_BBOX_LABEL = "Selected result"
_PAGE_IMAGE_SIZE_CACHE: dict[str, tuple[int, int]] = {}


def normalize_review_status(status: object) -> str:
    """Normalize reviewer status values into accepted, rejected, or pending."""

    if status is None:
        return ""

    normalized = str(status).strip().lower()
    if not normalized:
        return ""
    if normalized in {"accepted", "rejected"}:
        return normalized
    return ""


def format_text(value: object, empty: str = "-") -> str:
    """Return a safe string for UI display, normalizing missing values."""

    if value is None or pd.isna(value):
        return empty

    text = str(value).strip()
    if not text:
        return empty

    return html.escape(text)


def format_reference_bounds(ref_min: object, ref_max: object, *, unit: object = None) -> str:
    """Format min/max reference bounds, including one-sided ranges."""

    min_text = format_text(ref_min, empty="")
    max_text = format_text(ref_max, empty="")
    unit_text = format_text(unit, empty="")

    if min_text and max_text:
        range_text = f"{min_text} - {max_text}"
    elif min_text:
        range_text = f">{min_text}"
    elif max_text:
        range_text = f"<{max_text}"
    else:
        return "-"

    if unit_text:
        return f"{range_text} {unit_text}".strip()

    return range_text


def format_reference_text(row: pd.Series) -> str:
    """Format the best available reference range for UI display."""

    raw_reference = format_text(row.get("raw_reference_range"), empty="")
    if raw_reference:
        return raw_reference
    return format_reference_bounds(row.get("reference_min"), row.get("reference_max"))


def format_mapped_reference_text(row: pd.Series) -> str:
    """Format the normalized reference range in the standardized unit."""

    return format_reference_bounds(
        row.get("reference_min"),
        row.get("reference_max"),
        unit=row.get("lab_unit"),
    )


def get_bbox_coordinates(entry: dict | pd.Series) -> tuple[float, float, float, float] | None:
    """Return viewer-usable bounding-box coordinates from a row-like object."""

    bbox_keys = ["bbox_left", "bbox_top", "bbox_right", "bbox_bottom"]
    coords: list[float] = []

    for key in bbox_keys:
        value = entry.get(key)
        if value is None or pd.isna(value):
            return None

        try:
            coords.append(float(value))
        except (TypeError, ValueError):
            return None

    left, top, right, bottom = coords
    if right <= left or bottom <= top:
        return None

    return left, top, right, bottom


def get_image_size(image_path: str) -> tuple[int, int] | None:
    """Return image dimensions with a shared in-memory cache."""

    if image_path in _PAGE_IMAGE_SIZE_CACHE:
        return _PAGE_IMAGE_SIZE_CACHE[image_path]

    try:
        with Image.open(image_path) as image:
            size = image.size
    except (FileNotFoundError, OSError) as exc:
        logger.warning(f"Failed to read source image size from {image_path}: {exc}")
        return None

    _PAGE_IMAGE_SIZE_CACHE[image_path] = size
    return size


def scale_bbox_to_pixels(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Scale normalized or absolute bbox coordinates into image pixels."""

    width, height = image_size
    left, top, right, bottom = bbox
    max_coord = max(left, top, right, bottom)

    if max_coord <= 1:
        scaled = (
            int(round(left * width)),
            int(round(top * height)),
            int(round(right * width)),
            int(round(bottom * height)),
        )
    elif max_coord <= 1000:
        scaled = (
            int(round(left * width / 1000)),
            int(round(top * height / 1000)),
            int(round(right * width / 1000)),
            int(round(bottom * height / 1000)),
        )
    else:
        scaled = (
            int(round(left)),
            int(round(top)),
            int(round(right)),
            int(round(bottom)),
        )

    left_px, top_px, right_px, bottom_px = scaled
    left_px = max(0, min(left_px, width - 1))
    top_px = max(0, min(top_px, height - 1))
    right_px = max(0, min(right_px, width))
    bottom_px = max(0, min(bottom_px, height))

    if right_px <= left_px or bottom_px <= top_px:
        return None

    return left_px, top_px, right_px, bottom_px


def build_annotated_image_value(
    image_path: str | Path | None,
    bbox: tuple[float, float, float, float] | None,
    *,
    label: str = SOURCE_BBOX_LABEL,
) -> tuple[str, list[tuple[tuple[int, int, int, int], str]]] | None:
    """Build an annotated-image payload from an image path and bbox."""

    if not image_path:
        return None

    image_path_str = str(image_path)
    annotations: list[tuple[tuple[int, int, int, int], str]] = []

    if bbox is None:
        return image_path_str, annotations

    image_size = get_image_size(image_path_str)
    if image_size is None:
        return image_path_str, annotations

    pixel_bbox = scale_bbox_to_pixels(bbox, image_size)
    if pixel_bbox is None:
        return image_path_str, annotations

    annotations.append((pixel_bbox, label))
    return image_path_str, annotations


def build_page_image_value_for_document(
    doc_dir: Path,
    row: pd.Series,
    *,
    label: str = SOURCE_BBOX_LABEL,
) -> tuple[str, list[tuple[tuple[int, int, int, int], str]]] | None:
    """Build an annotated-image payload for a document-review row."""

    page_number = row.get("page_number")
    if page_number is None or pd.isna(page_number):
        return None

    image_path = get_page_image_path_from_store(doc_dir, int(page_number))
    if image_path is None:
        return None

    return build_annotated_image_value(
        image_path,
        get_bbox_coordinates(row),
        label=label,
    )


def build_page_image_value_for_entry(
    entry: dict,
    output_path: Path,
    *,
    label: str = SOURCE_BBOX_LABEL,
) -> tuple[str, list[tuple[tuple[int, int, int, int], str]]] | None:
    """Build an annotated-image payload for a merged review row."""

    page_number = entry.get("page_number")
    if page_number is not None and pd.notna(page_number):
        page_number = int(page_number)
    else:
        page_number = None

    image_path = resolve_page_path(output_path, entry.get("source_file", ""), page_number, ".jpg")
    if not image_path.parts or not image_path.exists():
        return None

    return build_annotated_image_value(
        image_path,
        get_bbox_coordinates(entry),
        label=label,
    )


def get_review_status(entry: dict) -> str | None:
    """Get review status for a row-like entry."""

    status = entry.get("review_status")
    if status is not None and pd.notna(status) and str(status).strip():
        return str(status).strip()
    return None


def _load_json_cached(json_path: Path, cache: dict[str, dict | None]) -> dict | None:
    """Load and cache page JSON for review-status backfills."""

    json_path_str = str(json_path)
    if json_path_str in cache:
        return cache[json_path_str]

    cache[json_path_str] = read_page_payload(json_path)
    return cache[json_path_str]


def _sync_review_statuses(df: pd.DataFrame, output_path: Path) -> list[str | None]:
    """Read review_status from page JSON for legacy merged review CSVs."""

    json_cache: dict[str, dict | None] = {}
    review_statuses: list[str | None] = []

    for row in df.itertuples():
        result_index = getattr(row, "result_index", None)
        if result_index is None or pd.isna(result_index):
            review_statuses.append(None)
            continue

        json_path = resolve_page_path(
            output_path,
            getattr(row, "source_file", ""),
            getattr(row, "page_number", None),
            ".json",
        )
        json_data = _load_json_cached(json_path, json_cache)
        if json_data and "lab_results" in json_data:
            result_idx = int(result_index)
            if result_idx < len(json_data["lab_results"]):
                review_statuses.append(json_data["lab_results"][result_idx].get("review_status"))
                continue

        review_statuses.append(None)

    return review_statuses


def _is_out_of_range(value: float, range_min, range_max) -> bool:
    """Return whether a value falls outside the given numeric bounds."""

    return (pd.notna(range_min) and value < range_min) or (pd.notna(range_max) and value > range_max)


def _format_reference_range(row) -> str:
    """Format reference_min/reference_max into a display string."""

    ref_min = row["reference_min"]
    ref_max = row["reference_max"]

    if pd.isna(ref_min) and pd.isna(ref_max):
        if row.get("lab_unit") == "boolean":
            return "0 - 1"
        return ""

    if pd.isna(ref_min):
        return f"< {ref_max}"
    if pd.isna(ref_max):
        return f"> {ref_min}"

    return f"{ref_min} - {ref_max}"


def _check_out_of_reference(row) -> bool | None:
    """Check if a value falls outside the PDF reference range."""

    value = row["value"]
    if pd.isna(value):
        return None

    ref_min = row["reference_min"]
    ref_max = row["reference_max"]
    if pd.isna(ref_min) and pd.isna(ref_max):
        if row.get("lab_unit") == "boolean":
            return value > 0
        return None

    return _is_out_of_range(value, ref_min, ref_max)


def _get_lab_spec_range(lab_name: str, lab_specs: LabSpecsConfig, gender: str | None, age: int | None) -> pd.Series:
    """Look up healthy range for a lab from lab_specs, adjusted for demographics."""

    range_min, range_max = lab_specs.get_healthy_range_for_demographics(lab_name, gender=gender, age=age)
    return pd.Series({"lab_specs_min": range_min, "lab_specs_max": range_max})


def _check_out_of_healthy_range(row) -> bool | None:
    """Check if a value falls outside the healthy range from lab specs."""

    value = row.get("value")
    if pd.isna(value):
        return None

    spec_min = row.get("lab_specs_min")
    spec_max = row.get("lab_specs_max")
    if pd.isna(spec_min) and pd.isna(spec_max):
        return None

    return _is_out_of_range(value, spec_min, spec_max)


def load_results_dataframe(
    output_path: Path,
    lab_specs: LabSpecsConfig,
    demographics: Demographics | None,
) -> pd.DataFrame:
    """Load review rows for the results explorer with one legacy fallback path."""

    df = build_corpus_review_rows(output_path, lab_specs)
    if df.empty:
        df = load_legacy_merged_review_dataframe(output_path)
        if df.empty:
            return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "review_status" not in df.columns:
        df["review_status"] = _sync_review_statuses(df, output_path)

    if "reference_min" in df.columns and "reference_max" in df.columns:
        df["reference_range"] = df.apply(_format_reference_range, axis=1)

    if "value" in df.columns and "reference_min" in df.columns and "reference_max" in df.columns:
        df["is_out_of_reference"] = df.apply(_check_out_of_reference, axis=1)

    gender = demographics.gender if demographics else None
    age = demographics.age if demographics else None

    if "lab_name" in df.columns and lab_specs.exists:
        range_df = df["lab_name"].apply(lambda name: _get_lab_spec_range(name, lab_specs, gender, age))
        df["lab_specs_min"] = range_df["lab_specs_min"]
        df["lab_specs_max"] = range_df["lab_specs_max"]

    if "value" in df.columns and "lab_specs_min" in df.columns and "lab_specs_max" in df.columns:
        df["is_out_of_healthy_range"] = df.apply(_check_out_of_healthy_range, axis=1)

    return df


__all__ = [
    "SOURCE_BBOX_LABEL",
    "build_annotated_image_value",
    "build_page_image_value_for_document",
    "build_page_image_value_for_entry",
    "format_mapped_reference_text",
    "format_reference_bounds",
    "format_reference_text",
    "format_text",
    "get_bbox_coordinates",
    "get_image_size",
    "get_review_status",
    "load_results_dataframe",
    "normalize_review_status",
    "scale_bbox_to_pixels",
]
