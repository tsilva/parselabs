"""Shared review helpers for the results explorer and document reviewer."""

from __future__ import annotations

import html
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from parselabs.store import (
    get_page_image_path as get_page_image_path_from_store,
)
from parselabs.store import resolve_page_path
from parselabs.types import ReviewRow, coerce_persisted_review_status

logger = logging.getLogger(__name__)

SOURCE_BBOX_LABEL = "Selected result"
_PAGE_IMAGE_SIZE_CACHE: dict[str, tuple[int, int]] = {}
_PENDING_STATUS_LABEL = "Pending"


@dataclass(frozen=True)
class ReviewStatusBadge:
    """Presentation-ready review status label and rendered HTML."""

    label: str
    html: str


def normalize_review_status(status: object) -> str:
    """Normalize reviewer status values into accepted, rejected, or pending."""

    return coerce_persisted_review_status(status) or ""


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


def format_raw_value(row: pd.Series) -> str:
    """Format the raw value and unit into one compact display string."""

    value_text = format_text(row.get("raw_value"))
    unit_text = format_text(row.get("raw_lab_unit"), empty="")

    # Omit the placeholder unit suffix when the unit is genuinely absent.
    if unit_text:
        return f"{value_text} {unit_text}".strip()

    return value_text


def format_mapped_value(row: pd.Series) -> str:
    """Format the standardized value and unit into one compact display string."""

    value_text = format_text(row.get("value"))
    unit_text = format_text(row.get("lab_unit"), empty="")

    # Omit the placeholder unit suffix when the unit is genuinely absent.
    if unit_text:
        return f"{value_text} {unit_text}".strip()

    return value_text


def get_bbox_coordinates(entry: Mapping[str, object] | pd.Series) -> tuple[float, float, float, float] | None:
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
    entry: ReviewRow,
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


def get_review_status(entry: ReviewRow) -> str | None:
    """Get review status for a row-like entry."""

    status = entry.get("review_status")
    if status is not None and pd.notna(status) and str(status).strip():
        return str(status).strip()
    return None


def get_review_status_label(entry: dict | pd.Series | None) -> str:
    """Return Accepted, Rejected, or Pending for one row-like entry."""

    # Guard: Empty selections default to the undecided state.
    if entry is None:
        return _PENDING_STATUS_LABEL

    status = normalize_review_status(get_review_status(entry))

    # Surface accepted rows as explicit approvals.
    if status == "accepted":
        return "Accepted"

    # Surface rejected rows as explicit rejections.
    if status == "rejected":
        return "Rejected"

    return _PENDING_STATUS_LABEL


def build_review_status_html(status_label: str) -> str:
    """Build the compact review-status badge used in the results viewer."""

    colors = {"Accepted": "#2e7d32", "Rejected": "#c62828", "Pending": "#757575"}
    color = colors.get(status_label, "#757575")
    return f'<div style="text-align:center;padding:4px 0;"><span style="color:{color};font-weight:bold;font-size:0.9em;">{status_label}</span></div>'


def build_review_status_badge(entry: dict | pd.Series | None) -> ReviewStatusBadge:
    """Return both the normalized label and the rendered HTML badge."""

    label = get_review_status_label(entry)
    return ReviewStatusBadge(label=label, html=build_review_status_html(label))


def build_reason_badges(review_reason: object) -> str:
    """Render validation reason codes as compact badges."""

    reason_text = format_text(review_reason, empty="")

    # Guard: Rows without validation flags do not need badge markup.
    if not reason_text:
        return '<span class="review-inline-muted">None</span>'

    badges: list[str] = []

    # Split semicolon-delimited reasons into individual badges for faster scanning.
    for reason in [part.strip() for part in html.unescape(reason_text).split(";") if part.strip()]:
        badges.append(f'<span class="review-reason-chip">{html.escape(reason)}</span>')

    return "".join(badges)


def build_details_html(entry: ReviewRow) -> str:
    """Build the raw-vs-standardized details table for one selected row."""

    # Guard: Empty selections render the stable placeholder.
    if not entry:
        return "<p>No entry selected</p>"

    paired_fields = [
        ("Lab Name", "raw_lab_name", "lab_name"),
        ("Value", "raw_value", "value"),
        ("Unit", "raw_lab_unit", "lab_unit"),
        ("Ref Min", "reference_min", "reference_min"),
        ("Ref Max", "reference_max", "reference_max"),
    ]

    def get_val(field: str) -> str:
        value = entry.get(field)
        if value is not None and pd.notna(value) and str(value).strip():
            return str(value)
        return "-"

    html_parts = [
        '<table style="width:100%; border-collapse:collapse; font-size:0.9em;">',
        '<thead><tr style="background:#f5f5f5;"><th style="padding:6px; text-align:left;">Field</th><th style="padding:6px; text-align:left;">Raw</th><th style="padding:6px; text-align:left;">Standardized</th></tr></thead>',
        "<tbody>",
    ]

    # Render each paired raw vs standardized field row.
    for label, raw_field, std_field in paired_fields:
        raw_val = get_val(raw_field)
        std_val = get_val(std_field)
        html_parts.append(f'<tr><td style="padding:6px; border-bottom:1px solid #ddd;">{label}</td>')
        html_parts.append(f'<td style="padding:6px; border-bottom:1px solid #ddd;">{raw_val}</td>')
        html_parts.append(f'<td style="padding:6px; border-bottom:1px solid #ddd;">{std_val}</td></tr>')

    html_parts.append("</tbody></table>")

    bbox = get_bbox_coordinates(entry)

    # Surface bbox metadata so reviewers can verify the stored highlight geometry.
    if bbox is not None:
        left, top, right, bottom = bbox
        html_parts.append(
            '<div style="margin-top:12px; color:#555; font-size:0.9em;">'
            f"<strong>Bounding Box:</strong> left={left:g}, top={top:g}, right={right:g}, bottom={bottom:g}"
            " (normalized page coordinates)</div>"
        )

    review_needed = entry.get("review_needed")
    review_reason = entry.get("review_reason")

    # Show review details when flags are present.
    if review_needed or review_reason:
        html_parts.append('<div style="margin-top:15px;">')
        if review_reason and pd.notna(review_reason):
            html_parts.append(f'<div class="status-warning">Reason: {review_reason}</div>')
        html_parts.append("</div>")

    return "".join(html_parts)


__all__ = [
    "ReviewStatusBadge",
    "SOURCE_BBOX_LABEL",
    "build_annotated_image_value",
    "build_details_html",
    "build_reason_badges",
    "build_page_image_value_for_document",
    "build_page_image_value_for_entry",
    "build_review_status_badge",
    "build_review_status_html",
    "format_mapped_value",
    "format_mapped_reference_text",
    "format_raw_value",
    "format_reference_bounds",
    "format_reference_text",
    "format_text",
    "get_bbox_coordinates",
    "get_image_size",
    "get_review_status",
    "get_review_status_label",
    "normalize_review_status",
    "scale_bbox_to_pixels",
]
