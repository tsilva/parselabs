"""Stateful review helpers shared by the results explorer and document reviewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from parselabs.review import (
    SOURCE_BBOX_LABEL,
    build_details_html,
    build_page_image_value_for_entry,
    build_review_status_badge,
)
from parselabs.store import apply_review_action, resolve_document_dir


@dataclass(frozen=True)
class ReviewTarget:
    """Filesystem-backed row identity for one persisted review row."""

    doc_dir: Path
    page_number: int
    result_index: int


@dataclass(frozen=True)
class ViewerRowContext:
    """Presentation payload for one selected row in the results explorer."""

    row_index: int
    position_text: str
    source_image_value: tuple[str, list[tuple[tuple[int, int, int, int], str]]] | None
    details_html: str
    status_label: str
    status_html: str
    banner_html: str
    plot_labs: list[str]
    selected_ref: tuple[float, float] | None


def get_selected_row(df: pd.DataFrame, row_index: int | None) -> pd.Series | None:
    """Return one selected row from a dataframe-like queue."""

    # Guard: Empty frames do not contain a selectable row.
    if df.empty:
        return None

    # Guard: Missing selections default to the first visible row.
    if row_index is None:
        return df.iloc[0]

    resolved_index = int(row_index)

    # Guard: Out-of-range selections do not resolve to a row.
    if resolved_index < 0 or resolved_index >= len(df):
        return None

    return df.iloc[resolved_index]


def build_viewer_row_context(
    filtered_df: pd.DataFrame,
    row_index: int,
    output_path: Path,
    *,
    selected_lab_name: str | None,
    banner_html: str,
) -> ViewerRowContext | None:
    """Build the common row payload for viewer selection and navigation flows."""

    row = get_selected_row(filtered_df, row_index)

    # Guard: Empty selections return no row context.
    if row is None:
        return None

    entry = row.to_dict()
    ref_min = row.get("reference_min")
    ref_max = row.get("reference_max")
    selected_ref = (ref_min, ref_max) if pd.notna(ref_min) or pd.notna(ref_max) else None
    plot_labs = [selected_lab_name] if selected_lab_name else [row.get("lab_name")]
    status_badge = build_review_status_badge(entry)

    return ViewerRowContext(
        row_index=row_index,
        position_text=f"**Row {row_index + 1} of {len(filtered_df)}**",
        source_image_value=build_page_image_value_for_entry(entry, output_path, label=SOURCE_BBOX_LABEL),
        details_html=build_details_html(entry),
        status_label=status_badge.label,
        status_html=status_badge.html,
        banner_html=banner_html,
        plot_labs=plot_labs,
        selected_ref=selected_ref,
    )


def resolve_review_target_for_entry(entry: dict, output_path: Path) -> tuple[ReviewTarget | None, str]:
    """Resolve the persisted row target for a merged-review entry."""

    source_file = entry.get("source_file", "")
    stem = source_file.rsplit(".", 1)[0] if "." in source_file else source_file
    doc_dir = resolve_document_dir(stem, output_path)
    result_index = entry.get("result_index")
    page_number = entry.get("page_number")

    # Guard: Review writes require stable row identity metadata.
    if result_index is None or pd.isna(result_index):
        return None, "Missing result_index for entry."

    # Guard: Review writes require a page number to locate the page JSON.
    if page_number is None or pd.isna(page_number):
        return None, "Missing page_number for entry."

    # Guard: Review writes require the hashed document directory to exist.
    if doc_dir is None:
        return None, f"Document directory not found for '{source_file}'."

    return ReviewTarget(
        doc_dir=doc_dir,
        page_number=int(page_number),
        result_index=int(result_index),
    ), ""


def apply_review_action_for_target(target: ReviewTarget, action: str) -> tuple[bool, str]:
    """Persist a supported review action for one resolved review target."""

    return apply_review_action(target.doc_dir, target.page_number, target.result_index, action)


def apply_review_action_for_entry(entry: dict, output_path: Path, action: str) -> tuple[bool, str]:
    """Persist a review action for one merged-review row entry."""

    target, error = resolve_review_target_for_entry(entry, output_path)

    # Guard: Unresolvable entries cannot be persisted.
    if target is None:
        return False, error

    return apply_review_action_for_target(target, action)


__all__ = [
    "ReviewTarget",
    "ViewerRowContext",
    "apply_review_action_for_entry",
    "apply_review_action_for_target",
    "build_viewer_row_context",
    "get_selected_row",
    "resolve_review_target_for_entry",
]
