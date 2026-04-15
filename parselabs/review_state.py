"""Stateful review helpers shared by the results explorer and document reviewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from parselabs.store import apply_review_action, resolve_document_dir
from parselabs.types import ReviewAction, ReviewRow, coerce_review_action


@dataclass(frozen=True)
class ReviewTarget:
    """Filesystem-backed row identity for one persisted review row."""

    doc_dir: Path
    page_number: int
    result_index: int


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


def resolve_review_target_for_entry(entry: ReviewRow, output_path: Path) -> tuple[ReviewTarget | None, str]:
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


def apply_review_action_for_target(target: ReviewTarget, action: ReviewAction) -> tuple[bool, str]:
    """Persist a supported review action for one resolved review target."""

    return apply_review_action(target.doc_dir, target.page_number, target.result_index, action)


def apply_review_action_for_entry(entry: ReviewRow, output_path: Path, action: object) -> tuple[bool, str]:
    """Persist a review action for one merged-review row entry."""

    normalized_action = coerce_review_action(action)
    if normalized_action is None:
        return False, f"Unsupported review action: {action}"

    target, error = resolve_review_target_for_entry(entry, output_path)

    # Guard: Unresolvable entries cannot be persisted.
    if target is None:
        return False, error

    return apply_review_action_for_target(target, normalized_action)


__all__ = [
    "ReviewTarget",
    "apply_review_action_for_entry",
    "apply_review_action_for_target",
    "get_selected_row",
    "resolve_review_target_for_entry",
]
