"""Document-centric reviewer for processed lab extraction outputs."""

from __future__ import annotations

import argparse
import html
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import gradio as gr
import pandas as pd

from parselabs.config import LabSpecsConfig, ProfileConfig
from parselabs.paths import get_static_dir
from parselabs.review_sync import (
    ProcessedDocument,
    get_document_review_summary,
    get_page_image_path,
    iter_processed_documents,
    rebuild_document_csv,
    save_missing_row_marker,
    save_review_status,
)

logger = logging.getLogger(__name__)

_STATIC_DIR = get_static_dir()

KEYBOARD_SHORTCUTS_JS = (_STATIC_DIR / "review_documents.js").read_text()
CUSTOM_CSS = (_STATIC_DIR / "review_documents.css").read_text()

_output_path: Path | None = None
_lab_specs: LabSpecsConfig | None = None

QUEUE_STATE_COLUMNS = [
    "actual_index",
    "status_code",
    "page_label",
    "row_label",
    "raw_lab_label",
    "raw_value_label",
    "mapped_lab_label",
]

QUEUE_DISPLAY_COLUMNS = [
    "Current",
    "St",
    "Pg",
    "Row",
    "Raw Lab",
    "Raw Value",
    "Mapped Lab",
]


@dataclass(frozen=True)
class DropdownState:
    """Resolved document-dropdown state after filter or refresh events."""

    selected_id: str | None
    choices: list[tuple[str, str]]
    status_text: str

    def as_update(self) -> dict:
        """Return the Gradio update payload for the document dropdown."""

        return gr.update(choices=self.choices, value=self.selected_id)


@dataclass(frozen=True)
class ReviewerView:
    """All UI fragments needed to render the active reviewer state."""

    current_index: int
    page_context_html: str
    image_value: str | None
    inspector_html: str
    queue_display: pd.DataFrame
    queue_state: pd.DataFrame
    progress_html: str

    def as_outputs(self) -> tuple[int, str, str | None, str, pd.DataFrame, pd.DataFrame, str]:
        """Return UI fragments in the order expected by Gradio callbacks."""

        return (
            self.current_index,
            self.page_context_html,
            self.image_value,
            self.inspector_html,
            self.queue_display,
            self.queue_state,
            self.progress_html,
        )


def set_output_path(path: Path) -> None:
    """Persist the active processed-output directory for UI callbacks."""

    global _output_path
    _output_path = path


def get_output_path() -> Path:
    """Return the active processed-output directory."""

    # Guard: Default to a local output directory when the CLI did not set a profile.
    if _output_path is None:
        return Path("./output")

    return _output_path


def get_lab_specs() -> LabSpecsConfig:
    """Return the shared lab specs config used by review helpers."""

    global _lab_specs

    # Initialize the config lazily so imports stay lightweight.
    if _lab_specs is None:
        _lab_specs = LabSpecsConfig()

    return _lab_specs


def load_profile(profile_name: str) -> ProfileConfig | None:
    """Load a configured profile and apply its output path."""

    profile_path = ProfileConfig.find_path(profile_name)

    # Guard: Unknown profiles cannot be used to resolve processed outputs.
    if not profile_path:
        return None

    profile = ProfileConfig.from_file(profile_path)

    # Guard: This reviewer needs an output path with processed documents.
    if profile.output_path:
        set_output_path(profile.output_path)

    return profile


def parse_args() -> argparse.Namespace:
    """Parse document-reviewer CLI arguments."""

    parser = argparse.ArgumentParser(description="Review processed lab documents line by line")
    parser.add_argument("--profile", help="Profile name used to locate the processed output directory")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles and exit")
    return parser.parse_args()


def _empty_queue_state() -> pd.DataFrame:
    """Return an empty queue-state dataframe with stable columns."""

    return pd.DataFrame(columns=QUEUE_STATE_COLUMNS)


def _empty_queue_display() -> pd.DataFrame:
    """Return an empty queue display with stable headers."""

    return pd.DataFrame(columns=QUEUE_DISPLAY_COLUMNS)


def _get_documents() -> list[ProcessedDocument]:
    """Return all processed documents for the active output path."""

    return iter_processed_documents(get_output_path())


def _get_document_by_id(doc_id: str | None) -> ProcessedDocument | None:
    """Resolve a processed document from its directory name."""

    # Guard: Empty selection means there is no active document.
    if not doc_id:
        return None

    for document in _get_documents():
        # Match the directory name because it is unique within the processed output path.
        if document.doc_dir.name == doc_id:
            return document

    return None


def _rebuild_all_document_csvs() -> list[ProcessedDocument]:
    """Refresh every per-document CSV from JSON before rendering UI state."""

    documents = _get_documents()
    lab_specs = get_lab_specs()

    # Rebuild each document CSV so the reviewer always starts from current JSON state.
    for document in documents:
        rebuild_document_csv(document.doc_dir, lab_specs)

    return documents


def _build_allowed_paths() -> list[str]:
    """Return filesystem roots Gradio may serve for the document reviewer."""

    allowed_paths: set[str] = set()

    # Read profile configs directly so the reviewer can switch across output roots safely.
    for profile_name in ProfileConfig.list_profiles():
        profile_path = ProfileConfig.find_path(profile_name)

        # Skip profiles that disappeared between discovery and load.
        if not profile_path:
            continue

        profile = ProfileConfig.from_file(profile_path)

        # Skip profiles without an output path because they cannot serve processed files.
        if not profile.output_path:
            continue

        allowed_paths.add(str(profile.output_path))

        # Allow the parent directory too because Gradio checks ancestor roots.
        if profile.output_path.parent != profile.output_path:
            allowed_paths.add(str(profile.output_path.parent))

    return sorted(allowed_paths)


def _get_review_frame(document: ProcessedDocument | None) -> pd.DataFrame:
    """Load the current review dataframe for a processed document."""

    # Guard: No selected document means no rows to render.
    if document is None:
        return pd.DataFrame()

    # Rebuild from JSON when the CSV is missing so the UI can recover gracefully.
    if not document.csv_path.exists():
        rebuild_document_csv(document.doc_dir, get_lab_specs())

    # Guard: Missing or empty CSVs still render a stable empty state.
    if not document.csv_path.exists() or document.csv_path.stat().st_size == 0:
        return pd.DataFrame()

    return pd.read_csv(document.csv_path, keep_default_na=False)


def _matches_document_filter(document: ProcessedDocument, filter_mode: str) -> bool:
    """Return whether a document should appear under the selected review filter."""

    review_df = _get_review_frame(document)
    summary = get_document_review_summary(document.doc_dir, review_df)

    # Show every document when no fixture-readiness filter is active.
    if filter_mode == "All":
        return True

    # Keep only incomplete documents when the reviewer wants remaining work.
    if filter_mode == "Not Fixture Ready":
        return not summary.fixture_ready

    # Otherwise surface only documents already promoted into reviewed truth.
    return summary.fixture_ready


def _normalize_status(status: object) -> str:
    """Normalize reviewer status values into accepted, rejected, or pending."""

    # Guard: Missing statuses stay pending.
    if status is None:
        return ""

    normalized = str(status).strip().lower()

    # Guard: Empty strings are treated as pending decisions.
    if not normalized:
        return ""

    # Guard: Only the two persisted review outcomes are surfaced explicitly.
    if normalized in {"accepted", "rejected"}:
        return normalized

    return ""


def _format_text(value: object, empty: str = "-") -> str:
    """Return a safe string for UI display, normalizing missing values."""

    # Guard: Missing values render as a short placeholder.
    if value is None or pd.isna(value):
        return empty

    text = str(value).strip()

    # Guard: Blank strings also render as a short placeholder.
    if not text:
        return empty

    return html.escape(text)


def _format_raw_value(row: pd.Series) -> str:
    """Format the raw value and raw unit into one compact queue cell."""

    value_text = _format_text(row.get("raw_value"))
    unit_text = _format_text(row.get("raw_lab_unit"), empty="")

    # Omit the placeholder unit suffix when the unit is genuinely absent.
    if unit_text:
        return f"{value_text} {unit_text}".strip()

    return value_text


def _format_mapped_value(row: pd.Series) -> str:
    """Format the normalized value and unit for inspector display."""

    value_text = _format_text(row.get("value"))
    unit_text = _format_text(row.get("lab_unit"), empty="")

    # Omit the placeholder unit suffix when the unit is genuinely absent.
    if unit_text:
        return f"{value_text} {unit_text}".strip()

    return value_text


def _format_reference_text(row: pd.Series) -> str:
    """Format the best available reference range for the inspector."""

    raw_reference = _format_text(row.get("raw_reference_range"), empty="")

    # Prefer the extracted range string when the model provided one.
    if raw_reference:
        return raw_reference

    ref_min = _format_text(row.get("reference_min"), empty="")
    ref_max = _format_text(row.get("reference_max"), empty="")

    # Show a min-max pair when either side is present.
    if ref_min or ref_max:
        return f"{ref_min} - {ref_max}".strip(" -")

    return "-"


def _build_queue_state(review_df: pd.DataFrame, show_reviewed: bool) -> pd.DataFrame:
    """Build the visible row queue for the active document."""

    # Guard: Empty documents render an empty queue state.
    if review_df.empty:
        return _empty_queue_state()

    queue_df = review_df.copy().reset_index().rename(columns={"index": "actual_index"})
    queue_df["status_normalized"] = queue_df["review_status"].apply(_normalize_status)
    queue_df["status_sort"] = queue_df["status_normalized"].apply(lambda status: 0 if not status else 1)

    # Hide accepted and rejected rows unless the reviewer explicitly asks to see them.
    if not show_reviewed:
        queue_df = queue_df[queue_df["status_normalized"] == ""].copy()

    # Guard: Fully reviewed documents have no visible queue when reviewed rows are hidden.
    if queue_df.empty:
        return _empty_queue_state()

    # Keep pending rows at the top, then preserve source order within the document.
    queue_df = queue_df.sort_values(["status_sort", "page_number", "result_index"], kind="mergesort").reset_index(drop=True)
    queue_df["status_code"] = queue_df["status_normalized"].map({"accepted": "A", "rejected": "R", "": "P"}).fillna("P")
    queue_df["page_label"] = queue_df["page_number"].apply(lambda value: str(int(value)) if str(value).strip() else "-")
    queue_df["row_label"] = queue_df["result_index"].apply(lambda value: str(int(value)) if str(value).strip() else "-")
    queue_df["raw_lab_label"] = queue_df["raw_lab_name"].apply(_format_text)
    queue_df["raw_value_label"] = queue_df.apply(_format_raw_value, axis=1)
    queue_df["mapped_lab_label"] = queue_df["lab_name"].apply(_format_text)
    return queue_df[QUEUE_STATE_COLUMNS].copy()


def _build_queue_display(queue_state: pd.DataFrame, current_index: int) -> pd.DataFrame:
    """Build the queue dataframe shown in the left navigation pane."""

    # Guard: Empty queue state renders stable table headers with no rows.
    if queue_state.empty:
        return _empty_queue_display()

    display_df = pd.DataFrame(
        {
            "Current": "",
            "St": queue_state["status_code"],
            "Pg": queue_state["page_label"],
            "Row": queue_state["row_label"],
            "Raw Lab": queue_state["raw_lab_label"],
            "Raw Value": queue_state["raw_value_label"],
            "Mapped Lab": queue_state["mapped_lab_label"],
        }
    )

    # Mark the active row so the queue remains navigable without relying on theme-specific selection styling.
    if current_index in set(queue_state["actual_index"].tolist()):
        selected_row = queue_state.index[queue_state["actual_index"] == current_index][0]
        display_df.loc[selected_row, "Current"] = "->"

    return display_df


def _resolve_current_index(queue_state: pd.DataFrame, requested_index: int | None, prefer_first_visible: bool) -> int:
    """Resolve the active row index from the visible queue."""

    # Guard: Empty queue state means there is no visible row to select.
    if queue_state.empty:
        return -1

    visible_indices = [int(value) for value in queue_state["actual_index"].tolist()]

    # Document switches should jump straight to the first visible row.
    if prefer_first_visible:
        return visible_indices[0]

    # Preserve the current row when it is still visible under the active queue filter.
    if requested_index in set(visible_indices):
        return int(requested_index)

    return visible_indices[0]


def _get_current_row(review_df: pd.DataFrame, current_index: int) -> pd.Series | None:
    """Return the active row from the full review dataframe."""

    # Guard: Empty documents or synthetic indexes have no active row.
    if review_df.empty or current_index < 0 or current_index >= len(review_df):
        return None

    return review_df.iloc[current_index]


def _build_reason_badges(review_reason: object) -> str:
    """Render validation reason codes as compact badges."""

    reason_text = _format_text(review_reason, empty="")

    # Guard: Rows without validation flags do not need badge markup.
    if not reason_text:
        return '<span class="review-inline-muted">None</span>'

    badges: list[str] = []

    # Split semicolon-delimited reasons into individual badges for faster scanning.
    for reason in [part.strip() for part in html.unescape(reason_text).split(";") if part.strip()]:
        badges.append(f'<span class="review-reason-chip">{html.escape(reason)}</span>')

    return "".join(badges)


def _build_pdf_link(document: ProcessedDocument | None) -> str:
    """Build a compact link to the source PDF for the active document."""

    # Guard: No selected document means there is no PDF to open.
    if document is None:
        return '<span class="review-pdf-link disabled">PDF unavailable</span>'

    pdf_href = f"/gradio_api/file={quote(str(document.pdf_path), safe='/:')}"
    return f'<a class="review-pdf-link" href="{pdf_href}" target="_blank" rel="noopener noreferrer">Open PDF</a>'


def _build_page_context_html(document: ProcessedDocument | None, review_df: pd.DataFrame, current_index: int) -> str:
    """Build the compact single-line header shown above the source page image."""

    pdf_link = _build_pdf_link(document)

    # Guard: No selected document gets a neutral placeholder header.
    if document is None:
        return f'<div class="review-pane-header compact"><div class="review-pane-meta">No document selected</div>{pdf_link}</div>'

    current_row = _get_current_row(review_df, current_index)

    # Surface the current page when there is an active row selection.
    if current_row is not None:
        page_title = f"Page {int(current_row['page_number'])}"
    else:
        page_title = "No active row"

    return (
        '<div class="review-pane-header compact">'
        f'<div class="review-pane-meta">{html.escape(document.stem)}<span class="review-pane-separator">•</span>{html.escape(page_title)}</div>'
        f"{pdf_link}"
        "</div>"
    )


def _build_progress_chip(label: str, value: str, tone: str = "neutral") -> str:
    """Build one progress chip for the sticky toolbar."""

    return f'<span class="review-progress-chip {tone}"><span class="label">{html.escape(label)}</span>{html.escape(value)}</span>'


def _build_progress_html(
    document: ProcessedDocument | None,
    review_df: pd.DataFrame,
    queue_state: pd.DataFrame,
    current_index: int,
    show_reviewed: bool,
) -> str:
    """Build the slim sticky toolbar summary for the active document."""

    # Guard: No selected document gets a short neutral summary.
    if document is None:
        return '<div class="review-progress-card"><div class="review-progress-title">No processed documents found</div></div>'

    summary = get_document_review_summary(document.doc_dir, review_df)
    current_row = _get_current_row(review_df, current_index)

    if current_row is not None:
        selection_chip = _build_progress_chip("At", f"P{int(current_row['page_number'])} R{int(current_row['result_index'])}", tone="neutral")
    elif review_df.empty:
        selection_chip = _build_progress_chip("State", "No rows", tone="neutral")
    elif not show_reviewed and summary.pending == 0:
        selection_chip = _build_progress_chip("State", "No pending rows", tone="neutral")
    else:
        selection_chip = _build_progress_chip("State", "No visible row", tone="neutral")

    chips = [
        selection_chip,
        _build_progress_chip("Reviewed", f"{summary.reviewed}/{summary.total}", tone="neutral"),
        _build_progress_chip("Pending", str(summary.pending), tone="warning"),
    ]
    if summary.rejected > 0:
        chips.append(_build_progress_chip("Rejected", str(summary.rejected), tone="danger"))
    if summary.missing_row_markers > 0:
        chips.append(_build_progress_chip("Missing", str(summary.missing_row_markers), tone="warning"))
    chips.append(_build_progress_chip("Fixture", "Ready" if summary.fixture_ready else "Blocked", tone="success" if summary.fixture_ready else "danger"))

    return (
        '<div class="review-progress-card compact">'
        f'<div class="review-progress-title">{html.escape(document.stem)}</div>'
        f'<div class="review-progress-row">{"".join(chips)}</div>'
        "</div>"
    )


def _build_inspector_html(document: ProcessedDocument | None, review_df: pd.DataFrame, current_index: int, show_reviewed: bool) -> str:
    """Render the compact inspector for the active review row."""

    # Guard: No selected document renders a short placeholder card.
    if document is None:
        return '<div class="review-card"><div class="review-empty-state">Select a processed document to start reviewing.</div></div>'

    # Guard: Documents without extracted rows still need a stable inspector.
    if review_df.empty:
        return '<div class="review-card"><div class="review-empty-state">This document has no extracted rows.</div></div>'

    current_row = _get_current_row(review_df, current_index)

    # Guard: Hidden reviewed rows should explain how to reveal them again.
    if current_row is None and not show_reviewed:
        return '<div class="review-card"><div class="review-empty-state">No pending rows remain in this document.</div><div class="review-empty-hint">Enable Show reviewed to inspect accepted and rejected rows.</div></div>'

    # Guard: Fallback placeholder for any other unselected state.
    if current_row is None:
        return '<div class="review-card"><div class="review-empty-state">No row selected.</div></div>'

    status = _normalize_status(current_row.get("review_status")) or "pending"
    status_label = status.capitalize()
    reference_text = _format_reference_text(current_row)
    comments_text = _format_text(current_row.get("raw_comments"), empty="")
    reason_badges_html = _build_reason_badges(current_row.get("review_reason"))

    meta_rows: list[str] = []
    if reason_badges_html != '<span class="review-inline-muted">None</span>':
        meta_rows.append(
            '<div class="review-meta-row"><span>Validation</span>'
            f'<div class="review-badge-row">{reason_badges_html}</div>'
            "</div>"
        )
    if comments_text:
        meta_rows.append(
            '<div class="review-meta-row"><span>Comments</span>'
            f"<strong>{comments_text}</strong>"
            "</div>"
        )

    meta_html = f'<div class="review-meta-list">{"".join(meta_rows)}</div>' if meta_rows else ""

    return (
        '<div class="review-card">'
        '<div class="review-card-header">'
        "<div>"
        '<div class="review-eyebrow">Verify this row</div>'
        f'<div class="review-title">Page {int(current_row["page_number"])} Row {int(current_row["result_index"])}</div>'
        "</div>"
        f'<span class="review-status-chip {status}">{html.escape(status_label)}</span>'
        "</div>"
        '<div class="review-compare-grid">'
        '<div class="review-compare-panel">'
        '<div class="review-panel-title">Raw</div>'
        f'<div class="review-field"><span>Lab</span><strong>{_format_text(current_row.get("raw_lab_name"))}</strong></div>'
        f'<div class="review-field"><span>Value</span><strong>{_format_raw_value(current_row)}</strong></div>'
        f'<div class="review-field"><span>Reference</span><strong>{reference_text}</strong></div>'
        "</div>"
        '<div class="review-compare-panel">'
        '<div class="review-panel-title">Mapped</div>'
        f'<div class="review-field"><span>Lab</span><strong>{_format_text(current_row.get("lab_name"))}</strong></div>'
        f'<div class="review-field"><span>Value</span><strong>{_format_mapped_value(current_row)}</strong></div>'
        f'<div class="review-field"><span>Unit</span><strong>{_format_text(current_row.get("lab_unit"))}</strong></div>'
        "</div>"
        "</div>"
        f"{meta_html}"
        "</div>"
    )


def _render_document(
    document: ProcessedDocument | None,
    requested_index: int | None,
    show_reviewed: bool,
    *,
    prefer_first_visible: bool,
) -> ReviewerView:
    """Render all UI fragments for the active document and queue selection."""

    review_df = _get_review_frame(document)
    queue_state = _build_queue_state(review_df, show_reviewed)
    current_index = _resolve_current_index(queue_state, requested_index, prefer_first_visible)
    current_row = _get_current_row(review_df, current_index)
    image_value = None

    # Resolve the current page image only when there is an active row selection.
    if document is not None and current_row is not None:
        page_number = int(current_row["page_number"])
        image_path = get_page_image_path(document.doc_dir, page_number)
        image_value = str(image_path) if image_path is not None else None

    return ReviewerView(
        current_index=current_index,
        page_context_html=_build_page_context_html(document, review_df, current_index),
        image_value=image_value,
        inspector_html=_build_inspector_html(document, review_df, current_index, show_reviewed),
        queue_display=_build_queue_display(queue_state, current_index),
        queue_state=queue_state,
        progress_html=_build_progress_html(document, review_df, queue_state, current_index, show_reviewed),
    )


def _build_dropdown_choices(documents: list[ProcessedDocument], filter_mode: str) -> list[tuple[str, str]]:
    """Build labeled dropdown choices from processed document summaries."""

    ranked_documents: list[tuple[int, int, str, str, str]] = []

    # Label each document with review progress so triage stays possible from the toolbar.
    for document in documents:
        # Skip documents excluded by the active fixture-readiness filter.
        if not _matches_document_filter(document, filter_mode):
            continue

        review_df = _get_review_frame(document)
        summary = get_document_review_summary(document.doc_dir, review_df)
        label = f"{document.stem} (pending {summary.pending}, rejected {summary.rejected}, missing {summary.missing_row_markers}, reviewed {summary.reviewed}/{summary.total})"
        ranked_documents.append((1 if summary.fixture_ready else 0, -summary.pending, document.stem.lower(), label, document.doc_dir.name))

    ranked_documents.sort()
    return [(label, doc_id) for _, _, _, label, doc_id in ranked_documents]


def _build_dropdown_state(current_doc_id: str | None, filter_mode: str, *, rebuild_all: bool) -> DropdownState:
    """Resolve dropdown choices and the selected document for the current toolbar filter."""

    documents = _rebuild_all_document_csvs() if rebuild_all else _get_documents()
    choices = _build_dropdown_choices(documents, filter_mode)
    available_ids = {value for _, value in choices}

    # Preserve the current document when it still matches the active filter.
    if current_doc_id in available_ids:
        selected_id = current_doc_id

    # Otherwise fall back to the highest-priority remaining document.
    elif choices:
        selected_id = choices[0][1]

    # Guard: Empty choice sets clear the current document selection.
    else:
        selected_id = None

    status_text = f"{len(choices)} shown / {len(documents)} processed document(s)"
    return DropdownState(selected_id=selected_id, choices=choices, status_text=status_text)


def _build_toolbar_outputs(dropdown_state: DropdownState, view: ReviewerView) -> tuple:
    """Compose the shared output tuple for toolbar-mutating callbacks."""

    return (
        dropdown_state.as_update(),
        dropdown_state.status_text,
        *view.as_outputs(),
    )


def _handle_document_list_refresh(
    current_doc_id: str | None,
    current_index: int,
    filter_mode: str,
    show_reviewed: bool,
) -> tuple:
    """Refresh the visible document list and rerender the active document."""

    dropdown_state = _build_dropdown_state(current_doc_id, filter_mode, rebuild_all=True)
    selected_document = _get_document_by_id(dropdown_state.selected_id)

    # Preserve the current row only when the selected document did not change.
    if dropdown_state.selected_id == current_doc_id:
        view = _render_document(selected_document, current_index, show_reviewed, prefer_first_visible=False)
    else:
        view = _render_document(selected_document, None, show_reviewed, prefer_first_visible=True)

    return _build_toolbar_outputs(dropdown_state, view)


def _handle_document_change(doc_id: str | None, show_reviewed: bool) -> tuple[int, str, str | None, str, pd.DataFrame, pd.DataFrame, str]:
    """Render a newly selected document, starting from its first visible queue row."""

    document = _get_document_by_id(doc_id)
    return _render_document(document, None, show_reviewed, prefer_first_visible=True).as_outputs()


def _handle_show_reviewed_change(
    doc_id: str | None,
    current_index: int,
    show_reviewed: bool,
) -> tuple[int, str, str | None, str, pd.DataFrame, pd.DataFrame, str]:
    """Toggle whether accepted and rejected rows stay visible in the queue."""

    document = _get_document_by_id(doc_id)
    return _render_document(document, current_index, show_reviewed, prefer_first_visible=False).as_outputs()


def _handle_queue_select(
    doc_id: str | None,
    queue_state: pd.DataFrame,
    show_reviewed: bool,
    evt: gr.SelectData,
) -> tuple[int, str, str | None, str, pd.DataFrame, pd.DataFrame, str]:
    """Select a row directly from the left queue pane."""

    document = _get_document_by_id(doc_id)

    # Guard: Ignore queue-selection events when there is no visible queue.
    if evt is None or queue_state.empty:
        return _render_document(document, None, show_reviewed, prefer_first_visible=False).as_outputs()

    selected_index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index

    # Guard: Ignore clicks that do not resolve to a visible queue row.
    if selected_index is None or selected_index < 0 or selected_index >= len(queue_state):
        return _render_document(document, None, show_reviewed, prefer_first_visible=False).as_outputs()

    actual_index = int(queue_state.iloc[int(selected_index)]["actual_index"])
    return _render_document(document, actual_index, show_reviewed, prefer_first_visible=False).as_outputs()


def _move_row(
    doc_id: str | None,
    current_index: int,
    delta: int,
    show_reviewed: bool,
) -> tuple[int, str, str | None, str, pd.DataFrame, pd.DataFrame, str]:
    """Move to the previous or next visible queue row."""

    document = _get_document_by_id(doc_id)
    review_df = _get_review_frame(document)
    queue_state = _build_queue_state(review_df, show_reviewed)

    # Guard: Empty queue state means there is nothing to navigate.
    if queue_state.empty:
        return _render_document(document, None, show_reviewed, prefer_first_visible=False).as_outputs()

    visible_indices = [int(value) for value in queue_state["actual_index"].tolist()]

    # Default to the first visible row when the current selection disappeared.
    if current_index not in set(visible_indices):
        return _render_document(document, visible_indices[0], show_reviewed, prefer_first_visible=False).as_outputs()

    current_position = visible_indices.index(int(current_index))
    next_position = max(0, min(current_position + delta, len(visible_indices) - 1))
    return _render_document(document, visible_indices[next_position], show_reviewed, prefer_first_visible=False).as_outputs()


def _choose_next_pending_index(review_df: pd.DataFrame, current_index: int) -> int:
    """Choose the next pending row after an accept or reject action."""

    # Guard: Empty documents have no pending row to advance into.
    if review_df.empty:
        return -1

    pending_indices = [idx for idx, status in enumerate(review_df["review_status"].tolist()) if _normalize_status(status) == ""]

    # Guard: Fully reviewed documents have no pending row to select.
    if not pending_indices:
        return -1

    current_row = _get_current_row(review_df, current_index)
    current_page = int(current_row["page_number"]) if current_row is not None else None

    # Stay on the same page first so the reviewer finishes local context before moving on.
    if current_page is not None:
        for idx in pending_indices:
            if idx > current_index and int(review_df.iloc[idx]["page_number"]) == current_page:
                return idx

    # Otherwise advance forward through the document in source order.
    for idx in pending_indices:
        if idx > current_index:
            return idx

    return pending_indices[0]


def _apply_review_action(
    doc_id: str | None,
    current_index: int,
    filter_mode: str,
    show_reviewed: bool,
    status: str | None,
) -> tuple:
    """Persist an accept, reject, or undo action and rerender the reviewer."""

    document = _get_document_by_id(doc_id)
    review_df = _get_review_frame(document)
    current_row = _get_current_row(review_df, current_index)

    # Guard: Ignore actions when there is no active row to mutate.
    if document is None or current_row is None:
        dropdown_state = _build_dropdown_state(doc_id, filter_mode, rebuild_all=False)
        view = _render_document(_get_document_by_id(dropdown_state.selected_id), current_index, show_reviewed, prefer_first_visible=False)
        return _build_toolbar_outputs(dropdown_state, view)

    success, error = save_review_status(
        document.doc_dir,
        int(current_row["page_number"]),
        int(current_row["result_index"]),
        status,
    )

    # Guard: Surface persistence errors without advancing away from the current row.
    if not success:
        gr.Warning(error)
        dropdown_state = _build_dropdown_state(doc_id, filter_mode, rebuild_all=False)
        view = _render_document(_get_document_by_id(dropdown_state.selected_id), current_index, show_reviewed, prefer_first_visible=False)
        return _build_toolbar_outputs(dropdown_state, view)

    # Rebuild the per-document CSV so the UI immediately reflects persisted JSON state.
    rebuild_document_csv(document.doc_dir, get_lab_specs())
    dropdown_state = _build_dropdown_state(doc_id, filter_mode, rebuild_all=False)
    selected_document = _get_document_by_id(dropdown_state.selected_id)

    # Switching documents should restart from the new document's first visible row.
    if dropdown_state.selected_id != doc_id:
        view = _render_document(selected_document, None, show_reviewed, prefer_first_visible=True)
        return _build_toolbar_outputs(dropdown_state, view)

    refreshed_df = _get_review_frame(document)

    # Undo keeps the current row selected so the reviewer can immediately decide again.
    if status is None:
        next_index = current_index

    # Accept and reject auto-advance to the best remaining pending row.
    else:
        next_index = _choose_next_pending_index(refreshed_df, current_index)

    view = _render_document(document, next_index, show_reviewed, prefer_first_visible=False)
    return _build_toolbar_outputs(dropdown_state, view)


def _mark_missing_row(
    doc_id: str | None,
    current_index: int,
    filter_mode: str,
    show_reviewed: bool,
) -> tuple:
    """Persist a missing-row marker and rerender the active document."""

    document = _get_document_by_id(doc_id)
    review_df = _get_review_frame(document)
    current_row = _get_current_row(review_df, current_index)

    # Guard: Ignore missing-row markers when there is no active row to anchor them to.
    if document is None or current_row is None:
        dropdown_state = _build_dropdown_state(doc_id, filter_mode, rebuild_all=False)
        view = _render_document(_get_document_by_id(dropdown_state.selected_id), current_index, show_reviewed, prefer_first_visible=False)
        return _build_toolbar_outputs(dropdown_state, view)

    success, error = save_missing_row_marker(
        document.doc_dir,
        int(current_row["page_number"]),
        int(current_row["result_index"]),
    )

    # Guard: Surface persistence errors without moving the current row.
    if not success:
        gr.Warning(error)
        dropdown_state = _build_dropdown_state(doc_id, filter_mode, rebuild_all=False)
        view = _render_document(_get_document_by_id(dropdown_state.selected_id), current_index, show_reviewed, prefer_first_visible=False)
        return _build_toolbar_outputs(dropdown_state, view)

    # Rebuild the per-document CSV so counters and queue state reflect the new marker.
    rebuild_document_csv(document.doc_dir, get_lab_specs())
    gr.Info("Missing-row marker recorded. Resolve it by editing the page JSON and clearing review_missing_rows.")

    dropdown_state = _build_dropdown_state(doc_id, filter_mode, rebuild_all=False)
    selected_document = _get_document_by_id(dropdown_state.selected_id)

    # Switching documents should restart from the new document's first visible row.
    if dropdown_state.selected_id != doc_id:
        view = _render_document(selected_document, None, show_reviewed, prefer_first_visible=True)
        return _build_toolbar_outputs(dropdown_state, view)

    view = _render_document(document, current_index, show_reviewed, prefer_first_visible=False)
    return _build_toolbar_outputs(dropdown_state, view)


def build_app() -> gr.Blocks:
    """Build the Gradio document-reviewer app."""

    initial_filter = "All"
    initial_show_reviewed = False
    dropdown_state = _build_dropdown_state(None, initial_filter, rebuild_all=True)
    initial_view = _render_document(
        _get_document_by_id(dropdown_state.selected_id),
        None,
        initial_show_reviewed,
        prefer_first_visible=True,
    )

    with gr.Blocks(title="Processed Document Reviewer") as demo:
        gr.Markdown("# Processed Document Reviewer")

        current_row_index = gr.State(initial_view.current_index)
        queue_state = gr.State(initial_view.queue_state)

        with gr.Column(elem_id="review-toolbar"):
            with gr.Row():
                document_filter = gr.Dropdown(
                    choices=["All", "Not Fixture Ready", "Fixture Ready"],
                    value=initial_filter,
                    label="Document Filter",
                    scale=1,
                )
                document_dropdown = gr.Dropdown(
                    choices=dropdown_state.choices,
                    value=dropdown_state.selected_id,
                    label="Document",
                    scale=3,
                )
                show_reviewed = gr.Checkbox(
                    label="Show reviewed",
                    value=initial_show_reviewed,
                    scale=1,
                    min_width=140,
                )
                refresh_btn = gr.Button("Refresh", scale=0, min_width=110)

            toolbar_status = gr.Markdown(dropdown_state.status_text, elem_id="review-toolbar-status")
            progress_html = gr.HTML(initial_view.progress_html)

        with gr.Row(elem_id="review-main-pane"):
            with gr.Column(scale=6, min_width=520, elem_id="review-image-pane"):
                page_context = gr.HTML(initial_view.page_context_html)
                page_image = gr.Image(
                    value=initial_view.image_value,
                    type="filepath",
                    show_label=False,
                    height=640,
                    elem_id="review-page-image",
                )

            with gr.Column(scale=5, min_width=380, elem_id="review-inspector-pane"):
                inspector_html = gr.HTML(initial_view.inspector_html, elem_id="review-inspector")

                with gr.Column(elem_id="review-action-bar"):
                    with gr.Row():
                        prev_btn = gr.Button("Previous [k]", elem_id="review-prev-btn")
                        next_btn = gr.Button("Next [j]", elem_id="review-next-btn")
                        undo_btn = gr.Button("Undo [u]", elem_id="review-undo-btn")

                    with gr.Row():
                        accept_btn = gr.Button("Accept [y]", variant="primary", elem_id="review-accept-btn")
                        reject_btn = gr.Button("Reject [n]", variant="stop", elem_id="review-reject-btn")
                        missing_btn = gr.Button("Missing [m]", elem_id="review-missing-btn")

        with gr.Column(elem_id="review-table-pane"):
            gr.Markdown("### Document Table")
            gr.Markdown("*Use the table to scan the document in bulk. Click a row to jump.*")
            queue_table = gr.Dataframe(
                value=initial_view.queue_display,
                interactive=False,
                show_label=False,
                wrap=True,
                max_height=300,
                elem_id="review-queue",
            )

        gr.Markdown("*Keyboard: Y=Accept, N=Reject, M=Missing, U=Undo, Arrow keys or J/K=Navigate*")

        view_outputs = [
            current_row_index,
            page_context,
            page_image,
            inspector_html,
            queue_table,
            queue_state,
            progress_html,
        ]

        toolbar_outputs = [
            document_dropdown,
            toolbar_status,
            *view_outputs,
        ]

        document_dropdown.change(
            fn=_handle_document_change,
            inputs=[document_dropdown, show_reviewed],
            outputs=view_outputs,
        )

        document_filter.change(
            fn=_handle_document_list_refresh,
            inputs=[document_dropdown, current_row_index, document_filter, show_reviewed],
            outputs=toolbar_outputs,
        )

        refresh_btn.click(
            fn=_handle_document_list_refresh,
            inputs=[document_dropdown, current_row_index, document_filter, show_reviewed],
            outputs=toolbar_outputs,
        )

        show_reviewed.change(
            fn=_handle_show_reviewed_change,
            inputs=[document_dropdown, current_row_index, show_reviewed],
            outputs=view_outputs,
        )

        queue_table.select(
            fn=_handle_queue_select,
            inputs=[document_dropdown, queue_state, show_reviewed],
            outputs=view_outputs,
        )

        prev_btn.click(
            fn=lambda doc_id, idx, visible: _move_row(doc_id, idx, -1, visible),
            inputs=[document_dropdown, current_row_index, show_reviewed],
            outputs=view_outputs,
        )

        next_btn.click(
            fn=lambda doc_id, idx, visible: _move_row(doc_id, idx, 1, visible),
            inputs=[document_dropdown, current_row_index, show_reviewed],
            outputs=view_outputs,
        )

        accept_btn.click(
            fn=lambda doc_id, idx, filter_mode, visible: _apply_review_action(doc_id, idx, filter_mode, visible, "accepted"),
            inputs=[document_dropdown, current_row_index, document_filter, show_reviewed],
            outputs=toolbar_outputs,
        )

        reject_btn.click(
            fn=lambda doc_id, idx, filter_mode, visible: _apply_review_action(doc_id, idx, filter_mode, visible, "rejected"),
            inputs=[document_dropdown, current_row_index, document_filter, show_reviewed],
            outputs=toolbar_outputs,
        )

        missing_btn.click(
            fn=_mark_missing_row,
            inputs=[document_dropdown, current_row_index, document_filter, show_reviewed],
            outputs=toolbar_outputs,
        )

        undo_btn.click(
            fn=lambda doc_id, idx, filter_mode, visible: _apply_review_action(doc_id, idx, filter_mode, visible, None),
            inputs=[document_dropdown, current_row_index, document_filter, show_reviewed],
            outputs=toolbar_outputs,
        )

    return demo


def main() -> None:
    """Document-reviewer CLI entry point."""

    args = parse_args()

    # Print the available profiles and exit when requested.
    if args.list_profiles:
        for profile_name in ProfileConfig.list_profiles():
            print(profile_name)
        return

    # Load the selected profile when one was provided.
    if args.profile:
        profile = load_profile(args.profile)

        # Guard: Unknown profiles cannot start the reviewer.
        if profile is None:
            raise SystemExit(f"Profile '{args.profile}' was not found.")

        # Guard: The reviewer needs a processed output directory.
        if profile.output_path is None:
            raise SystemExit(f"Profile '{args.profile}' has no output_path configured.")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    demo = build_app()
    allowed_paths = _build_allowed_paths()
    logger.info("Starting Processed Document Reviewer on http://localhost:7863")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7863,
        show_error=True,
        inbrowser=False,
        allowed_paths=allowed_paths,
        css=CUSTOM_CSS,
        head=KEYBOARD_SHORTCUTS_JS,
    )


if __name__ == "__main__":
    main()
