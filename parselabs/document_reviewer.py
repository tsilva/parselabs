"""Document-centric reviewer for processed lab extraction outputs."""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import pandas as pd

from parselabs.config import LabSpecsConfig
from parselabs.paths import get_static_dir
from parselabs.review import (
    SOURCE_BBOX_LABEL,
    build_page_image_value_for_document,
    build_reason_badges,
    format_mapped_reference_text,
    format_mapped_value,
    format_raw_value,
    format_reference_text,
    format_text,
    normalize_review_status,
)
from parselabs.review_state import ReviewTarget, apply_review_action_for_target, get_selected_row
from parselabs.rows import ProcessedDocument, build_review_rows, get_document_review_summary, iter_processed_documents
from parselabs.runtime import RuntimeContext

logger = logging.getLogger(__name__)

_STATIC_DIR = get_static_dir()
KEYBOARD_SHORTCUTS_JS = (_STATIC_DIR / "review_documents.js").read_text()
CUSTOM_CSS = (_STATIC_DIR / "review_documents.css").read_text()

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
    image_value: tuple[str, list[tuple[tuple[int, int, int, int], str]]] | None
    inspector_html: str
    queue_display: pd.DataFrame
    queue_state: pd.DataFrame

    def as_outputs(self) -> tuple[int, tuple[str, list[tuple[tuple[int, int, int, int], str]]] | None, str, pd.DataFrame, pd.DataFrame]:
        """Return UI fragments in the order expected by Gradio callbacks."""

        return (
            self.current_index,
            self.image_value,
            self.inspector_html,
            self.queue_display,
            self.queue_state,
        )


def _empty_queue_state() -> pd.DataFrame:
    """Return an empty queue-state dataframe with stable columns."""

    return pd.DataFrame(columns=QUEUE_STATE_COLUMNS)


def _empty_queue_display() -> pd.DataFrame:
    """Return an empty queue display with stable headers."""

    return pd.DataFrame(columns=QUEUE_DISPLAY_COLUMNS)


def _get_documents(output_path: Path) -> list[ProcessedDocument]:
    """Return all processed documents for the active output path."""

    return iter_processed_documents(output_path)


def _get_document_by_id(doc_id: str | None, output_path: Path) -> ProcessedDocument | None:
    """Resolve a processed document from its directory name."""

    # Guard: Empty selection means there is no active document.
    if not doc_id:
        return None

    for document in _get_documents(output_path):
        # Match the directory name because it is unique within the processed output path.
        if document.doc_dir.name == doc_id:
            return document

    return None


def _get_review_frame(document: ProcessedDocument | None, lab_specs: LabSpecsConfig) -> pd.DataFrame:
    """Load the current review dataframe for a processed document."""

    # Guard: No selected document means no rows to render.
    if document is None:
        return pd.DataFrame()

    return build_review_rows(document.doc_dir, lab_specs).fillna("")


def _matches_document_filter(
    document: ProcessedDocument,
    filter_mode: str,
    lab_specs: LabSpecsConfig,
) -> bool:
    """Return whether a document should appear under the selected review filter."""

    review_df = _get_review_frame(document, lab_specs)
    summary = get_document_review_summary(document.doc_dir, review_df)

    # Show every document when no fixture-readiness filter is active.
    if filter_mode == "All":
        return True

    # Keep only incomplete documents when the reviewer wants remaining work.
    if filter_mode == "Not Fixture Ready":
        return not summary.fixture_ready

    # Otherwise surface only documents already promoted into reviewed truth.
    return summary.fixture_ready


def _format_queue_status_icon(status_code: object) -> str:
    """Render the document-table status column as compact symbols."""

    if status_code == "A":
        return "✅"
    if status_code == "R":
        return "❌"
    return ""


def _build_queue_state(review_df: pd.DataFrame, show_reviewed: bool) -> pd.DataFrame:
    """Build the visible row queue for the active document."""

    # Guard: Empty documents render an empty queue state.
    if review_df.empty:
        return _empty_queue_state()

    queue_df = review_df.copy().reset_index().rename(columns={"index": "actual_index"})
    queue_df["status_normalized"] = queue_df["review_status"].apply(normalize_review_status)
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
    queue_df["raw_lab_label"] = queue_df["raw_lab_name"].apply(format_text)
    queue_df["raw_value_label"] = queue_df.apply(format_raw_value, axis=1)
    queue_df["mapped_lab_label"] = queue_df["lab_name"].apply(format_text)
    return queue_df[QUEUE_STATE_COLUMNS].copy()


def _build_queue_display(queue_state: pd.DataFrame, current_index: int) -> pd.DataFrame:
    """Build the queue dataframe shown in the left navigation pane."""

    # Guard: Empty queue state renders stable table headers with no rows.
    if queue_state.empty:
        return _empty_queue_display()

    display_df = pd.DataFrame(
        {
            "Current": "",
            "St": queue_state["status_code"].apply(_format_queue_status_icon),
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

    return get_selected_row(review_df, current_index)


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

    status = normalize_review_status(current_row.get("review_status")) or "pending"
    status_label = status.capitalize()
    reference_text = format_reference_text(current_row)
    mapped_reference_text = format_mapped_reference_text(current_row)
    comments_text = format_text(current_row.get("raw_comments"), empty="")
    reason_badges_html = build_reason_badges(current_row.get("review_reason"))

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
        '<div class="review-panel-title">Mapped</div>'
        f'<div class="review-field"><span>Lab</span><strong>{format_text(current_row.get("lab_name"))}</strong></div>'
        f'<div class="review-field"><span>Value</span><strong>{format_mapped_value(current_row)}</strong></div>'
        f'<div class="review-field"><span>Reference</span><strong>{mapped_reference_text}</strong></div>'
        "</div>"
        '<div class="review-compare-panel">'
        '<div class="review-panel-title">Raw</div>'
        f'<div class="review-field"><span>Lab</span><strong>{format_text(current_row.get("raw_lab_name"))}</strong></div>'
        f'<div class="review-field"><span>Value</span><strong>{format_raw_value(current_row)}</strong></div>'
        f'<div class="review-field"><span>Reference</span><strong>{reference_text}</strong></div>'
        "</div>"
        "</div>"
        f"{meta_html}"
        "</div>"
    )


def _render_document(
    document: ProcessedDocument | None,
    requested_index: int | None,
    show_reviewed: bool,
    output_path: Path,
    lab_specs: LabSpecsConfig,
    *,
    prefer_first_visible: bool,
) -> ReviewerView:
    """Render all UI fragments for the active document and queue selection."""

    review_df = _get_review_frame(document, lab_specs)
    queue_state = _build_queue_state(review_df, show_reviewed)
    current_index = _resolve_current_index(queue_state, requested_index, prefer_first_visible)
    current_row = _get_current_row(review_df, current_index)
    # Resolve the current page image and overlay only when there is an active row selection.
    image_value = None
    if document is not None and current_row is not None:
        image_value = build_page_image_value_for_document(
            document.doc_dir,
            current_row,
            label=SOURCE_BBOX_LABEL,
        )

    return ReviewerView(
        current_index=current_index,
        image_value=image_value,
        inspector_html=_build_inspector_html(document, review_df, current_index, show_reviewed),
        queue_display=_build_queue_display(queue_state, current_index),
        queue_state=queue_state,
    )


def _build_dropdown_choices(
    documents: list[ProcessedDocument],
    filter_mode: str,
    lab_specs: LabSpecsConfig,
) -> list[tuple[str, str]]:
    """Build labeled dropdown choices from processed document summaries."""

    ranked_documents: list[tuple[int, int, str, str, str]] = []

    # Label each document with review progress so triage stays possible from the toolbar.
    for document in documents:
        # Skip documents excluded by the active fixture-readiness filter.
        if not _matches_document_filter(document, filter_mode, lab_specs):
            continue

        review_df = _get_review_frame(document, lab_specs)
        summary = get_document_review_summary(document.doc_dir, review_df)
        label = f"{document.stem} (pending {summary.pending}, rejected {summary.rejected}, missing {summary.missing_row_markers}, reviewed {summary.reviewed}/{summary.total})"
        ranked_documents.append((1 if summary.fixture_ready else 0, -summary.pending, document.stem.lower(), label, document.doc_dir.name))

    ranked_documents.sort()
    return [(label, doc_id) for _, _, _, label, doc_id in ranked_documents]


def _build_dropdown_state(
    current_doc_id: str | None,
    filter_mode: str,
    output_path: Path,
    lab_specs: LabSpecsConfig,
    *,
    rebuild_all: bool,
) -> DropdownState:
    """Resolve dropdown choices and the selected document for the current toolbar filter."""

    documents = _get_documents(output_path)
    choices = _build_dropdown_choices(documents, filter_mode, lab_specs)
    available_ids = {value for _, value in choices}

    # Preserve the current document when it still matches the active filter.
    if current_doc_id in available_ids:
        current_document = _get_document_by_id(current_doc_id, output_path)
        current_review_df = _get_review_frame(current_document, lab_specs)
        current_summary = get_document_review_summary(current_document.doc_dir, current_review_df) if current_document is not None else None

        # Advance to the next ranked document once the current document has no pending rows left.
        if current_summary is not None and current_summary.pending == 0 and choices and choices[0][1] != current_doc_id:
            selected_id = choices[0][1]

        # Otherwise stay on the current document.
        else:
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
    """Compose the shared output tuple for callbacks that may switch documents."""

    return (
        dropdown_state.selected_id,
        *view.as_outputs(),
    )


def _rerender_toolbar_state(
    current_doc_id: str | None,
    current_index: int | None,
    filter_mode: str,
    show_reviewed: bool,
    output_path: Path,
    lab_specs: LabSpecsConfig,
    *,
    rebuild_all: bool,
    prefer_first_visible: bool,
) -> tuple:
    """Rebuild dropdown state and rerender the selected document in one path."""

    dropdown_state = _build_dropdown_state(current_doc_id, filter_mode, output_path, lab_specs, rebuild_all=rebuild_all)
    selected_document = _get_document_by_id(dropdown_state.selected_id, output_path)
    selected_changed = dropdown_state.selected_id != current_doc_id
    view = _render_document(
        selected_document,
        None if prefer_first_visible or selected_changed else current_index,
        show_reviewed,
        output_path,
        lab_specs,
        prefer_first_visible=prefer_first_visible or selected_changed,
    )
    return _build_toolbar_outputs(dropdown_state, view)


def _persist_row_action(document: ProcessedDocument, current_row: pd.Series, action: str) -> tuple[bool, str]:
    """Persist one review action for the currently selected row."""

    target = ReviewTarget(
        doc_dir=document.doc_dir,
        page_number=int(current_row["page_number"]),
        result_index=int(current_row["result_index"]),
    )
    return apply_review_action_for_target(target, action)


def _handle_document_list_refresh(
    current_doc_id: str | None,
    current_index: int,
    filter_mode: str,
    show_reviewed: bool,
    output_path: Path,
    lab_specs: LabSpecsConfig,
) -> tuple:
    """Refresh the visible document list and rerender the active document."""

    return _rerender_toolbar_state(
        current_doc_id,
        current_index,
        filter_mode,
        show_reviewed,
        output_path,
        lab_specs,
        rebuild_all=True,
        prefer_first_visible=False,
    )


def _handle_document_change(
    doc_id: str | None,
    show_reviewed: bool,
    output_path: Path,
    lab_specs: LabSpecsConfig,
) -> tuple[int, tuple[str, list[tuple[tuple[int, int, int, int], str]]] | None, str, pd.DataFrame, pd.DataFrame, str]:
    """Render a newly selected document, starting from its first visible queue row."""

    document = _get_document_by_id(doc_id, output_path)
    return _render_document(document, None, show_reviewed, output_path, lab_specs, prefer_first_visible=True).as_outputs()


def _handle_show_reviewed_change(
    doc_id: str | None,
    current_index: int,
    show_reviewed: bool,
    output_path: Path,
    lab_specs: LabSpecsConfig,
) -> tuple[int, tuple[str, list[tuple[tuple[int, int, int, int], str]]] | None, str, pd.DataFrame, pd.DataFrame, str]:
    """Toggle whether accepted and rejected rows stay visible in the queue."""

    document = _get_document_by_id(doc_id, output_path)
    return _render_document(document, current_index, show_reviewed, output_path, lab_specs, prefer_first_visible=False).as_outputs()


def _handle_queue_select(
    doc_id: str | None,
    queue_state: pd.DataFrame,
    show_reviewed: bool,
    evt: gr.SelectData,
    output_path: Path,
    lab_specs: LabSpecsConfig,
) -> tuple[int, tuple[str, list[tuple[tuple[int, int, int, int], str]]] | None, str, pd.DataFrame, pd.DataFrame, str]:
    """Select a row directly from the left queue pane."""

    document = _get_document_by_id(doc_id, output_path)

    # Guard: Ignore queue-selection events when there is no visible queue.
    if evt is None or queue_state.empty:
        return _render_document(document, None, show_reviewed, output_path, lab_specs, prefer_first_visible=False).as_outputs()

    selected_index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index

    # Guard: Ignore clicks that do not resolve to a visible queue row.
    if selected_index is None or selected_index < 0 or selected_index >= len(queue_state):
        return _render_document(document, None, show_reviewed, output_path, lab_specs, prefer_first_visible=False).as_outputs()

    actual_index = int(queue_state.iloc[int(selected_index)]["actual_index"])
    return _render_document(document, actual_index, show_reviewed, output_path, lab_specs, prefer_first_visible=False).as_outputs()


def _dispatch_queue_select(
    doc_id: str | None,
    queue_state: pd.DataFrame,
    evt: gr.SelectData,
    show_reviewed: bool,
    output_path: Path,
    lab_specs: LabSpecsConfig,
) -> tuple[int, tuple[str, list[tuple[tuple[int, int, int, int], str]]] | None, str, pd.DataFrame, pd.DataFrame]:
    """Adapt Gradio's input-first select callback order to the queue-select handler."""

    return _handle_queue_select(
        doc_id,
        queue_state,
        show_reviewed,
        evt,
        output_path,
        lab_specs,
    )


def _move_row(
    doc_id: str | None,
    current_index: int,
    delta: int,
    show_reviewed: bool,
    output_path: Path,
    lab_specs: LabSpecsConfig,
) -> tuple[int, tuple[str, list[tuple[tuple[int, int, int, int], str]]] | None, str, pd.DataFrame, pd.DataFrame, str]:
    """Move to the previous or next visible queue row."""

    document = _get_document_by_id(doc_id, output_path)
    review_df = _get_review_frame(document, lab_specs)
    queue_state = _build_queue_state(review_df, show_reviewed)

    # Guard: Empty queue state means there is nothing to navigate.
    if queue_state.empty:
        return _render_document(document, None, show_reviewed, output_path, lab_specs, prefer_first_visible=False).as_outputs()

    visible_indices = [int(value) for value in queue_state["actual_index"].tolist()]

    # Default to the first visible row when the current selection disappeared.
    if current_index not in set(visible_indices):
        return _render_document(document, visible_indices[0], show_reviewed, output_path, lab_specs, prefer_first_visible=False).as_outputs()

    current_position = visible_indices.index(int(current_index))
    next_position = max(0, min(current_position + delta, len(visible_indices) - 1))
    return _render_document(document, visible_indices[next_position], show_reviewed, output_path, lab_specs, prefer_first_visible=False).as_outputs()


def _choose_next_pending_index(review_df: pd.DataFrame, current_index: int) -> int:
    """Choose the next pending row after an accept or reject action."""

    # Guard: Empty documents have no pending row to advance into.
    if review_df.empty:
        return -1

    pending_indices = [idx for idx, status in enumerate(review_df["review_status"].tolist()) if normalize_review_status(status) == ""]

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
    output_path: Path,
    lab_specs: LabSpecsConfig,
) -> tuple:
    """Persist an accept, reject, or undo action and rerender the reviewer."""

    document = _get_document_by_id(doc_id, output_path)
    review_df = _get_review_frame(document, lab_specs)
    current_row = _get_current_row(review_df, current_index)

    # Guard: Ignore actions when there is no active row to mutate.
    if document is None or current_row is None:
        return _rerender_toolbar_state(
            doc_id,
            current_index,
            filter_mode,
            show_reviewed,
            output_path,
            lab_specs,
            rebuild_all=False,
            prefer_first_visible=False,
        )

    action = "clear" if status is None else ("accept" if status == "accepted" else "reject")
    success, error = _persist_row_action(document, current_row, action)

    # Guard: Surface persistence errors without advancing away from the current row.
    if not success:
        gr.Warning(error)
        return _rerender_toolbar_state(
            doc_id,
            current_index,
            filter_mode,
            show_reviewed,
            output_path,
            lab_specs,
            rebuild_all=False,
            prefer_first_visible=False,
        )

    refreshed_df = _get_review_frame(document, lab_specs)

    # Undo keeps the current row selected so the reviewer can immediately decide again.
    if status is None:
        next_index = current_index

    # Accept and reject auto-advance to the best remaining pending row.
    else:
        next_index = _choose_next_pending_index(refreshed_df, current_index)

    return _rerender_toolbar_state(
        doc_id,
        next_index,
        filter_mode,
        show_reviewed,
        output_path,
        lab_specs,
        rebuild_all=False,
        prefer_first_visible=False,
    )


def _mark_missing_row(
    doc_id: str | None,
    current_index: int,
    filter_mode: str,
    show_reviewed: bool,
    output_path: Path,
    lab_specs: LabSpecsConfig,
) -> tuple:
    """Persist a missing-row marker and rerender the active document."""

    document = _get_document_by_id(doc_id, output_path)
    review_df = _get_review_frame(document, lab_specs)
    current_row = _get_current_row(review_df, current_index)

    # Guard: Ignore missing-row markers when there is no active row to anchor them to.
    if document is None or current_row is None:
        return _rerender_toolbar_state(
            doc_id,
            current_index,
            filter_mode,
            show_reviewed,
            output_path,
            lab_specs,
            rebuild_all=False,
            prefer_first_visible=False,
        )

    success, error = _persist_row_action(document, current_row, "missing_row")

    # Guard: Surface persistence errors without moving the current row.
    if not success:
        gr.Warning(error)
        return _rerender_toolbar_state(
            doc_id,
            current_index,
            filter_mode,
            show_reviewed,
            output_path,
            lab_specs,
            rebuild_all=False,
            prefer_first_visible=False,
        )

    gr.Info("Missing-row marker recorded. Resolve it by editing the page JSON and clearing review_missing_rows.")
    return _rerender_toolbar_state(
        doc_id,
        current_index,
        filter_mode,
        show_reviewed,
        output_path,
        lab_specs,
        rebuild_all=False,
        prefer_first_visible=False,
    )


def build_app(context: RuntimeContext) -> gr.Blocks:
    """Build the Gradio document-reviewer app."""

    initial_filter = "All"
    initial_show_reviewed = True
    output_path = context.output_path if context.output_path is not None else Path("./output")
    lab_specs = context.lab_specs
    dropdown_state = _build_dropdown_state(None, initial_filter, output_path, lab_specs, rebuild_all=True)
    initial_view = _render_document(
        _get_document_by_id(dropdown_state.selected_id, output_path),
        None,
        initial_show_reviewed,
        output_path,
        lab_specs,
        prefer_first_visible=True,
    )

    with gr.Blocks(
        title="Processed Document Reviewer",
        fill_width=True,
        fill_height=True,
    ) as demo:
        current_document_id = gr.State(dropdown_state.selected_id)
        current_row_index = gr.State(initial_view.current_index)
        queue_state = gr.State(initial_view.queue_state)

        with gr.Row(elem_id="review-main-pane", equal_height=True):
            with gr.Column(scale=7, min_width=680, elem_id="review-image-pane"):
                page_image = gr.AnnotatedImage(
                    value=initial_view.image_value,
                    color_map={SOURCE_BBOX_LABEL: "#dc2626"},
                    show_legend=False,
                    show_label=False,
                    elem_id="review-page-image",
                )

            with gr.Column(scale=4, min_width=360, elem_id="review-inspector-pane"):
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
            page_image,
            inspector_html,
            queue_table,
            queue_state,
        ]

        action_outputs = [
            current_document_id,
            *view_outputs,
        ]

        def _handle_queue_table_select(
            doc_id: str | None,
            queue_state: pd.DataFrame,
            evt: gr.SelectData,
        ) -> tuple[int, tuple[str, list[tuple[tuple[int, int, int, int], str]]] | None, str, pd.DataFrame, pd.DataFrame]:
            """Route typed queue-select event data into the shared reviewer handler."""

            return _dispatch_queue_select(
                doc_id,
                queue_state,
                evt,
                initial_show_reviewed,
                output_path,
                lab_specs,
            )

        queue_table.select(
            fn=_handle_queue_table_select,
            inputs=[current_document_id, queue_state],
            outputs=view_outputs,
        )

        prev_btn.click(
            fn=lambda doc_id, idx: _move_row(doc_id, idx, -1, initial_show_reviewed, output_path, lab_specs),
            inputs=[current_document_id, current_row_index],
            outputs=view_outputs,
        )

        next_btn.click(
            fn=lambda doc_id, idx: _move_row(doc_id, idx, 1, initial_show_reviewed, output_path, lab_specs),
            inputs=[current_document_id, current_row_index],
            outputs=view_outputs,
        )

        accept_btn.click(
            fn=lambda doc_id, idx: _apply_review_action(
                doc_id,
                idx,
                initial_filter,
                initial_show_reviewed,
                "accepted",
                output_path,
                lab_specs,
            ),
            inputs=[current_document_id, current_row_index],
            outputs=action_outputs,
        )

        reject_btn.click(
            fn=lambda doc_id, idx: _apply_review_action(
                doc_id,
                idx,
                initial_filter,
                initial_show_reviewed,
                "rejected",
                output_path,
                lab_specs,
            ),
            inputs=[current_document_id, current_row_index],
            outputs=action_outputs,
        )

        missing_btn.click(
            fn=lambda doc_id, idx: _mark_missing_row(
                doc_id,
                idx,
                initial_filter,
                initial_show_reviewed,
                output_path,
                lab_specs,
            ),
            inputs=[current_document_id, current_row_index],
            outputs=action_outputs,
        )

        undo_btn.click(
            fn=lambda doc_id, idx: _apply_review_action(
                doc_id,
                idx,
                initial_filter,
                initial_show_reviewed,
                None,
                output_path,
                lab_specs,
            ),
            inputs=[current_document_id, current_row_index],
            outputs=action_outputs,
        )

    return demo
