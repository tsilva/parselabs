"""Document-centric reviewer for processed lab extraction outputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gradio as gr
import pandas as pd

from parselabs.config import LabSpecsConfig, ProfileConfig
from parselabs.review_sync import (
    ProcessedDocument,
    count_review_missing_rows,
    get_document_review_summary,
    get_page_image_path,
    get_review_summary,
    iter_processed_documents,
    rebuild_document_csv,
    save_missing_row_marker,
    save_review_status,
)

logger = logging.getLogger(__name__)

_output_path: Path | None = None
_lab_specs: LabSpecsConfig | None = None

KEYBOARD_SHORTCUTS_JS = """
<script>
(function() {
    document.addEventListener('keydown', function(event) {
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
            return;
        }

        switch (event.key.toLowerCase()) {
            case 'y':
                document.querySelector('#review-accept-btn')?.click();
                event.preventDefault();
                break;
            case 'n':
                document.querySelector('#review-reject-btn')?.click();
                event.preventDefault();
                break;
            case 'm':
                document.querySelector('#review-missing-btn')?.click();
                event.preventDefault();
                break;
            case 'arrowright':
            case 'j':
                document.querySelector('#review-next-btn')?.click();
                event.preventDefault();
                break;
            case 'arrowleft':
            case 'k':
                document.querySelector('#review-prev-btn')?.click();
                event.preventDefault();
                break;
        }
    });
})();
</script>
"""


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


def _build_dropdown_choices(
    documents: list[ProcessedDocument],
    filter_mode: str,
) -> list[tuple[str, str]]:
    """Build labeled dropdown choices from processed document summaries."""

    choices: list[tuple[str, str]] = []

    # Label each document with its review progress so users can triage quickly.
    for document in documents:
        # Skip documents excluded by the current fixture-readiness filter.
        if not _matches_document_filter(document, filter_mode):
            continue

        review_df = _get_review_frame(document)
        summary = get_document_review_summary(document.doc_dir, review_df)
        label = (
            f"{document.stem}"
            f" (reviewed {summary.reviewed}/{summary.total},"
            f" rejected {summary.rejected},"
            f" pending {summary.pending},"
            f" missing {summary.missing_row_markers})"
        )
        choices.append((label, document.doc_dir.name))

    return choices


def _get_review_frame(document: ProcessedDocument | None) -> pd.DataFrame:
    """Load the current review dataframe for a processed document."""

    # Guard: No selected document means no rows to render.
    if document is None:
        return pd.DataFrame()

    # Rebuild from JSON when the CSV is missing so the UI can recover gracefully.
    if not document.csv_path.exists():
        rebuild_document_csv(document.doc_dir, get_lab_specs())

    if not document.csv_path.exists() or document.csv_path.stat().st_size == 0:
        return pd.DataFrame()

    return pd.read_csv(document.csv_path, keep_default_na=False)


def _first_pending_row(review_df: pd.DataFrame) -> int:
    """Return the first pending row index for a document."""

    # Guard: Empty frames default to the first row index.
    if review_df.empty:
        return 0

    statuses = review_df["review_status"].fillna("").astype(str).str.strip().str.lower()
    pending_rows = review_df.index[statuses == ""].tolist()

    # Prefer the first pending row so review resumes where work remains.
    if pending_rows:
        return int(pending_rows[0])

    return 0


def _clamp_row_index(review_df: pd.DataFrame, row_index: int | None) -> int:
    """Clamp a requested row index into the valid bounds for a dataframe."""

    # Guard: Empty frames always render the synthetic first row.
    if review_df.empty:
        return 0

    # Guard: Missing row indexes should start from the first pending row.
    if row_index is None:
        return _first_pending_row(review_df)

    return max(0, min(int(row_index), len(review_df) - 1))


def _build_page_summary(doc_dir: Path, review_df: pd.DataFrame, page_number: int) -> str:
    """Build a short per-page summary for the active review row."""

    page_df = review_df[review_df["page_number"] == page_number].copy()
    page_summary = get_review_summary(
        page_df,
        missing_row_markers=count_review_missing_rows(doc_dir, page_number=page_number),
    )
    return (
        f"page {page_number}: reviewed {page_summary.reviewed}/{page_summary.total}"
        f" | rejected {page_summary.rejected}"
        f" | pending {page_summary.pending}"
        f" | missing {page_summary.missing_row_markers}"
    )


def _build_progress_text(document: ProcessedDocument | None, review_df: pd.DataFrame, current_index: int) -> str:
    """Build a short progress label for the active document."""

    # Guard: No document means there is nothing to summarize.
    if document is None:
        return "No processed documents found."

    summary = get_document_review_summary(document.doc_dir, review_df)

    # Guard: Empty documents still need a stable label in the UI.
    if review_df.empty:
        return (
            f"{document.stem}: no extracted rows"
            f" | missing {summary.missing_row_markers}"
            f" | fixture-ready {'yes' if summary.fixture_ready else 'no'}"
        )

    page_number = int(review_df.iloc[current_index]["page_number"])
    document_line = (
        f"{document.stem}: row {current_index + 1}/{len(review_df)}"
        f" | reviewed {summary.reviewed}/{summary.total}"
        f" | rejected {summary.rejected}"
        f" | pending {summary.pending}"
        f" | missing {summary.missing_row_markers}"
        f" | fixture-ready {'yes' if summary.fixture_ready else 'no'}"
    )
    page_line = _build_page_summary(document.doc_dir, review_df, page_number)
    return f"{document_line}\n\n{page_line}"


def _build_table(review_df: pd.DataFrame, current_index: int) -> pd.DataFrame:
    """Build a compact, read-only table for the current document."""

    # Guard: Empty documents render an empty table with stable columns.
    if review_df.empty:
        return pd.DataFrame(columns=["current", "page", "row", "raw_lab_name", "raw_value", "lab_name", "value", "lab_unit", "review_status"])

    table_df = review_df[
        [
            "page_number",
            "result_index",
            "raw_lab_name",
            "raw_value",
            "lab_name",
            "value",
            "lab_unit",
            "review_status",
        ]
    ].copy()
    table_df.insert(0, "current", "")
    table_df.rename(columns={"page_number": "page", "result_index": "row"}, inplace=True)

    # Mark the active row so keyboard-less navigation is still obvious.
    if 0 <= current_index < len(table_df):
        table_df.loc[current_index, "current"] = "->"

    return table_df


def _format_entry_html(review_df: pd.DataFrame, current_index: int) -> str:
    """Render the current review row as a compact HTML card."""

    # Guard: Empty documents need a simple placeholder message.
    if review_df.empty:
        return "<div><strong>No extracted rows</strong></div>"

    row = review_df.iloc[current_index]
    status = str(row.get("review_status", "") or "pending").strip() or "pending"
    reason = str(row.get("review_reason", "") or "").strip()
    comments = str(row.get("raw_comments", "") or "").strip()
    mapped_lab = row.get("lab_name") or ""
    normalized_value = row.get("value") if str(row.get("value", "")) != "nan" else ""
    normalized_unit = row.get("lab_unit") or ""
    reference_min = row.get("reference_min") if str(row.get("reference_min", "")) != "nan" else ""
    reference_max = row.get("reference_max") if str(row.get("reference_max", "")) != "nan" else ""
    reference_text = f"{reference_min} - {reference_max}".strip(" -")

    parts = [
        "<div>",
        "<div><strong>Review these fields</strong></div>",
        f"<div><strong>Mapped lab:</strong> <strong>{mapped_lab}</strong></div>",
        f"<div><strong>Normalized value:</strong> <strong>{normalized_value} {normalized_unit}</strong></div>",
        f"<div><strong>Reference:</strong> <strong>{reference_text}</strong></div>",
        "<hr>",
        f"<div><strong>Status:</strong> {status}</div>",
        f"<div><strong>Page:</strong> {row.get('page_number')}</div>",
        f"<div><strong>Row:</strong> {row.get('result_index')}</div>",
        f"<div><strong>Raw lab:</strong> {row.get('raw_lab_name') or ''}</div>",
        f"<div><strong>Raw value:</strong> {row.get('raw_value') or ''} {row.get('raw_lab_unit') or ''}</div>",
    ]

    # Surface validation flags only when they exist for the current row.
    if reason:
        parts.append(f"<div><strong>Validation:</strong> {reason}</div>")

    # Surface extracted comments only when they exist.
    if comments:
        parts.append(f"<div><strong>Comments:</strong> {comments}</div>")

    parts.append(
        "<div><strong>Missing Row:</strong> "
        "Use the Missing Row action when the source page has an omitted lab line. "
        "After manual JSON repair, remove the page's review_missing_rows marker.</div>"
    )
    parts.append("</div>")

    return "<div>" + "".join(parts) + "</div>"


def _render_document(document: ProcessedDocument | None, requested_row_index: int | None) -> tuple[int, str | None, str | None, str, pd.DataFrame, str]:
    """Render all UI fragments for the active document and row."""

    review_df = _get_review_frame(document)
    current_index = _clamp_row_index(review_df, requested_row_index)
    pdf_value = str(document.pdf_path) if document is not None else None
    image_value = None

    # Resolve the current page image when there is an active document row.
    if document is not None and not review_df.empty:
        page_number = int(review_df.iloc[current_index]["page_number"])
        image_path = get_page_image_path(document.doc_dir, page_number)
        image_value = str(image_path) if image_path is not None else None

    progress_text = _build_progress_text(document, review_df, current_index)
    entry_html = _format_entry_html(review_df, current_index)
    table_df = _build_table(review_df, current_index)
    return current_index, pdf_value, image_value, entry_html, table_df, progress_text


def _refresh_document_choices(current_doc_id: str | None, filter_mode: str) -> tuple[gr.Dropdown, str]:
    """Refresh all document CSVs and rebuild the dropdown choices."""

    documents = _rebuild_all_document_csvs()
    choices = _build_dropdown_choices(documents, filter_mode)
    available_ids = {value for _, value in choices}

    # Keep the current document when it still exists after refresh.
    if current_doc_id in available_ids:
        selected_id = current_doc_id
    elif choices:
        selected_id = choices[0][1]
    else:
        selected_id = None

    status_text = f"{len(choices)} shown / {len(documents)} processed document(s)"
    return gr.update(choices=choices, value=selected_id), status_text


def _select_document(doc_id: str | None) -> tuple[int, str | None, str | None, str, pd.DataFrame, str]:
    """Handle document selection changes."""

    document = _get_document_by_id(doc_id)

    # Default to the first pending row when the user switches documents.
    review_df = _get_review_frame(document)
    target_index = _first_pending_row(review_df)
    return _render_document(document, target_index)


def _move_row(doc_id: str | None, current_index: int, delta: int) -> tuple[int, str | None, str | None, str, pd.DataFrame, str]:
    """Move to the previous or next row within the selected document."""

    document = _get_document_by_id(doc_id)
    return _render_document(document, current_index + delta)


def _apply_review_action(doc_id: str | None, current_index: int, status: str) -> tuple[int, str | None, str | None, str, pd.DataFrame, str]:
    """Persist a review action and refresh the active document view."""

    document = _get_document_by_id(doc_id)
    review_df = _get_review_frame(document)
    current_index = _clamp_row_index(review_df, current_index)

    # Guard: Ignore actions when there is no active row to mutate.
    if document is None or review_df.empty:
        return _render_document(document, current_index)

    row = review_df.iloc[current_index]
    success, error = save_review_status(
        document.doc_dir,
        int(row["page_number"]),
        int(row["result_index"]),
        status,
    )

    # Guard: Surface persistence errors without moving the current row.
    if not success:
        gr.Warning(error)
        return _render_document(document, current_index)

    # Rebuild the per-document CSV so the CSV immediately reflects the persisted JSON status.
    rebuild_document_csv(document.doc_dir, get_lab_specs())
    refreshed_df = _get_review_frame(document)
    next_index = _clamp_row_index(refreshed_df, current_index + 1)
    pending_index = _first_pending_row(refreshed_df)

    # Prefer the next pending row when one still exists after the current position.
    if not refreshed_df.empty:
        pending_candidates = refreshed_df.index[
            refreshed_df["review_status"].fillna("").astype(str).str.strip().eq("")
        ].tolist()
        future_candidates = [candidate for candidate in pending_candidates if candidate >= current_index]
        if future_candidates:
            next_index = int(future_candidates[0])
        elif pending_index < len(refreshed_df):
            next_index = pending_index

    return _render_document(document, next_index)


def _mark_missing_row(doc_id: str | None, current_index: int) -> tuple[int, str | None, str | None, str, pd.DataFrame, str]:
    """Persist a missing-row marker without changing the selected extracted row."""

    document = _get_document_by_id(doc_id)
    review_df = _get_review_frame(document)
    current_index = _clamp_row_index(review_df, current_index)

    # Guard: Ignore missing-row markers when there is no active row to anchor them to.
    if document is None or review_df.empty:
        return _render_document(document, current_index)

    row = review_df.iloc[current_index]
    success, error = save_missing_row_marker(
        document.doc_dir,
        int(row["page_number"]),
        int(row["result_index"]),
    )

    # Guard: Surface persistence errors without moving the current row.
    if not success:
        gr.Warning(error)
        return _render_document(document, current_index)

    # Rebuild the per-document CSV so counters and warnings reflect the new missing-row marker.
    rebuild_document_csv(document.doc_dir, get_lab_specs())
    gr.Info("Missing-row marker recorded. Resolve it by editing the page JSON and clearing review_missing_rows.")
    return _render_document(document, current_index)


def build_app() -> gr.Blocks:
    """Build the Gradio document-reviewer app."""

    documents = _rebuild_all_document_csvs()
    initial_filter = "All"
    dropdown_choices = _build_dropdown_choices(documents, initial_filter)
    initial_doc_id = dropdown_choices[0][1] if dropdown_choices else None
    initial_row_index, initial_pdf, initial_image, initial_html, initial_table, initial_progress = _select_document(initial_doc_id)

    with gr.Blocks(title="Processed Document Reviewer") as demo:
        gr.Markdown("# Processed Document Reviewer")
        gr.Markdown(
            "Review each extracted row against its source page, persist decisions back to JSON, "
            "and record missing-row markers for omissions you will fix manually later."
        )

        with gr.Row():
            document_filter = gr.Dropdown(
                choices=["All", "Not Fixture Ready", "Fixture Ready"],
                value=initial_filter,
                label="Document Filter",
            )
            document_dropdown = gr.Dropdown(choices=dropdown_choices, value=initial_doc_id, label="Document")
            refresh_btn = gr.Button("Refresh")
            refresh_status = gr.Markdown(f"{len(dropdown_choices)} shown / {len(documents)} processed document(s)")

        current_row_index = gr.State(initial_row_index)

        with gr.Row():
            with gr.Column(scale=1):
                pdf_file = gr.File(value=initial_pdf, label="Document PDF")
                page_image = gr.Image(value=initial_image, label="Current Page", type="filepath")
            with gr.Column(scale=1):
                progress_markdown = gr.Markdown(initial_progress)
                entry_html = gr.HTML(initial_html)

                with gr.Row():
                    prev_btn = gr.Button("Previous", elem_id="review-prev-btn")
                    next_btn = gr.Button("Next", elem_id="review-next-btn")

                with gr.Row():
                    accept_btn = gr.Button("Approve [y]", variant="primary", elem_id="review-accept-btn")
                    reject_btn = gr.Button("Reject [n]", variant="stop", elem_id="review-reject-btn")
                    missing_btn = gr.Button("Missing Row [m]", elem_id="review-missing-btn")

        document_table = gr.Dataframe(value=initial_table, interactive=False, label="Document CSV Rows")
        gr.Markdown("*Keyboard: Y=Approve, N=Reject, M=Missing Row, Arrow keys/J/K=Navigate*")

        document_dropdown.change(
            fn=_select_document,
            inputs=[document_dropdown],
            outputs=[current_row_index, pdf_file, page_image, entry_html, document_table, progress_markdown],
        )

        document_filter.change(
            fn=_refresh_document_choices,
            inputs=[document_dropdown, document_filter],
            outputs=[document_dropdown, refresh_status],
        ).then(
            fn=_select_document,
            inputs=[document_dropdown],
            outputs=[current_row_index, pdf_file, page_image, entry_html, document_table, progress_markdown],
        )

        prev_btn.click(
            fn=lambda doc_id, idx: _move_row(doc_id, idx, -1),
            inputs=[document_dropdown, current_row_index],
            outputs=[current_row_index, pdf_file, page_image, entry_html, document_table, progress_markdown],
        )

        next_btn.click(
            fn=lambda doc_id, idx: _move_row(doc_id, idx, 1),
            inputs=[document_dropdown, current_row_index],
            outputs=[current_row_index, pdf_file, page_image, entry_html, document_table, progress_markdown],
        )

        accept_btn.click(
            fn=lambda doc_id, idx: _apply_review_action(doc_id, idx, "accepted"),
            inputs=[document_dropdown, current_row_index],
            outputs=[current_row_index, pdf_file, page_image, entry_html, document_table, progress_markdown],
        )

        reject_btn.click(
            fn=lambda doc_id, idx: _apply_review_action(doc_id, idx, "rejected"),
            inputs=[document_dropdown, current_row_index],
            outputs=[current_row_index, pdf_file, page_image, entry_html, document_table, progress_markdown],
        )

        missing_btn.click(
            fn=_mark_missing_row,
            inputs=[document_dropdown, current_row_index],
            outputs=[current_row_index, pdf_file, page_image, entry_html, document_table, progress_markdown],
        )

        refresh_btn.click(
            fn=_refresh_document_choices,
            inputs=[document_dropdown, document_filter],
            outputs=[document_dropdown, refresh_status],
        ).then(
            fn=_select_document,
            inputs=[document_dropdown],
            outputs=[current_row_index, pdf_file, page_image, entry_html, document_table, progress_markdown],
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
        if profile is None:
            raise SystemExit(f"Profile '{args.profile}' was not found.")
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
        head=KEYBOARD_SHORTCUTS_JS,
    )


if __name__ == "__main__":
    main()
