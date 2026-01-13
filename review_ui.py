"""
Fast Lab Extraction Review UI

Gradio app for rapidly reviewing lab extraction accuracy.
Shows source page image side-by-side with extracted data.
Keyboard-driven: Y=Accept, N=Reject, S=Skip

Review status is stored directly in the extraction JSON files.
"""

import os
import json
import pandas as pd
import gradio as gr
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load .env from repo root
load_dotenv(Path(__file__).parent / '.env')

# =============================================================================
# Keyboard Shortcuts (JavaScript)
# =============================================================================

KEYBOARD_JS = """
<script>
document.addEventListener('keydown', function(event) {
    // Skip if user is typing in an input field
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
        return;
    }

    switch(event.key.toLowerCase()) {
        case 'y':
            document.querySelector('#accept-btn')?.click();
            break;
        case 'n':
            document.querySelector('#reject-btn')?.click();
            break;
        case 's':
            document.querySelector('#skip-btn')?.click();
            break;
        case 'arrowright':
            document.querySelector('#next-btn')?.click();
            break;
        case 'arrowleft':
            document.querySelector('#prev-btn')?.click();
            break;
    }
});
</script>
"""

# =============================================================================
# Custom CSS
# =============================================================================

CUSTOM_CSS = """
.status-accepted { background-color: #198754; color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-weight: 600; }
.status-rejected { background-color: #dc3545; color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-weight: 600; }
.status-warning { background-color: #fd7e14; color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-weight: 600; }
.status-info { background-color: #0d6efd; color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-weight: 600; }
.entry-counter { text-align: center; }
.footer { text-align: center; color: #666; }
"""

# =============================================================================
# Configuration
# =============================================================================

def get_output_path() -> Path:
    """Get output path from environment."""
    return Path(os.getenv('OUTPUT_PATH', './output'))


# =============================================================================
# JSON File Operations
# =============================================================================

def get_json_path(entry: dict, output_path: Path) -> Path:
    """Get the JSON file path for an entry.

    source_file in the CSV is the CSV filename like "2001-12-27 - analises.csv"
    page_number is the page index (1-based)
    The JSON is at output_path/{stem}/{stem}.{page:03d}.json
    """
    source_file = entry.get('source_file', '')
    page_number = entry.get('page_number')

    if not source_file:
        return Path()

    # source_file = "2001-12-27 - analises.csv" -> stem = "2001-12-27 - analises"
    stem = source_file.rsplit('.', 1)[0] if '.' in source_file else source_file

    # page_number = 1 -> "001"
    if page_number is not None and pd.notna(page_number):
        page_str = f"{int(page_number):03d}"
    else:
        page_str = "001"  # fallback

    return output_path / stem / f"{stem}.{page_str}.json"


def save_review_to_json(entry: dict, status: str, output_path: Path) -> Tuple[bool, str]:
    """Save review status directly to the source JSON file.

    Args:
        entry: The entry dict containing source_file and result_index
        status: 'accepted' or 'rejected'
        output_path: Base output directory

    Returns:
        Tuple of (success, error_message)
    """
    json_path = get_json_path(entry, output_path)
    result_index = entry.get('result_index')

    if not json_path.exists():
        return False, f"JSON file not found: {json_path}"

    if result_index is None or pd.isna(result_index):
        return False, (
            "Missing result_index for entry. "
            "Please re-run main.py to regenerate the CSV with result_index column."
        )

    result_index = int(result_index)

    try:
        # Load the JSON file
        data = json.loads(json_path.read_text(encoding='utf-8'))

        # Update the specific lab result
        if 'lab_results' not in data:
            return False, "No lab_results in JSON file"

        if result_index >= len(data['lab_results']):
            return False, f"result_index {result_index} out of range (max: {len(data['lab_results'])-1})"

        # Update the review fields
        data['lab_results'][result_index]['review_status'] = status
        data['lab_results'][result_index]['reviewed_at'] = datetime.utcnow().isoformat() + 'Z'

        # Save back to file
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
        return True, ""

    except Exception as e:
        return False, f"Failed to save review: {e}"


# =============================================================================
# Data Loading
# =============================================================================

def load_entries(output_path: str) -> list:
    """Load all lab results from all.csv and sync review status from JSON files."""
    output_path = Path(output_path)
    csv_path = output_path / "all.csv"
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    entries = []

    # Cache JSON data to avoid repeated file reads
    json_cache = {}

    for idx, row in df.iterrows():
        entry = row.to_dict()
        entry['_row_idx'] = idx

        # Sync review status from JSON file
        result_index = entry.get('result_index')
        if result_index is not None and pd.notna(result_index):
            json_path = get_json_path(entry, output_path)
            json_path_str = str(json_path)

            # Load JSON (with caching)
            if json_path_str not in json_cache:
                if json_path.exists():
                    try:
                        json_cache[json_path_str] = json.loads(
                            json_path.read_text(encoding='utf-8')
                        )
                    except Exception:
                        json_cache[json_path_str] = None
                else:
                    json_cache[json_path_str] = None

            json_data = json_cache.get(json_path_str)
            if json_data and 'lab_results' in json_data:
                result_idx = int(result_index)
                if result_idx < len(json_data['lab_results']):
                    json_entry = json_data['lab_results'][result_idx]
                    # Sync review fields from JSON
                    if 'review_status' in json_entry:
                        entry['review_status'] = json_entry['review_status']
                    if 'reviewed_at' in json_entry:
                        entry['reviewed_at'] = json_entry['reviewed_at']

        entries.append(entry)

    return entries


def get_image_path(entry: dict, output_path: Path) -> Path:
    """Get page image path from source_file and page_number."""
    source_file = entry.get('source_file', '')
    page_number = entry.get('page_number')

    if not source_file:
        return Path()

    # source_file = "2001-12-27 - analises.csv" -> stem = "2001-12-27 - analises"
    stem = source_file.rsplit('.', 1)[0] if '.' in source_file else source_file

    # page_number = 1 -> "001"
    if page_number is not None and pd.notna(page_number):
        page_str = f"{int(page_number):03d}"
    else:
        page_str = "001"  # fallback

    # Image path: output_path / stem / stem.001.jpg
    return output_path / stem / f"{stem}.{page_str}.jpg"


# =============================================================================
# Review Status Helpers
# =============================================================================

def is_reviewed(entry: dict) -> bool:
    """Check if entry has been reviewed."""
    status = entry.get('review_status')
    return status is not None and pd.notna(status) and str(status).strip() != ''


def get_review_status(entry: dict) -> Optional[str]:
    """Get review status for an entry (accepted/rejected/None)."""
    status = entry.get('review_status')
    if status is not None and pd.notna(status) and str(status).strip():
        return str(status).strip()
    return None


# =============================================================================
# Filtering
# =============================================================================

def filter_entries(entries: list, filter_mode: str) -> list:
    """Filter entries based on selected mode."""
    if filter_mode == 'All':
        return entries

    elif filter_mode == 'Unreviewed':
        return [e for e in entries if not is_reviewed(e)]

    elif filter_mode == 'Low Confidence':
        return [
            e for e in entries
            if e.get('confidence_score', 1.0) is not None
            and float(e.get('confidence_score', 1.0)) < 0.7
            and not is_reviewed(e)
        ]

    elif filter_mode == 'Needs Review':
        return [
            e for e in entries
            if e.get('needs_review', False) == True
            and not is_reviewed(e)
        ]

    elif filter_mode == 'Rejected':
        return [
            e for e in entries
            if get_review_status(e) == 'rejected'
        ]

    elif filter_mode == 'Accepted':
        return [
            e for e in entries
            if get_review_status(e) == 'accepted'
        ]

    return entries


# =============================================================================
# Display Helpers
# =============================================================================

def build_details_table(entry: dict) -> str:
    """Build markdown table for entry details."""
    paired_fields = [
        ('lab_name', 'lab_name_raw', 'lab_name_standardized'),
        ('value', 'value_raw', 'value_primary'),
        ('unit', 'lab_unit_raw', 'lab_unit_standardized'),
        ('unit_primary', None, 'lab_unit_primary'),
        ('reference_range', 'reference_range', None),
        ('reference_min', 'reference_min_raw', 'reference_min_primary'),
        ('reference_max', 'reference_max_raw', 'reference_max_primary'),
        ('comments', 'comments', None),
    ]

    def get_val(field: str) -> str:
        if not field:
            return ""
        val = entry.get(field)
        if val is not None and pd.notna(val) and str(val).strip():
            return str(val)
        return ""

    # Build table rows
    table_rows = []
    for label, raw_field, std_field in paired_fields:
        raw_val = get_val(raw_field)
        std_val = get_val(std_field)
        if raw_val or std_val:
            table_rows.append((label, raw_val, std_val))

    if not table_rows:
        return "No data available"

    # Render as markdown table
    table_md = "| Field | Raw | Standardized |\n|-------|-----|---------------|\n"
    for label, raw_val, std_val in table_rows:
        table_md += f"| {label} | {raw_val} | {std_val} |\n"

    return table_md


def build_verification_display(entry: dict) -> str:
    """Build verification section content."""
    verification_fields = [
        'verification_status',
        'verification_confidence',
        'verification_method',
        'cross_model_verified',
        'verification_corrected',
        'value_raw_original',
    ]

    lines = []
    for field in verification_fields:
        val = entry.get(field)
        if val is not None and pd.notna(val) and str(val).strip():
            lines.append(f"`{field}`: {val}")

    return "\n\n".join(lines) if lines else "No verification data"


def get_display_updates(state: dict):
    """Generate all display component updates from current state."""
    if state is None:
        return (
            "Loading...",      # progress_display
            "**Loading...**",  # entry_counter
            None,              # source_image
            "",                # image_caption
            "",                # status_display
            "",                # review_reason_display
            "",                # confidence_display
            "",                # details_table
            "",                # verification_display
            "",                # source_text_display
            gr.update(interactive=False),  # prev_btn
            gr.update(interactive=False),  # next_btn
        )

    output_path = Path(state["output_path"])
    entries = state["entries"]
    filter_mode = state["filter_mode"]
    current_index = state["current_index"]

    filtered = filter_entries(entries, filter_mode)

    # Calculate stats
    total = len(entries)
    reviewed = sum(1 for e in entries if is_reviewed(e))
    accepted = sum(1 for e in entries if get_review_status(e) == 'accepted')
    rejected = sum(1 for e in entries if get_review_status(e) == 'rejected')

    # Progress display
    pct = reviewed / total * 100 if total > 0 else 0
    progress_md = f"Reviewed: {reviewed}/{total} ({pct:.1f}%) | Accepted: {accepted} | Rejected: {rejected}"

    if not filtered:
        return (
            progress_md,                  # progress_display
            "**All done!**",              # entry_counter
            None,                         # source_image
            "",                           # image_caption
            "",                           # status_display
            "",                           # review_reason_display
            "",                           # confidence_display
            "All entries in this filter have been reviewed!",  # details_table
            "",                           # verification_display
            "",                           # source_text_display
            gr.update(interactive=False), # prev_btn
            gr.update(interactive=False), # next_btn
        )

    # Ensure valid index
    if current_index >= len(filtered):
        current_index = 0

    entry = filtered[current_index]

    # Entry counter
    counter_md = f"**Entry {current_index + 1} of {len(filtered)}**"

    # Image
    image_path = get_image_path(entry, output_path)
    image_value = str(image_path) if image_path.exists() else None
    caption = f"Page: {entry.get('source_file', 'Unknown')}"

    # Status badge
    status = get_review_status(entry)
    if status == "accepted":
        status_md = '<div class="status-accepted">Accepted</div>'
    elif status == "rejected":
        status_md = '<div class="status-rejected">Rejected</div>'
    else:
        status_md = ""

    # Review reason
    review_reason = entry.get("review_reason")
    if review_reason and pd.notna(review_reason) and str(review_reason).strip():
        reason_md = f'<div class="status-warning">review_reason: {review_reason}</div>'
    else:
        reason_md = ""

    # Confidence
    confidence = entry.get("confidence_score")
    if confidence is not None and pd.notna(confidence):
        conf_val = float(confidence)
        css_class = "status-warning" if conf_val < 0.7 else "status-info"
        conf_md = f'<div class="{css_class}">confidence_score: {conf_val:.2f}</div>'
    else:
        conf_md = ""

    # Details table
    details_md = build_details_table(entry)

    # Verification
    verification_md = build_verification_display(entry)

    # Source text
    source_text = entry.get("source_text", "N/A")
    if source_text is None or (isinstance(source_text, float) and pd.isna(source_text)):
        source_text = "N/A"

    # Button states
    prev_interactive = current_index > 0
    next_interactive = current_index < len(filtered) - 1

    return (
        progress_md,
        counter_md,
        image_value,
        caption,
        status_md,
        reason_md,
        conf_md,
        details_md,
        verification_md,
        str(source_text),
        gr.update(interactive=prev_interactive),
        gr.update(interactive=next_interactive),
    )


# =============================================================================
# Event Handlers
# =============================================================================

def initialize_state():
    """Load entries and create initial state."""
    output_path = get_output_path()
    entries = load_entries(str(output_path))

    return {
        "current_index": 0,
        "filter_mode": "Unreviewed",
        "entries": entries,
        "output_path": str(output_path),
    }


def handle_filter_change(filter_mode: str, state: dict):
    """Handle filter mode change."""
    if state is None:
        state = initialize_state()
    state = state.copy()
    state["filter_mode"] = filter_mode
    state["current_index"] = 0  # Reset to first entry
    return state, *get_display_updates(state)


def handle_previous(state: dict):
    """Navigate to previous entry."""
    if state is None:
        return state, *get_display_updates(state)

    state = state.copy()
    if state["current_index"] > 0:
        state["current_index"] -= 1
    return state, *get_display_updates(state)


def handle_next(state: dict):
    """Navigate to next entry."""
    if state is None:
        return state, *get_display_updates(state)

    state = state.copy()
    filtered = filter_entries(state["entries"], state["filter_mode"])
    if state["current_index"] < len(filtered) - 1:
        state["current_index"] += 1
    return state, *get_display_updates(state)


def handle_review_action(state: dict, status: str):
    """Generic review action handler."""
    if state is None:
        return state, *get_display_updates(state)

    state = state.copy()
    output_path = Path(state["output_path"])
    filtered = filter_entries(state["entries"], state["filter_mode"])

    if not filtered:
        return state, *get_display_updates(state)

    current_index = state["current_index"]
    if current_index >= len(filtered):
        current_index = 0

    current_entry = filtered[current_index]

    # Save to JSON
    success, error = save_review_to_json(current_entry, status, output_path)

    if not success:
        # Show error to user via gr.Warning
        gr.Warning(f"Failed to save review: {error}")
        return state, *get_display_updates(state)

    # Update entry in state
    for i, e in enumerate(state["entries"]):
        if e.get("_row_idx") == current_entry.get("_row_idx"):
            state["entries"][i] = state["entries"][i].copy()
            state["entries"][i]["review_status"] = status
            state["entries"][i]["reviewed_at"] = datetime.utcnow().isoformat() + 'Z'
            break

    # Re-filter after update (entry may leave current filter)
    new_filtered = filter_entries(state["entries"], state["filter_mode"])

    # Adjust index if needed
    if len(new_filtered) == 0:
        state["current_index"] = 0
    elif state["current_index"] >= len(new_filtered):
        state["current_index"] = max(0, len(new_filtered) - 1)
    # If not at end, stay at same index (next item slides into position)

    return state, *get_display_updates(state)


def handle_accept(state: dict):
    """Mark current entry as accepted."""
    return handle_review_action(state, "accepted")


def handle_reject(state: dict):
    """Mark current entry as rejected."""
    return handle_review_action(state, "rejected")


def handle_skip(state: dict):
    """Skip to next entry without marking."""
    return handle_next(state)


# =============================================================================
# Main App
# =============================================================================

def create_app():
    """Create and configure the Gradio app."""

    with gr.Blocks(title="Lab Review") as demo:

        # State
        app_state = gr.State(value=None)

        # Header
        gr.Markdown("# Lab Extraction Review")

        # Filter radio
        filter_radio = gr.Radio(
            choices=['Unreviewed', 'All', 'Low Confidence', 'Needs Review', 'Accepted', 'Rejected'],
            value='Unreviewed',
            label="Filter:",
            interactive=True
        )

        # Progress display
        progress_display = gr.Markdown("Loading...")

        # Navigation row
        with gr.Row():
            prev_btn = gr.Button("Previous", elem_id="prev-btn", scale=1)
            entry_counter = gr.Markdown("**Entry 0 of 0**", elem_classes=["entry-counter"])
            next_btn = gr.Button("Next", elem_id="next-btn", scale=1)

        gr.HTML("<hr>")

        # Main content: Image left, Data right
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Source Document")
                source_image = gr.Image(
                    type="filepath",
                    label="Page Image",
                    show_label=False,
                    interactive=False
                )
                image_caption = gr.Markdown("")

            with gr.Column(scale=1):
                gr.Markdown("### Extracted Data")
                status_display = gr.HTML("")
                review_reason_display = gr.HTML("")
                confidence_display = gr.HTML("")
                details_table = gr.Markdown("")

                # Verification accordion
                with gr.Accordion("Verification", open=False):
                    verification_display = gr.Markdown("")

                # Source text accordion
                with gr.Accordion("source_text", open=False):
                    source_text_display = gr.Code(label="", language=None)

                gr.HTML("<hr>")

                # Action buttons
                with gr.Row():
                    accept_btn = gr.Button(
                        "Accept [Y]",
                        variant="primary",
                        elem_id="accept-btn",
                        scale=1
                    )
                    reject_btn = gr.Button(
                        "Reject [N]",
                        variant="secondary",
                        elem_id="reject-btn",
                        scale=1
                    )
                    skip_btn = gr.Button(
                        "Skip [S]",
                        elem_id="skip-btn",
                        scale=1
                    )

        gr.HTML("<hr>")
        gr.Markdown("Keyboard: Y=Accept, N=Reject, S=Skip, Arrow keys=Navigate", elem_classes=["footer"])

        # Define all output components
        all_outputs = [
            app_state,
            progress_display,
            entry_counter,
            source_image,
            image_caption,
            status_display,
            review_reason_display,
            confidence_display,
            details_table,
            verification_display,
            source_text_display,
            prev_btn,
            next_btn,
        ]

        # Initialize on load
        demo.load(
            fn=lambda: (initialize_state(), *get_display_updates(initialize_state())),
            outputs=all_outputs
        )

        # Filter change
        filter_radio.change(
            fn=handle_filter_change,
            inputs=[filter_radio, app_state],
            outputs=all_outputs
        )

        # Navigation
        prev_btn.click(
            fn=handle_previous,
            inputs=[app_state],
            outputs=all_outputs
        )
        next_btn.click(
            fn=handle_next,
            inputs=[app_state],
            outputs=all_outputs
        )

        # Actions
        accept_btn.click(
            fn=handle_accept,
            inputs=[app_state],
            outputs=all_outputs
        )
        reject_btn.click(
            fn=handle_reject,
            inputs=[app_state],
            outputs=all_outputs
        )
        skip_btn.click(
            fn=handle_skip,
            inputs=[app_state],
            outputs=all_outputs
        )

    return demo


if __name__ == "__main__":
    demo = create_app()
    output_path = get_output_path()

    # Build list of allowed paths for serving images
    # Include output_path and its parent to handle various output locations
    allowed_paths = [str(output_path)]
    if output_path.parent != output_path:
        allowed_paths.append(str(output_path.parent))

    # Run with `gradio review_ui.py` for auto-reload on code changes
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        allowed_paths=allowed_paths,
        head=KEYBOARD_JS,
        css=CUSTOM_CSS,
    )
