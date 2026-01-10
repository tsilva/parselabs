"""
Fast Lab Extraction Review UI

Streamlit app for rapidly reviewing lab extraction accuracy.
Shows source page image side-by-side with extracted data.
Keyboard-driven: Y=Accept, N=Reject, S=Skip

Review status is stored directly in the extraction JSON files.
"""

import os
import json
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Load .env from parent directory (repo root)
load_dotenv(Path(__file__).parent.parent / '.env')

# Try to import keyboard shortcuts, fall back gracefully
try:
    from streamlit_shortcuts import add_keyboard_shortcuts
    HAS_SHORTCUTS = True
except ImportError:
    HAS_SHORTCUTS = False

# =============================================================================
# Configuration
# =============================================================================

def get_output_path() -> Path:
    """Get output path from environment or secrets."""
    # Try streamlit secrets first (with safe check)
    try:
        if 'OUTPUT_PATH' in st.secrets:
            return Path(st.secrets['OUTPUT_PATH'])
    except Exception:
        pass  # No secrets file, fall through
    # Fall back to environment variable
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


def save_review_to_json(entry: dict, status: str, output_path: Path) -> bool:
    """Save review status directly to the source JSON file.

    Args:
        entry: The entry dict containing source_file and result_index
        status: 'accepted' or 'rejected'
        output_path: Base output directory

    Returns:
        True if save succeeded, False otherwise
    """
    json_path = get_json_path(entry, output_path)
    result_index = entry.get('result_index')

    if not json_path.exists():
        st.error(f"JSON file not found: {json_path}")
        return False

    if result_index is None or pd.isna(result_index):
        st.error(
            "Missing result_index for entry. "
            "Please re-run main.py to regenerate the CSV with result_index column, "
            "or run: python -c \"from review_ui.backfill import backfill_result_index; backfill_result_index()\""
        )
        return False

    result_index = int(result_index)

    try:
        # Load the JSON file
        data = json.loads(json_path.read_text(encoding='utf-8'))

        # Update the specific lab result
        if 'lab_results' not in data:
            st.error(f"No lab_results in JSON file")
            return False

        if result_index >= len(data['lab_results']):
            st.error(f"result_index {result_index} out of range (max: {len(data['lab_results'])-1})")
            return False

        # Update the review fields
        data['lab_results'][result_index]['review_status'] = status
        data['lab_results'][result_index]['reviewed_at'] = datetime.utcnow().isoformat() + 'Z'

        # Save back to file
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
        return True

    except Exception as e:
        st.error(f"Failed to save review: {e}")
        return False


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
# UI Components
# =============================================================================

def display_entry_details(entry: dict):
    """Display lab result details."""
    # Current review status
    status = get_review_status(entry)
    if status == 'accepted':
        st.success("Accepted")
    elif status == 'rejected':
        st.error("Rejected")

    # Review reason if flagged
    review_reason = entry.get('review_reason')
    if review_reason and pd.notna(review_reason) and review_reason.strip():
        st.warning(f"review_reason: {review_reason}")

    # Confidence indicator
    confidence = entry.get('confidence_score')
    if confidence is not None and pd.notna(confidence):
        conf_val = float(confidence)
        if conf_val < 0.7:
            st.warning(f"confidence_score: {conf_val:.2f}")
        else:
            st.info(f"confidence_score: {conf_val:.2f}")

    # Define paired fields: (label, raw_field, standardized_field)
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

    verification_fields = [
        'verification_status',
        'verification_confidence',
        'verification_method',
        'cross_model_verified',
        'verification_corrected',
        'value_raw_original',
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

    # Render as markdown table
    if table_rows:
        table_md = "| Field | Raw | Standardized |\n|-------|-----|---------------|\n"
        for label, raw_val, std_val in table_rows:
            table_md += f"| {label} | {raw_val} | {std_val} |\n"
        st.markdown(table_md)

    # Verification (collapsible if has data)
    has_verification = any(
        entry.get(f) is not None and pd.notna(entry.get(f))
        for f in verification_fields
    )
    if has_verification:
        with st.expander("Verification"):
            for field in verification_fields:
                val = entry.get(field)
                if val is not None and pd.notna(val) and str(val).strip():
                    st.markdown(f"`{field}`: {val}")

    # Source text for verification
    with st.expander("source_text"):
        st.code(entry.get('source_text', 'N/A'))


def display_progress(total: int, reviewed: int, accepted: int, rejected: int):
    """Display progress metrics."""
    if total > 0:
        pct = reviewed / total
        st.progress(pct)
        st.caption(
            f"Reviewed: {reviewed}/{total} ({pct*100:.1f}%) | "
            f"Accepted: {accepted} | Rejected: {rejected}"
        )
    else:
        st.progress(0.0)
        st.caption("No entries to review")


# =============================================================================
# Main App
# =============================================================================

def main():
    st.set_page_config(
        layout="wide",
        page_title="Lab Review",
        page_icon="🔬"
    )

    output_path = get_output_path()

    # Initialize session state
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    if 'filter_mode' not in st.session_state:
        st.session_state.filter_mode = 'Unreviewed'

    # Load entries (without caching to see review updates)
    entries = load_entries(str(output_path))

    if not entries:
        st.error(f"No data found. Check OUTPUT_PATH: {output_path}")
        st.info("Set OUTPUT_PATH environment variable or configure in .streamlit/secrets.toml")
        return

    # Header
    st.title("Lab Extraction Review")

    # Filter selection
    filter_mode = st.radio(
        "Filter:",
        ['Unreviewed', 'All', 'Low Confidence', 'Needs Review', 'Accepted', 'Rejected'],
        horizontal=True,
        key='filter_radio'
    )

    # Update filter mode and reset index if changed
    if filter_mode != st.session_state.filter_mode:
        st.session_state.filter_mode = filter_mode
        st.session_state.current_index = 0

    # Get filtered entries
    filtered = filter_entries(entries, filter_mode)

    # Progress stats
    total_entries = len(entries)
    reviewed_count = sum(1 for e in entries if is_reviewed(e))
    accepted_count = sum(1 for e in entries if get_review_status(e) == 'accepted')
    rejected_count = sum(1 for e in entries if get_review_status(e) == 'rejected')

    display_progress(total_entries, reviewed_count, accepted_count, rejected_count)

    # Check if done
    if not filtered:
        st.success("All entries in this filter have been reviewed!")
        if filter_mode == 'Unreviewed':
            st.balloons()
        return

    # Ensure index is valid
    if st.session_state.current_index >= len(filtered):
        st.session_state.current_index = 0

    current_entry = filtered[st.session_state.current_index]

    # Navigation
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("Previous", disabled=st.session_state.current_index == 0):
            st.session_state.current_index -= 1
            st.rerun()
    with nav_col2:
        st.markdown(f"**Entry {st.session_state.current_index + 1} of {len(filtered)}**")
    with nav_col3:
        if st.button("Next", disabled=st.session_state.current_index >= len(filtered) - 1):
            st.session_state.current_index += 1
            st.rerun()

    st.divider()

    # Main layout: image left, details right
    img_col, data_col = st.columns([1, 1])

    with img_col:
        st.subheader("Source Document")
        image_path = get_image_path(current_entry, output_path)

        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
            st.caption(f"Page: {current_entry.get('source_file', 'Unknown')}")
        else:
            st.warning(f"Image not found: {image_path}")

    with data_col:
        st.subheader("Extracted Data")
        display_entry_details(current_entry)

        st.divider()

        # Action buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            accept_clicked = st.button(
                "Accept [Y]",
                type="primary",
                use_container_width=True,
                key="accept_btn"
            )

        with btn_col2:
            reject_clicked = st.button(
                "Reject [N]",
                type="secondary",
                use_container_width=True,
                key="reject_btn"
            )

        with btn_col3:
            skip_clicked = st.button(
                "Skip [S]",
                use_container_width=True,
                key="skip_btn"
            )

        # Handle actions
        def mark_reviewed(status: str):
            success = save_review_to_json(current_entry, status, output_path)
            if success:
                # Clear the cache so we reload fresh data
                st.cache_data.clear()

                # Auto-advance
                if st.session_state.current_index < len(filtered) - 1:
                    st.session_state.current_index += 1
                st.rerun()

        def skip_entry():
            if st.session_state.current_index < len(filtered) - 1:
                st.session_state.current_index += 1
            st.rerun()

        if accept_clicked:
            mark_reviewed('accepted')
        elif reject_clicked:
            mark_reviewed('rejected')
        elif skip_clicked:
            skip_entry()

    # Keyboard shortcuts (if available)
    if HAS_SHORTCUTS:
        shortcuts = add_keyboard_shortcuts({
            'y': 'Accept',
            'n': 'Reject',
            's': 'Skip',
            'ArrowRight': 'Next',
            'ArrowLeft': 'Previous',
        })

        if shortcuts == 'Accept':
            mark_reviewed('accepted')
        elif shortcuts == 'Reject':
            mark_reviewed('rejected')
        elif shortcuts == 'Skip':
            skip_entry()
        elif shortcuts == 'Next' and st.session_state.current_index < len(filtered) - 1:
            st.session_state.current_index += 1
            st.rerun()
        elif shortcuts == 'Previous' and st.session_state.current_index > 0:
            st.session_state.current_index -= 1
            st.rerun()

    # Footer with instructions
    st.divider()
    if HAS_SHORTCUTS:
        st.caption("Keyboard: Y=Accept, N=Reject, S=Skip, Arrow keys=Navigate")
    else:
        st.caption("Install streamlit-shortcuts for keyboard navigation: pip install streamlit-shortcuts")


if __name__ == "__main__":
    main()
