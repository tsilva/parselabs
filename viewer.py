"""
Lab Results Viewer

Interactive UI for browsing and reviewing extracted lab results.
Shows data table with interactive plots and review actions side-by-side.

Usage:
  python viewer.py --profile tiago
  python viewer.py --list-profiles
  python viewer.py --profile tiago --env local

Keyboard: Y=Accept, N=Reject, Arrow keys/j/k=Navigate
"""

from utils import load_dotenv_with_env
load_dotenv_with_env()

import os
import sys
import json
import argparse
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

from config import ProfileConfig, Demographics, LabSpecsConfig

# =============================================================================
# Keyboard Shortcuts (JavaScript)
# =============================================================================

KEYBOARD_JS = r"""
<script>
(function() {
    document.addEventListener('keydown', function(event) {
        // Skip if user is typing in an input field
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
            return;
        }
        switch(event.key.toLowerCase()) {
            case 'y':
                document.querySelector('#accept-btn')?.click();
                event.preventDefault();
                break;
            case 'n':
                document.querySelector('#reject-btn')?.click();
                event.preventDefault();
                break;
            case 'arrowright':
            case 'j':
                document.querySelector('#next-btn')?.click();
                event.preventDefault();
                break;
            case 'arrowleft':
            case 'k':
                document.querySelector('#prev-btn')?.click();
                event.preventDefault();
                break;
        }
    });
})();
</script>
"""

# =============================================================================
# Custom CSS
# =============================================================================

CUSTOM_CSS = """
/* Status badges */
.status-accepted { background-color: #198754; color: #ffffff; padding: 8px 12px; border-radius: 5px; font-weight: 600; display: inline-block; }
.status-rejected { background-color: #dc3545; color: #ffffff; padding: 8px 12px; border-radius: 5px; font-weight: 600; display: inline-block; }
.status-pending { background-color: #6c757d; color: #ffffff; padding: 8px 12px; border-radius: 5px; font-weight: 600; display: inline-block; }
.status-warning { background-color: #fd7e14; color: #ffffff; padding: 8px 12px; border-radius: 5px; font-weight: 600; display: inline-block; margin-top: 5px; }
.status-info { background-color: #0d6efd; color: #ffffff; padding: 8px 12px; border-radius: 5px; font-weight: 600; display: inline-block; margin-top: 5px; }
.review-actions { margin-top: 10px; }

/* Summary cards - dark mode compatible */
.summary-row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; padding: 8px 0; }
.stat-card {
    display: inline-flex;
    align-items: center;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.9em;
    background-color: rgba(75, 85, 99, 0.6);
    border: 1px solid rgba(107, 114, 128, 0.5);
    color: #e5e7eb;
}
.stat-card.warning {
    background-color: rgba(245, 158, 11, 0.25);
    border-color: #f59e0b;
    color: #fcd34d;
}
.stat-card.danger {
    background-color: rgba(239, 68, 68, 0.25);
    border-color: #ef4444;
    color: #fca5a5;
}
.stat-card.success {
    background-color: rgba(16, 185, 129, 0.25);
    border-color: #10b981;
    color: #6ee7b7;
}

/* Review reason banner - dark mode compatible */
.review-banner {
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 12px;
    font-size: 0.9em;
}
.review-banner.warning {
    background-color: rgba(245, 158, 11, 0.2);
    border: 1px solid #f59e0b;
    color: #fcd34d;
}
.review-banner.info {
    background-color: rgba(59, 130, 246, 0.2);
    border: 1px solid #3b82f6;
    color: #93c5fd;
}
.review-banner-title {
    font-weight: 600;
    margin-bottom: 4px;
}
.review-banner-reasons {
    font-size: 0.85em;
    opacity: 0.9;
}

/* Quick filter pills - dark mode compatible */
.quick-filter-pills {
    padding: 20px 15px;
}
.quick-filter-pills .wrap {
    gap: 8px !important;
}
.quick-filter-pills label {
    padding: 6px 14px !important;
    border-radius: 20px !important;
    font-size: 0.85em !important;
    font-weight: 500 !important;
    border: 1px solid rgba(107, 114, 128, 0.5) !important;
    background-color: rgba(55, 65, 81, 0.6) !important;
    color: #e5e7eb !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    min-height: 32px !important;
    display: inline-flex !important;
    align-items: center !important;
}
.quick-filter-pills label:hover {
    background-color: rgba(75, 85, 99, 0.8) !important;
    border-color: rgba(156, 163, 175, 0.6) !important;
}
.quick-filter-pills input:checked + label {
    background-color: #3b82f6 !important;
    border-color: #3b82f6 !important;
    color: white !important;
}

/* Toggle pill checkbox styling (for Latest Only) */
.toggle-pill {
    padding: 26px 15px;
}
.toggle-pill label {
    padding: 6px 14px !important;
    border-radius: 20px !important;
    font-size: 0.85em !important;
    font-weight: 500 !important;
    border: 1px solid rgba(107, 114, 128, 0.5) !important;
    background-color: rgba(55, 65, 81, 0.6) !important;
    color: #e5e7eb !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
.toggle-pill label:hover {
    background-color: rgba(75, 85, 99, 0.8) !important;
    border-color: rgba(156, 163, 175, 0.6) !important;
}
.toggle-pill input:checked + span {
    background-color: #3b82f6 !important;
    border-color: #3b82f6 !important;
    color: white !important;
}
/* Hide the checkbox itself, style the container */
.toggle-pill .wrap {
    gap: 0 !important;
}
.toggle-pill > label > span {
    padding: 6px 14px !important;
    border-radius: 20px !important;
    font-size: 0.85em !important;
    font-weight: 500 !important;
    border: 1px solid rgba(107, 114, 128, 0.5) !important;
    background-color: rgba(55, 65, 81, 0.6) !important;
    color: #e5e7eb !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    min-height: 32px !important;
    display: inline-flex !important;
    align-items: center !important;
}
.toggle-pill > label > span:hover {
    background-color: rgba(75, 85, 99, 0.8) !important;
    border-color: rgba(156, 163, 175, 0.6) !important;
}
.toggle-pill input[type="checkbox"]:checked + span {
    background-color: #3b82f6 !important;
    border-color: #3b82f6 !important;
    color: white !important;
}

/* Compact lab dropdown in filter row */
.lab-dropdown-compact {
    min-width: 200px;
}
.lab-dropdown-compact .wrap {
    padding: 0 !important;
}
.lab-dropdown-compact input {
    padding: 6px 14px !important;
    font-size: 0.85em !important;
    min-height: 32px !important;
}

/* Filter row vertical alignment */
.filter-row {
    display: flex;
    align-items: center !important;
    gap: 16px;
}
.filter-row > div {
    display: flex;
    align-items: center;
}

/* Compact table display - dark mode compatible */
#lab-data-table table {
    font-size: 0.85em;
}
#lab-data-table td, #lab-data-table th {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 200px;
    padding: 6px 10px !important;
}
#lab-data-table tr:hover {
    background-color: rgba(75, 85, 99, 0.5) !important;
}
#lab-data-table tr.selected {
    background-color: rgba(59, 130, 246, 0.3) !important;
    border-left: 3px solid #3b82f6;
}
"""

# =============================================================================
# Configuration - Global State
# =============================================================================

# Global output path (set from profile or environment)
_configured_output_path: Optional[Path] = None

# Demographics for personalized healthy ranges (set from profile)
_configured_demographics: Optional[Demographics] = None

# Lab specs config (loaded once)
_lab_specs: Optional[LabSpecsConfig] = None

# Current profile name
_current_profile_name: Optional[str] = None


def set_output_path(path: Path) -> None:
    """Set the output path (called from main when using profile)."""
    global _configured_output_path
    _configured_output_path = path


def get_output_path() -> Path:
    """Get output path from configuration, profile, or environment."""
    global _configured_output_path
    if _configured_output_path:
        return _configured_output_path
    return Path(os.getenv('OUTPUT_PATH', './output'))


def set_current_profile(name: str) -> None:
    """Set the current profile name."""
    global _current_profile_name
    _current_profile_name = name


def get_current_profile() -> Optional[str]:
    """Get the current profile name."""
    return _current_profile_name


def set_demographics(demographics: Optional[Demographics]) -> None:
    """Set demographics for personalized range selection."""
    global _configured_demographics
    _configured_demographics = demographics


def get_demographics() -> Optional[Demographics]:
    """Get configured demographics."""
    return _configured_demographics


def get_lab_specs() -> LabSpecsConfig:
    """Get or initialize lab specs config."""
    global _lab_specs
    if _lab_specs is None:
        _lab_specs = LabSpecsConfig()
    return _lab_specs


def load_profile(profile_name: str) -> Optional[ProfileConfig]:
    """Load a profile by name and update global configuration."""
    profile_path = None
    for ext in ('.yaml', '.yml', '.json'):
        p = Path(f"profiles/{profile_name}{ext}")
        if p.exists():
            profile_path = p
            break

    if not profile_path:
        return None

    profile = ProfileConfig.from_file(profile_path)

    set_current_profile(profile_name)

    if profile.output_path:
        set_output_path(profile.output_path)

    if profile.demographics:
        set_demographics(profile.demographics)
    else:
        set_demographics(None)

    return profile


def get_available_profiles() -> list[str]:
    """Get list of available profile names (excluding templates)."""
    profiles = ProfileConfig.list_profiles()
    return [p for p in profiles if not p.startswith('_')]


def get_lab_name_choices(df: pd.DataFrame) -> list[str]:
    """Get sorted list of unique lab names from DataFrame, excluding unknowns."""
    if df.empty or 'lab_name' not in df.columns:
        return []
    return sorted([
        name for name in df['lab_name'].dropna().unique()
        if name and not str(name).startswith('$UNKNOWN')
    ])


# =============================================================================
# Display Configuration
# =============================================================================

# Display columns for the data table
DISPLAY_COLUMNS = [
    'date',
    'lab_name',
    'value',
    'unit',
    'reference_range',
    'is_out_of_reference',
    'review_status',
]

# Column display names
COLUMN_LABELS = {
    'date': 'Date',
    'lab_name': 'Lab',
    'value': 'Value',
    'unit': 'Unit',
    'reference_range': 'Ref',
    'is_out_of_reference': 'Abn',
    'review_status': 'Review',
}


# =============================================================================
# Path Resolution
# =============================================================================

def get_image_path(entry: dict, output_path: Path) -> Optional[str]:
    """Get page image path from source_file and page_number."""
    source_file = entry.get('source_file', '')
    page_number = entry.get('page_number')

    if not source_file:
        return None

    stem = source_file.rsplit('.', 1)[0] if '.' in source_file else source_file

    if page_number is not None and pd.notna(page_number):
        page_str = f"{int(page_number):03d}"
    else:
        page_str = "001"

    image_path = output_path / stem / f"{stem}.{page_str}.jpg"

    if image_path.exists():
        return str(image_path)
    return None


def get_json_path(entry: dict, output_path: Path) -> Path:
    """Get the JSON file path for an entry."""
    source_file = entry.get('source_file', '')
    page_number = entry.get('page_number')

    if not source_file:
        return Path()

    stem = source_file.rsplit('.', 1)[0] if '.' in source_file else source_file

    if page_number is not None and pd.notna(page_number):
        page_str = f"{int(page_number):03d}"
    else:
        page_str = "001"

    return output_path / stem / f"{stem}.{page_str}.json"


# =============================================================================
# JSON File Operations (Review Persistence)
# =============================================================================

def save_review_to_json(entry: dict, status: str, output_path: Path) -> Tuple[bool, str]:
    """Save review status directly to the source JSON file."""
    json_path = get_json_path(entry, output_path)
    result_index = entry.get('result_index')

    if not json_path.exists():
        return False, f"JSON file not found: {json_path}"

    if result_index is None or pd.isna(result_index):
        return False, "Missing result_index for entry."

    result_index = int(result_index)

    try:
        data = json.loads(json_path.read_text(encoding='utf-8'))

        if 'lab_results' not in data:
            return False, "No lab_results in JSON file"

        if result_index >= len(data['lab_results']):
            return False, f"result_index {result_index} out of range"

        data['lab_results'][result_index]['review_status'] = status
        data['lab_results'][result_index]['review_completed_at'] = datetime.utcnow().isoformat() + 'Z'

        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
        return True, ""

    except Exception as e:
        return False, f"Failed to save review: {e}"


# =============================================================================
# Data Loading
# =============================================================================

def load_data(output_path: Path) -> pd.DataFrame:
    """Load lab results from all.csv and sync review status from JSON files."""
    csv_path = output_path / "all.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Sync review status from JSON files (cache to avoid repeated reads)
    json_cache = {}
    review_statuses = []

    for row in df.itertuples():
        result_index = getattr(row, 'result_index', None)
        review_status = None

        if result_index is not None and pd.notna(result_index):
            # Build entry dict with only fields needed for get_json_path
            entry = {
                'source_file': getattr(row, 'source_file', ''),
                'page_number': getattr(row, 'page_number', None)
            }
            json_path = get_json_path(entry, output_path)
            json_path_str = str(json_path)

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
                    review_status = json_entry.get('review_status')

        review_statuses.append(review_status)

    df['review_status'] = review_statuses

    # Compute reference_range from reference_min and reference_max
    if 'reference_min' in df.columns and 'reference_max' in df.columns:
        def format_range(row):
            ref_min = row['reference_min']
            ref_max = row['reference_max']
            if pd.isna(ref_min) and pd.isna(ref_max):
                if row.get('unit') == 'boolean':
                    return '0 - 1'
                return ''
            if pd.isna(ref_min):
                return f'< {ref_max}'
            if pd.isna(ref_max):
                return f'> {ref_min}'
            return f'{ref_min} - {ref_max}'
        df['reference_range'] = df.apply(format_range, axis=1)

    # Compute is_out_of_reference
    if 'value' in df.columns and 'reference_min' in df.columns and 'reference_max' in df.columns:
        def check_out_of_range(row):
            val = row['value']
            ref_min = row['reference_min']
            ref_max = row['reference_max']
            if pd.isna(val):
                return None
            if pd.isna(ref_min) and pd.isna(ref_max):
                if row.get('unit') == 'boolean':
                    return val > 0
                return None
            if pd.notna(ref_min) and val < ref_min:
                return True
            if pd.notna(ref_max) and val > ref_max:
                return True
            return False
        df['is_out_of_reference'] = df.apply(check_out_of_range, axis=1)

    # Compute lab_specs healthy ranges based on demographics
    lab_specs = get_lab_specs()
    demographics = get_demographics()
    gender = demographics.gender if demographics else None
    age = demographics.age if demographics else None

    if 'lab_name' in df.columns and lab_specs.exists:
        def get_lab_spec_range(lab_name):
            range_min, range_max = lab_specs.get_healthy_range_for_demographics(
                lab_name, gender=gender, age=age
            )
            return pd.Series({'lab_specs_min': range_min, 'lab_specs_max': range_max})

        range_df = df['lab_name'].apply(get_lab_spec_range)
        df['lab_specs_min'] = range_df['lab_specs_min']
        df['lab_specs_max'] = range_df['lab_specs_max']

    # Compute is_out_of_healthy_range (based on lab_specs healthy ranges)
    if 'value' in df.columns and 'lab_specs_min' in df.columns and 'lab_specs_max' in df.columns:
        def check_out_of_healthy_range(row):
            val = row.get('value')
            spec_min = row.get('lab_specs_min')
            spec_max = row.get('lab_specs_max')
            if pd.isna(val):
                return None
            if pd.isna(spec_min) and pd.isna(spec_max):
                return None  # No healthy range defined
            if pd.notna(spec_min) and val < spec_min:
                return True
            if pd.notna(spec_max) and val > spec_max:
                return True
            return False
        df['is_out_of_healthy_range'] = df.apply(check_out_of_healthy_range, axis=1)

    return df


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
# Summary Statistics
# =============================================================================

def get_summary_stats(df: pd.DataFrame) -> str:
    """Generate summary statistics markdown."""
    if df.empty:
        return "No data loaded"

    total = len(df)
    unique_tests = df['lab_name'].nunique() if 'lab_name' in df.columns else 0

    # Date range
    date_range = ""
    if 'date' in df.columns and df['date'].notna().any():
        min_date = df['date'].min()
        max_date = df['date'].max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = f" | {min_date.strftime('%Y')}-{max_date.strftime('%Y')}"

    # Abnormal count
    abnormal_count = 0
    if 'is_out_of_reference' in df.columns:
        abnormal_count = int(df['is_out_of_reference'].sum())

    # Review counts
    reviewed_count = 0
    needs_review_count = 0
    if 'review_status' in df.columns:
        reviewed_count = df['review_status'].notna().sum()
    if 'review_needed' in df.columns:
        needs_review_count = int(df['review_needed'].sum())

    return f"**{total:,} results** | {unique_tests} tests | {abnormal_count} abnormal | {needs_review_count} need review | {reviewed_count} reviewed{date_range}"


def build_summary_cards(df: pd.DataFrame) -> str:
    """Generate HTML summary cards with color coding."""
    if df.empty:
        return '<div class="summary-row"><span class="stat-card">No data loaded</span></div>'

    total = len(df)
    unique_tests = df['lab_name'].nunique() if 'lab_name' in df.columns else 0

    # Date range
    date_range = ""
    if 'date' in df.columns and df['date'].notna().any():
        min_date = df['date'].min()
        max_date = df['date'].max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = f"{min_date.strftime('%Y')}-{max_date.strftime('%Y')}"

    # Abnormal count (outside PDF reference range)
    abnormal_count = 0
    if 'is_out_of_reference' in df.columns:
        abnormal_count = int(df['is_out_of_reference'].sum())

    # Unhealthy count (outside lab specs healthy range)
    unhealthy_count = 0
    if 'is_out_of_healthy_range' in df.columns:
        unhealthy_count = int(df['is_out_of_healthy_range'].sum())

    # Review counts
    reviewed_count = 0
    needs_review_count = 0
    if 'review_status' in df.columns:
        reviewed_count = int(df['review_status'].notna().sum())
    if 'review_needed' in df.columns:
        # Only count those that need review AND haven't been reviewed yet
        needs_review_count = int(
            ((df['review_needed'] == True) &
             (df['review_status'].isna() | (df['review_status'] == ''))).sum()
        )

    # Build HTML cards
    cards = []
    cards.append(f'<span class="stat-card">{total:,} results</span>')
    cards.append(f'<span class="stat-card">{unique_tests} tests</span>')

    if date_range:
        cards.append(f'<span class="stat-card">{date_range}</span>')

    if needs_review_count > 0:
        cards.append(f'<span class="stat-card warning">{needs_review_count} need review</span>')

    if abnormal_count > 0:
        cards.append(f'<span class="stat-card danger">{abnormal_count} abnormal</span>')

    if unhealthy_count > 0:
        cards.append(f'<span class="stat-card warning">{unhealthy_count} unhealthy</span>')

    if reviewed_count > 0:
        cards.append(f'<span class="stat-card success">{reviewed_count} reviewed</span>')

    return f'<div class="summary-row">{" ".join(cards)}</div>'


# Human-readable descriptions for validation reason codes
REASON_DESCRIPTIONS = {
    "IMPOSSIBLE_VALUE": "Biologically impossible value detected",
    "RELATIONSHIP_MISMATCH": "Calculated value doesn't match related labs",
    "TEMPORAL_ANOMALY": "Implausible change between consecutive tests",
    "FORMAT_ARTIFACT": "Possible extraction/formatting error",
    "RANGE_INCONSISTENCY": "Reference range appears invalid",
    "PERCENTAGE_BOUNDS": "Percentage value outside 0-100%",
    "NEGATIVE_VALUE": "Unexpected negative value",
    "EXTREME_DEVIATION": "Value extremely far outside reference range",
}


def build_review_reason_banner(entry: dict) -> str:
    """Build HTML banner showing why a result needs review."""
    if not entry:
        return ""

    review_needed = entry.get('review_needed')
    review_reason = entry.get('review_reason')
    review_confidence = entry.get('review_confidence')

    if not review_needed and not review_reason:
        return ""

    # Parse reason codes (semicolon-separated)
    reasons = []
    if review_reason and pd.notna(review_reason):
        reason_str = str(review_reason).strip()
        for code in reason_str.split(';'):
            code = code.strip()
            if code and code in REASON_DESCRIPTIONS:
                reasons.append(REASON_DESCRIPTIONS[code])
            elif code:
                reasons.append(code)

    if not reasons and not (review_confidence and pd.notna(review_confidence) and float(review_confidence) < 0.7):
        return ""

    # Build banner HTML
    banner_class = "warning" if reasons else "info"
    title = "Review Needed" if reasons else "Low Confidence"

    html = f'<div class="review-banner {banner_class}">'
    html += f'<div class="review-banner-title">⚠️ {title}</div>'

    if reasons:
        html += '<div class="review-banner-reasons">'
        html += '<br>'.join(f"• {r}" for r in reasons)
        html += '</div>'

    if review_confidence and pd.notna(review_confidence):
        conf_val = float(review_confidence)
        html += f'<div style="margin-top: 4px; font-size: 0.85em;">Confidence: {conf_val:.0%}</div>'

    html += '</div>'
    return html


# =============================================================================
# Filtering
# =============================================================================

def apply_filters(
    df: pd.DataFrame,
    lab_names: Optional[str],
    latest_only: bool,
    review_filter: str
) -> pd.DataFrame:
    """Apply all filters to DataFrame and sort by date descending.

    Args:
        df: Full DataFrame
        lab_names: Lab name to filter by
        latest_only: Whether to show only latest result per lab
        review_filter: Filter option ('All', 'Needs Review', 'Abnormal', 'Unhealthy', 'Unreviewed')
    """
    if df.empty:
        return df

    filtered = df.copy()

    # Filter by lab name (single selection)
    if lab_names:
        filtered = filtered[filtered['lab_name'] == lab_names]

    # Status filter pills options
    if review_filter == 'Unreviewed':
        filtered = filtered[filtered['review_status'].isna() | (filtered['review_status'] == '')]
    elif review_filter == 'Abnormal':
        # Quick filter for abnormal results
        if 'is_out_of_reference' in filtered.columns:
            filtered = filtered[filtered['is_out_of_reference'] == True]
    elif review_filter == 'Needs Review':
        if 'review_needed' in filtered.columns:
            filtered = filtered[
                (filtered['review_needed'] == True) &
                (filtered['review_status'].isna() | (filtered['review_status'] == ''))
            ]
    elif review_filter == 'Unhealthy':
        if 'is_out_of_healthy_range' in filtered.columns:
            filtered = filtered[filtered['is_out_of_healthy_range'] == True]

    # Sort by date descending, then by lab_name ascending
    if 'date' in filtered.columns:
        filtered = filtered.sort_values(['date', 'lab_name'], ascending=[False, True], na_position='last')

    # Latest only: keep only the most recent value per lab test
    if latest_only and 'lab_name' in filtered.columns and 'date' in filtered.columns:
        filtered = filtered.drop_duplicates(subset=['lab_name'], keep='first')

    # Reset index so iloc positions match displayed row positions
    filtered = filtered.reset_index(drop=True)

    return filtered


def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for display (subset and format columns)."""
    if df.empty:
        return pd.DataFrame(columns=[COLUMN_LABELS.get(c, c) for c in DISPLAY_COLUMNS])

    display_df = df[[col for col in DISPLAY_COLUMNS if col in df.columns]].copy()

    # Format date column
    if 'date' in display_df.columns:
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')

    # Format boolean columns
    if 'is_out_of_reference' in display_df.columns:
        display_df['is_out_of_reference'] = display_df['is_out_of_reference'].map(
            {True: 'Yes', False: 'No', None: ''}
        )

    # Format review status
    if 'review_status' in display_df.columns:
        display_df['review_status'] = display_df['review_status'].fillna('').apply(
            lambda x: x.capitalize() if x else ''
        )

    # Round numeric columns
    if 'value' in display_df.columns:
        display_df['value'] = display_df['value'].round(2)

    # Rename columns to display labels
    display_df = display_df.rename(columns=COLUMN_LABELS)

    return display_df


# =============================================================================
# Plotting
# =============================================================================

def create_single_lab_plot(
    df: pd.DataFrame,
    lab_name: str,
    selected_ref: Optional[tuple[float, float]] = None
) -> tuple[go.Figure, str]:
    """Generate a single plot for one lab test. Returns (figure, unit).

    Args:
        df: DataFrame with lab data
        lab_name: Name of the lab test to plot
        selected_ref: Optional (ref_min, ref_max) from the selected row. When provided,
                      this is used for the PDF reference range instead of computing the mode.
    """
    lab_df = df[df['lab_name'] == lab_name].copy()

    if lab_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data for {lab_name}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(template='plotly_white', height=300)
        return fig, ""

    lab_df['date'] = pd.to_datetime(lab_df['date'], errors='coerce')
    lab_df = lab_df.dropna(subset=['date', 'value'])
    lab_df = lab_df.sort_values('date')

    if lab_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No valid data points for {lab_name}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(template='plotly_white', height=300)
        return fig, ""

    unit = ""
    if 'unit' in lab_df.columns:
        units = lab_df['unit'].dropna()
        if not units.empty:
            unit = str(units.iloc[0])

    fig = go.Figure()

    # Add data trace
    fig.add_trace(go.Scatter(
        x=lab_df['date'],
        y=lab_df['value'],
        mode='lines+markers',
        name="Values",
        marker=dict(size=10, color='#1f77b4'),
        line=dict(width=2),
        hovertemplate=(
            '<b>Date:</b> %{x|%Y-%m-%d}<br>'
            f'<b>Value:</b> %{{y:.2f}} {unit}<br>'
            '<extra></extra>'
        )
    ))

    has_lab_specs_range = False
    has_pdf_range = False

    # Add lab_specs healthy range (blue band)
    if 'lab_specs_min' in lab_df.columns and 'lab_specs_max' in lab_df.columns:
        min_vals = lab_df['lab_specs_min'].dropna()
        max_vals = lab_df['lab_specs_max'].dropna()

        if not min_vals.empty and not max_vals.empty:
            spec_min = float(min_vals.iloc[0])
            spec_max = float(max_vals.iloc[0])
            has_lab_specs_range = True

            fig.add_hrect(
                y0=spec_min, y1=spec_max,
                fillcolor="rgba(37, 99, 235, 0.20)",
                line_width=0,
            )
            fig.add_hline(y=spec_min, line_dash="dot", line_color="rgba(37, 99, 235, 0.8)")
            fig.add_hline(y=spec_max, line_dash="dot", line_color="rgba(37, 99, 235, 0.8)")

    # Add PDF reference range (orange band)
    if 'reference_min' in lab_df.columns and 'reference_max' in lab_df.columns:
        min_vals = lab_df['reference_min'].dropna()
        max_vals = lab_df['reference_max'].dropna()

        ref_min = None
        ref_max = None

        # Use selected row's reference range if provided
        if selected_ref is not None:
            sel_min, sel_max = selected_ref
            if sel_min is not None and pd.notna(sel_min):
                ref_min = float(sel_min)
            if sel_max is not None and pd.notna(sel_max):
                ref_max = float(sel_max)
        else:
            # Fallback to mode-based logic for historical view
            if not min_vals.empty:
                ref_min = float(min_vals.mode().iloc[0]) if len(min_vals.mode()) > 0 else float(min_vals.iloc[0])
            if not max_vals.empty:
                ref_max = float(max_vals.mode().iloc[0]) if len(max_vals.mode()) > 0 else float(max_vals.iloc[0])

        if ref_min is not None or ref_max is not None:
            has_pdf_range = True

            if ref_min is not None and ref_max is not None:
                fig.add_hrect(
                    y0=ref_min, y1=ref_max,
                    fillcolor="rgba(245, 158, 11, 0.20)",
                    line_width=0,
                )
                fig.add_hline(y=ref_min, line_dash="dash", line_color="rgba(245, 158, 11, 0.8)")
                fig.add_hline(y=ref_max, line_dash="dash", line_color="rgba(245, 158, 11, 0.8)")
            elif ref_max is not None:
                data_min = lab_df['value'].min()
                y_bottom = min(0, data_min * 0.9) if data_min > 0 else data_min * 1.1
                fig.add_hrect(
                    y0=y_bottom, y1=ref_max,
                    fillcolor="rgba(245, 158, 11, 0.20)",
                    line_width=0,
                )
                fig.add_hline(y=ref_max, line_dash="dash", line_color="rgba(245, 158, 11, 0.8)",
                              annotation_text=f"< {ref_max}", annotation_position="top right")
            else:
                data_max = lab_df['value'].max()
                y_top = max(data_max * 1.2, ref_min * 2) if data_max > 0 else data_max * 0.8
                fig.add_hrect(
                    y0=ref_min, y1=y_top,
                    fillcolor="rgba(245, 158, 11, 0.20)",
                    line_width=0,
                )
                fig.add_hline(y=ref_min, line_dash="dash", line_color="rgba(245, 158, 11, 0.8)",
                              annotation_text=f"> {ref_min}", annotation_position="bottom right")

    # Add legend entries
    if has_lab_specs_range:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='rgba(37, 99, 235, 0.6)', symbol='square'),
            name='Healthy Range',
            showlegend=True
        ))

    if has_pdf_range:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='rgba(245, 158, 11, 0.6)', symbol='square'),
            name='PDF Reference',
            showlegend=True
        ))

    xaxis_config = dict(
        title="Date",
        tickformat='%Y',
    )

    if len(lab_df) == 1:
        single_date = lab_df['date'].iloc[0]
        xaxis_config['tickvals'] = [single_date]
        xaxis_config['ticktext'] = [single_date.strftime('%Y')]

    fig.update_layout(
        title=dict(
            text=f"{lab_name}" + (f" [{unit}]" if unit else ""),
            font=dict(size=14)
        ),
        xaxis=xaxis_config,
        yaxis_title=f"Value ({unit})" if unit else "Value",
        hovermode='x unified',
        template='plotly_white',
        height=300,
        margin=dict(l=60, r=20, t=40, b=40),
        showlegend=has_lab_specs_range or has_pdf_range,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )

    return fig, unit


def create_interactive_plot(
    df: pd.DataFrame,
    lab_names: Optional[list],
    selected_ref: Optional[tuple[float, float]] = None
) -> go.Figure:
    """Generate interactive Plotly plot(s) for selected lab tests.

    Args:
        df: DataFrame with lab data
        lab_names: List of lab names to plot
        selected_ref: Optional (ref_min, ref_max) from the selected row. Only applies
                      to single-lab plots or the first lab in multi-lab plots.
    """
    if not lab_names or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Select lab tests to view time series",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        return fig

    if len(lab_names) == 1:
        fig, _ = create_single_lab_plot(df, lab_names[0], selected_ref=selected_ref)
        fig.update_layout(height=400)
        return fig

    # Multiple labs - create stacked subplots
    n_labs = len(lab_names)
    fig = make_subplots(
        rows=n_labs,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=lab_names
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, lab_name in enumerate(lab_names):
        lab_df = df[df['lab_name'] == lab_name].copy()

        if lab_df.empty:
            continue

        lab_df['date'] = pd.to_datetime(lab_df['date'], errors='coerce')
        lab_df = lab_df.dropna(subset=['date', 'value'])
        lab_df = lab_df.sort_values('date')

        if lab_df.empty:
            continue

        unit = ""
        if 'unit' in lab_df.columns:
            units = lab_df['unit'].dropna()
            if not units.empty:
                unit = str(units.iloc[0])

        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=lab_df['date'],
                y=lab_df['value'],
                mode='lines+markers',
                name="Values",
                marker=dict(size=8, color=color),
                line=dict(width=2, color=color),
                hovertemplate=(
                    '<b>Date:</b> %{x|%Y-%m-%d}<br>'
                    f'<b>Value:</b> %{{y:.2f}} {unit}<br>'
                    '<extra></extra>'
                )
            ),
            row=i + 1,
            col=1
        )

        # Add lab_specs healthy range (blue band)
        if 'lab_specs_min' in lab_df.columns and 'lab_specs_max' in lab_df.columns:
            min_vals = lab_df['lab_specs_min'].dropna()
            max_vals = lab_df['lab_specs_max'].dropna()

            if not min_vals.empty and not max_vals.empty:
                spec_min = float(min_vals.iloc[0])
                spec_max = float(max_vals.iloc[0])

                fig.add_hrect(
                    y0=spec_min, y1=spec_max,
                    fillcolor="rgba(37, 99, 235, 0.20)",
                    line_width=0,
                    row=i + 1, col=1
                )

        # Add PDF reference range (orange band)
        if 'reference_min' in lab_df.columns and 'reference_max' in lab_df.columns:
            min_vals = lab_df['reference_min'].dropna()
            max_vals = lab_df['reference_max'].dropna()

            ref_min = None
            ref_max = None

            # Use selected row's reference range for the first lab (primary selection)
            if i == 0 and selected_ref is not None:
                sel_min, sel_max = selected_ref
                if sel_min is not None and pd.notna(sel_min):
                    ref_min = float(sel_min)
                if sel_max is not None and pd.notna(sel_max):
                    ref_max = float(sel_max)
            else:
                # Fallback to mode-based logic for other labs or when no selection
                if not min_vals.empty:
                    ref_min = float(min_vals.mode().iloc[0]) if len(min_vals.mode()) > 0 else float(min_vals.iloc[0])
                if not max_vals.empty:
                    ref_max = float(max_vals.mode().iloc[0]) if len(max_vals.mode()) > 0 else float(max_vals.iloc[0])

            if ref_min is not None and ref_max is not None:
                fig.add_hrect(
                    y0=ref_min, y1=ref_max,
                    fillcolor="rgba(245, 158, 11, 0.20)",
                    line_width=0,
                    row=i + 1, col=1
                )

        fig.update_yaxes(title_text=unit if unit else "Value", row=i + 1, col=1)

    height_per_chart = 250
    fig.update_layout(
        template='plotly_white',
        height=height_per_chart * n_labs,
        margin=dict(l=60, r=20, t=40, b=40),
        showlegend=False,
        hovermode='x unified'
    )

    fig.update_xaxes(tickformat='%Y')

    return fig


# =============================================================================
# Details Display (from review.py)
# =============================================================================

def build_details_html(entry: dict) -> str:
    """Build HTML for entry details (raw vs standardized comparison)."""
    if not entry:
        return "<p>No entry selected</p>"

    paired_fields = [
        ('Lab Name', 'lab_name_raw', 'lab_name'),
        ('Value', 'value_raw', 'value'),
        ('Unit', 'unit_raw', 'unit'),
        ('Ref Min', 'reference_min', 'reference_min'),
        ('Ref Max', 'reference_max', 'reference_max'),
    ]

    def get_val(field: str) -> str:
        val = entry.get(field)
        if val is not None and pd.notna(val) and str(val).strip():
            return str(val)
        return "-"

    html = '<table style="width:100%; border-collapse:collapse; font-size:0.9em;">'
    html += '<thead><tr style="background:#f5f5f5;"><th style="padding:6px; text-align:left;">Field</th><th style="padding:6px; text-align:left;">Raw</th><th style="padding:6px; text-align:left;">Standardized</th></tr></thead>'
    html += '<tbody>'

    for label, raw_field, std_field in paired_fields:
        raw_val = get_val(raw_field)
        std_val = get_val(std_field)
        html += f'<tr><td style="padding:6px; border-bottom:1px solid #ddd;">{label}</td>'
        html += f'<td style="padding:6px; border-bottom:1px solid #ddd;">{raw_val}</td>'
        html += f'<td style="padding:6px; border-bottom:1px solid #ddd;">{std_val}</td></tr>'

    html += '</tbody></table>'

    # Add review info if present
    review_needed = entry.get('review_needed')
    review_reason = entry.get('review_reason')
    review_confidence = entry.get('review_confidence')

    if review_needed or review_reason or (review_confidence and pd.notna(review_confidence)):
        html += '<div style="margin-top:15px;">'
        if review_reason and pd.notna(review_reason):
            html += f'<div class="status-warning">Reason: {review_reason}</div>'
        if review_confidence and pd.notna(review_confidence):
            conf_val = float(review_confidence)
            css_class = "status-warning" if conf_val < 0.7 else "status-info"
            html += f'<div class="{css_class}" style="margin-top:5px;">Confidence: {conf_val:.2f}</div>'
        html += '</div>'

    return html


def build_review_status_html(entry: dict) -> str:
    """Build HTML for current review status badge."""
    if not entry:
        return ""

    status = get_review_status(entry)
    if status == "accepted":
        return '<span class="status-accepted">Accepted</span>'
    elif status == "rejected":
        return '<span class="status-rejected">Rejected</span>'
    else:
        return '<span class="status-pending">Pending</span>'


# =============================================================================
# Event Handlers
# =============================================================================

def handle_filter_change(
    lab_names: Optional[str],
    latest_only: bool,
    review_filter: str,
    full_df: pd.DataFrame
):
    """Handle filter changes and update display."""
    filtered_df = apply_filters(full_df, lab_names, latest_only, review_filter)
    display_df = prepare_display_df(filtered_df)
    summary = build_summary_cards(filtered_df)

    current_idx = 0
    position_text = "No results"
    image_path = None
    details_html = "<p>No entry selected</p>"
    status_html = ""
    banner_html = ""

    # Determine which labs to plot
    if lab_names:
        plot_labs = [lab_names]
    elif not filtered_df.empty:
        plot_labs = [filtered_df.iloc[0].get('lab_name')]
    else:
        plot_labs = []

    selected_ref = None
    if not filtered_df.empty:
        first_row = filtered_df.iloc[0]
        position_text = f"**Row 1 of {len(filtered_df)}**"
        image_path = get_image_path(first_row.to_dict(), get_output_path())
        details_html = build_details_html(first_row.to_dict())
        status_html = build_review_status_html(first_row.to_dict())
        banner_html = build_review_reason_banner(first_row.to_dict())
        # Extract first row's reference range for the plot
        ref_min = first_row.get('reference_min')
        ref_max = first_row.get('reference_max')
        selected_ref = (ref_min, ref_max) if pd.notna(ref_min) or pd.notna(ref_max) else None

    plot = create_interactive_plot(full_df, plot_labs, selected_ref=selected_ref)

    return display_df, summary, plot, filtered_df, current_idx, position_text, image_path, details_html, status_html, banner_html


def handle_row_select(
    evt: gr.SelectData,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    lab_names: Optional[str]
):
    """Handle row selection to update plot, details, and current index."""
    if evt is None or filtered_df.empty:
        return create_interactive_plot(full_df, []), 0, "No results", None, "<p>No entry selected</p>", "", ""

    if isinstance(evt.index, (list, tuple)):
        row_idx = evt.index[0]
    else:
        row_idx = evt.index

    if row_idx < 0 or row_idx >= len(filtered_df):
        row_idx = 0

    row = filtered_df.iloc[row_idx]
    position_text = f"**Row {row_idx + 1} of {len(filtered_df)}**"
    image_path = get_image_path(row.to_dict(), get_output_path())
    details_html = build_details_html(row.to_dict())
    status_html = build_review_status_html(row.to_dict())
    banner_html = build_review_reason_banner(row.to_dict())

    if lab_names:
        plot_labs = [lab_names]
    else:
        plot_labs = [row.get('lab_name')]

    # Extract selected row's reference range for the plot
    ref_min = row.get('reference_min')
    ref_max = row.get('reference_max')
    selected_ref = (ref_min, ref_max) if pd.notna(ref_min) or pd.notna(ref_max) else None

    return create_interactive_plot(full_df, plot_labs, selected_ref=selected_ref), row_idx, position_text, image_path, details_html, status_html, banner_html


def handle_previous(
    current_idx: int,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    lab_names: Optional[str]
):
    """Navigate to previous row."""
    if filtered_df.empty:
        return create_interactive_plot(full_df, []), 0, "No results", None, "<p>No entry selected</p>", "", ""

    new_idx = current_idx - 1
    if new_idx < 0:
        new_idx = len(filtered_df) - 1

    row = filtered_df.iloc[new_idx]
    position_text = f"**Row {new_idx + 1} of {len(filtered_df)}**"
    image_path = get_image_path(row.to_dict(), get_output_path())
    details_html = build_details_html(row.to_dict())
    status_html = build_review_status_html(row.to_dict())
    banner_html = build_review_reason_banner(row.to_dict())

    if lab_names:
        plot_labs = [lab_names]
    else:
        plot_labs = [row.get('lab_name')]

    # Extract selected row's reference range for the plot
    ref_min = row.get('reference_min')
    ref_max = row.get('reference_max')
    selected_ref = (ref_min, ref_max) if pd.notna(ref_min) or pd.notna(ref_max) else None

    return create_interactive_plot(full_df, plot_labs, selected_ref=selected_ref), new_idx, position_text, image_path, details_html, status_html, banner_html


def handle_next(
    current_idx: int,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    lab_names: Optional[str]
):
    """Navigate to next row."""
    if filtered_df.empty:
        return create_interactive_plot(full_df, []), 0, "No results", None, "<p>No entry selected</p>", "", ""

    new_idx = current_idx + 1
    if new_idx >= len(filtered_df):
        new_idx = 0

    row = filtered_df.iloc[new_idx]
    position_text = f"**Row {new_idx + 1} of {len(filtered_df)}**"
    image_path = get_image_path(row.to_dict(), get_output_path())
    details_html = build_details_html(row.to_dict())
    status_html = build_review_status_html(row.to_dict())
    banner_html = build_review_reason_banner(row.to_dict())

    if lab_names:
        plot_labs = [lab_names]
    else:
        plot_labs = [row.get('lab_name')]

    # Extract selected row's reference range for the plot
    ref_min = row.get('reference_min')
    ref_max = row.get('reference_max')
    selected_ref = (ref_min, ref_max) if pd.notna(ref_min) or pd.notna(ref_max) else None

    return create_interactive_plot(full_df, plot_labs, selected_ref=selected_ref), new_idx, position_text, image_path, details_html, status_html, banner_html


def handle_review_action(
    current_idx: int,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    lab_names: Optional[str],
    latest_only: bool,
    review_filter: str,
    status: str
):
    """Handle review action (accept/reject)."""
    if filtered_df.empty:
        return (
            full_df,  # full_df unchanged
            filtered_df,  # filtered_df unchanged
            prepare_display_df(filtered_df),  # display_df
            create_interactive_plot(full_df, []),  # plot
            0,  # current_idx
            "No results",  # position_text
            None,  # image
            "<p>No entry selected</p>",  # details
            "",  # status_html
            build_summary_cards(full_df),  # summary
            "",  # banner_html
        )

    if current_idx >= len(filtered_df):
        current_idx = 0

    current_entry = filtered_df.iloc[current_idx].to_dict()
    output_path = get_output_path()

    # Save to JSON
    success, error = save_review_to_json(current_entry, status, output_path)

    if not success:
        gr.Warning(f"Failed to save review: {error}")
    else:
        # Update the entry in full_df
        # Find the matching row by source_file, page_number, and result_index
        mask = (
            (full_df['source_file'] == current_entry.get('source_file')) &
            (full_df['page_number'] == current_entry.get('page_number')) &
            (full_df['result_index'] == current_entry.get('result_index'))
        )
        full_df.loc[mask, 'review_status'] = status

    # Re-filter
    filtered_df = apply_filters(full_df, lab_names, latest_only, review_filter)
    display_df = prepare_display_df(filtered_df)
    summary = build_summary_cards(full_df)

    # Adjust index if needed
    if len(filtered_df) == 0:
        current_idx = 0
        return (
            full_df,
            filtered_df,
            display_df,
            create_interactive_plot(full_df, [lab_names] if lab_names else []),
            current_idx,
            "All done!",
            None,
            "<p>All entries reviewed in this filter!</p>",
            "",
            summary,
            "",  # banner_html
        )

    if current_idx >= len(filtered_df):
        current_idx = max(0, len(filtered_df) - 1)

    row = filtered_df.iloc[current_idx]
    position_text = f"**Row {current_idx + 1} of {len(filtered_df)}**"
    image_path = get_image_path(row.to_dict(), get_output_path())
    details_html = build_details_html(row.to_dict())
    status_html = build_review_status_html(row.to_dict())
    banner_html = build_review_reason_banner(row.to_dict())

    if lab_names:
        plot_labs = [lab_names]
    else:
        plot_labs = [row.get('lab_name')]

    # Extract selected row's reference range for the plot
    ref_min = row.get('reference_min')
    ref_max = row.get('reference_max')
    selected_ref = (ref_min, ref_max) if pd.notna(ref_min) or pd.notna(ref_max) else None

    plot = create_interactive_plot(full_df, plot_labs, selected_ref=selected_ref)

    return (
        full_df,
        filtered_df,
        display_df,
        plot,
        current_idx,
        position_text,
        image_path,
        details_html,
        status_html,
        summary,
        banner_html,
    )


def handle_accept(
    current_idx: int,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    lab_names: Optional[list],
    latest_only: bool,
    review_filter: str
):
    """Mark current entry as accepted."""
    return handle_review_action(
        current_idx, filtered_df, full_df, lab_names,
        latest_only, review_filter, "accepted"
    )


def handle_reject(
    current_idx: int,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    lab_names: Optional[list],
    latest_only: bool,
    review_filter: str
):
    """Mark current entry as rejected."""
    return handle_review_action(
        current_idx, filtered_df, full_df, lab_names,
        latest_only, review_filter, "rejected"
    )


def export_csv(filtered_df: pd.DataFrame):
    """Export filtered data to CSV file."""
    if filtered_df.empty:
        return None

    output_path = get_output_path()
    export_path = output_path / "filtered_export.csv"
    filtered_df.to_csv(export_path, index=False)
    return str(export_path)


def handle_profile_change(profile_name: str):
    """Handle profile switch - reload all data for the new profile."""
    if not profile_name:
        return (
            pd.DataFrame(),  # display_df
            '<div class="summary-row"><span class="stat-card">No profile selected</span></div>',  # summary
            go.Figure(),  # plot
            pd.DataFrame(),  # full_df
            pd.DataFrame(),  # filtered_df
            0,  # current_idx
            "No results",  # position_text
            None,  # image
            [],  # lab_name_choices
            "<p>No entry selected</p>",  # details
            "",  # status_html
            "",  # banner_html
        )

    profile = load_profile(profile_name)
    if not profile or not profile.output_path:
        return (
            pd.DataFrame(),
            f'<div class="summary-row"><span class="stat-card warning">Profile \'{profile_name}\' not found or has no output path</span></div>',
            go.Figure(),
            pd.DataFrame(),
            pd.DataFrame(),
            0,
            "No results",
            None,
            [],
            "<p>No entry selected</p>",
            "",
            "",  # banner_html
        )

    output_path = get_output_path()
    full_df = load_data(output_path)

    if not full_df.empty and 'date' in full_df.columns:
        full_df = full_df.sort_values(['date', 'lab_name'], ascending=[False, True], na_position='last').reset_index(drop=True)

    lab_name_choices = get_lab_name_choices(full_df)

    display_df = prepare_display_df(full_df)
    summary = build_summary_cards(full_df)

    position_text = f"**Row 1 of {len(full_df)}**" if not full_df.empty else "No results"
    image_path = get_image_path(full_df.iloc[0].to_dict(), output_path) if not full_df.empty else None
    details_html = build_details_html(full_df.iloc[0].to_dict()) if not full_df.empty else "<p>No entry selected</p>"
    status_html = build_review_status_html(full_df.iloc[0].to_dict()) if not full_df.empty else ""
    banner_html = build_review_reason_banner(full_df.iloc[0].to_dict()) if not full_df.empty else ""

    # Auto-select first row - show its plot
    initial_plot_labs = []
    if not full_df.empty:
        first_lab = full_df.iloc[0].get('lab_name')
        if first_lab:
            initial_plot_labs = [first_lab]

    plot = create_interactive_plot(full_df, initial_plot_labs)

    return (
        display_df,
        summary,
        plot,
        full_df,
        full_df,
        0,
        position_text,
        image_path,
        gr.update(choices=lab_name_choices, value=None),
        details_html,
        status_html,
        banner_html,
    )


# =============================================================================
# Main App
# =============================================================================

def create_app():
    """Create and configure the Gradio app."""
    output_path = get_output_path()
    full_df = load_data(output_path)

    if not full_df.empty and 'date' in full_df.columns:
        full_df = full_df.sort_values(['date', 'lab_name'], ascending=[False, True], na_position='last').reset_index(drop=True)

    lab_name_choices = get_lab_name_choices(full_df)

    initial_position = f"**Row 1 of {len(full_df)}**" if not full_df.empty else "No results"
    initial_image = get_image_path(full_df.iloc[0].to_dict(), output_path) if not full_df.empty else None
    initial_details = build_details_html(full_df.iloc[0].to_dict()) if not full_df.empty else "<p>No entry selected</p>"
    initial_status = build_review_status_html(full_df.iloc[0].to_dict()) if not full_df.empty else ""

    # Auto-select first row - get its lab name for the initial plot
    initial_plot_labs = []
    if not full_df.empty:
        first_lab = full_df.iloc[0].get('lab_name')
        if first_lab:
            initial_plot_labs = [first_lab]

    available_profiles = get_available_profiles()
    current_profile = get_current_profile()

    with gr.Blocks(title="Lab Results Viewer") as demo:

        # State variables
        full_df_state = gr.State(value=full_df)
        filtered_df_state = gr.State(value=full_df)
        current_idx_state = gr.State(value=0)

        # Header with profile selector
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("# Lab Results Viewer")
            with gr.Column(scale=1):
                profile_selector = gr.Dropdown(
                    choices=available_profiles,
                    value=current_profile,
                    label="Profile",
                    allow_custom_value=False,
                    interactive=True
                )

        gr.Markdown("Browse, analyze, and review extracted lab results.")

        # Unified filter row: Lab dropdown | Status pills | Latest toggle
        with gr.Row(elem_classes="filter-row"):
            with gr.Column(scale=1, min_width=200):
                lab_name_filter = gr.Dropdown(
                    choices=lab_name_choices,
                    multiselect=False,
                    value=None,
                    label="Lab",
                    allow_custom_value=False,
                    elem_classes="lab-dropdown-compact"
                )
            with gr.Column(scale=2):
                review_filter = gr.Radio(
                    choices=['All', 'Needs Review', 'Abnormal', 'Unhealthy', 'Unreviewed'],
                    value='All',
                    label="Status",
                    elem_classes="quick-filter-pills"
                )
            with gr.Column(scale=1, min_width=120):
                latest_filter = gr.Checkbox(
                    label="Latest Only",
                    value=False,
                    elem_classes="toggle-pill"
                )

        # Summary cards
        summary_display = gr.HTML(build_summary_cards(full_df))

        gr.Markdown("---")

        # Main content: Table + Right Panel side by side
        with gr.Row():
            # Left column: Data Table
            with gr.Column(scale=3):
                gr.Markdown("### Data Table")
                gr.Markdown("*Click a row or use arrow keys to navigate*")
                data_table = gr.DataFrame(
                    value=prepare_display_df(full_df),
                    interactive=False,
                    wrap=True,
                    max_height=500,
                    elem_id="lab-data-table"
                )

                with gr.Row():
                    export_btn = gr.Button("Export Filtered CSV", size="sm")
                    export_file = gr.File(label="Download", visible=False)

            # Right column: Navigation + Tabs + Review Actions
            with gr.Column(scale=2):
                # Navigation controls
                with gr.Row():
                    prev_btn = gr.Button("< Prev [k]", elem_id="prev-btn", size="sm")
                    position_display = gr.Markdown(initial_position, elem_id="position-display")
                    next_btn = gr.Button("Next [j] >", elem_id="next-btn", size="sm")

                # Tabs for Plot, Source Image, and Details
                with gr.Tabs():
                    with gr.TabItem("Plot"):
                        plot_display = gr.Plot(
                            value=create_interactive_plot(full_df, initial_plot_labs),
                            label=""
                        )
                    with gr.TabItem("Source"):
                        source_image = gr.Image(
                            value=initial_image,
                            label="Source Document Page",
                            type="filepath",
                            show_label=False,
                            height=400
                        )
                    with gr.TabItem("Details"):
                        details_display = gr.HTML(value=initial_details)

                gr.Markdown("---")

                # Review section with reason banner
                gr.Markdown("### Review")

                # Review reason banner (shows why item needs review)
                initial_banner = build_review_reason_banner(full_df.iloc[0].to_dict()) if not full_df.empty else ""
                review_reason_banner = gr.HTML(value=initial_banner)

                with gr.Row():
                    review_status_display = gr.HTML(value=initial_status)

                with gr.Row():
                    accept_btn = gr.Button(
                        "Accept [Y]",
                        variant="primary",
                        elem_id="accept-btn",
                        size="sm"
                    )
                    reject_btn = gr.Button(
                        "Reject [N]",
                        variant="secondary",
                        elem_id="reject-btn",
                        size="sm"
                    )

        gr.Markdown("---")
        gr.Markdown("*Keyboard: Y=Accept, N=Reject, Arrow keys/j/k=Navigate*")

        # Wire up profile selector
        profile_selector.change(
            fn=handle_profile_change,
            inputs=[profile_selector],
            outputs=[
                data_table,
                summary_display,
                plot_display,
                full_df_state,
                filtered_df_state,
                current_idx_state,
                position_display,
                source_image,
                lab_name_filter,
                details_display,
                review_status_display,
                review_reason_banner,
            ]
        )

        # Filter inputs and outputs
        filter_inputs = [lab_name_filter, latest_filter, review_filter, full_df_state]
        filter_outputs = [data_table, summary_display, plot_display, filtered_df_state, current_idx_state, position_display, source_image, details_display, review_status_display, review_reason_banner]

        lab_name_filter.change(fn=handle_filter_change, inputs=filter_inputs, outputs=filter_outputs)
        latest_filter.change(fn=handle_filter_change, inputs=filter_inputs, outputs=filter_outputs)
        review_filter.change(fn=handle_filter_change, inputs=filter_inputs, outputs=filter_outputs)

        # Row selection
        data_table.select(
            fn=handle_row_select,
            inputs=[filtered_df_state, full_df_state, lab_name_filter],
            outputs=[plot_display, current_idx_state, position_display, source_image, details_display, review_status_display, review_reason_banner]
        )

        # Navigation buttons
        nav_inputs = [current_idx_state, filtered_df_state, full_df_state, lab_name_filter]
        nav_outputs = [plot_display, current_idx_state, position_display, source_image, details_display, review_status_display, review_reason_banner]

        prev_btn.click(fn=handle_previous, inputs=nav_inputs, outputs=nav_outputs)
        next_btn.click(fn=handle_next, inputs=nav_inputs, outputs=nav_outputs)

        # Review action buttons
        review_inputs = [current_idx_state, filtered_df_state, full_df_state, lab_name_filter, latest_filter, review_filter]
        review_outputs = [full_df_state, filtered_df_state, data_table, plot_display, current_idx_state, position_display, source_image, details_display, review_status_display, summary_display, review_reason_banner]

        accept_btn.click(fn=handle_accept, inputs=review_inputs, outputs=review_outputs)
        reject_btn.click(fn=handle_reject, inputs=review_inputs, outputs=review_outputs)

        # Export
        export_btn.click(
            fn=export_csv,
            inputs=[filtered_df_state],
            outputs=[export_file]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[export_file]
        )

    return demo


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Lab Results Viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python viewer.py --profile tiago     # Start with specific profile
  python viewer.py                     # Uses first available profile
  python viewer.py --list-profiles     # List available profiles
  python viewer.py --profile tiago --env local  # Use .env.local
        """
    )
    parser.add_argument(
        '--profile', '-p',
        type=str,
        help='Profile name (defaults to first available profile)'
    )
    parser.add_argument(
        '--list-profiles',
        action='store_true',
        help='List available profiles and exit'
    )
    parser.add_argument(
        '--env',
        type=str,
        default=None,
        help='Environment name to load (loads .env.{name} instead of .env)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Handle --list-profiles
    if args.list_profiles:
        profiles = ProfileConfig.list_profiles()
        if profiles:
            print("Available profiles:")
            for name in profiles:
                print(f"  - {name}")
        else:
            print("No profiles found. Create profiles in the 'profiles/' directory.")
        sys.exit(0)

    # Determine which profile to use
    profile_name = args.profile
    if not profile_name:
        available = get_available_profiles()
        if not available:
            print("Error: No profiles found.")
            print("Create profiles in the 'profiles/' directory.")
            sys.exit(1)
        profile_name = available[0]
        print(f"No profile specified, defaulting to: {profile_name}")

    # Load profile
    profile = load_profile(profile_name)
    if not profile:
        print(f"Error: Profile '{profile_name}' not found")
        print("Use --list-profiles to see available profiles.")
        sys.exit(1)

    print(f"Using profile: {profile.name}")

    if profile.demographics:
        demo_info = []
        if profile.demographics.gender:
            demo_info.append(f"gender={profile.demographics.gender}")
        if profile.demographics.age is not None:
            demo_info.append(f"age={profile.demographics.age}")
        if demo_info:
            print(f"Demographics: {', '.join(demo_info)}")

    if not profile.output_path:
        print(f"Error: Profile '{profile_name}' has no output_path defined.")
        sys.exit(1)

    # Verify output path has all.csv
    output_path = get_output_path()
    csv_path = output_path / "all.csv"
    if not csv_path.exists():
        print(f"Error: No all.csv found at {csv_path}")
        print("Run extract.py first to extract lab results.")
        sys.exit(1)

    # Build allowed paths for serving files from all profiles
    # Read profile configs directly without modifying global state
    available = get_available_profiles()
    print(f"Available profiles: {', '.join(available)}")

    allowed_paths = set()
    for pname in available:
        for ext in ('.yaml', '.yml', '.json'):
            profile_path = Path(f"profiles/{pname}{ext}")
            if profile_path.exists():
                p = ProfileConfig.from_file(profile_path)
                if p.output_path:
                    allowed_paths.add(str(p.output_path))
                    if p.output_path.parent != p.output_path:
                        allowed_paths.add(str(p.output_path.parent))
                break

    allowed_paths = list(allowed_paths)

    print(f"Output path: {output_path}")
    print(f"Starting Lab Results Viewer on http://localhost:7862")

    demo = create_app()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,  # New port for unified viewer
        show_error=True,
        allowed_paths=allowed_paths,
        head=KEYBOARD_JS,
        css=CUSTOM_CSS,
    )
