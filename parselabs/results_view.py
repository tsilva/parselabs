"""Unified review workspace for exploring and reviewing extracted lab results."""

from __future__ import annotations

import html
import json
import logging  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from parselabs.config import Demographics, LabSpecsConfig  # noqa: E402
from parselabs.paths import get_static_dir  # noqa: E402
from parselabs.review import (  # noqa: E402
    SOURCE_BBOX_LABEL,
    build_review_status_badge,
    format_mapped_reference_text,
    format_mapped_value,
    format_raw_value,
    format_reference_text,
    format_text,
    get_bbox_coordinates,
    load_results_dataframe,
    normalize_review_status,
)
from parselabs.review_state import (  # noqa: E402
    apply_review_action_for_entry,
    build_viewer_row_context,
    get_selected_row,
)
from parselabs.runtime import RuntimeContext  # noqa: E402

# Initialize module logger
logger = logging.getLogger(__name__)

_STATIC_DIR = get_static_dir()
KEYBOARD_JS = (_STATIC_DIR / "viewer.js").read_text()
CUSTOM_CSS = (_STATIC_DIR / "viewer.css").read_text()


@dataclass(frozen=True)
class ViewerRenderState:
    """Standard output payload shared by the viewer callbacks."""

    display_df: pd.DataFrame
    summary_html: str
    plot: go.Figure
    filtered_df: pd.DataFrame
    current_idx: int
    position_text: str
    source_image_value: tuple[str, list[tuple[tuple[int, int, int, int], str]]] | None
    inspector_html: str
    selection_html: str
    prev_button_props: dict
    next_button_props: dict

    def as_filter_outputs(self) -> tuple:
        """Return outputs for filter-driven viewer updates."""

        return (
            self.display_df,
            self.summary_html,
            self.filtered_df,
            self.current_idx,
            self.position_text,
            self.inspector_html,
            self.source_image_value,
            self.plot,
            self.selection_html,
            self.prev_button_props,
            self.next_button_props,
        )

    def as_review_outputs(self, full_df: pd.DataFrame) -> tuple:
        """Return outputs for review-action callbacks that also update full_df state."""

        return (
            full_df,
            self.filtered_df,
            self.display_df,
            self.current_idx,
            self.position_text,
            self.inspector_html,
            self.source_image_value,
            self.plot,
            self.summary_html,
            self.selection_html,
            self.prev_button_props,
            self.next_button_props,
        )


def get_lab_name_choices(df: pd.DataFrame) -> list[str]:
    """Get sorted list of unique lab names from DataFrame, excluding unknowns."""

    # Guard: no data available
    if df.empty or "lab_name" not in df.columns:
        return []
    return sorted([name for name in df["lab_name"].dropna().unique() if name and not str(name).startswith("$UNKNOWN")])


def _format_document_label(source_file: object) -> str:
    """Return a short, readable document label for one source file."""

    source_text = str(source_file or "").strip()
    if not source_text:
        return "-"
    return source_text.rsplit(".", 1)[0]


def _build_document_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-document summary rows used for dropdown ordering and defaults."""

    if df.empty or "source_file" not in df.columns:
        return pd.DataFrame(columns=["source_file", "result_count", "bbox_count", "needs_review_count", "unreviewed_count"])

    summary_df = df[df["source_file"].notna()].copy()
    if summary_df.empty:
        return pd.DataFrame(columns=["source_file", "result_count", "bbox_count", "needs_review_count", "unreviewed_count"])

    bbox_columns = ["bbox_left", "bbox_top", "bbox_right", "bbox_bottom"]
    has_bbox = (
        summary_df[bbox_columns].notna().all(axis=1)
        if set(bbox_columns).issubset(summary_df.columns)
        else pd.Series(False, index=summary_df.index)
    )
    review_status = summary_df["review_status"] if "review_status" in summary_df.columns else pd.Series("", index=summary_df.index)
    review_needed = summary_df["review_needed"] if "review_needed" in summary_df.columns else pd.Series(False, index=summary_df.index)
    is_unreviewed = review_status.isna() | review_status.astype(str).str.strip().eq("")

    summary_df = summary_df.assign(
        has_bbox=has_bbox,
        needs_review_unresolved=review_needed.fillna(False) & is_unreviewed,
        is_unreviewed=is_unreviewed,
    )

    return (
        summary_df.groupby("source_file", dropna=True)
        .agg(
            result_count=("source_file", "size"),
            bbox_count=("has_bbox", "sum"),
            needs_review_count=("needs_review_unresolved", "sum"),
            unreviewed_count=("is_unreviewed", "sum"),
        )
        .reset_index()
    )


def get_document_choices(df: pd.DataFrame, *, prioritize_review_sources: bool = False) -> list[tuple[str, str]]:
    """Return readable document dropdown choices keyed by source file."""

    document_summary = _build_document_summary(df)
    if document_summary.empty:
        return []

    if prioritize_review_sources:
        document_summary = document_summary.sort_values(
            ["bbox_count", "needs_review_count", "unreviewed_count", "result_count", "source_file"],
            ascending=[False, False, False, False, True],
            na_position="last",
        )
    else:
        document_summary = document_summary.sort_values("source_file", ascending=True, na_position="last")

    return [
        (f"{_format_document_label(row.source_file)} ({int(row.result_count)})", str(row.source_file))
        for row in document_summary.itertuples()
        if str(row.source_file).strip()
    ]


def get_initial_document(df: pd.DataFrame, *, prioritize_review_sources: bool) -> str | None:
    """Pick the default document for the current workspace launch mode."""

    if not prioritize_review_sources:
        return None

    document_summary = _build_document_summary(df)
    if document_summary.empty:
        return None

    ranked = document_summary.sort_values(
        ["bbox_count", "needs_review_count", "unreviewed_count", "result_count", "source_file"],
        ascending=[False, False, False, False, True],
        na_position="last",
    )
    top_row = ranked.iloc[0]
    source_file = str(top_row.get("source_file") or "").strip()
    return source_file or None


# =============================================================================
# Display Configuration
# =============================================================================

# Display columns for the data table
DISPLAY_COLUMNS = [
    "date",
    "lab_name",
    "value",
    "lab_unit",
    "source_document",
    "page_number",
    "reference_range",
    "review_status",
]

# Column display names
COLUMN_LABELS = {
    "date": "Date",
    "lab_name": "Lab",
    "value": "Value",
    "lab_unit": "Unit",
    "source_document": "Document",
    "page_number": "Page",
    "reference_range": "Ref",
    "review_status": "Review",
}


def _build_row_token(source_file: object, page_number: object, result_index: object) -> str:
    """Build a stable UI token for one merged results row."""

    normalized_values = []
    for value in (source_file, page_number, result_index):
        normalized_values.append(value.item() if hasattr(value, "item") else value)

    return json.dumps(normalized_values, separators=(",", ":"))


def _build_row_token_for_entry(entry: pd.Series | dict) -> str:
    """Build the stable UI token for a dataframe row or dict entry."""

    return _build_row_token(
        entry.get("source_file"),
        entry.get("page_number"),
        entry.get("result_index"),
    )


def _build_display_row_match_values(entry: pd.Series | dict) -> list[str]:
    """Build the formatted visible table cells for one row."""

    entry_df = pd.DataFrame([dict(entry)])
    if "date" in entry_df.columns:
        entry_df["date"] = pd.to_datetime(entry_df["date"], errors="coerce")

    display_df = prepare_display_df(entry_df)
    if display_df.empty:
        return []

    return [str(value).strip() for value in display_df.iloc[0].tolist()]


# =============================================================================
# Summary Statistics
# =============================================================================


def build_summary_cards(df: pd.DataFrame) -> str:
    """Generate HTML summary cards with color coding."""

    # Guard: no data
    if df.empty:
        return '<div class="summary-line"><span class="summary-item">No data loaded</span></div>'

    total = len(df)
    unique_tests = df["lab_name"].nunique() if "lab_name" in df.columns else 0

    # Date range
    date_range = ""
    if "date" in df.columns and df["date"].notna().any():
        min_date = df["date"].min()
        max_date = df["date"].max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = f"{min_date.strftime('%Y')}-{max_date.strftime('%Y')}"

    # Abnormal count (outside PDF reference range)
    abnormal_count = 0
    if "is_out_of_reference" in df.columns:
        abnormal_count = int(df["is_out_of_reference"].sum())

    # Unhealthy count (outside lab specs healthy range)
    unhealthy_count = 0
    if "is_out_of_healthy_range" in df.columns:
        unhealthy_count = int(df["is_out_of_healthy_range"].sum())

    # Review counts
    reviewed_count = 0
    needs_review_count = 0

    if "review_status" in df.columns:
        reviewed_count = int(df["review_status"].notna().sum())

    if "review_needed" in df.columns:
        # Only count those that need review AND haven't been reviewed yet
        needs_review_count = int(((df["review_needed"]) & (df["review_status"].isna() | (df["review_status"] == ""))).sum())

    # Build HTML cards
    cards = []
    cards.append(f'<span class="summary-item"><strong>{total:,}</strong> results</span>')
    cards.append(f'<span class="summary-item"><strong>{unique_tests}</strong> tests</span>')

    if date_range:
        cards.append(f'<span class="summary-item"><strong>{date_range}</strong></span>')

    if needs_review_count > 0:
        cards.append(f'<span class="summary-item warning"><strong>{needs_review_count}</strong> need review</span>')

    if abnormal_count > 0:
        cards.append(f'<span class="summary-item danger"><strong>{abnormal_count}</strong> abnormal</span>')

    if unhealthy_count > 0:
        cards.append(f'<span class="summary-item warning"><strong>{unhealthy_count}</strong> unhealthy</span>')

    if reviewed_count > 0:
        cards.append(f'<span class="summary-item success"><strong>{reviewed_count}</strong> reviewed</span>')

    return f'<div class="summary-line">{"".join(cards)}</div>'


# Human-readable descriptions for validation reason codes
REASON_DESCRIPTIONS = {
    "EXTRACTION_FAILED": "Extraction failed on this page",
    "UNKNOWN_LAB_MAPPING": "Lab name did not map to a known standardized test",
    "UNKNOWN_UNIT_MAPPING": "Unit did not map to a publishable standardized unit",
    "AMBIGUOUS_PERCENTAGE_VARIANT": "Unit and standardized test variant disagree (% vs absolute)",
    "SUSPICIOUS_REFERENCE_RANGE": "Reference range looks incompatible with the standardized lab",
    "IMPOSSIBLE_VALUE": "Biologically impossible value detected",
    "RELATIONSHIP_MISMATCH": "Calculated value doesn't match related labs",
    "TEMPORAL_ANOMALY": "Implausible change between consecutive tests",
    "FORMAT_ARTIFACT": "Possible extraction/formatting error",
    "RANGE_INCONSISTENCY": "Reference range appears invalid",
    "PERCENTAGE_BOUNDS": "Percentage value outside 0-100%",
    "NEGATIVE_VALUE": "Unexpected negative value",
    "EXTREME_DEVIATION": "Value extremely far outside reference range",
    "DUPLICATE_ENTRY": "Possible duplicate result for the same date and lab",
}


def build_review_reason_banner(entry: dict) -> str:
    """Build HTML banner showing why a result needs review."""

    # Guard: no entry
    if not entry:
        return ""

    review_reason = entry.get("review_reason")

    # Guard: no review reason
    if not review_reason or not pd.notna(review_reason):
        return ""

    # Parse reason codes (semicolon-separated)
    reasons = []
    reason_str = str(review_reason).strip()
    for code in reason_str.split(";"):
        code = code.strip()
        # Known reason code — use human-readable description
        if code and code in REASON_DESCRIPTIONS:
            reasons.append(REASON_DESCRIPTIONS[code])
        # Unknown reason code — show as-is
        elif code:
            reasons.append(code)

    # Guard: no reasons after parsing
    if not reasons:
        return ""

    # Build banner HTML
    html = '<div class="review-banner warning">'
    html += '<div class="review-banner-title">⚠️ Review Needed</div>'
    html += '<div class="review-banner-reasons">'
    html += "<br>".join(f"• {r}" for r in reasons)
    html += "</div>"
    html += "</div>"
    return html


def build_selection_inspector_html(entry: dict | pd.Series | None) -> str:
    """Render the shared selection inspector used for both exploration and review."""

    if entry is None:
        return (
            '<div class="review-card workspace-empty-card">'
            '<div class="review-empty-state">No result selected.</div>'
            '<div class="review-empty-hint">Adjust filters or pick a row from the table to inspect it.</div>'
            "</div>"
        )

    row = entry if isinstance(entry, pd.Series) else pd.Series(entry)
    status_badge = build_review_status_badge(row)
    reason_text = format_text(row.get("review_reason"), empty="")
    comments_text = format_text(row.get("raw_comments"), empty="")
    source_bbox = get_bbox_coordinates(row)
    date_value = row.get("date")
    if date_value is not None and pd.notna(date_value):
        try:
            date_text = pd.to_datetime(date_value).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            date_text = format_text(date_value)
    else:
        date_text = "-"
    meta_rows: list[str] = []

    if reason_text:
        meta_rows.append(
            '<div class="review-meta-row compact"><span>Validation</span>'
            f'<strong>{reason_text}</strong>'
            "</div>"
        )

    if comments_text:
        meta_rows.append(
            '<div class="review-meta-row compact"><span>Comments</span>'
            f"<strong>{comments_text}</strong>"
            "</div>"
        )

    meta_rows.append(
        '<div class="review-meta-row compact"><span>Source Box</span>'
        f"<strong>{'Available' if source_bbox is not None else 'Not stored for this row'}</strong>"
        "</div>"
    )

    context_items = [
        f'<span class="review-inline-meta"><span class="label">Date</span>{date_text}</span>',
        f'<span class="review-inline-meta"><span class="label">Document</span>{format_text(_format_document_label(row.get("source_file")))}</span>',
        f'<span class="review-inline-meta"><span class="label">Page</span>{format_text(row.get("page_number"))}</span>',
        f'<span class="review-inline-meta"><span class="label">Row</span>{format_text(row.get("result_index"))}</span>',
    ]
    meta_html = f'<div class="review-meta-list">{"".join(meta_rows)}</div>' if meta_rows else ""

    return (
        '<div class="review-card">'
        '<div class="review-card-header compact">'
        '<div class="review-header-line">'
        '<span class="review-eyebrow">Selected Result</span>'
        f'<span class="review-inline-status {html.escape(normalize_review_status(row.get("review_status")) or "pending")}">{status_badge.label}</span>'
        "</div>"
        f'<div class="review-title">{format_text(row.get("lab_name"))}</div>'
        "</div>"
        f'<div class="review-inline-meta-row workspace-context-row">{"".join(context_items)}</div>'
        '<div class="review-compare-grid compact">'
        '<div class="review-compare-panel compact">'
        '<div class="review-panel-title">Mapped</div>'
        f'<div class="review-field"><span>Lab</span><strong>{format_text(row.get("lab_name"))}</strong></div>'
        f'<div class="review-field"><span>Value</span><strong>{format_mapped_value(row)}</strong></div>'
        f'<div class="review-field"><span>Reference</span><strong>{format_mapped_reference_text(row)}</strong></div>'
        "</div>"
        '<div class="review-compare-panel compact">'
        '<div class="review-panel-title">Raw</div>'
        f'<div class="review-field"><span>Lab</span><strong>{format_text(row.get("raw_lab_name"))}</strong></div>'
        f'<div class="review-field"><span>Value</span><strong>{format_raw_value(row)}</strong></div>'
        f'<div class="review-field"><span>Reference</span><strong>{format_reference_text(row)}</strong></div>'
        "</div>"
        "</div>"
        f"{meta_html}"
        "</div>"
    )


# =============================================================================
# Filtering
# =============================================================================


def apply_filters(
    df: pd.DataFrame,
    lab_names: str | None,
    latest_only: bool,
    review_filter: str,
    document_name: str | None = None,
) -> pd.DataFrame:
    """Apply all filters to DataFrame and sort by date descending.

    Args:
        df: Full DataFrame
        lab_names: Lab name to filter by
        latest_only: Whether to show only latest result per lab
        review_filter: Filter option ('All', 'Needs Review', 'Abnormal', 'Unhealthy', 'Unreviewed', 'Accepted', 'Rejected')
    """

    # Guard: no data to filter
    if df.empty:
        return df

    filtered = df.copy()

    if document_name and "source_file" in filtered.columns:
        filtered = filtered[filtered["source_file"] == document_name]

    if document_name and {"page_number", "result_index"}.issubset(filtered.columns):
        filtered = filtered.sort_values(
            ["page_number", "result_index", "lab_name"],
            ascending=[True, True, True],
            na_position="last",
        )
    elif "date" in filtered.columns:
        filtered = filtered.sort_values(
            ["date", "lab_name", "source_file", "page_number", "result_index"],
            ascending=[False, True, True, True, True],
            na_position="last",
        )

    # Latest only: keep only the most recent value per lab test
    # This must run BEFORE status filters so we get the latest result first,
    # then apply status filters to those latest results
    if latest_only and "lab_name" in filtered.columns and "date" in filtered.columns:
        filtered = filtered.drop_duplicates(subset=["lab_name"], keep="first")

    # Filter by lab name (single selection)
    if lab_names:
        filtered = filtered[filtered["lab_name"] == lab_names]

    # Status filter pills options
    if review_filter == "Unreviewed":
        filtered = filtered[filtered["review_status"].isna() | (filtered["review_status"] == "")]
    elif review_filter == "Abnormal":
        # Results outside PDF reference range
        if "is_out_of_reference" in filtered.columns:
            filtered = filtered[filtered["is_out_of_reference"].fillna(False)]
    elif review_filter == "Needs Review":
        # Flagged by validation and not yet reviewed
        if "review_needed" in filtered.columns:
            filtered = filtered[(filtered["review_needed"].fillna(False)) & (filtered["review_status"].isna() | (filtered["review_status"] == ""))]
    elif review_filter == "Unhealthy":
        # Results outside lab_specs healthy range
        if "is_out_of_healthy_range" in filtered.columns:
            filtered = filtered[filtered["is_out_of_healthy_range"].fillna(False)]
    elif review_filter == "Accepted":
        if "review_status" in filtered.columns:
            filtered = filtered[filtered["review_status"] == "accepted"]
    elif review_filter == "Rejected":
        if "review_status" in filtered.columns:
            filtered = filtered[filtered["review_status"] == "rejected"]

    # Reset index so iloc positions match displayed row positions
    filtered = filtered.reset_index(drop=True)

    return filtered


def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for display (subset and format columns)."""

    # Guard: no data
    if df.empty:
        return pd.DataFrame(columns=[COLUMN_LABELS.get(c, c) for c in DISPLAY_COLUMNS])

    display_df = df.copy()

    if "reference_range" not in display_df.columns:
        display_df["reference_range"] = display_df.apply(format_reference_text, axis=1)

    if "source_document" not in display_df.columns:
        display_df["source_document"] = display_df["source_file"].apply(_format_document_label) if "source_file" in display_df.columns else "-"

    display_df = display_df[[col for col in DISPLAY_COLUMNS if col in display_df.columns]].copy()

    # Format date column
    if "date" in display_df.columns:
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

    # Format review status
    if "review_status" in display_df.columns:
        display_df["review_status"] = display_df["review_status"].fillna("").apply(lambda x: x.capitalize() if x else "")

    # Round numeric columns
    if "value" in display_df.columns:
        display_df["value"] = display_df["value"].round(2)

    if "page_number" in display_df.columns:
        display_df["page_number"] = display_df["page_number"].apply(
            lambda value: "" if value is None or pd.isna(value) else str(int(value))
        )

    # Rename columns to display labels
    display_df = display_df.rename(columns=COLUMN_LABELS)

    return display_df


# =============================================================================
# Plotting
# =============================================================================


def create_single_lab_plot(
    df: pd.DataFrame,
    lab_name: str,
    selected_ref: tuple[float, float] | None = None,
) -> tuple[go.Figure, str]:
    """Generate a single plot for one lab test. Returns (figure, unit).

    Args:
        df: DataFrame with lab data
        lab_name: Name of the lab test to plot
        selected_ref: Optional (ref_min, ref_max) from the selected row. When provided,
                      this is used for the PDF reference range instead of computing the mode.
    """

    lab_df = df[df["lab_name"] == lab_name].copy()

    # Guard: no data for this lab
    if lab_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data for {lab_name}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(template="plotly_white", height=300)
        return fig, ""

    # Prepare time series data
    lab_df["date"] = pd.to_datetime(lab_df["date"], errors="coerce")
    lab_df = lab_df.dropna(subset=["date", "value"])
    lab_df = lab_df.sort_values("date")

    # Guard: no valid data points after cleanup
    if lab_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No valid data points for {lab_name}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(template="plotly_white", height=300)
        return fig, ""

    unit = ""
    if "lab_unit" in lab_df.columns:
        units = lab_df["lab_unit"].dropna()
        if not units.empty:
            unit = str(units.iloc[0])

    fig = go.Figure()
    point_customdata = lab_df.apply(_build_row_token_for_entry, axis=1).tolist()

    # Add data trace
    fig.add_trace(
        go.Scatter(
            x=lab_df["date"],
            y=lab_df["value"],
            customdata=point_customdata,
            mode="lines+markers",
            name="Values",
            marker=dict(size=10, color="#1f77b4"),
            line=dict(width=2),
            hovertemplate=(f"<b>Date:</b> %{{x|%Y-%m-%d}}<br><b>Value:</b> %{{y:.2f}} {unit}<br><extra></extra>"),
        )
    )

    has_lab_specs_range = False
    has_pdf_range = False

    # Add lab_specs healthy range (blue band)
    if "lab_specs_min" in lab_df.columns and "lab_specs_max" in lab_df.columns:
        min_vals = lab_df["lab_specs_min"].dropna()
        max_vals = lab_df["lab_specs_max"].dropna()

        if not min_vals.empty and not max_vals.empty:
            spec_min = float(min_vals.iloc[0])
            spec_max = float(max_vals.iloc[0])
            has_lab_specs_range = True

            fig.add_hrect(
                y0=spec_min,
                y1=spec_max,
                fillcolor="rgba(37, 99, 235, 0.20)",
                line_width=0,
            )
            fig.add_hline(
                y=spec_min,
                line_dash="dot",
                line_color="rgba(37, 99, 235, 0.8)",
            )
            fig.add_hline(
                y=spec_max,
                line_dash="dot",
                line_color="rgba(37, 99, 235, 0.8)",
            )

    # Add PDF reference range (orange band)
    if "reference_min" in lab_df.columns and "reference_max" in lab_df.columns:
        min_vals = lab_df["reference_min"].dropna()
        max_vals = lab_df["reference_max"].dropna()

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
                    y0=ref_min,
                    y1=ref_max,
                    fillcolor="rgba(245, 158, 11, 0.20)",
                    line_width=0,
                )
                fig.add_hline(
                    y=ref_min,
                    line_dash="dash",
                    line_color="rgba(245, 158, 11, 0.8)",
                )
                fig.add_hline(
                    y=ref_max,
                    line_dash="dash",
                    line_color="rgba(245, 158, 11, 0.8)",
                )
            elif ref_max is not None:
                data_min = lab_df["value"].min()
                y_bottom = min(0, data_min * 0.9) if data_min > 0 else data_min * 1.1
                fig.add_hrect(
                    y0=y_bottom,
                    y1=ref_max,
                    fillcolor="rgba(245, 158, 11, 0.20)",
                    line_width=0,
                )
                fig.add_hline(
                    y=ref_max,
                    line_dash="dash",
                    line_color="rgba(245, 158, 11, 0.8)",
                    annotation_text=f"< {ref_max}",
                    annotation_position="top right",
                )
            else:
                data_max = lab_df["value"].max()
                y_top = max(data_max * 1.2, ref_min * 2) if data_max > 0 else data_max * 0.8
                fig.add_hrect(
                    y0=ref_min,
                    y1=y_top,
                    fillcolor="rgba(245, 158, 11, 0.20)",
                    line_width=0,
                )
                fig.add_hline(
                    y=ref_min,
                    line_dash="dash",
                    line_color="rgba(245, 158, 11, 0.8)",
                    annotation_text=f"> {ref_min}",
                    annotation_position="bottom right",
                )

    # Add legend entries
    if has_lab_specs_range:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="rgba(37, 99, 235, 0.6)", symbol="square"),
                name="Healthy Range",
                showlegend=True,
            )
        )

    if has_pdf_range:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="rgba(245, 158, 11, 0.6)", symbol="square"),
                name="PDF Reference",
                showlegend=True,
            )
        )

    xaxis_config = dict(
        title="Date",
        tickformat="%Y",
    )

    if len(lab_df) == 1:
        single_date = lab_df["date"].iloc[0]
        xaxis_config["tickvals"] = [single_date]
        xaxis_config["ticktext"] = [single_date.strftime("%Y")]

    fig.update_layout(
        title=dict(
            text=f"{lab_name}" + (f" [{unit}]" if unit else ""),
            font=dict(size=14),
        ),
        xaxis=xaxis_config,
        yaxis_title=f"Value ({unit})" if unit else "Value",
        hovermode="x unified",
        template="plotly_white",
        height=300,
        margin=dict(l=60, r=20, t=40, b=40),
        showlegend=has_lab_specs_range or has_pdf_range,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    )

    return fig, unit


def create_interactive_plot(
    df: pd.DataFrame,
    lab_names: list | None,
    selected_ref: tuple[float, float] | None = None,
) -> go.Figure:
    """Generate interactive Plotly plot(s) for selected lab tests.

    Args:
        df: DataFrame with lab data
        lab_names: List of lab names to plot
        selected_ref: Optional (ref_min, ref_max) from the selected row. Only applies
                      to single-lab plots or the first lab in multi-lab plots.
    """

    # Guard: no labs selected or no data
    if not lab_names or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Select lab tests to view time series",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(template="plotly_white", height=220)
        return fig

    # Single lab — delegate to dedicated single-lab plot
    if len(lab_names) == 1:
        fig, _ = create_single_lab_plot(df, lab_names[0], selected_ref=selected_ref)
        fig.update_layout(height=240)
        return fig

    # Multiple labs — create stacked subplots
    n_labs = len(lab_names)
    fig = make_subplots(
        rows=n_labs,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=lab_names,
    )

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for i, lab_name in enumerate(lab_names):
        lab_df = df[df["lab_name"] == lab_name].copy()

        if lab_df.empty:
            continue

        lab_df["date"] = pd.to_datetime(lab_df["date"], errors="coerce")
        lab_df = lab_df.dropna(subset=["date", "value"])
        lab_df = lab_df.sort_values("date")

        if lab_df.empty:
            continue

        unit = ""
        if "lab_unit" in lab_df.columns:
            units = lab_df["lab_unit"].dropna()
            if not units.empty:
                unit = str(units.iloc[0])

        color = colors[i % len(colors)]
        point_customdata = lab_df.apply(_build_row_token_for_entry, axis=1).tolist()

        fig.add_trace(
            go.Scatter(
                x=lab_df["date"],
                y=lab_df["value"],
                customdata=point_customdata,
                mode="lines+markers",
                name="Values",
                marker=dict(size=8, color=color),
                line=dict(width=2, color=color),
                hovertemplate=(f"<b>Date:</b> %{{x|%Y-%m-%d}}<br><b>Value:</b> %{{y:.2f}} {unit}<br><extra></extra>"),
            ),
            row=i + 1,
            col=1,
        )

        # Add lab_specs healthy range (blue band)
        if "lab_specs_min" in lab_df.columns and "lab_specs_max" in lab_df.columns:
            min_vals = lab_df["lab_specs_min"].dropna()
            max_vals = lab_df["lab_specs_max"].dropna()

            if not min_vals.empty and not max_vals.empty:
                spec_min = float(min_vals.iloc[0])
                spec_max = float(max_vals.iloc[0])

                fig.add_hrect(
                    y0=spec_min,
                    y1=spec_max,
                    fillcolor="rgba(37, 99, 235, 0.20)",
                    line_width=0,
                    row=i + 1,
                    col=1,
                )

        # Add PDF reference range (orange band)
        if "reference_min" in lab_df.columns and "reference_max" in lab_df.columns:
            min_vals = lab_df["reference_min"].dropna()
            max_vals = lab_df["reference_max"].dropna()

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
                    y0=ref_min,
                    y1=ref_max,
                    fillcolor="rgba(245, 158, 11, 0.20)",
                    line_width=0,
                    row=i + 1,
                    col=1,
                )

        fig.update_yaxes(title_text=unit if unit else "Value", row=i + 1, col=1)

    height_per_chart = 200
    fig.update_layout(
        template="plotly_white",
        height=height_per_chart * n_labs,
        margin=dict(l=60, r=20, t=40, b=40),
        showlegend=False,
        hovermode="x unified",
    )

    fig.update_xaxes(tickformat="%Y")

    return fig


# =============================================================================
# Event Handlers
# =============================================================================


def _load_output_data(
    output_path: Path,
    lab_specs: LabSpecsConfig,
    demographics: Demographics | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load and sort data from output path, return (full_df, lab_name_choices)."""

    full_df = load_results_dataframe(output_path, lab_specs, demographics)
    if not full_df.empty and "date" in full_df.columns:
        full_df = full_df.sort_values(["date", "lab_name"], ascending=[False, True], na_position="last").reset_index(drop=True)
    return full_df, get_lab_name_choices(full_df)


def _build_empty_viewer_state(
    full_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    *,
    summary_df: pd.DataFrame,
    plot_labs: list[str] | None = None,
    position_text: str = "No results",
) -> ViewerRenderState:
    """Build the stable empty-state payload for viewer callbacks."""

    prev_button_props, next_button_props = _build_navigation_button_props(None, len(filtered_df))
    return ViewerRenderState(
        display_df=prepare_display_df(filtered_df),
        summary_html=build_summary_cards(summary_df),
        plot=create_interactive_plot(full_df, plot_labs or []),
        filtered_df=filtered_df,
        current_idx=0,
        position_text=position_text,
        source_image_value=None,
        inspector_html=build_selection_inspector_html(None),
        selection_html=_build_selection_state_html(None, len(filtered_df)),
        prev_button_props=prev_button_props,
        next_button_props=next_button_props,
    )


def _build_selection_state_html(
    selected_row_index: int | None,
    row_count: int,
    selected_row_token: str | None = None,
    selected_display_values: list[str] | None = None,
) -> str:
    """Return a hidden DOM marker that frontend code can use to sync row highlighting."""

    selected_value = "" if selected_row_index is None else str(int(selected_row_index))
    selected_token = "" if not selected_row_token else html.escape(selected_row_token, quote=True)
    selected_display = html.escape(json.dumps(selected_display_values or [], separators=(",", ":")), quote=True)
    return (
        '<div id="viewer-selection-state" '
        f'data-selected-row="{selected_value}" '
        f'data-row-count="{int(row_count)}" '
        f"data-selected-token='{selected_token}' "
        f"data-selected-display='{selected_display}' "
        'aria-hidden="true"></div>'
    )


def _build_navigation_button_props(selected_row_index: int | None, row_count: int) -> tuple[dict, dict]:
    """Return Gradio button updates for the current selection boundaries."""

    if selected_row_index is None or row_count <= 0:
        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    resolved_index = max(0, min(int(selected_row_index), row_count - 1))
    return (
        gr.update(interactive=resolved_index > 0),
        gr.update(interactive=resolved_index < row_count - 1),
    )


def _render_viewer_state(
    full_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    output_path: Path,
    lab_names: str | None,
    *,
    row_index: int = 0,
    summary_df: pd.DataFrame | None = None,
    empty_position_text: str = "No results",
    document_name: str | None = None,
) -> ViewerRenderState:
    """Build the unified viewer render payload for the current filtered dataframe."""

    resolved_summary_df = filtered_df if summary_df is None else summary_df

    # Guard: Empty result sets render the stable placeholder state.
    if filtered_df.empty:
        return _build_empty_viewer_state(
            full_df,
            filtered_df,
            summary_df=resolved_summary_df,
            plot_labs=[lab_names] if lab_names else [],
            position_text=empty_position_text,
        )

    resolved_row_index = max(0, min(int(row_index), len(filtered_df) - 1))
    selected_row = get_selected_row(filtered_df, resolved_row_index)

    # Guard: Missing selections fall back to the stable empty placeholder.
    if selected_row is None:
        return _build_empty_viewer_state(
            full_df,
            filtered_df,
            summary_df=resolved_summary_df,
            plot_labs=[lab_names] if lab_names else [],
            position_text=empty_position_text,
        )

    row_context = build_viewer_row_context(
        filtered_df,
        resolved_row_index,
        output_path,
        selected_lab_name=lab_names,
        banner_html="",
    )

    # Guard: Row-context resolution failures should degrade gracefully.
    if row_context is None:
        return _build_empty_viewer_state(
            full_df,
            filtered_df,
            summary_df=resolved_summary_df,
            plot_labs=[lab_names] if lab_names else [],
            position_text=empty_position_text,
        )

    prev_button_props, next_button_props = _build_navigation_button_props(row_context.row_index, len(filtered_df))
    return ViewerRenderState(
        display_df=prepare_display_df(filtered_df),
        summary_html=build_summary_cards(resolved_summary_df),
        plot=create_interactive_plot(full_df, row_context.plot_labs, selected_ref=row_context.selected_ref),
        filtered_df=filtered_df,
        current_idx=row_context.row_index,
        position_text=(
            f"**{row_context.row_index + 1} of {len(filtered_df)} in {_format_document_label(document_name)}**"
            if document_name
            else row_context.position_text.replace("Row", "Result")
        ),
        source_image_value=row_context.source_image_value,
        inspector_html=build_selection_inspector_html(selected_row),
        selection_html=_build_selection_state_html(
            row_context.row_index,
            len(filtered_df),
            _build_row_token_for_entry(selected_row),
            _build_display_row_match_values(selected_row),
        ),
        prev_button_props=prev_button_props,
        next_button_props=next_button_props,
    )


def handle_filter_change(
    lab_names: str | None,
    latest_only: bool,
    review_filter: str,
    full_df: pd.DataFrame,
    output_path: Path,
    document_name: str | None = None,
) -> tuple:
    """Handle filter changes and update viewer state from the first visible row."""

    filtered_df = apply_filters(full_df, lab_names, latest_only, review_filter, document_name=document_name)
    return _render_viewer_state(
        full_df,
        filtered_df,
        output_path,
        lab_names,
        summary_df=filtered_df,
        document_name=document_name,
    ).as_filter_outputs()


def handle_row_select(
    evt: gr.SelectData,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    lab_names: str | None,
    output_path: Path,
    document_name: str | None = None,
) -> tuple:
    """Handle row selection to update plot, details, and current index."""

    # Guard: Empty selections render the stable placeholder state.
    if evt is None or filtered_df.empty:
        render_state = _build_empty_viewer_state(full_df, filtered_df, summary_df=filtered_df)
    else:
        row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        selected_idx = 0 if row_idx is None or row_idx < 0 or row_idx >= len(filtered_df) else int(row_idx)
        render_state = _render_viewer_state(
            full_df,
            filtered_df,
            output_path,
            lab_names,
            row_index=selected_idx,
            summary_df=filtered_df,
            document_name=document_name,
        )

    return (
        render_state.current_idx,
        render_state.position_text,
        render_state.inspector_html,
        render_state.source_image_value,
        render_state.plot,
        render_state.selection_html,
        render_state.prev_button_props,
        render_state.next_button_props,
    )


def _dispatch_row_select(
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    lab_names: str | None,
    evt: gr.SelectData,
    output_path: Path,
    document_name: str | None = None,
) -> tuple:
    """Adapt Gradio's input-first select callback order to the row-select handler."""

    return handle_row_select(
        evt,
        filtered_df,
        full_df,
        lab_names,
        output_path,
        document_name=document_name,
    )


def _resolve_plot_point_row_index(filtered_df: pd.DataFrame, point_token: str | None) -> int | None:
    """Resolve a Plotly point token back to a filtered dataframe row index."""

    if filtered_df.empty or not point_token:
        return None

    try:
        token_data = json.loads(point_token)
    except (TypeError, json.JSONDecodeError):
        return None

    if isinstance(token_data, str):
        selected_token = token_data
    elif isinstance(token_data, dict):
        source_file = token_data.get("source_file")
        page_number = token_data.get("page_number")
        result_index = token_data.get("result_index")
        selected_token = _build_row_token(source_file, page_number, result_index)
    elif isinstance(token_data, list | tuple) and len(token_data) >= 3:
        source_file, page_number, result_index = token_data[:3]
        selected_token = _build_row_token(source_file, page_number, result_index)
    else:
        return None

    row_tokens = filtered_df.apply(_build_row_token_for_entry, axis=1)
    matches = row_tokens.index[row_tokens == selected_token]
    if len(matches) == 0:
        return None

    return int(matches[0])


def handle_navigation(
    current_idx: int,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    lab_names: str | None,
    delta: int,
    output_path: Path,
    document_name: str | None = None,
) -> tuple:
    """Move the current selection backward or forward through the filtered rows."""

    # Guard: Empty result sets render the stable placeholder state.
    if filtered_df.empty:
        render_state = _build_empty_viewer_state(full_df, filtered_df, summary_df=filtered_df)
    else:
        resolved_index = max(0, min(int(current_idx), len(filtered_df) - 1))
        next_idx = max(0, min(resolved_index + delta, len(filtered_df) - 1))
        render_state = _render_viewer_state(
            full_df,
            filtered_df,
            output_path,
            lab_names,
            row_index=next_idx,
            summary_df=filtered_df,
            document_name=document_name,
        )

    return (
        render_state.current_idx,
        render_state.position_text,
        render_state.inspector_html,
        render_state.source_image_value,
        render_state.plot,
        render_state.selection_html,
        render_state.prev_button_props,
        render_state.next_button_props,
    )


def handle_plot_point_select(
    point_token: str | None,
    current_idx: int,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    lab_names: str | None,
    output_path: Path,
    document_name: str | None = None,
) -> tuple:
    """Handle a plotly point click by selecting the matching table row when visible."""

    if filtered_df.empty:
        render_state = _build_empty_viewer_state(full_df, filtered_df, summary_df=filtered_df)
    else:
        fallback_idx = max(0, min(int(current_idx), len(filtered_df) - 1))
        matched_idx = _resolve_plot_point_row_index(filtered_df, point_token)
        render_state = _render_viewer_state(
            full_df,
            filtered_df,
            output_path,
            lab_names,
            row_index=fallback_idx if matched_idx is None else matched_idx,
            summary_df=filtered_df,
            document_name=document_name,
        )

    return (
        render_state.current_idx,
        render_state.position_text,
        render_state.inspector_html,
        render_state.source_image_value,
        render_state.plot,
        render_state.selection_html,
        render_state.prev_button_props,
        render_state.next_button_props,
    )


def handle_review_action(
    current_idx: int,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    lab_names: str | None,
    latest_only: bool,
    review_filter: str,
    status: str,
    output_path: Path,
    document_name: str | None = None,
) -> tuple:
    """Handle review action (accept or reject) and rerender the current filter."""

    # Guard: Empty result sets render the stable placeholder state.
    if filtered_df.empty:
        return _build_empty_viewer_state(full_df, filtered_df, summary_df=full_df).as_review_outputs(full_df)

    resolved_index = max(0, min(int(current_idx), len(filtered_df) - 1))
    current_entry = filtered_df.iloc[resolved_index].to_dict()
    action = {
        "accepted": "accept",
        "rejected": "reject",
        "missing_row": "missing_row",
    }.get(status, "clear")
    success, error = apply_review_action_for_entry(current_entry, output_path, action)

    # Surface persistence errors without mutating the local dataframe mirror.
    if not success:
        gr.Warning(f"Failed to save review: {error}")

    # Mirror successful review writes into the in-memory full dataframe state.
    elif status in {"accepted", "rejected", "clear"}:
        mask = (full_df["source_file"] == current_entry.get("source_file")) & (full_df["page_number"] == current_entry.get("page_number")) & (full_df["result_index"] == current_entry.get("result_index"))
        full_df.loc[mask, "review_status"] = "" if status == "clear" else status
    else:
        gr.Info("Missing-row marker recorded for this page.")

    filtered_df = apply_filters(full_df, lab_names, latest_only, review_filter, document_name=document_name)
    render_state = _render_viewer_state(
        full_df,
        filtered_df,
        output_path,
        lab_names,
        row_index=min(resolved_index, max(0, len(filtered_df) - 1)) if not filtered_df.empty else 0,
        summary_df=full_df,
        empty_position_text="All done!",
        document_name=document_name,
    )
    return render_state.as_review_outputs(full_df)


# =============================================================================
# Main App
# =============================================================================


def create_app(context: RuntimeContext, *, launch_mode: str = "results-explorer"):
    """Create and configure the unified review workspace for one runtime context."""

    output_path = context.output_path if context.output_path is not None else Path("./output")
    prioritize_review_sources = str(launch_mode).strip().lower() == "review-queue"
    full_df, lab_name_choices = _load_output_data(
        output_path,
        context.lab_specs,
        context.demographics,
    )
    document_choices = get_document_choices(full_df, prioritize_review_sources=prioritize_review_sources)
    initial_document = get_initial_document(full_df, prioritize_review_sources=prioritize_review_sources)
    initial_filtered_df = apply_filters(
        full_df,
        None,
        False,
        "All",
        document_name=initial_document,
    )
    initial_view = _render_viewer_state(
        full_df,
        initial_filtered_df,
        output_path,
        None,
        summary_df=initial_filtered_df,
        document_name=initial_document,
    )

    with gr.Blocks(
        title="Parselabs Review Workspace",
        fill_width=True,
        fill_height=True,
    ) as demo:
        full_df_state = gr.State(value=full_df)
        filtered_df_state = gr.State(value=initial_view.filtered_df)
        current_idx_state = gr.State(value=initial_view.current_idx)

        with gr.Column(elem_id="workspace-shell"):
            with gr.Row(elem_id="workspace-filter-row", elem_classes="filter-row"):
                with gr.Column(scale=2, min_width=220):
                    document_filter = gr.Dropdown(
                        choices=document_choices,
                        multiselect=False,
                        value=initial_document,
                        label="Document",
                        allow_custom_value=False,
                    )
                with gr.Column(scale=2, min_width=220):
                    lab_name_filter = gr.Dropdown(
                        choices=lab_name_choices,
                        multiselect=False,
                        value=None,
                        label="Lab",
                        allow_custom_value=False,
                        elem_classes="lab-dropdown-compact",
                    )
                with gr.Column(scale=2, min_width=180):
                    review_filter = gr.Dropdown(
                        choices=[
                            "All",
                            "Needs Review",
                            "Abnormal",
                            "Unhealthy",
                            "Unreviewed",
                            "Accepted",
                            "Rejected",
                        ],
                        value="All",
                        label="Status",
                        allow_custom_value=False,
                    )
                with gr.Column(scale=1, min_width=120):
                    latest_filter = gr.Checkbox(
                        label="Latest Only",
                        value=False,
                        elem_classes="toggle-pill",
                    )

            summary_display = gr.HTML(initial_view.summary_html, elem_id="workspace-summary")

            with gr.Row(elem_id="workspace-main-row"):
                with gr.Column(scale=5, min_width=460, elem_id="workspace-document-col"):
                    source_image = gr.AnnotatedImage(
                        value=initial_view.source_image_value,
                        label="Source Document Page",
                        color_map={SOURCE_BBOX_LABEL: "#dc2626"},
                        show_legend=False,
                        show_label=False,
                        elem_id="workspace-source-image",
                    )

                with gr.Column(scale=3, min_width=260, elem_id="workspace-analysis-col"):
                    inspector_display = gr.HTML(value=initial_view.inspector_html, elem_id="workspace-inspector-card")

                    plot_display = gr.Plot(
                        value=initial_view.plot,
                        label="",
                        elem_id="viewer-plot",
                    )
                    plot_point_selection = gr.Textbox(value="", container=False, elem_id="plot-point-selection")
                    plot_point_select_btn = gr.Button("Select Plot Point", elem_id="plot-point-select-btn")

                    with gr.Column(elem_id="workspace-action-bar"):
                        with gr.Row(elem_id="workspace-primary-controls"):
                            prev_btn = gr.Button("< Prev [k]", elem_id="prev-btn", size="sm", interactive=False)
                            position_display = gr.Markdown(initial_view.position_text, elem_id="position-display")
                            next_btn = gr.Button("Next [j] >", elem_id="next-btn", size="sm", interactive=len(initial_view.filtered_df) > 1)
                            accept_btn = gr.Button("Accept [y]", elem_id="accept-btn", variant="primary")
                            reject_btn = gr.Button("Reject [n]", elem_id="reject-btn", variant="stop")
                        with gr.Row(elem_id="workspace-secondary-controls"):
                            undo_btn = gr.Button("Undo [u]", elem_id="undo-btn", size="sm")
                            missing_btn = gr.Button("Missing [m]", elem_id="missing-btn", size="sm")

                with gr.Column(scale=4, min_width=360, elem_id="workspace-results-col"):
                    data_table = gr.DataFrame(
                        value=initial_view.display_df,
                        interactive=False,
                        wrap=True,
                        elem_id="lab-data-table",
                    )
                    selection_state = gr.HTML(
                        value=initial_view.selection_html,
                        elem_id="viewer-selection-state-host",
                    )

            gr.Markdown(
                "*Keyboard: Y=Accept, N=Reject, U=Undo, M=Missing, Arrow keys/J/K=Navigate*",
                elem_id="workspace-keyboard-hint",
            )

        filter_inputs = [
            document_filter,
            lab_name_filter,
            latest_filter,
            review_filter,
            full_df_state,
        ]
        filter_outputs = [
            data_table,
            summary_display,
            filtered_df_state,
            current_idx_state,
            position_display,
            inspector_display,
            source_image,
            plot_display,
            selection_state,
            prev_btn,
            next_btn,
        ]

        def _handle_data_table_select(
            filtered_df: pd.DataFrame,
            full_df: pd.DataFrame,
            document_name: str | None,
            lab_name: str | None,
            evt: gr.SelectData,
        ) -> tuple:
            return _dispatch_row_select(
                filtered_df,
                full_df,
                lab_name,
                evt,
                output_path,
                document_name=document_name,
            )

        for trigger in [document_filter, lab_name_filter, latest_filter, review_filter]:
            trigger.change(
                fn=lambda document_name, lab_names, latest_only, review_filter, full_df: handle_filter_change(
                    lab_names,
                    latest_only,
                    review_filter,
                    full_df,
                    output_path,
                    document_name=document_name,
                ),
                inputs=filter_inputs,
                outputs=filter_outputs,
            )

        data_table.select(
            fn=_handle_data_table_select,
            inputs=[filtered_df_state, full_df_state, document_filter, lab_name_filter],
            outputs=[
                current_idx_state,
                position_display,
                inspector_display,
                source_image,
                plot_display,
                selection_state,
                prev_btn,
                next_btn,
            ],
        )

        nav_inputs = [
            current_idx_state,
            filtered_df_state,
            full_df_state,
            document_filter,
            lab_name_filter,
        ]
        nav_outputs = [
            current_idx_state,
            position_display,
            inspector_display,
            source_image,
            plot_display,
            selection_state,
            prev_btn,
            next_btn,
        ]

        plot_point_select_btn.click(
            fn=lambda point_token, current_idx, filtered_df, full_df, document_name, lab_name: handle_plot_point_select(
                point_token,
                current_idx,
                filtered_df,
                full_df,
                lab_name,
                output_path,
                document_name=document_name,
            ),
            inputs=[
                plot_point_selection,
                current_idx_state,
                filtered_df_state,
                full_df_state,
                document_filter,
                lab_name_filter,
            ],
            outputs=nav_outputs,
        )

        prev_btn.click(
            fn=lambda current_idx, filtered_df, full_df, document_name, lab_name: handle_navigation(
                current_idx,
                filtered_df,
                full_df,
                lab_name,
                -1,
                output_path,
                document_name=document_name,
            ),
            inputs=nav_inputs,
            outputs=nav_outputs,
        )
        next_btn.click(
            fn=lambda current_idx, filtered_df, full_df, document_name, lab_name: handle_navigation(
                current_idx,
                filtered_df,
                full_df,
                lab_name,
                1,
                output_path,
                document_name=document_name,
            ),
            inputs=nav_inputs,
            outputs=nav_outputs,
        )

        review_btn_inputs = [
            current_idx_state,
            filtered_df_state,
            full_df_state,
            document_filter,
            lab_name_filter,
            latest_filter,
            review_filter,
        ]
        review_outputs = [
            full_df_state,
            filtered_df_state,
            data_table,
            current_idx_state,
            position_display,
            inspector_display,
            source_image,
            plot_display,
            summary_display,
            selection_state,
            prev_btn,
            next_btn,
        ]

        accept_btn.click(
            fn=lambda current_idx, filtered_df, full_df, document_name, lab_name, latest_only, review_filter: handle_review_action(
                current_idx,
                filtered_df,
                full_df,
                lab_name,
                latest_only,
                review_filter,
                "accepted",
                output_path,
                document_name=document_name,
            ),
            inputs=review_btn_inputs,
            outputs=review_outputs,
        )
        reject_btn.click(
            fn=lambda current_idx, filtered_df, full_df, document_name, lab_name, latest_only, review_filter: handle_review_action(
                current_idx,
                filtered_df,
                full_df,
                lab_name,
                latest_only,
                review_filter,
                "rejected",
                output_path,
                document_name=document_name,
            ),
            inputs=review_btn_inputs,
            outputs=review_outputs,
        )
        undo_btn.click(
            fn=lambda current_idx, filtered_df, full_df, document_name, lab_name, latest_only, review_filter: handle_review_action(
                current_idx,
                filtered_df,
                full_df,
                lab_name,
                latest_only,
                review_filter,
                "clear",
                output_path,
                document_name=document_name,
            ),
            inputs=review_btn_inputs,
            outputs=review_outputs,
        )
        missing_btn.click(
            fn=lambda current_idx, filtered_df, full_df, document_name, lab_name, latest_only, review_filter: handle_review_action(
                current_idx,
                filtered_df,
                full_df,
                lab_name,
                latest_only,
                review_filter,
                "missing_row",
                output_path,
                document_name=document_name,
            ),
            inputs=review_btn_inputs,
            outputs=review_outputs,
        )

    return demo
