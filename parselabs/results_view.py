"""
Lab Results Viewer

Interactive UI for browsing and reviewing extracted lab results.
Shows data table with interactive plots and review actions side-by-side.

Usage:
  parselabs review --profile tiago
  parselabs review --list-profiles

Keyboard: Y=Accept, N=Reject, Arrow keys/j/k=Navigate
"""

from __future__ import annotations

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
    build_review_status_html,
    load_results_dataframe,
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
    details_html: str
    status_html: str
    banner_html: str

    def as_filter_outputs(self) -> tuple:
        """Return outputs for filter-driven viewer updates."""

        return (
            self.display_df,
            self.summary_html,
            self.plot,
            self.filtered_df,
            self.current_idx,
            self.position_text,
            self.source_image_value,
            self.details_html,
            self.status_html,
            self.banner_html,
        )

    def as_review_outputs(self, full_df: pd.DataFrame) -> tuple:
        """Return outputs for review-action callbacks that also update full_df state."""

        return (
            full_df,
            self.filtered_df,
            self.display_df,
            self.plot,
            self.current_idx,
            self.position_text,
            self.source_image_value,
            self.details_html,
            self.status_html,
            self.summary_html,
            self.banner_html,
        )


def get_lab_name_choices(df: pd.DataFrame) -> list[str]:
    """Get sorted list of unique lab names from DataFrame, excluding unknowns."""

    # Guard: no data available
    if df.empty or "lab_name" not in df.columns:
        return []
    return sorted([name for name in df["lab_name"].dropna().unique() if name and not str(name).startswith("$UNKNOWN")])


# =============================================================================
# Display Configuration
# =============================================================================

# Display columns for the data table
DISPLAY_COLUMNS = [
    "date",
    "lab_name",
    "value",
    "lab_unit",
    "reference_range",
    "is_out_of_reference",
    "review_status",
]

# Column display names
COLUMN_LABELS = {
    "date": "Date",
    "lab_name": "Lab",
    "value": "Value",
    "lab_unit": "Unit",
    "reference_range": "Ref",
    "is_out_of_reference": "Abn",
    "review_status": "Review",
}


# =============================================================================
# Summary Statistics
# =============================================================================


def build_summary_cards(df: pd.DataFrame) -> str:
    """Generate HTML summary cards with color coding."""

    # Guard: no data
    if df.empty:
        return '<div class="summary-row"><span class="stat-card">No data loaded</span></div>'

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


# =============================================================================
# Filtering
# =============================================================================


def apply_filters(
    df: pd.DataFrame,
    lab_names: str | None,
    latest_only: bool,
    review_filter: str,
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

    # Sort by date descending, then by lab_name ascending
    if "date" in filtered.columns:
        filtered = filtered.sort_values(["date", "lab_name"], ascending=[False, True], na_position="last")

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

    display_df = df[[col for col in DISPLAY_COLUMNS if col in df.columns]].copy()

    # Format date column
    if "date" in display_df.columns:
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

    # Format boolean columns
    if "is_out_of_reference" in display_df.columns:
        display_df["is_out_of_reference"] = display_df["is_out_of_reference"].map({True: "Yes", False: "No", None: ""})

    # Format review status
    if "review_status" in display_df.columns:
        display_df["review_status"] = display_df["review_status"].fillna("").apply(lambda x: x.capitalize() if x else "")

    # Round numeric columns
    if "value" in display_df.columns:
        display_df["value"] = display_df["value"].round(2)

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

    # Add data trace
    fig.add_trace(
        go.Scatter(
            x=lab_df["date"],
            y=lab_df["value"],
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
        fig.update_layout(template="plotly_white", height=400)
        return fig

    # Single lab — delegate to dedicated single-lab plot
    if len(lab_names) == 1:
        fig, _ = create_single_lab_plot(df, lab_names[0], selected_ref=selected_ref)
        fig.update_layout(height=400)
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

        fig.add_trace(
            go.Scatter(
                x=lab_df["date"],
                y=lab_df["value"],
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

    height_per_chart = 250
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
    details_html: str = "<p>No entry selected</p>",
) -> ViewerRenderState:
    """Build the stable empty-state payload for viewer callbacks."""

    return ViewerRenderState(
        display_df=prepare_display_df(filtered_df),
        summary_html=build_summary_cards(summary_df),
        plot=create_interactive_plot(full_df, plot_labs or []),
        filtered_df=filtered_df,
        current_idx=0,
        position_text=position_text,
        source_image_value=None,
        details_html=details_html,
        status_html=build_review_status_html("Pending"),
        banner_html="",
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
    empty_details_html: str = "<p>No entry selected</p>",
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
            details_html=empty_details_html,
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
            details_html=empty_details_html,
        )

    row_context = build_viewer_row_context(
        filtered_df,
        resolved_row_index,
        output_path,
        selected_lab_name=lab_names,
        banner_html=build_review_reason_banner(selected_row.to_dict()),
    )

    # Guard: Row-context resolution failures should degrade gracefully.
    if row_context is None:
        return _build_empty_viewer_state(
            full_df,
            filtered_df,
            summary_df=resolved_summary_df,
            plot_labs=[lab_names] if lab_names else [],
            position_text=empty_position_text,
            details_html=empty_details_html,
        )

    return ViewerRenderState(
        display_df=prepare_display_df(filtered_df),
        summary_html=build_summary_cards(resolved_summary_df),
        plot=create_interactive_plot(full_df, row_context.plot_labs, selected_ref=row_context.selected_ref),
        filtered_df=filtered_df,
        current_idx=row_context.row_index,
        position_text=row_context.position_text,
        source_image_value=row_context.source_image_value,
        details_html=row_context.details_html,
        status_html=row_context.status_html,
        banner_html=row_context.banner_html,
    )


def handle_filter_change(
    lab_names: str | None,
    latest_only: bool,
    review_filter: str,
    full_df: pd.DataFrame,
    output_path: Path,
) -> tuple:
    """Handle filter changes and update viewer state from the first visible row."""

    filtered_df = apply_filters(full_df, lab_names, latest_only, review_filter)
    return _render_viewer_state(
        full_df,
        filtered_df,
        output_path,
        lab_names,
        summary_df=filtered_df,
    ).as_filter_outputs()


def handle_row_select(
    evt: gr.SelectData,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    lab_names: str | None,
    output_path: Path,
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
        )

    return (
        render_state.plot,
        render_state.current_idx,
        render_state.position_text,
        render_state.source_image_value,
        render_state.details_html,
        render_state.status_html,
        render_state.banner_html,
    )


def handle_navigation(
    current_idx: int,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    lab_names: str | None,
    delta: int,
    output_path: Path,
) -> tuple:
    """Move the current selection backward or forward through the filtered rows."""

    # Guard: Empty result sets render the stable placeholder state.
    if filtered_df.empty:
        render_state = _build_empty_viewer_state(full_df, filtered_df, summary_df=filtered_df)
    else:
        next_idx = (int(current_idx) + delta) % len(filtered_df)
        render_state = _render_viewer_state(
            full_df,
            filtered_df,
            output_path,
            lab_names,
            row_index=next_idx,
            summary_df=filtered_df,
        )

    return (
        render_state.plot,
        render_state.current_idx,
        render_state.position_text,
        render_state.source_image_value,
        render_state.details_html,
        render_state.status_html,
        render_state.banner_html,
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
) -> tuple:
    """Handle review action (accept or reject) and rerender the current filter."""

    # Guard: Empty result sets render the stable placeholder state.
    if filtered_df.empty:
        return _build_empty_viewer_state(full_df, filtered_df, summary_df=full_df).as_review_outputs(full_df)

    resolved_index = max(0, min(int(current_idx), len(filtered_df) - 1))
    current_entry = filtered_df.iloc[resolved_index].to_dict()
    action = {"accepted": "accept", "rejected": "reject"}.get(status, "clear")
    success, error = apply_review_action_for_entry(current_entry, output_path, action)

    # Surface persistence errors without mutating the local dataframe mirror.
    if not success:
        gr.Warning(f"Failed to save review: {error}")

    # Mirror successful review writes into the in-memory full dataframe state.
    else:
        mask = (full_df["source_file"] == current_entry.get("source_file")) & (full_df["page_number"] == current_entry.get("page_number")) & (full_df["result_index"] == current_entry.get("result_index"))
        full_df.loc[mask, "review_status"] = status

    filtered_df = apply_filters(full_df, lab_names, latest_only, review_filter)
    render_state = _render_viewer_state(
        full_df,
        filtered_df,
        output_path,
        lab_names,
        row_index=min(resolved_index, max(0, len(filtered_df) - 1)) if not filtered_df.empty else 0,
        summary_df=full_df,
        empty_position_text="All done!",
        empty_details_html="<p>All entries reviewed in this filter!</p>",
    )
    return render_state.as_review_outputs(full_df)


def export_csv(filtered_df: pd.DataFrame, output_path: Path):
    """Export filtered data to CSV file."""

    # Guard: no data to export
    if filtered_df.empty:
        return None

    export_path = output_path / "filtered_export.csv"
    filtered_df.to_csv(export_path, index=False)
    return str(export_path)


# =============================================================================
# Main App
# =============================================================================


def create_app(context: RuntimeContext):
    """Create and configure the Gradio app for one runtime context."""

    # Resolve the launch-selected profile once because review runs no longer switch profiles in-app.
    output_path = context.output_path if context.output_path is not None else Path("./output")

    full_df, lab_name_choices = _load_output_data(
        output_path,
        context.lab_specs,
        context.demographics,
    )
    initial_view = _render_viewer_state(
        full_df,
        full_df,
        output_path,
        None,
        summary_df=full_df,
    )

    with gr.Blocks(title="Lab Results Viewer") as demo:
        # State variables
        full_df_state = gr.State(value=full_df)
        filtered_df_state = gr.State(value=full_df)
        current_idx_state = gr.State(value=0)

        # Unified filter row: Lab dropdown | Status pills | Latest toggle
        with gr.Row(elem_classes="filter-row"):
            with gr.Column(scale=1, min_width=200):
                lab_name_filter = gr.Dropdown(
                    choices=lab_name_choices,
                    multiselect=False,
                    value=None,
                    label="Lab",
                    allow_custom_value=False,
                    elem_classes="lab-dropdown-compact",
                )
            with gr.Column(scale=2):
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

        # Summary cards
        summary_display = gr.HTML(initial_view.summary_html)

        gr.Markdown("---")

        # Main content: Table + Right Panel side by side
        with gr.Row():
            # Left column: Data Table
            with gr.Column(scale=3):
                gr.Markdown("### Data Table")
                gr.Markdown("*Click a row or use arrow keys to navigate*")
                data_table = gr.DataFrame(
                    value=initial_view.display_df,
                    interactive=False,
                    wrap=True,
                    max_height=500,
                    elem_id="lab-data-table",
                )

                with gr.Row():
                    export_btn = gr.Button("Export Filtered CSV", size="sm")
                    export_file = gr.File(label="Download", visible=False)

            # Right column: Navigation + Tabs + Review Actions
            with gr.Column(scale=2):
                # Navigation controls
                with gr.Row():
                    prev_btn = gr.Button("< Prev [k]", elem_id="prev-btn", size="sm")
                    position_display = gr.Markdown(initial_view.position_text, elem_id="position-display")
                    next_btn = gr.Button("Next [j] >", elem_id="next-btn", size="sm")

                # Tabs for Plot, Source Image, and Details
                with gr.Tabs():
                    with gr.TabItem("Plot"):
                        plot_display = gr.Plot(
                            value=initial_view.plot,
                            label="",
                        )
                    with gr.TabItem("Source"):
                        source_image = gr.AnnotatedImage(
                            value=initial_view.source_image_value,
                            label="Source Document Page",
                            color_map={SOURCE_BBOX_LABEL: "#dc2626"},
                            show_legend=False,
                            show_label=False,
                            height=400,
                        )
                    with gr.TabItem("Details"):
                        details_display = gr.HTML(value=initial_view.details_html)

                gr.Markdown("---")

                # Review section with reason banner
                gr.Markdown("### Review")

                # Review reason banner (shows why item needs review)
                review_reason_banner = gr.HTML(value=initial_view.banner_html)

                with gr.Row():
                    accept_btn = gr.Button("Accept [y]", elem_id="accept-btn", size="sm")
                    review_status_display = gr.HTML(value=initial_view.status_html)
                    reject_btn = gr.Button("Reject [n]", elem_id="reject-btn", size="sm")

        gr.Markdown("---")
        gr.Markdown("*Keyboard: Y=Accept, N=Reject, Arrow keys/j/k=Navigate*")

        # Filter inputs and outputs
        filter_inputs = [
            lab_name_filter,
            latest_filter,
            review_filter,
            full_df_state,
        ]
        filter_outputs = [
            data_table,
            summary_display,
            plot_display,
            filtered_df_state,
            current_idx_state,
            position_display,
            source_image,
            details_display,
            review_status_display,
            review_reason_banner,
        ]

        lab_name_filter.change(
            fn=lambda lab_names, latest_only, review_filter, full_df: handle_filter_change(
                lab_names,
                latest_only,
                review_filter,
                full_df,
                output_path,
            ),
            inputs=filter_inputs,
            outputs=filter_outputs,
        )
        latest_filter.change(
            fn=lambda lab_names, latest_only, review_filter, full_df: handle_filter_change(
                lab_names,
                latest_only,
                review_filter,
                full_df,
                output_path,
            ),
            inputs=filter_inputs,
            outputs=filter_outputs,
        )
        review_filter.change(
            fn=lambda lab_names, latest_only, review_filter, full_df: handle_filter_change(
                lab_names,
                latest_only,
                review_filter,
                full_df,
                output_path,
            ),
            inputs=filter_inputs,
            outputs=filter_outputs,
        )

        # Row selection
        data_table.select(
            fn=lambda evt, filtered_df, full_df, lab_name: handle_row_select(
                evt,
                filtered_df,
                full_df,
                lab_name,
                output_path,
            ),
            inputs=[filtered_df_state, full_df_state, lab_name_filter],
            outputs=[
                plot_display,
                current_idx_state,
                position_display,
                source_image,
                details_display,
                review_status_display,
                review_reason_banner,
            ],
        )

        # Navigation buttons
        nav_inputs = [
            current_idx_state,
            filtered_df_state,
            full_df_state,
            lab_name_filter,
        ]
        nav_outputs = [
            plot_display,
            current_idx_state,
            position_display,
            source_image,
            details_display,
            review_status_display,
            review_reason_banner,
        ]

        prev_btn.click(
            fn=lambda current_idx, filtered_df, full_df, lab_name: handle_navigation(
                current_idx,
                filtered_df,
                full_df,
                lab_name,
                -1,
                output_path,
            ),
            inputs=nav_inputs,
            outputs=nav_outputs,
        )
        next_btn.click(
            fn=lambda current_idx, filtered_df, full_df, lab_name: handle_navigation(
                current_idx,
                filtered_df,
                full_df,
                lab_name,
                1,
                output_path,
            ),
            inputs=nav_inputs,
            outputs=nav_outputs,
        )

        # Review action buttons
        review_btn_inputs = [
            current_idx_state,
            filtered_df_state,
            full_df_state,
            lab_name_filter,
            latest_filter,
            review_filter,
        ]
        review_outputs = [
            full_df_state,
            filtered_df_state,
            data_table,
            plot_display,
            current_idx_state,
            position_display,
            source_image,
            details_display,
            review_status_display,
            summary_display,
            review_reason_banner,
        ]

        accept_btn.click(
            fn=lambda current_idx, filtered_df, full_df, lab_name, latest_only, review_filter: handle_review_action(
                current_idx,
                filtered_df,
                full_df,
                lab_name,
                latest_only,
                review_filter,
                "accepted",
                output_path,
            ),
            inputs=review_btn_inputs,
            outputs=review_outputs,
        )
        reject_btn.click(
            fn=lambda current_idx, filtered_df, full_df, lab_name, latest_only, review_filter: handle_review_action(
                current_idx,
                filtered_df,
                full_df,
                lab_name,
                latest_only,
                review_filter,
                "rejected",
                output_path,
            ),
            inputs=review_btn_inputs,
            outputs=review_outputs,
        )

        # Export
        export_btn.click(
            fn=lambda filtered_df: export_csv(filtered_df, output_path),
            inputs=[filtered_df_state],
            outputs=[export_file],
        ).then(fn=lambda: gr.update(visible=True), outputs=[export_file])

    return demo
