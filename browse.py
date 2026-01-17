"""
Lab Results Browser UI

Gradio app for browsing and analyzing extracted lab results.
Provides Excel-like filtering/sorting with interactive Plotly plots.

Run with: python results_browser.py
Or for auto-reload: gradio results_browser.py
"""

import os
import sys
import argparse
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from config import ProfileConfig, Demographics, LabSpecsConfig

# Load .env from repo root
load_dotenv(Path(__file__).parent / '.env')

# =============================================================================
# Keyboard Shortcuts (JavaScript)
# =============================================================================

KEYBOARD_JS = r"""
<script>
(function() {
    // Keyboard navigation only
    document.addEventListener('keydown', function(event) {
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
            return;
        }
        switch(event.key) {
            case 'ArrowRight':
            case 'j':
                document.querySelector('#next-btn')?.click();
                event.preventDefault();
                break;
            case 'ArrowLeft':
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
# Configuration
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


def set_current_profile(name: str) -> None:
    """Set the current profile name."""
    global _current_profile_name
    _current_profile_name = name


def get_current_profile() -> Optional[str]:
    """Get the current profile name."""
    return _current_profile_name


def load_profile(profile_name: str) -> Optional[ProfileConfig]:
    """Load a profile by name and update global configuration.

    Returns the loaded ProfileConfig or None if not found.
    """
    # Find profile file
    profile_path = None
    for ext in ('.yaml', '.yml', '.json'):
        p = Path(f"profiles/{profile_name}{ext}")
        if p.exists():
            profile_path = p
            break

    if not profile_path:
        return None

    profile = ProfileConfig.from_file(profile_path)

    # Update global state
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
    # Filter out template profiles
    return [p for p in profiles if not p.startswith('_')]


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


# Display columns for the data table (in order) - standardized values only
DISPLAY_COLUMNS = [
    'date',
    'lab_name',
    'value',
    'unit',
    'reference_range',
    'is_out_of_reference',
]

# Column display names (human-readable, kept short to match content width)
COLUMN_LABELS = {
    'date': 'Date',
    'lab_name': 'Lab',
    'value': 'Value',
    'unit': 'Unit',
    'reference_range': 'Ref',
    'is_out_of_reference': 'Abn',
}


def get_output_path() -> Path:
    """Get output path from configuration, profile, or environment."""
    global _configured_output_path
    if _configured_output_path:
        return _configured_output_path
    return Path(os.getenv('OUTPUT_PATH', './output'))


def get_image_path(entry: dict, output_path: Path) -> Optional[str]:
    """Get page image path from source_file and page_number."""
    source_file = entry.get('source_file', '')
    page_number = entry.get('page_number')

    if not source_file:
        return None

    # source_file = "2001-12-27 - analises.csv" -> stem = "2001-12-27 - analises"
    stem = source_file.rsplit('.', 1)[0] if '.' in source_file else source_file

    # page_number = 1 -> "001"
    if page_number is not None and pd.notna(page_number):
        page_str = f"{int(page_number):03d}"
    else:
        page_str = "001"  # fallback

    # Image path: output_path / stem / stem.001.jpg
    image_path = output_path / stem / f"{stem}.{page_str}.jpg"

    if image_path.exists():
        return str(image_path)
    return None


# =============================================================================
# Data Loading
# =============================================================================

def load_data(output_path: Path) -> pd.DataFrame:
    """Load lab results from all.csv."""
    csv_path = output_path / "all.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Compute reference_range from reference_min and reference_max
    if 'reference_min' in df.columns and 'reference_max' in df.columns:
        def format_range(row):
            ref_min = row['reference_min']
            ref_max = row['reference_max']
            if pd.isna(ref_min) and pd.isna(ref_max):
                # Boolean values default to 0 - 1 range
                if row.get('unit') == 'boolean':
                    return '0 - 1'
                return ''
            if pd.isna(ref_min):
                return f'< {ref_max}'
            if pd.isna(ref_max):
                return f'> {ref_min}'
            return f'{ref_min} - {ref_max}'
        df['reference_range'] = df.apply(format_range, axis=1)

    # Compute is_out_of_reference by checking if value is outside range
    if 'value' in df.columns and 'reference_min' in df.columns and 'reference_max' in df.columns:
        def check_out_of_range(row):
            val = row['value']
            ref_min = row['reference_min']
            ref_max = row['reference_max']
            if pd.isna(val):
                return None
            if pd.isna(ref_min) and pd.isna(ref_max):
                # Boolean values: value > 0 is abnormal (positive), value == 0 is normal
                if row.get('unit') == 'boolean':
                    return val > 0
                return None  # No reference range to compare against
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
            """Get lab_specs range for a lab name."""
            range_min, range_max = lab_specs.get_healthy_range_for_demographics(
                lab_name, gender=gender, age=age
            )
            return pd.Series({'lab_specs_min': range_min, 'lab_specs_max': range_max})

        range_df = df['lab_name'].apply(get_lab_spec_range)
        df['lab_specs_min'] = range_df['lab_specs_min']
        df['lab_specs_max'] = range_df['lab_specs_max']

    return df


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
            date_range = f" | Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"

    # Abnormal count
    abnormal_count = 0
    if 'is_out_of_reference' in df.columns:
        abnormal_count = df['is_out_of_reference'].sum()

    return f"**{total:,} results** | {unique_tests} unique tests | {abnormal_count} abnormal{date_range}"


# =============================================================================
# Filtering
# =============================================================================

def apply_filters(
    df: pd.DataFrame,
    lab_names: Optional[list],
    abnormal_only: bool,
    latest_only: bool = False
) -> pd.DataFrame:
    """Apply filters to DataFrame and sort by date descending."""
    if df.empty:
        return df

    filtered = df.copy()

    # Filter by lab names (multi-selection)
    if lab_names:
        filtered = filtered[filtered['lab_name'].isin(lab_names)]

    # Filter abnormal only
    if abnormal_only and 'is_out_of_reference' in filtered.columns:
        filtered = filtered[filtered['is_out_of_reference'] == True]

    # Sort by date descending (most recent first), then by lab_name ascending (A-Z)
    if 'date' in filtered.columns:
        filtered = filtered.sort_values(['date', 'lab_name'], ascending=[False, True], na_position='last')

    # Latest only: keep only the most recent value per lab test
    if latest_only and 'lab_name' in filtered.columns and 'date' in filtered.columns:
        # Already sorted by date desc, so first occurrence per lab is the latest
        filtered = filtered.drop_duplicates(subset=['lab_name'], keep='first')

    # Reset index so iloc positions match displayed row positions
    filtered = filtered.reset_index(drop=True)

    return filtered


def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for display (subset and format columns).

    Note: Does NOT sort - sorting is done in apply_filters() to ensure
    the displayed row order matches filtered_df_state for correct row selection.
    """
    if df.empty:
        return pd.DataFrame(columns=[COLUMN_LABELS.get(c, c) for c in DISPLAY_COLUMNS])

    # Select and order columns (no sorting - already sorted by apply_filters)
    display_df = df[[col for col in DISPLAY_COLUMNS if col in df.columns]].copy()

    # Format date column
    if 'date' in display_df.columns:
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')

    # Format boolean columns
    if 'is_out_of_reference' in display_df.columns:
        display_df['is_out_of_reference'] = display_df['is_out_of_reference'].map(
            {True: 'Yes', False: 'No', None: ''}
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

def create_single_lab_plot(df: pd.DataFrame, lab_name: str) -> tuple[go.Figure, str]:
    """Generate a single plot for one lab test. Returns (figure, unit)."""
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

    # Ensure date is datetime and sort
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

    # Get unit
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

    # Track which ranges we have for legend
    has_lab_specs_range = False
    has_pdf_range = False

    # Add lab_specs healthy range (blue band - from lab_specs.json)
    if 'lab_specs_min' in lab_df.columns and 'lab_specs_max' in lab_df.columns:
        min_vals = lab_df['lab_specs_min'].dropna()
        max_vals = lab_df['lab_specs_max'].dropna()

        if not min_vals.empty and not max_vals.empty:
            spec_min = float(min_vals.iloc[0])
            spec_max = float(max_vals.iloc[0])
            has_lab_specs_range = True

            # Blue band for lab_specs healthy range
            fig.add_hrect(
                y0=spec_min, y1=spec_max,
                fillcolor="rgba(37, 99, 235, 0.20)",
                line_width=0,
            )
            fig.add_hline(y=spec_min, line_dash="dot", line_color="rgba(37, 99, 235, 0.8)")
            fig.add_hline(y=spec_max, line_dash="dot", line_color="rgba(37, 99, 235, 0.8)")

    # Add PDF reference range (orange band - from extracted data)
    if 'reference_min' in lab_df.columns and 'reference_max' in lab_df.columns:
        min_vals = lab_df['reference_min'].dropna()
        max_vals = lab_df['reference_max'].dropna()

        ref_min = None
        ref_max = None
        if not min_vals.empty:
            ref_min = float(min_vals.mode().iloc[0]) if len(min_vals.mode()) > 0 else float(min_vals.iloc[0])
        if not max_vals.empty:
            ref_max = float(max_vals.mode().iloc[0]) if len(max_vals.mode()) > 0 else float(max_vals.iloc[0])

        if ref_min is not None or ref_max is not None:
            has_pdf_range = True

            if ref_min is not None and ref_max is not None:
                # Both bounds: draw band
                fig.add_hrect(
                    y0=ref_min, y1=ref_max,
                    fillcolor="rgba(245, 158, 11, 0.20)",
                    line_width=0,
                )
                fig.add_hline(y=ref_min, line_dash="dash", line_color="rgba(245, 158, 11, 0.8)")
                fig.add_hline(y=ref_max, line_dash="dash", line_color="rgba(245, 158, 11, 0.8)")
            elif ref_max is not None:
                # Only upper bound (< ref_max): healthy zone is below ref_max
                # Calculate y-axis extent based on data
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
                # Only lower bound (> ref_min): healthy zone extends from ref_min to top
                # Calculate y-axis extent based on data
                data_max = lab_df['value'].max()
                y_top = max(data_max * 1.2, ref_min * 2) if data_max > 0 else data_max * 0.8
                fig.add_hrect(
                    y0=ref_min, y1=y_top,
                    fillcolor="rgba(245, 158, 11, 0.20)",
                    line_width=0,
                )
                fig.add_hline(y=ref_min, line_dash="dash", line_color="rgba(245, 158, 11, 0.8)",
                              annotation_text=f"> {ref_min}", annotation_position="bottom right")

    # Add legend entries for range types
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

    # Configure x-axis for dates (show year only)
    xaxis_config = dict(
        title="Date",
        tickformat='%Y',
    )

    # For single data point, show just that year without surrounding timestamps
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


def create_interactive_plot(df: pd.DataFrame, lab_names: Optional[list]) -> go.Figure:
    """Generate interactive Plotly plot(s) for selected lab tests.

    When multiple labs are selected, creates vertically stacked subplots.
    """
    # Handle empty selection
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

    # Single lab - simple case
    if len(lab_names) == 1:
        fig, _ = create_single_lab_plot(df, lab_names[0])
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

        # Get unit
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
            elif ref_max is not None:
                # Only upper bound (< ref_max): healthy zone is below ref_max
                data_min = lab_df['value'].min()
                y_bottom = min(0, data_min * 0.9) if data_min > 0 else data_min * 1.1
                fig.add_hrect(
                    y0=y_bottom, y1=ref_max,
                    fillcolor="rgba(245, 158, 11, 0.20)",
                    line_width=0,
                    row=i + 1, col=1
                )
                fig.add_hline(y=ref_max, line_dash="dash", line_color="rgba(245, 158, 11, 0.8)",
                              row=i + 1, col=1)
            elif ref_min is not None:
                # Only lower bound (> ref_min): healthy zone extends from ref_min to top
                data_max = lab_df['value'].max()
                y_top = max(data_max * 1.2, ref_min * 2) if data_max > 0 else data_max * 0.8
                fig.add_hrect(
                    y0=ref_min, y1=y_top,
                    fillcolor="rgba(245, 158, 11, 0.20)",
                    line_width=0,
                    row=i + 1, col=1
                )
                fig.add_hline(y=ref_min, line_dash="dash", line_color="rgba(245, 158, 11, 0.8)",
                              row=i + 1, col=1)

        # Update y-axis label
        fig.update_yaxes(title_text=unit if unit else "Value", row=i + 1, col=1)

    # Layout
    height_per_chart = 250
    fig.update_layout(
        template='plotly_white',
        height=height_per_chart * n_labs,
        margin=dict(l=60, r=20, t=40, b=40),
        showlegend=False,
        hovermode='x unified'
    )

    # Configure x-axis to show year only
    fig.update_xaxes(tickformat='%Y')

    return fig


# =============================================================================
# Event Handlers
# =============================================================================

def handle_filter_change(
    lab_names: Optional[list],
    abnormal_only: bool,
    latest_only: bool,
    full_df: pd.DataFrame
):
    """Handle filter changes and update display."""
    filtered_df = apply_filters(full_df, lab_names, abnormal_only, latest_only)
    display_df = prepare_display_df(filtered_df)
    summary = get_summary_stats(filtered_df)

    # Reset to first row
    current_idx = 0
    position_text = "No results"
    image_path = None

    # Determine which labs to plot
    if lab_names:
        # Use filtered labs for plot
        plot_labs = lab_names
    elif not filtered_df.empty:
        # No filter - show first row's lab
        plot_labs = [filtered_df.iloc[0].get('lab_name')]
    else:
        plot_labs = []

    if not filtered_df.empty:
        first_row = filtered_df.iloc[0]
        position_text = f"**Row 1 of {len(filtered_df)}**"
        image_path = get_image_path(first_row.to_dict(), get_output_path())

    plot = create_interactive_plot(full_df, plot_labs)

    return display_df, summary, plot, filtered_df, current_idx, position_text, image_path


def handle_row_select(
    evt: gr.SelectData,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    lab_names: Optional[list]
):
    """Handle row selection to update plot and current index."""
    if evt is None or filtered_df.empty:
        return create_interactive_plot(full_df, []), 0, "No results", None

    # Get the selected row index - evt.index is (row, col) tuple
    # Use the row component which is the visual position in the displayed table
    if isinstance(evt.index, (list, tuple)):
        row_idx = evt.index[0]
    else:
        row_idx = evt.index

    # Ensure row_idx is within bounds of the filtered DataFrame
    if row_idx < 0 or row_idx >= len(filtered_df):
        row_idx = 0

    # Get row data using iloc (positional index) which matches displayed order
    row = filtered_df.iloc[row_idx]
    position_text = f"**Row {row_idx + 1} of {len(filtered_df)}**"
    image_path = get_image_path(row.to_dict(), get_output_path())

    # Determine which labs to plot
    if lab_names:
        plot_labs = lab_names
    else:
        plot_labs = [row.get('lab_name')]

    return create_interactive_plot(full_df, plot_labs), row_idx, position_text, image_path


def handle_previous(
    current_idx: int,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    lab_names: Optional[list]
):
    """Navigate to previous row."""
    if filtered_df.empty:
        return create_interactive_plot(full_df, []), 0, "No results", None

    # Move to previous row (with wrap-around)
    new_idx = current_idx - 1
    if new_idx < 0:
        new_idx = len(filtered_df) - 1  # Wrap to last

    row = filtered_df.iloc[new_idx]
    position_text = f"**Row {new_idx + 1} of {len(filtered_df)}**"
    image_path = get_image_path(row.to_dict(), get_output_path())

    # Determine which labs to plot
    if lab_names:
        plot_labs = lab_names
    else:
        plot_labs = [row.get('lab_name')]

    return create_interactive_plot(full_df, plot_labs), new_idx, position_text, image_path


def handle_next(
    current_idx: int,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame,
    lab_names: Optional[list]
):
    """Navigate to next row."""
    if filtered_df.empty:
        return create_interactive_plot(full_df, []), 0, "No results", None

    # Move to next row (with wrap-around)
    new_idx = current_idx + 1
    if new_idx >= len(filtered_df):
        new_idx = 0  # Wrap to first

    row = filtered_df.iloc[new_idx]
    position_text = f"**Row {new_idx + 1} of {len(filtered_df)}**"
    image_path = get_image_path(row.to_dict(), get_output_path())

    # Determine which labs to plot
    if lab_names:
        plot_labs = lab_names
    else:
        plot_labs = [row.get('lab_name')]

    return create_interactive_plot(full_df, plot_labs), new_idx, position_text, image_path


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
            "No profile selected",  # summary
            go.Figure(),  # plot
            pd.DataFrame(),  # full_df
            pd.DataFrame(),  # filtered_df
            0,  # current_idx
            "No results",  # position_text
            None,  # image
            [],  # lab_name_choices
        )

    profile = load_profile(profile_name)
    if not profile or not profile.output_path:
        return (
            pd.DataFrame(),
            f"Profile '{profile_name}' not found or has no output path",
            go.Figure(),
            pd.DataFrame(),
            pd.DataFrame(),
            0,
            "No results",
            None,
            [],
        )

    # Load fresh data for the new profile
    output_path = get_output_path()
    full_df = load_data(output_path)

    # Sort initial data by date descending, then by lab_name ascending (A-Z)
    if not full_df.empty and 'date' in full_df.columns:
        full_df = full_df.sort_values(['date', 'lab_name'], ascending=[False, True], na_position='last').reset_index(drop=True)

    # Get lab name choices
    lab_name_choices = []
    if not full_df.empty and 'lab_name' in full_df.columns:
        lab_name_choices = sorted(full_df['lab_name'].dropna().unique().tolist())

    # Prepare display
    display_df = prepare_display_df(full_df)
    summary = get_summary_stats(full_df)

    # Position and image
    position_text = f"**Row 1 of {len(full_df)}**" if not full_df.empty else "No results"
    image_path = get_image_path(full_df.iloc[0].to_dict(), output_path) if not full_df.empty else None

    # Empty plot (no labs selected initially)
    plot = create_interactive_plot(full_df, [])

    return (
        display_df,
        summary,
        plot,
        full_df,
        full_df,  # filtered_df starts as full_df
        0,
        position_text,
        image_path,
        gr.update(choices=lab_name_choices, value=[]),  # Update dropdown choices and clear selection
    )


# =============================================================================
# Main App
# =============================================================================

def create_app():
    """Create and configure the Gradio app."""
    output_path = get_output_path()
    full_df = load_data(output_path)

    # Sort initial data by date descending, then by lab_name ascending (A-Z), and reset index
    if not full_df.empty and 'date' in full_df.columns:
        full_df = full_df.sort_values(['date', 'lab_name'], ascending=[False, True], na_position='last').reset_index(drop=True)

    # Get unique lab names for dropdown
    lab_name_choices = []
    if not full_df.empty and 'lab_name' in full_df.columns:
        lab_name_choices = sorted(full_df['lab_name'].dropna().unique().tolist())

    # Initial position text
    initial_position = f"**Row 1 of {len(full_df)}**" if not full_df.empty else "No results"

    # Initial plot (empty - no labs selected) and image
    initial_labs = []  # Start with no filter applied
    initial_image = get_image_path(full_df.iloc[0].to_dict(), output_path) if not full_df.empty else None

    # Get available profiles for dropdown
    available_profiles = get_available_profiles()
    current_profile = get_current_profile()

    with gr.Blocks(title="Lab Results Browser") as demo:

        # State variables
        full_df_state = gr.State(value=full_df)
        filtered_df_state = gr.State(value=full_df)
        current_idx_state = gr.State(value=0)

        # Header with profile selector
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("# Lab Results Browser")
            with gr.Column(scale=1):
                profile_selector = gr.Dropdown(
                    choices=available_profiles,
                    value=current_profile,
                    label="Profile",
                    info="Switch between profiles",
                    allow_custom_value=False,
                    interactive=True
                )
        gr.Markdown("Browse and analyze extracted lab results with interactive filtering and plots.")

        # Filters Row
        with gr.Row():
            with gr.Column(scale=3):
                lab_name_filter = gr.Dropdown(
                    choices=lab_name_choices,
                    multiselect=True,
                    value=[],
                    label="Lab Names",
                    info="Filter by lab tests (select multiple or none)",
                    allow_custom_value=False
                )
            with gr.Column(scale=1):
                gr.Markdown("**Filters**")
                abnormal_filter = gr.Checkbox(
                    label="Abnormal Only",
                    value=False
                )
                latest_filter = gr.Checkbox(
                    label="Latest Only",
                    value=False
                )

        # Summary statistics
        summary_display = gr.Markdown(get_summary_stats(full_df))

        gr.Markdown("---")

        # Main content: Table + Plot side by side
        with gr.Row():
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

            with gr.Column(scale=2):
                # Navigation controls
                with gr.Row():
                    prev_btn = gr.Button("< Previous [←]", elem_id="prev-btn", size="sm")
                    position_display = gr.Markdown(initial_position, elem_id="position-display")
                    next_btn = gr.Button("Next [→] >", elem_id="next-btn", size="sm")

                # Tabs for Plot and Source Image
                with gr.Tabs():
                    with gr.TabItem("Time Series Plot"):
                        plot_display = gr.Plot(
                            value=create_interactive_plot(full_df, initial_labs),
                            label=""
                        )
                    with gr.TabItem("Source Page"):
                        source_image = gr.Image(
                            value=initial_image,
                            label="Source Document Page",
                            type="filepath",
                            show_label=False,
                            height=500
                        )

        gr.Markdown("---")
        gr.Markdown("*Keyboard: ← / k = Previous, → / j = Next*", elem_id="footer")

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
                lab_name_filter,  # Update choices when profile changes
            ]
        )

        # Wire up filter events
        filter_inputs = [lab_name_filter, abnormal_filter, latest_filter, full_df_state]
        filter_outputs = [data_table, summary_display, plot_display, filtered_df_state, current_idx_state, position_display, source_image]

        lab_name_filter.change(
            fn=handle_filter_change,
            inputs=filter_inputs,
            outputs=filter_outputs
        )

        abnormal_filter.change(
            fn=handle_filter_change,
            inputs=filter_inputs,
            outputs=filter_outputs
        )

        latest_filter.change(
            fn=handle_filter_change,
            inputs=filter_inputs,
            outputs=filter_outputs
        )

        # Wire up row selection
        data_table.select(
            fn=handle_row_select,
            inputs=[filtered_df_state, full_df_state, lab_name_filter],
            outputs=[plot_display, current_idx_state, position_display, source_image]
        )

        # Wire up navigation buttons
        nav_inputs = [current_idx_state, filtered_df_state, full_df_state, lab_name_filter]
        nav_outputs = [plot_display, current_idx_state, position_display, source_image]

        prev_btn.click(
            fn=handle_previous,
            inputs=nav_inputs,
            outputs=nav_outputs
        )

        next_btn.click(
            fn=handle_next,
            inputs=nav_inputs,
            outputs=nav_outputs
        )

        # Wire up export
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
        description='Lab Results Browser UI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python browse.py                     # Uses first available profile
  python browse.py --profile tiago     # Uses specific profile
  python browse.py --list-profiles     # List available profiles
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
        # Default to first available profile
        available = get_available_profiles()
        if not available:
            print("Error: No profiles found.")
            print("Create profiles in the 'profiles/' directory.")
            sys.exit(1)
        profile_name = available[0]
        print(f"No profile specified, defaulting to: {profile_name}")

    # Load profile using the helper function
    profile = load_profile(profile_name)
    if not profile:
        print(f"Error: Profile '{profile_name}' not found")
        print("Use --list-profiles to see available profiles.")
        sys.exit(1)

    print(f"Using profile: {profile.name}")

    # Print demographics info if available
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

    # Build list of allowed paths for serving files from all profiles
    # This enables profile switching without restart
    available = get_available_profiles()
    print(f"Available profiles: {', '.join(available)}")

    allowed_paths = set()
    for pname in available:
        p = load_profile(pname)
        if p and p.output_path:
            allowed_paths.add(str(p.output_path))
            if p.output_path.parent != p.output_path:
                allowed_paths.add(str(p.output_path.parent))

    # Reload the original profile before creating app
    load_profile(profile_name)
    allowed_paths = list(allowed_paths)

    demo = create_app()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from review_ui
        show_error=True,
        allowed_paths=allowed_paths,
        head=KEYBOARD_JS,  # Add keyboard shortcuts
    )
