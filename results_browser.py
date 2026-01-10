"""
Lab Results Browser UI

Gradio app for browsing and analyzing extracted lab results.
Provides Excel-like filtering/sorting with interactive Plotly plots.

Run with: python results_browser.py
Or for auto-reload: gradio results_browser.py
"""

import os
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env from repo root
load_dotenv(Path(__file__).parent / '.env')

# =============================================================================
# Keyboard Shortcuts (JavaScript)
# =============================================================================

KEYBOARD_JS = r"""
<script>
document.addEventListener('keydown', function(event) {
    // Skip if user is typing in an input field
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

// Function to highlight and scroll to a row
function highlightRow(rowIndex) {
    // Find all table rows in the data table (skip header)
    const table = document.querySelector('table.table');
    if (!table) return;

    const rows = table.querySelectorAll('tbody tr');

    // Remove existing highlights
    rows.forEach(row => row.classList.remove('highlighted-row'));

    // Add highlight to selected row
    if (rowIndex >= 0 && rowIndex < rows.length) {
        const targetRow = rows[rowIndex];
        targetRow.classList.add('highlighted-row');

        // Scroll row into view
        targetRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

// Watch for changes to the position display
const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if (mutation.target.id === 'position-display' ||
            mutation.target.closest('#position-display')) {
            const text = mutation.target.textContent || '';
            const match = text.match(/Row\s+(\d+)\s+of/);
            if (match) {
                const rowNum = parseInt(match[1], 10) - 1; // 0-indexed
                highlightRow(rowNum);
            }
        }
    });
});

// Start observing once DOM is ready
function startObserver() {
    const posDisplay = document.querySelector('#position-display');
    if (posDisplay) {
        observer.observe(posDisplay, {
            childList: true,
            subtree: true,
            characterData: true
        });
        // Initial highlight
        const text = posDisplay.textContent || '';
        const match = text.match(/Row\s+(\d+)\s+of/);
        if (match) {
            highlightRow(parseInt(match[1], 10) - 1);
        }
    } else {
        // Retry if not found yet
        setTimeout(startObserver, 500);
    }
}

// Wait for page load then start observer
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startObserver);
} else {
    setTimeout(startObserver, 500);
}
</script>

<style>
.highlighted-row {
    background-color: #3b82f6 !important;
    color: white !important;
}
.highlighted-row td {
    background-color: #3b82f6 !important;
    color: white !important;
}
</style>
"""

# =============================================================================
# Configuration
# =============================================================================

# Display columns for the data table (in order) - standardized values only
DISPLAY_COLUMNS = [
    'date',
    'lab_name_standardized',
    'value_primary',
    'lab_unit_primary',
    'reference_range',
    'is_out_of_reference',
]

# Column display names (human-readable)
COLUMN_LABELS = {
    'date': 'Date',
    'lab_name_standardized': 'Lab Name',
    'value_primary': 'Value',
    'lab_unit_primary': 'Unit',
    'reference_range': 'Reference Range',
    'is_out_of_reference': 'Abnormal',
}


def get_output_path() -> Path:
    """Get output path from environment."""
    return Path(os.getenv('OUTPUT_PATH', './output'))


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

    return df


def get_summary_stats(df: pd.DataFrame) -> str:
    """Generate summary statistics markdown."""
    if df.empty:
        return "No data loaded"

    total = len(df)
    unique_tests = df['lab_name_standardized'].nunique() if 'lab_name_standardized' in df.columns else 0

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
    lab_names: list,
    abnormal_only: bool,
    search_text: str
) -> pd.DataFrame:
    """Apply filters to DataFrame and sort by date descending."""
    if df.empty:
        return df

    filtered = df.copy()

    # Filter by lab name(s)
    if lab_names and len(lab_names) > 0:
        filtered = filtered[filtered['lab_name_standardized'].isin(lab_names)]

    # Filter abnormal only
    if abnormal_only and 'is_out_of_reference' in filtered.columns:
        filtered = filtered[filtered['is_out_of_reference'] == True]

    # Text search (case-insensitive across all string columns)
    if search_text and search_text.strip():
        search_lower = search_text.lower().strip()
        mask = pd.Series([False] * len(filtered), index=filtered.index)

        for col in filtered.columns:
            if filtered[col].dtype == 'object':
                col_mask = filtered[col].astype(str).str.lower().str.contains(search_lower, na=False)
                mask = mask | col_mask

        filtered = filtered[mask]

    # Sort by date descending (most recent first)
    if 'date' in filtered.columns:
        filtered = filtered.sort_values('date', ascending=False, na_position='last')

    return filtered


def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for display (subset and format columns)."""
    if df.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)

    # Sort by date descending (most recent first)
    sorted_df = df.copy()
    if 'date' in sorted_df.columns:
        sorted_df = sorted_df.sort_values('date', ascending=False, na_position='last')

    # Select and order columns
    display_df = sorted_df[[col for col in DISPLAY_COLUMNS if col in sorted_df.columns]].copy()

    # Format date column
    if 'date' in display_df.columns:
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')

    # Format boolean columns
    if 'is_out_of_reference' in display_df.columns:
        display_df['is_out_of_reference'] = display_df['is_out_of_reference'].map(
            {True: 'Yes', False: 'No', None: ''}
        )

    # Round numeric columns
    if 'value_primary' in display_df.columns:
        display_df['value_primary'] = display_df['value_primary'].round(2)

    return display_df


# =============================================================================
# Plotting
# =============================================================================

def create_interactive_plot(df: pd.DataFrame, lab_name: Optional[str]) -> go.Figure:
    """Generate interactive Plotly plot for a specific lab test."""
    if not lab_name or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Select a lab test to view its time series",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        return fig

    # Filter for selected lab
    lab_df = df[df['lab_name_standardized'] == lab_name].copy()

    if lab_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data for {lab_name}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(template='plotly_white', height=400)
        return fig

    # Ensure date is datetime and sort
    lab_df['date'] = pd.to_datetime(lab_df['date'], errors='coerce')
    lab_df = lab_df.dropna(subset=['date', 'value_primary'])
    lab_df = lab_df.sort_values('date')

    if lab_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No valid data points for {lab_name}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(template='plotly_white', height=400)
        return fig

    # Get unit
    unit = ""
    if 'lab_unit_primary' in lab_df.columns:
        units = lab_df['lab_unit_primary'].dropna()
        if not units.empty:
            unit = str(units.iloc[0])

    fig = go.Figure()

    # Add data trace
    fig.add_trace(go.Scatter(
        x=lab_df['date'],
        y=lab_df['value_primary'],
        mode='lines+markers',
        name=lab_name,
        marker=dict(size=10, color='#1f77b4'),
        line=dict(width=2),
        hovertemplate=(
            '<b>Date:</b> %{x|%Y-%m-%d}<br>'
            f'<b>Value:</b> %{{y:.2f}} {unit}<br>'
            '<extra></extra>'
        )
    ))

    # Add reference range bands if available
    if 'healthy_range_min' in lab_df.columns and 'healthy_range_max' in lab_df.columns:
        min_vals = lab_df['healthy_range_min'].dropna()
        max_vals = lab_df['healthy_range_max'].dropna()

        if not min_vals.empty and not max_vals.empty:
            # Use mode (most common value) for reference range
            ref_min = float(min_vals.mode().iloc[0]) if len(min_vals.mode()) > 0 else float(min_vals.iloc[0])
            ref_max = float(max_vals.mode().iloc[0]) if len(max_vals.mode()) > 0 else float(max_vals.iloc[0])

            # Green band for healthy range
            fig.add_hrect(
                y0=ref_min, y1=ref_max,
                fillcolor="rgba(75, 192, 75, 0.15)",
                line_width=0,
                annotation_text="Healthy Range",
                annotation_position="top left",
                annotation=dict(font_size=10, font_color="green")
            )

            # Add reference lines
            fig.add_hline(
                y=ref_min,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Min: {ref_min:.1f}",
                annotation_position="bottom right"
            )
            fig.add_hline(
                y=ref_max,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Max: {ref_max:.1f}",
                annotation_position="top right"
            )

    # Layout
    fig.update_layout(
        title=dict(
            text=f"{lab_name}" + (f" [{unit}]" if unit else ""),
            font=dict(size=16)
        ),
        xaxis_title="Date",
        yaxis_title=f"Value ({unit})" if unit else "Value",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        margin=dict(l=60, r=20, t=60, b=60),
        showlegend=False
    )

    return fig


# =============================================================================
# Event Handlers
# =============================================================================

def handle_filter_change(
    lab_names: list,
    abnormal_only: bool,
    search_text: str,
    full_df: pd.DataFrame
):
    """Handle filter changes and update display."""
    filtered_df = apply_filters(full_df, lab_names, abnormal_only, search_text)
    display_df = prepare_display_df(filtered_df)
    summary = get_summary_stats(filtered_df)

    # Reset to first row, get its lab for plot
    current_idx = 0
    selected_lab = None
    position_text = "No results"

    if not filtered_df.empty:
        selected_lab = filtered_df.iloc[0].get('lab_name_standardized')
        position_text = f"**Row 1 of {len(filtered_df)}**"

    plot = create_interactive_plot(full_df, selected_lab)

    return display_df, summary, plot, filtered_df, current_idx, position_text


def handle_row_select(
    evt: gr.SelectData,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame
):
    """Handle row selection to update plot and current index."""
    if evt is None or filtered_df.empty:
        return create_interactive_plot(full_df, None), 0, "No results"

    # Get the selected row index
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index

    if row_idx >= len(filtered_df):
        return create_interactive_plot(full_df, None), 0, "No results"

    # Get lab name from filtered DataFrame
    lab_name = filtered_df.iloc[row_idx].get('lab_name_standardized')
    position_text = f"**Row {row_idx + 1} of {len(filtered_df)}**"

    return create_interactive_plot(full_df, lab_name), row_idx, position_text


def handle_previous(
    current_idx: int,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame
):
    """Navigate to previous row."""
    if filtered_df.empty:
        return create_interactive_plot(full_df, None), 0, "No results"

    # Move to previous row (with wrap-around)
    new_idx = current_idx - 1
    if new_idx < 0:
        new_idx = len(filtered_df) - 1  # Wrap to last

    lab_name = filtered_df.iloc[new_idx].get('lab_name_standardized')
    position_text = f"**Row {new_idx + 1} of {len(filtered_df)}**"

    return create_interactive_plot(full_df, lab_name), new_idx, position_text


def handle_next(
    current_idx: int,
    filtered_df: pd.DataFrame,
    full_df: pd.DataFrame
):
    """Navigate to next row."""
    if filtered_df.empty:
        return create_interactive_plot(full_df, None), 0, "No results"

    # Move to next row (with wrap-around)
    new_idx = current_idx + 1
    if new_idx >= len(filtered_df):
        new_idx = 0  # Wrap to first

    lab_name = filtered_df.iloc[new_idx].get('lab_name_standardized')
    position_text = f"**Row {new_idx + 1} of {len(filtered_df)}**"

    return create_interactive_plot(full_df, lab_name), new_idx, position_text


def export_csv(filtered_df: pd.DataFrame):
    """Export filtered data to CSV file."""
    if filtered_df.empty:
        return None

    output_path = get_output_path()
    export_path = output_path / "filtered_export.csv"
    filtered_df.to_csv(export_path, index=False)
    return str(export_path)


# =============================================================================
# Main App
# =============================================================================

def create_app():
    """Create and configure the Gradio app."""
    output_path = get_output_path()
    full_df = load_data(output_path)

    # Sort initial data by date descending
    if not full_df.empty and 'date' in full_df.columns:
        full_df = full_df.sort_values('date', ascending=False, na_position='last')

    # Get unique lab names for dropdown
    lab_name_choices = []
    if not full_df.empty and 'lab_name_standardized' in full_df.columns:
        lab_name_choices = sorted(full_df['lab_name_standardized'].dropna().unique().tolist())

    # Initial position text
    initial_position = f"**Row 1 of {len(full_df)}**" if not full_df.empty else "No results"

    # Initial plot (first row's lab)
    initial_lab = full_df.iloc[0].get('lab_name_standardized') if not full_df.empty else None

    with gr.Blocks(title="Lab Results Browser") as demo:

        # State variables
        full_df_state = gr.State(value=full_df)
        filtered_df_state = gr.State(value=full_df)
        current_idx_state = gr.State(value=0)

        # Header
        gr.Markdown("# Lab Results Browser")
        gr.Markdown("Browse and analyze extracted lab results with interactive filtering and plots.")

        # Filters Row
        with gr.Row():
            with gr.Column(scale=2):
                lab_name_filter = gr.Dropdown(
                    choices=lab_name_choices,
                    multiselect=True,
                    label="Lab Name",
                    info="Select one or more lab tests to filter",
                    allow_custom_value=False
                )
            with gr.Column(scale=1):
                abnormal_filter = gr.Checkbox(
                    label="Abnormal Only",
                    value=False,
                    info="Show only out-of-reference results"
                )
            with gr.Column(scale=2):
                text_search = gr.Textbox(
                    label="Search",
                    placeholder="Search across all columns...",
                    info="Case-insensitive substring match"
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
                    headers=list(COLUMN_LABELS.values()),
                    interactive=False,
                    wrap=True,
                    max_height=500
                )

                with gr.Row():
                    export_btn = gr.Button("Export Filtered CSV", size="sm")
                    export_file = gr.File(label="Download", visible=False)

            with gr.Column(scale=2):
                gr.Markdown("### Time Series Plot")

                # Navigation controls
                with gr.Row():
                    prev_btn = gr.Button("< Previous [←]", elem_id="prev-btn", size="sm")
                    position_display = gr.Markdown(initial_position, elem_id="position-display")
                    next_btn = gr.Button("Next [→] >", elem_id="next-btn", size="sm")

                plot_display = gr.Plot(
                    value=create_interactive_plot(full_df, initial_lab),
                    label=""
                )

        gr.Markdown("---")
        gr.Markdown("*Keyboard: ← / k = Previous, → / j = Next*", elem_id="footer")

        # Wire up filter events
        filter_inputs = [lab_name_filter, abnormal_filter, text_search, full_df_state]
        filter_outputs = [data_table, summary_display, plot_display, filtered_df_state, current_idx_state, position_display]

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

        text_search.change(
            fn=handle_filter_change,
            inputs=filter_inputs,
            outputs=filter_outputs
        )

        # Wire up row selection
        data_table.select(
            fn=handle_row_select,
            inputs=[filtered_df_state, full_df_state],
            outputs=[plot_display, current_idx_state, position_display]
        )

        # Wire up navigation buttons
        nav_inputs = [current_idx_state, filtered_df_state, full_df_state]
        nav_outputs = [plot_display, current_idx_state, position_display]

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


if __name__ == "__main__":
    demo = create_app()
    output_path = get_output_path()

    # Build list of allowed paths for serving files
    allowed_paths = [str(output_path)]
    if output_path.parent != output_path:
        allowed_paths.append(str(output_path.parent))

    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from review_ui
        show_error=True,
        allowed_paths=allowed_paths,
        head=KEYBOARD_JS,  # Add keyboard shortcuts
    )
