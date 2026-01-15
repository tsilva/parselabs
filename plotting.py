"""Plotting functionality for lab results time series."""

import re
import logging
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)


class LabPlotter:
    """Generate time-series plots for lab results."""

    def __init__(
        self,
        date_col: str = "date",
        value_col: str = "value_primary",
        group_col: str = "lab_name_standardized",
        unit_col: str = "lab_unit_primary"
    ):
        """
        Initialize plotter with column names.

        Args:
            date_col: Column name for dates
            value_col: Column name for values
            group_col: Column name for grouping (lab names)
            unit_col: Column name for units
        """
        self.date_col = date_col
        self.value_col = value_col
        self.group_col = group_col
        self.unit_col = unit_col

    def generate_all_plots(
        self,
        df: pd.DataFrame,
        output_dirs: list[Path],
        max_workers: int = None
    ) -> None:
        """
        Generate plots for all lab tests with sufficient data points.

        Args:
            df: DataFrame with lab results
            output_dirs: List of directories to save plots to
            max_workers: Number of parallel workers (default: cpu_count - 1)
        """
        if df.empty:
            logger.warning("Empty DataFrame, skipping plotting")
            return

        # Validate required columns
        required_cols = [self.date_col, self.value_col, self.group_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns for plotting: {missing_cols}")
            return

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
            df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")

        # Find lab tests with at least 2 valid data points
        unique_labs = df[df[self.value_col].notna()][self.group_col].dropna().unique()

        plot_tasks = []
        for lab_name in unique_labs:
            lab_df = df[
                (df[self.group_col] == lab_name) &
                df[self.date_col].notna() &
                df[self.value_col].notna()
            ]
            if len(lab_df) >= 2:
                plot_tasks.append((lab_name, df, output_dirs))

        if not plot_tasks:
            logger.info("No lab tests with sufficient data for plotting")
            return

        # Determine number of workers
        import os
        if max_workers is None:
            max_workers = max(1, (os.cpu_count() or 1) - 1)

        max_workers = min(max_workers, len(plot_tasks))

        logger.info(f"Plotting {len(plot_tasks)} lab tests using {max_workers} worker(s)")

        # Use multiprocessing for parallel plotting
        with Pool(processes=max_workers) as pool:
            pool.map(self._plot_single_lab, plot_tasks)

        logger.info("Plotting finished")

    def _plot_single_lab(self, args: tuple) -> None:
        """
        Plot a single lab test (designed for multiprocessing).

        Args:
            args: Tuple of (lab_name, df, output_dirs)
        """
        lab_name, df, output_dirs = args

        try:
            # Filter data for this lab
            lab_df = df[df[self.group_col] == lab_name].copy()
            lab_df = lab_df[lab_df[self.date_col].notna() & lab_df[self.value_col].notna()]

            if lab_df.empty or len(lab_df) < 2:
                return

            lab_df = lab_df.sort_values(self.date_col, ascending=True)

            # Get unit for y-axis label
            unit_str = ""
            if self.unit_col in lab_df.columns:
                units = lab_df[self.unit_col].dropna().astype(str).unique()
                unit_str = next((u for u in units if u), "")

            y_label = f"Value ({unit_str})" if unit_str else "Value"
            title = f"{lab_name}" + (f" [{unit_str}]" if unit_str else "")

            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(lab_df[self.date_col], lab_df[self.value_col], marker='o', linestyle='-')

            # Configure x-axis to show first and last year
            ax = plt.gca()
            start_date = lab_df[self.date_col].min()
            end_date = lab_df[self.date_col].max()
            ax.set_xlim(start_date, end_date)

            year_ticks = pd.date_range(start_date, end_date, freq="YS").to_pydatetime().tolist()
            ticks = []
            if start_date not in year_ticks:
                ticks.append(start_date)
            ticks.extend(year_ticks)
            if end_date not in year_ticks:
                ticks.append(end_date)

            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

            # Add reference range bands if available (handles open-ended ranges too)
            if "healthy_range_min" in lab_df.columns or "healthy_range_max" in lab_df.columns:
                self._add_reference_bands(ax, lab_df, start_date, end_date)

            # Add reference range lines (mode)
            if "healthy_range_min" in lab_df.columns and lab_df["healthy_range_min"].notna().any():
                y_min_line = float(lab_df["healthy_range_min"].mode()[0])
                plt.axhline(y=y_min_line, color='gray', linestyle='--', label='Ref Min (mode)')
                cur_ymin, cur_ymax = ax.get_ylim()
                ax.set_ylim(min(cur_ymin, y_min_line), cur_ymax)

            if "healthy_range_max" in lab_df.columns and lab_df["healthy_range_max"].notna().any():
                y_max_line = float(lab_df["healthy_range_max"].mode()[0])
                plt.axhline(y=y_max_line, color='gray', linestyle='--', label='Ref Max (mode)')
                cur_ymin, cur_ymax = ax.get_ylim()
                ax.set_ylim(cur_ymin, max(cur_ymax, y_max_line))

            # Show legend if reference ranges are present
            if ("healthy_range_min" in lab_df.columns and lab_df["healthy_range_min"].notna().any()) or \
               ("healthy_range_max" in lab_df.columns and lab_df["healthy_range_max"].notna().any()):
                plt.legend()

            # Formatting
            plt.title(title, fontsize=16)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save to all output directories
            safe_filename = self._sanitize_filename(lab_name)
            for output_dir in output_dirs:
                output_dir = Path(output_dir)
                output_dir.mkdir(exist_ok=True, parents=True)
                plt.savefig(output_dir / f"{safe_filename}.png")

            plt.close()

        except Exception as e:
            logger.error(f"Plotting error for {lab_name}: {e}")

    def _add_reference_bands(self, ax, lab_df: pd.DataFrame, start_date, end_date) -> None:
        """Add colored reference range bands to plot."""
        min_vals = lab_df["healthy_range_min"].dropna() if "healthy_range_min" in lab_df.columns else pd.Series(dtype=float)
        max_vals = lab_df["healthy_range_max"].dropna() if "healthy_range_max" in lab_df.columns else pd.Series(dtype=float)

        if min_vals.empty and max_vals.empty:
            return

        y_min_mode = float(min_vals.mode()[0]) if not min_vals.empty else None
        y_max_mode = float(max_vals.mode()[0]) if not max_vals.empty else None

        light_green = "#b7e6a1"
        light_red = "#e6b7b7"

        # Get current y-axis limits
        cur_ymin, cur_ymax = ax.get_ylim()

        if y_min_mode is not None and y_max_mode is not None:
            # Both bounds: green band between them, red bands outside
            ax.set_ylim(min(cur_ymin, y_min_mode), max(cur_ymax, y_max_mode))
            cur_ymin, cur_ymax = ax.get_ylim()

            plt.fill_between(
                [start_date, end_date],
                y_min_mode,
                y_max_mode,
                color=light_green,
                alpha=0.6,
                label="Reference Range"
            )
            plt.fill_between(
                [start_date, end_date],
                cur_ymin,
                y_min_mode,
                color=light_red,
                alpha=0.3,
                label="Below Range"
            )
            plt.fill_between(
                [start_date, end_date],
                y_max_mode,
                cur_ymax,
                color=light_red,
                alpha=0.3,
                label="Above Range"
            )
        elif y_min_mode is not None:
            # Only lower bound (> y_min_mode): healthy extends from min to top
            data_max = lab_df[self.value_col].max()
            y_top = max(cur_ymax, data_max * 1.2, y_min_mode * 2)
            ax.set_ylim(min(cur_ymin, y_min_mode * 0.8), y_top)
            cur_ymin, cur_ymax = ax.get_ylim()

            plt.fill_between(
                [start_date, end_date],
                y_min_mode,
                cur_ymax,
                color=light_green,
                alpha=0.6,
                label="Reference Range (≥)"
            )
            plt.fill_between(
                [start_date, end_date],
                cur_ymin,
                y_min_mode,
                color=light_red,
                alpha=0.3,
                label="Below Range"
            )
        else:
            # Only upper bound (< y_max_mode): healthy extends from bottom to max
            data_min = lab_df[self.value_col].min()
            y_bottom = min(cur_ymin, 0, data_min * 0.9 if data_min > 0 else data_min * 1.1)
            ax.set_ylim(y_bottom, max(cur_ymax, y_max_mode * 1.2))
            cur_ymin, cur_ymax = ax.get_ylim()

            plt.fill_between(
                [start_date, end_date],
                cur_ymin,
                y_max_mode,
                color=light_green,
                alpha=0.6,
                label="Reference Range (≤)"
            )
            plt.fill_between(
                [start_date, end_date],
                y_max_mode,
                cur_ymax,
                color=light_red,
                alpha=0.3,
                label="Above Range"
            )

    @staticmethod
    def _sanitize_filename(lab_name: str) -> str:
        """Sanitize lab name for use as filename."""
        # Convert '(%)' to '(percentage)' for better filename compatibility
        lab_name = str(lab_name).replace('(%)', '(percentage)')
        # Allow parentheses but replace other special chars
        safe_name = re.sub(r'[^\w\-_. ()]', '_', lab_name)
        return safe_name
