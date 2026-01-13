"""End-of-run reporting for lab results pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

from config import UNKNOWN_VALUE


@dataclass
class PipelineMetrics:
    """Container for pipeline run metrics."""
    # Timing
    run_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # PDF Processing
    pdfs_found: int = 0
    pdfs_skipped: int = 0
    pdfs_processed: int = 0
    pdfs_failed: int = 0

    # Row counts
    rows_after_merge: int = 0
    rows_after_dedup: int = 0

    # Standardization
    unknown_names_count: int = 0
    unknown_units_count: int = 0
    unknown_names: list = field(default_factory=list)
    unknown_units: list = field(default_factory=list)

    # Edge cases
    needs_review_count: int = 0
    high_priority_count: int = 0  # confidence < 0.7
    edge_case_breakdown: dict = field(default_factory=dict)

    # Validation results
    validation_errors: list = field(default_factory=list)
    validation_warnings: list = field(default_factory=list)

    # Output paths
    csv_path: Optional[Path] = None
    excel_path: Optional[Path] = None
    report_path: Optional[Path] = None


class RunReportGenerator:
    """Generates end-of-run reports for the pipeline."""

    def __init__(self, metrics: PipelineMetrics, df: pd.DataFrame):
        self.metrics = metrics
        self.df = df

    def compute_metrics(self) -> None:
        """Compute all metrics from the DataFrame."""
        self._compute_standardization_metrics()
        self._compute_edge_case_metrics()

    def _compute_standardization_metrics(self) -> None:
        """Compute standardization success/failure metrics."""
        # Unknown lab names
        if 'lab_name_standardized' in self.df.columns:
            unknown_names_mask = self.df['lab_name_standardized'] == UNKNOWN_VALUE
            self.metrics.unknown_names_count = int(unknown_names_mask.sum())
            if self.metrics.unknown_names_count > 0:
                self.metrics.unknown_names = (
                    self.df.loc[unknown_names_mask, 'lab_name_raw']
                    .dropna().unique().tolist()
                )

        # Unknown units
        if 'lab_unit_standardized' in self.df.columns:
            unknown_units_mask = self.df['lab_unit_standardized'] == UNKNOWN_VALUE
            self.metrics.unknown_units_count = int(unknown_units_mask.sum())
            if self.metrics.unknown_units_count > 0:
                self.metrics.unknown_units = (
                    self.df.loc[unknown_units_mask, ['lab_name_raw', 'lab_unit_raw']]
                    .drop_duplicates()
                    .to_dict('records')
                )

    def _compute_edge_case_metrics(self) -> None:
        """Compute edge case and review metrics."""
        if 'needs_review' not in self.df.columns:
            return

        self.metrics.needs_review_count = int(self.df['needs_review'].sum())

        if 'confidence_score' in self.df.columns:
            self.metrics.high_priority_count = int((self.df['confidence_score'] < 0.7).sum())

        # Breakdown by reason
        if self.metrics.needs_review_count > 0 and 'review_reason' in self.df.columns:
            review_df = self.df[self.df['needs_review'] == True]
            reasons = review_df['review_reason'].str.split('; ').explode()
            reasons = reasons[reasons != '']
            self.metrics.edge_case_breakdown = reasons.value_counts().to_dict()

    def run_validations(self) -> None:
        """Run validation checks (from test.py)."""
        from test import (
            test_all_rows_have_dates_and_no_duplicates,
            test_lab_unit_percent_vs_lab_name,
            test_lab_unit_percent_value_range,
            test_lab_unit_boolean_value,
            test_lab_name_standardized_unit_consistency,
            test_lab_value_outliers_by_lab_name_standardized,
            test_unique_date_lab_name_standardized,
        )

        report = {}
        test_all_rows_have_dates_and_no_duplicates(report)
        test_lab_unit_percent_vs_lab_name(report)
        test_lab_unit_percent_value_range(report)
        test_lab_unit_boolean_value(report)
        test_lab_name_standardized_unit_consistency(report)
        test_lab_value_outliers_by_lab_name_standardized(report)
        test_unique_date_lab_name_standardized(report)

        # Categorize as errors vs warnings
        for file_key, errors in report.items():
            for error in errors:
                if 'outlier' in error.lower():
                    self.metrics.validation_warnings.append(error)
                else:
                    self.metrics.validation_errors.append(error)

    def get_status(self) -> str:
        """Determine overall pipeline status."""
        if self.metrics.validation_errors:
            return "FAILURE"
        elif (self.metrics.unknown_names_count > 0 or
              self.metrics.unknown_units_count > 0 or
              self.metrics.high_priority_count > 0):
            return "NEEDS_REVIEW"
        else:
            return "SUCCESS"

    def generate_console_report(self) -> str:
        """Generate concise console output."""
        status = self.get_status()
        status_indicator = {"SUCCESS": "[OK]", "NEEDS_REVIEW": "[!!]", "FAILURE": "[FAIL]"}[status]

        lines = [
            "",
            "=" * 60,
            f"  PIPELINE RUN REPORT - {status_indicator} {status}",
            "=" * 60,
            "",
            f"  PDFs: {self.metrics.pdfs_processed} processed, "
            f"{self.metrics.pdfs_skipped} skipped, {self.metrics.pdfs_failed} failed",
            f"  Rows: {self.metrics.rows_after_dedup} (after deduplication)",
            "",
        ]

        # Standardization summary
        if self.metrics.unknown_names_count > 0 or self.metrics.unknown_units_count > 0:
            lines.append("  STANDARDIZATION ISSUES:")
            if self.metrics.unknown_names_count > 0:
                lines.append(f"    - {self.metrics.unknown_names_count} unknown lab names")
            if self.metrics.unknown_units_count > 0:
                lines.append(f"    - {self.metrics.unknown_units_count} unknown units")
            lines.append("")

        # Review summary
        if self.metrics.needs_review_count > 0:
            lines.append(f"  ITEMS NEEDING REVIEW: {self.metrics.needs_review_count}")
            lines.append(f"    - High priority (conf < 0.7): {self.metrics.high_priority_count}")
            lines.append("")

        # Validation issues
        if self.metrics.validation_errors:
            lines.append(f"  VALIDATION ERRORS: {len(self.metrics.validation_errors)}")
            for err in self.metrics.validation_errors[:3]:
                truncated = err[:77] + "..." if len(err) > 80 else err
                lines.append(f"    - {truncated}")
            if len(self.metrics.validation_errors) > 3:
                lines.append(f"    - ... and {len(self.metrics.validation_errors) - 3} more")
            lines.append("")

        lines.append(f"  Full report: {self.metrics.report_path}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def generate_markdown_report(self) -> str:
        """Generate full markdown report."""
        status = self.get_status()

        lines = [
            f"# Pipeline Run Report",
            f"",
            f"**Status**: {status}",
            f"**Timestamp**: {self.metrics.run_timestamp}",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| PDFs Found | {self.metrics.pdfs_found} |",
            f"| PDFs Processed | {self.metrics.pdfs_processed} |",
            f"| PDFs Skipped (cached) | {self.metrics.pdfs_skipped} |",
            f"| PDFs Failed | {self.metrics.pdfs_failed} |",
            f"| Rows After Merge | {self.metrics.rows_after_merge} |",
            f"| Rows After Dedup | {self.metrics.rows_after_dedup} |",
            f"",
        ]

        # Standardization section
        lines.extend([
            f"## Standardization",
            f"",
        ])

        if self.metrics.unknown_names_count == 0 and self.metrics.unknown_units_count == 0:
            lines.append("All lab names and units successfully standardized.")
        else:
            if self.metrics.unknown_names_count > 0:
                lines.extend([
                    f"### Unknown Lab Names ({self.metrics.unknown_names_count} rows)",
                    f"",
                    f"Add these to `config/lab_specs.json`:",
                    f"",
                ])
                for name in self.metrics.unknown_names[:20]:
                    lines.append(f"- `{name}`")
                if len(self.metrics.unknown_names) > 20:
                    lines.append(f"- ... and {len(self.metrics.unknown_names) - 20} more")
                lines.append("")

            if self.metrics.unknown_units_count > 0:
                lines.extend([
                    f"### Unknown Units ({self.metrics.unknown_units_count} rows)",
                    f"",
                    f"Add these to `config/lab_specs.json`:",
                    f"",
                ])
                for item in self.metrics.unknown_units[:20]:
                    lines.append(f"- `{item.get('lab_unit_raw')}` for `{item.get('lab_name_raw')}`")
                if len(self.metrics.unknown_units) > 20:
                    lines.append(f"- ... and {len(self.metrics.unknown_units) - 20} more")
                lines.append("")

        # Edge cases section
        lines.extend([
            f"## Quality Review",
            f"",
        ])

        if self.metrics.needs_review_count == 0:
            lines.append("No items flagged for review.")
        else:
            lines.extend([
                f"**{self.metrics.needs_review_count}** items need review",
                f"**{self.metrics.high_priority_count}** high priority (confidence < 0.7)",
                f"",
            ])

            if self.metrics.edge_case_breakdown:
                lines.extend([
                    f"### Breakdown by Category",
                    f"",
                    f"| Category | Count |",
                    f"|----------|-------|",
                ])
                for category, count in self.metrics.edge_case_breakdown.items():
                    lines.append(f"| {category} | {count} |")
                lines.append("")

        lines.append("")

        # Validation section
        lines.extend([
            f"## Data Validation",
            f"",
        ])

        if not self.metrics.validation_errors and not self.metrics.validation_warnings:
            lines.append("All validation checks passed.")
        else:
            if self.metrics.validation_errors:
                lines.extend([
                    f"### Errors ({len(self.metrics.validation_errors)})",
                    f"",
                ])
                for err in self.metrics.validation_errors[:10]:
                    lines.append(f"- {err}")
                if len(self.metrics.validation_errors) > 10:
                    lines.append(f"- ... and {len(self.metrics.validation_errors) - 10} more")
                lines.append("")

            if self.metrics.validation_warnings:
                lines.extend([
                    f"### Warnings ({len(self.metrics.validation_warnings)})",
                    f"",
                ])
                for warn in self.metrics.validation_warnings[:10]:
                    lines.append(f"- {warn}")
                if len(self.metrics.validation_warnings) > 10:
                    lines.append(f"- ... and {len(self.metrics.validation_warnings) - 10} more")

        # Output files section
        lines.extend([
            f"",
            f"## Output Files",
            f"",
            f"- CSV: `{self.metrics.csv_path}`",
            f"- Excel: `{self.metrics.excel_path}`",
            f"- Report: `{self.metrics.report_path}`",
        ])

        return "\n".join(lines)

    def write_report(self, output_path: Path) -> Path:
        """Write markdown report to file and return path."""
        report_path = output_path / "run_report.md"
        self.metrics.report_path = report_path

        report_content = self.generate_markdown_report()
        report_path.write_text(report_content, encoding='utf-8')

        return report_path


def generate_run_report(
    df: pd.DataFrame,
    output_path: Path,
    pdfs_found: int,
    pdfs_skipped: int,
    pdfs_processed: int,
    pdfs_failed: int,
    rows_after_merge: int,
    rows_after_dedup: int,
    csv_path: Path,
    excel_path: Path,
) -> Path:
    """
    Generate and write end-of-run report.

    Returns path to the markdown report file.
    """
    metrics = PipelineMetrics(
        pdfs_found=pdfs_found,
        pdfs_skipped=pdfs_skipped,
        pdfs_processed=pdfs_processed,
        pdfs_failed=pdfs_failed,
        rows_after_merge=rows_after_merge,
        rows_after_dedup=rows_after_dedup,
        csv_path=csv_path,
        excel_path=excel_path,
    )

    generator = RunReportGenerator(metrics, df)
    generator.compute_metrics()
    generator.run_validations()

    report_path = generator.write_report(output_path)

    # Print console summary
    print(generator.generate_console_report())

    return report_path
