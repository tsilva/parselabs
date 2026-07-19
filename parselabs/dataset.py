"""Integrity-report helpers for lab specs and merged exports."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from parselabs.exceptions import ConfigurationError
from parselabs.lab_specs_validation import LabSpecsValidator
from parselabs.paths import get_lab_specs_path
from parselabs.runtime import list_non_template_profiles, load_profile_config

logger = logging.getLogger(__name__)

LAB_SPECS_PATH = get_lab_specs_path()
PERCENT_UPPER_BOUND_EXCEPTIONS = {
    "Blood - Prothrombin Time (PT) (%)",
}


def _append_report_error(report: dict[str, list[str]], file_key: str, error: str) -> None:
    """Append one error line under a report key."""

    report.setdefault(file_key, []).append(error)


def _load_results_dataframe(csv_path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    """Load one merged review CSV for integrity checks."""

    errors: list[str] = []

    try:
        return pd.read_csv(csv_path), errors
    except FileNotFoundError as exc:
        errors.append(f"Results file not found: {exc}")
    except pd.errors.EmptyDataError as exc:
        errors.append(f"Results file is empty: {exc}")
    except pd.errors.ParserError as exc:
        errors.append(f"Failed to parse CSV: {exc}")
    except PermissionError as exc:
        errors.append(f"Permission denied reading results file: {exc}")

    return None, errors


def _check_all_rows_have_dates_and_no_duplicates(df: pd.DataFrame, report: dict[str, list[str]], csv_path: Path) -> None:
    """Validate that rows have dates and are not fully duplicated."""

    if not df["date"].notnull().all():
        _append_report_error(report, str(csv_path), "Some rows have null date")

    if not (df["date"].astype(str).str.strip() != "").all():
        _append_report_error(report, str(csv_path), "Some rows have empty date")

    row_hashes = df.apply(
        lambda row: hashlib.sha256(("|".join(str(value) for value in row.tolist())).encode("utf-8")).hexdigest(),
        axis=1,
    )
    duplicates = row_hashes[row_hashes.duplicated(keep=False)]

    if duplicates.empty:
        return

    for dup_hash in duplicates.unique():
        dup_indices = duplicates[duplicates == dup_hash].index.tolist()
        for idx in dup_indices:
            row = df.loc[idx]
            source_file = row.get("source_file", "unknown")
            _append_report_error(report, str(source_file), f"Duplicate row at index {idx}: {row.to_dict()}")


def _check_lab_unit_percent_vs_lab_name(df: pd.DataFrame, report: dict[str, list[str]]) -> None:
    """Validate that percentage units keep percentage lab names."""

    if "lab_unit" not in df.columns or "lab_name" not in df.columns:
        return

    mask = (df["lab_unit"] == "%") & (~df["lab_name"].astype(str).str.endswith("(%)"))
    for idx in df[mask].index:
        row = df.loc[idx]
        source_file = row.get("source_file", "unknown")
        lab_name = row.get("lab_name", "")
        _append_report_error(report, str(source_file), f'Row at index {idx} (lab_name="{lab_name}") has unit="%" but lab_name="{lab_name}"')


def _check_lab_unit_percent_value_range(df: pd.DataFrame, report: dict[str, list[str]]) -> None:
    """Validate that percentage values stay within 0-100."""

    if "lab_unit" not in df.columns or "value" not in df.columns:
        return

    lab_names = df["lab_name"].astype(str) if "lab_name" in df.columns else pd.Series("", index=df.index)
    exceeds_upper_bound = (df["value"] > 100) & ~lab_names.isin(PERCENT_UPPER_BOUND_EXCEPTIONS)
    mask = (df["lab_unit"] == "%") & ((df["value"] < 0) | exceeds_upper_bound)
    for idx in df[mask].index:
        row = df.loc[idx]
        source_file = row.get("source_file", "unknown")
        lab_name = row.get("lab_name", "")
        value = row.get("value")
        _append_report_error(report, str(source_file), f'Row at index {idx} (lab_name="{lab_name}") has unit="%" but value={value} (should be between 0 and 100)')


def _check_lab_unit_boolean_value(df: pd.DataFrame, report: dict[str, list[str]]) -> None:
    """Validate that boolean labs keep 0/1 values."""

    if "lab_unit" not in df.columns or "value" not in df.columns:
        return

    mask = (df["lab_unit"] == "boolean") & (~df["value"].isin([0, 1]))
    for idx in df[mask].index:
        row = df.loc[idx]
        source_file = row.get("source_file", "unknown")
        lab_name = row.get("lab_name", "")
        value = row.get("value")
        _append_report_error(report, str(source_file), f'Row at index {idx} (lab_name="{lab_name}") has unit="boolean" but value={value} (should be 0 or 1)')


def _check_lab_name_unit_consistency(df: pd.DataFrame, report: dict[str, list[str]], csv_path: Path) -> None:
    """Validate that one lab name does not emit multiple units."""

    if "lab_name" not in df.columns or "lab_unit" not in df.columns:
        return

    grouped = df.groupby("lab_name")["lab_unit"].unique()
    for lab_name, units in grouped.items():
        non_null_units = [unit for unit in units if pd.notnull(unit)]
        if len(non_null_units) > 1:
            indices = df[df["lab_name"] == lab_name].index.tolist()
            _append_report_error(report, str(csv_path), f'lab_name="{lab_name}" has inconsistent unit values: {non_null_units} (rows: {indices})')


def _check_lab_value_outliers_by_lab_name(df: pd.DataFrame, report: dict[str, list[str]], csv_path: Path) -> None:
    """Validate that grouped values do not contain obvious statistical outliers."""

    if "value" not in df.columns or "lab_name" not in df.columns:
        return

    numeric_df = df[pd.notnull(df["value"])]

    for lab_name, group in numeric_df.groupby("lab_name"):
        if "lab_unit" in group.columns:
            unit_counts = group["lab_unit"].value_counts()
            if unit_counts.empty:
                continue
            most_freq_unit = unit_counts.idxmax()
            values = group[group["lab_unit"] == most_freq_unit]["value"]
        else:
            values = group["value"]
            most_freq_unit = "N/A"

        values = pd.to_numeric(values, errors="coerce").dropna()
        if len(values) < 5:
            continue

        mean = values.mean()
        std = values.std()
        if std == 0 or pd.isnull(std):
            continue

        if "lab_unit" in group.columns:
            outliers = group[(group["lab_unit"] == most_freq_unit) & ((group["value"] > mean + 3 * std) | (group["value"] < mean - 3 * std))]
        else:
            outliers = group[(group["value"] > mean + 3 * std) | (group["value"] < mean - 3 * std)]

        if outliers.empty:
            continue

        source_files = set(outliers["source_file"].dropna().astype(str))
        outlier_values = outliers["value"].tolist()
        page_numbers = outliers["page_number"].tolist() if "page_number" in outliers.columns else ["unknown"] * len(outlier_values)
        _append_report_error(
            report,
            str(csv_path),
            f'lab_name="{lab_name}", unit="{most_freq_unit}" has outlier value (>3 std from mean {mean:.2f}±{std:.2f}) in files: {list(sorted(source_files))} outlier values: {outlier_values} page numbers: {page_numbers}',
        )


def _check_unique_date_lab_name(df: pd.DataFrame, report: dict[str, list[str]]) -> None:
    """Validate uniqueness of the (date, lab_name) key."""

    if "date" not in df.columns or "lab_name" not in df.columns:
        return

    duplicates = df.duplicated(subset=["date", "lab_name"], keep=False)
    if not duplicates.any():
        return

    dup_df = df[duplicates]
    for row in dup_df.itertuples():
        source_file = getattr(row, "source_file", "unknown") or "unknown"
        date_val = getattr(row, "date", "")
        lab_name = getattr(row, "lab_name", "")
        _append_report_error(report, str(source_file), f"Duplicate (date, lab_name) at index {row.Index}: date={date_val}, lab_name={lab_name}")


def _check_lab_specs_schema(report: dict[str, list[str]]) -> None:
    """Validate lab_specs.json schema and completeness."""

    file_key = str(LAB_SPECS_PATH)

    try:
        validator = LabSpecsValidator()
        validator.validate()

        for error in validator.errors:
            _append_report_error(report, file_key, error)
    except ImportError as exc:
        _append_report_error(report, file_key, f"Exception during schema validation: {exc}")


def build_integrity_report(profile_names: list[str] | None = None) -> dict[str, list[str]]:
    """Run schema and merged-output integrity checks and return a grouped report."""

    report: dict[str, list[str]] = {}

    _check_lab_specs_schema(report)

    if profile_names is None:
        profile_names = list_non_template_profiles()

    if not profile_names:
        logger.warning("No profiles found. Only schema checks were run.")
        return report

    for profile_name in profile_names:
        try:
            profile = load_profile_config(profile_name)
        except ConfigurationError as exc:
            _append_report_error(report, profile_name, str(exc))
            continue

        if not profile.output_path:
            _append_report_error(report, profile_name, f"Profile '{profile_name}' has no output_path defined")
            continue

        csv_path = profile.output_path / "all.csv"
        df, errors = _load_results_dataframe(csv_path)
        for error in errors:
            _append_report_error(report, str(csv_path), error)
        if df is None:
            continue

        _check_all_rows_have_dates_and_no_duplicates(df, report, csv_path)
        _check_lab_unit_percent_vs_lab_name(df, report)
        _check_lab_unit_percent_value_range(df, report)
        _check_lab_unit_boolean_value(df, report)
        _check_lab_name_unit_consistency(df, report, csv_path)
        _check_lab_value_outliers_by_lab_name(df, report, csv_path)
        _check_unique_date_lab_name(df, report)

    return report


def print_integrity_report(report: dict[str, list[str]]) -> None:
    """Print the grouped integrity report to stdout via the root logger."""

    logger.info("\n=== Integrity Report ===")
    if not report:
        logger.info("All checks passed.")
        return

    for file_key, errors in report.items():
        logger.info(f"\nFile: {file_key}")
        for error in errors:
            logger.warning(f"  - {error}")


__all__ = [
    "build_integrity_report",
    "print_integrity_report",
]
