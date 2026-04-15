"""Data-loading helpers for the review workspace."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from parselabs.config import Demographics, LabSpecsConfig
from parselabs.rows import build_corpus_review_rows
from parselabs.store import load_legacy_merged_review_dataframe, read_page_payload, resolve_page_path
from parselabs.types import PagePayload, PersistedReviewStatus, coerce_persisted_review_status


def _load_json_cached(json_path: Path, cache: dict[str, PagePayload | None]) -> PagePayload | None:
    """Load and cache page JSON for review-status backfills."""

    json_path_str = str(json_path)
    if json_path_str in cache:
        return cache[json_path_str]

    cache[json_path_str] = read_page_payload(json_path)
    return cache[json_path_str]


def _sync_review_statuses(df: pd.DataFrame, output_path: Path) -> list[PersistedReviewStatus | None]:
    """Read review_status from page JSON for legacy merged review CSVs."""

    json_cache: dict[str, PagePayload | None] = {}
    review_statuses: list[PersistedReviewStatus | None] = []

    for row in df.itertuples():
        result_index = getattr(row, "result_index", None)
        if result_index is None or pd.isna(result_index):
            review_statuses.append(None)
            continue

        json_path = resolve_page_path(
            output_path,
            getattr(row, "source_file", ""),
            getattr(row, "page_number", None),
            ".json",
        )
        json_data = _load_json_cached(json_path, json_cache)
        if json_data and "lab_results" in json_data:
            result_idx = int(result_index)
            if result_idx < len(json_data["lab_results"]):
                review_statuses.append(
                    coerce_persisted_review_status(
                        json_data["lab_results"][result_idx].get("review_status")
                    )
                )
                continue

        review_statuses.append(None)

    return review_statuses


def _is_out_of_range(value: float, range_min: object, range_max: object) -> bool:
    """Return whether a value falls outside the given numeric bounds."""

    lower_bound = _coerce_optional_float(range_min)
    upper_bound = _coerce_optional_float(range_max)
    return (lower_bound is not None and value < lower_bound) or (upper_bound is not None and value > upper_bound)


def _coerce_optional_float(value: object) -> float | None:
    """Return a float when the value is numeric-like, otherwise None."""

    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, float, str)):
        return float(value)

    return None


def _format_reference_range(row: pd.Series) -> str:
    """Format reference_min/reference_max into a display string."""

    ref_min = row["reference_min"]
    ref_max = row["reference_max"]

    if pd.isna(ref_min) and pd.isna(ref_max):
        if row.get("lab_unit") == "boolean":
            return "0 - 1"
        return ""

    if pd.isna(ref_min):
        return f"< {ref_max}"
    if pd.isna(ref_max):
        return f"> {ref_min}"

    return f"{ref_min} - {ref_max}"


def _check_out_of_reference(row: pd.Series) -> bool | None:
    """Check if a value falls outside the PDF reference range."""

    value = row["value"]
    if pd.isna(value):
        return None

    ref_min = row["reference_min"]
    ref_max = row["reference_max"]
    if pd.isna(ref_min) and pd.isna(ref_max):
        if row.get("lab_unit") == "boolean":
            return value > 0
        return None

    return _is_out_of_range(value, ref_min, ref_max)


def _get_lab_spec_range(lab_name: str, lab_specs: LabSpecsConfig, gender: str | None, age: int | None) -> pd.Series:
    """Look up the configured optimal range for a lab, adjusted for demographics."""

    range_min, range_max = lab_specs.get_optimal_range_for_demographics(lab_name, gender=gender, age=age)
    return pd.Series({"lab_specs_min": range_min, "lab_specs_max": range_max})


def _check_out_of_optimal_range(row: pd.Series) -> bool | None:
    """Check if a value falls outside the configured optimal range from lab specs."""

    value = row.get("value")
    if pd.isna(value):
        return None

    spec_min = row.get("lab_specs_min")
    spec_max = row.get("lab_specs_max")
    if pd.isna(spec_min) and pd.isna(spec_max):
        return None

    return _is_out_of_range(value, spec_min, spec_max)


def load_results_dataframe(
    output_path: Path,
    lab_specs: LabSpecsConfig,
    demographics: Demographics | None,
) -> pd.DataFrame:
    """Load review rows for the results explorer with one legacy fallback path."""

    df = build_corpus_review_rows(output_path, lab_specs)
    if df.empty:
        df = load_legacy_merged_review_dataframe(output_path)
        if df.empty:
            return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "review_status" not in df.columns:
        df["review_status"] = _sync_review_statuses(df, output_path)

    if "reference_min" in df.columns and "reference_max" in df.columns:
        df["reference_range"] = df.apply(_format_reference_range, axis=1)

    if "value" in df.columns and "reference_min" in df.columns and "reference_max" in df.columns:
        df["is_out_of_reference"] = df.apply(_check_out_of_reference, axis=1)

    gender = demographics.gender if demographics else None
    age = demographics.age if demographics else None

    if "lab_name" in df.columns and lab_specs.exists:
        range_df = df["lab_name"].apply(lambda name: _get_lab_spec_range(name, lab_specs, gender, age))
        df["lab_specs_min"] = range_df["lab_specs_min"]
        df["lab_specs_max"] = range_df["lab_specs_max"]

    if "value" in df.columns and "lab_specs_min" in df.columns and "lab_specs_max" in df.columns:
        optimal_mask = df.apply(_check_out_of_optimal_range, axis=1)
        df["is_out_of_optimal_range"] = optimal_mask
        df["is_out_of_healthy_range"] = optimal_mask

    return df


__all__ = ["load_results_dataframe"]
