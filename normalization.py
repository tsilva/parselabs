"""DataFrame normalization and transformation logic."""

import logging
import pandas as pd
from typing import Optional

from openai import OpenAI

from config import LabSpecsConfig, UNKNOWN_VALUE
from utils import slugify, ensure_columns

logger = logging.getLogger(__name__)


def apply_normalizations(
    df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
    client: Optional[OpenAI] = None,
    model_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Apply all normalization transformations to the DataFrame in batched operations.

    This function consolidates multiple row-wise operations into efficient vectorized
    operations where possible.

    Args:
        df: DataFrame with raw lab results
        lab_specs: Lab specifications configuration
        client: OpenAI client for qualitative value standardization
        model_id: Model ID for qualitative value standardization

    Returns:
        DataFrame with normalized columns added
    """
    if df.empty:
        return df

    # Ensure required columns exist
    ensure_columns(df, ["lab_name_standardized", "lab_unit_standardized", "lab_name_raw"], default=None)

    # Look up lab_type from config (vectorized)
    if lab_specs.exists:
        df["lab_type"] = df["lab_name_standardized"].apply(
            lambda name: lab_specs.get_lab_type(name) if pd.notna(name) and name != UNKNOWN_VALUE else "blood"
        )
    else:
        df["lab_type"] = "blood"

    # Create lab_name_slug (vectorized)
    if "lab_name_raw" in df.columns:
        df["lab_name_slug"] = df.apply(
            lambda row: f"{row.get('lab_type', 'blood')}-{slugify(row.get('lab_name_raw', ''))}",
            axis=1
        )
    else:
        df["lab_name_slug"] = ""

    # Convert to primary units (batched)
    if lab_specs.exists:
        df = apply_unit_conversions(df, lab_specs, client, model_id)
    else:
        # No conversions, just copy values
        df["value_primary"] = df.get("value_raw")
        df["lab_unit_primary"] = df.get("lab_unit_standardized")
        df["reference_min_primary"] = df.get("reference_min_raw")
        df["reference_max_primary"] = df.get("reference_max_raw")

    # Compute health status columns (vectorized where possible)
    if lab_specs.exists:
        df = compute_health_status(df, lab_specs)
    else:
        df["is_out_of_reference"] = None
        df["healthy_range_min"] = None
        df["healthy_range_max"] = None
        df["is_in_healthy_range"] = None

    return df


def apply_unit_conversions(
    df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
    client: Optional[OpenAI] = None,
    model_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Convert values to primary units defined in lab specs.

    This function applies unit conversions in a vectorized manner where possible.
    Handles boolean tests by converting qualitative text values (e.g., "negativo", "positivo")
    to 0/1 using LLM-based classification with caching.
    """
    # Initialize primary unit columns
    # Convert value_raw to numeric for calculations (text values will become NaN)
    df["value_primary"] = pd.to_numeric(df["value_raw"], errors='coerce')
    df["lab_unit_primary"] = df["lab_unit_standardized"]
    df["reference_min_primary"] = df["reference_min_raw"]
    df["reference_max_primary"] = df["reference_max_raw"]

    # Handle boolean tests: convert qualitative values to 0/1
    if client and model_id:
        # Import here to avoid circular imports
        from standardization import standardize_qualitative_values

        # Find labs with boolean primary unit
        boolean_labs = [
            lab_name for lab_name in df["lab_name_standardized"].unique()
            if pd.notna(lab_name) and lab_name != UNKNOWN_VALUE
            and lab_specs.get_primary_unit(lab_name) == "boolean"
        ]

        if boolean_labs:
            # First pass: convert qualitative text in value_raw
            value_raw_mask = (
                df["lab_name_standardized"].isin(boolean_labs) &
                df["value_primary"].isna() &
                df["value_raw"].notna()
            )

            if value_raw_mask.any():
                raw_values = df.loc[value_raw_mask, "value_raw"].tolist()
                qual_map = standardize_qualitative_values(raw_values, model_id, client)

                # Apply converted values
                df.loc[value_raw_mask, "value_primary"] = df.loc[value_raw_mask, "value_raw"].map(
                    lambda v: qual_map.get(v)
                )
                df.loc[value_raw_mask, "lab_unit_primary"] = "boolean"
                logger.info(f"[normalization] Converted {value_raw_mask.sum()} qualitative values from value_raw")

            # Second pass: use comments field for boolean tests where value_raw is NaN
            comments_mask = (
                df["lab_name_standardized"].isin(boolean_labs) &
                df["value_primary"].isna() &
                df["value_raw"].isna() &
                df["comments"].notna()
            )

            if comments_mask.any():
                comment_values = df.loc[comments_mask, "comments"].tolist()
                qual_map = standardize_qualitative_values(comment_values, model_id, client)

                # Apply converted values from comments
                df.loc[comments_mask, "value_primary"] = df.loc[comments_mask, "comments"].map(
                    lambda v: qual_map.get(v)
                )
                df.loc[comments_mask, "lab_unit_primary"] = "boolean"
                logger.info(f"[normalization] Converted {comments_mask.sum()} qualitative values from comments")

        # Third pass: for ANY test (not just boolean), convert negative-like comments to 0
        # This handles presence/absence tests where "Negativo"/"Normal" means "not detected" = 0
        negative_comments_mask = (
            df["value_primary"].isna() &
            df["value_raw"].isna() &
            df["comments"].notna()
        )

        if negative_comments_mask.any():
            comment_values = df.loc[negative_comments_mask, "comments"].tolist()
            qual_map = standardize_qualitative_values(comment_values, model_id, client)

            # Only apply 0 for negative results (qual_map returns 0 for negative)
            # Keep NaN for positive (1) or unknown (None) since we don't have a numeric value
            for idx in df.loc[negative_comments_mask].index:
                comment = df.at[idx, "comments"]
                converted = qual_map.get(comment)
                if converted == 0:
                    df.at[idx, "value_primary"] = 0.0

            converted_count = sum(1 for c in comment_values if qual_map.get(c) == 0)
            if converted_count > 0:
                logger.info(f"[normalization] Converted {converted_count} negative-like comments to 0 for numeric tests")

    # Group by lab_name_standardized to apply conversions efficiently
    for lab_name_standardized in df["lab_name_standardized"].unique():
        if pd.isna(lab_name_standardized) or lab_name_standardized == UNKNOWN_VALUE:
            continue

        primary_unit = lab_specs.get_primary_unit(lab_name_standardized)
        if not primary_unit:
            continue

        # Get rows for this lab
        mask = df["lab_name_standardized"] == lab_name_standardized

        # For each unique unit in this lab
        for unit in df.loc[mask, "lab_unit_standardized"].unique():
            if pd.isna(unit):
                continue

            unit_mask = mask & (df["lab_unit_standardized"] == unit)

            # If already in primary unit, just update lab_unit_primary
            if unit == primary_unit:
                df.loc[unit_mask, "lab_unit_primary"] = primary_unit
                continue

            # Get conversion factor
            factor = lab_specs.get_conversion_factor(lab_name_standardized, unit)
            if factor is None:
                continue

            # Apply conversion to all matching rows (vectorized)
            # Use already-converted value_primary (text values are already NaN)
            df.loc[unit_mask, "value_primary"] = df.loc[unit_mask, "value_primary"] * factor
            df.loc[unit_mask, "reference_min_primary"] = df.loc[unit_mask, "reference_min_raw"] * factor
            df.loc[unit_mask, "reference_max_primary"] = df.loc[unit_mask, "reference_max_raw"] * factor
            df.loc[unit_mask, "lab_unit_primary"] = primary_unit

    return df


def compute_health_status(df: pd.DataFrame, lab_specs: LabSpecsConfig) -> pd.DataFrame:
    """
    Compute health status columns (is_out_of_reference, is_in_healthy_range).

    Uses vectorized operations where possible.
    """
    # Get healthy ranges for all unique lab names
    lab_names = df["lab_name_standardized"].unique()
    healthy_ranges = {}
    for lab_name_standardized in lab_names:
        if pd.notna(lab_name_standardized) and lab_name_standardized != UNKNOWN_VALUE:
            healthy_ranges[lab_name_standardized] = lab_specs.get_healthy_range(lab_name_standardized)

    # Apply healthy ranges (vectorized lookup)
    df["healthy_range_min"] = df["lab_name_standardized"].map(lambda name: healthy_ranges.get(name, (None, None))[0])
    df["healthy_range_max"] = df["lab_name_standardized"].map(lambda name: healthy_ranges.get(name, (None, None))[1])

    # Compute is_out_of_reference (vectorized)
    value_series = pd.to_numeric(df["value_primary"], errors='coerce')
    ref_min_series = pd.to_numeric(df["reference_min_primary"], errors='coerce')
    ref_max_series = pd.to_numeric(df["reference_max_primary"], errors='coerce')

    is_low = value_series < ref_min_series
    is_high = value_series > ref_max_series
    has_ref = ref_min_series.notna() | ref_max_series.notna()

    df["is_out_of_reference"] = None
    df.loc[has_ref, "is_out_of_reference"] = (is_low | is_high)[has_ref]

    # Compute is_in_healthy_range (vectorized)
    healthy_min_series = pd.to_numeric(df["healthy_range_min"], errors='coerce')
    healthy_max_series = pd.to_numeric(df["healthy_range_max"], errors='coerce')

    too_low = value_series < healthy_min_series
    too_high = value_series > healthy_max_series
    has_healthy = healthy_min_series.notna() | healthy_max_series.notna()

    df["is_in_healthy_range"] = None
    df.loc[has_healthy, "is_in_healthy_range"] = ~(too_low | too_high)[has_healthy]

    return df


def deduplicate_results(df: pd.DataFrame, lab_specs: LabSpecsConfig) -> pd.DataFrame:
    """
    Deduplicate by (date, lab_name_standardized), keeping best match (prefer primary unit).

    Args:
        df: DataFrame with lab results
        lab_specs: Lab specifications configuration

    Returns:
        Deduplicated DataFrame
    """
    if df.empty or "date" not in df.columns or "lab_name_standardized" not in df.columns:
        return df

    def pick_best_dupe(group):
        """Pick best duplicate: prefer primary unit if multiple entries exist."""
        # Get lab_name_standardized from first row (grouping columns are included)
        lab_name_standardized = group.iloc[0]["lab_name_standardized"]
        primary_unit = lab_specs.get_primary_unit(lab_name_standardized) if lab_specs.exists else None

        if primary_unit and "lab_unit_standardized" in group.columns and (group["lab_unit_standardized"] == primary_unit).any():
            return group[group["lab_unit_standardized"] == primary_unit].iloc[0]
        else:
            return group.iloc[0]

    # Group and apply deduplication (using deprecated behavior to preserve columns)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        deduplicated = (
            df
            .groupby(["date", "lab_name_standardized"], dropna=False, as_index=False, group_keys=False)
            .apply(pick_best_dupe)
            .reset_index(drop=True)
        )

    return deduplicated


def apply_dtype_conversions(df: pd.DataFrame, dtype_map: dict) -> pd.DataFrame:
    """
    Apply dtype conversions to DataFrame columns based on schema.

    Args:
        df: DataFrame to convert
        dtype_map: Dictionary mapping column names to dtype strings

    Returns:
        DataFrame with converted dtypes
    """
    for col, target_dtype in dtype_map.items():
        if col not in df.columns:
            continue

        try:
            if target_dtype == "datetime64[ns]":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif target_dtype == "boolean":
                df[col] = df[col].astype("boolean")
            elif target_dtype == "Int64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif "float" in target_dtype:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            # Other types like string are often fine as 'object'
        except Exception as e:
            logger.warning(f"Dtype conversion failed for {col} to {target_dtype}: {e}")

    return df
