"""DataFrame normalization and transformation logic."""

import logging
import re

import pandas as pd

from parselabs.config import UNKNOWN_VALUE, LabSpecsConfig
from parselabs.utils import ensure_columns

logger = logging.getLogger(__name__)

INTERVAL_VALUE_PATTERN = re.compile(
    r"^\s*(?P<min>\d+(?:\.\d+)?)\s*(?:-|a)\s*(?P<max>\d+(?:\.\d+)?)(?:\s*/.*)?\s*$",
    re.IGNORECASE,
)

QUALITATIVE_VARIANT_MAP = {
    "Blood - Anti-Tissue Transglutaminase Antibody IgA (Anti-tTG IgA)": "Blood - Anti-Tissue Transglutaminase Antibody IgA (Anti-tTG IgA), Qualitative",
    "Urine Type II - Bilirubin": "Urine Type II - Bilirubin, Qualitative",
    "Urine Type II - Glucose": "Urine Type II - Glucose, Qualitative",
    "Urine Type II - Ketones": "Urine Type II - Ketones, Qualitative",
    "Urine Type II - Proteins": "Urine Type II - Proteins, Qualitative",
    "Urine Type II - Urobilinogen": "Urine Type II - Urobilinogen, Qualitative",
}


def preprocess_numeric_value(value) -> str:
    """
    Clean raw value for numeric conversion.

    Handles:
    - Strip trailing "=" (e.g., "0.9=" → "0.9")
    - Extract first number from embedded metadata (e.g., "52.6=1946" → "52.6")
    - Remove space thousands separators (e.g., "256 000" → "256000")
    - European decimal format (comma → period)

    Args:
        value: Raw value from extraction

    Returns:
        Cleaned string ready for numeric conversion
    """

    # Pass through missing values
    if pd.isna(value):
        return value

    s = str(value).strip()

    # Normalize OCR spacing around decimal separators before any parsing logic.
    # Examples: "13 .0" -> "13.0", "4 ,04" -> "4,04"
    s = re.sub(r"(?<=\d)\s*([.,])\s*(?=\d)", r"\1", s)

    # Handle embedded metadata: "52.6=1946" → "52.6" (keep first number before =)
    # Must be done before stripping trailing "=" to handle both cases
    if "=" in s and not s.endswith("="):
        s = s.split("=")[0].strip()

    # Strip trailing "=" (e.g., "0.9=" → "0.9")
    s = s.rstrip("=")

    # Remove space thousands separators (e.g., "256 000" → "256000")
    # Only if result looks like a number with spaces between digits
    if re.match(r"^\d[\d\s]+$", s):
        s = s.replace(" ", "")

    # European decimal format (comma → period)
    s = s.replace(",", ".")

    return s


def extract_comparison_value(value) -> tuple[str, bool, bool]:
    """
    Extract numeric value from comparison operators.

    Args:
        value: Raw value from extraction

    Returns:
        Tuple of (numeric_str, is_below_limit, is_above_limit)
    """

    # Pass through missing values
    if pd.isna(value):
        return str(value), False, False

    s = str(value).strip()
    is_below = s.startswith("<")
    is_above = s.startswith(">")

    # Strip operator prefix and extract numeric part
    if is_below or is_above:
        # Extract numeric part: "<0.05" → "0.05", "< 10" → "10"
        numeric = re.sub(r"^[<>]\s*", "", s)
        # Handle cases like "<20=NR" → "20"
        numeric = numeric.split("=")[0].strip()
        return numeric, is_below, is_above

    # No operator found
    return s, False, False


def parse_interval_midpoint(value) -> float | None:
    """
    Parse interval-style result text and return its midpoint.

    Supports deterministic numeric ranges such as ``1 a 2/campo`` or ``0 - 5``.
    Returns ``None`` when the value is not a simple ascending interval.
    """

    # Guard: Missing values cannot encode an interval midpoint.
    if pd.isna(value):
        return None

    text = str(value).strip()
    match = INTERVAL_VALUE_PATTERN.match(text)

    # Guard: Non-interval values should fall back to the regular numeric parser.
    if match is None:
        return None

    lower_bound = float(match.group("min"))
    upper_bound = float(match.group("max"))

    # Guard: Descending intervals are more likely malformed OCR than valid ranges.
    if upper_bound < lower_bound:
        return None

    # Use the deterministic midpoint while preserving the original raw interval text.
    return (lower_bound + upper_bound) / 2


def apply_normalizations(
    df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """
    Apply normalization transformations to the DataFrame.

    This focuses on unit conversions to enable cross-provider comparison.
    Health status calculations have been moved to the review tool.

    Args:
        df: DataFrame with raw lab results
        lab_specs: Lab specifications configuration

    Returns:
        DataFrame with normalized columns added
    """

    # Nothing to normalize
    if df.empty:
        return df

    # Ensure required columns exist
    ensure_columns(
        df,
        ["lab_name_standardized", "lab_unit_standardized", "raw_lab_name"],
        default=None,
    )

    # Look up lab_type from config (vectorized)
    if lab_specs.exists:
        df["lab_type"] = df["lab_name_standardized"].apply(lambda name: lab_specs.get_lab_type(name) if pd.notna(name) and name != UNKNOWN_VALUE else "blood")
    # Default all to blood when no specs available
    else:
        df["lab_type"] = "blood"

    # Convert to primary units (batched)
    if lab_specs.exists:
        df = apply_unit_conversions(df, lab_specs)
    # No conversions available, just copy values as-is
    else:
        df["value_primary"] = df.get("raw_value")
        df["lab_unit_primary"] = df.get("lab_unit_standardized")
        df["reference_min_primary"] = df.get("raw_reference_min")
        df["reference_max_primary"] = df.get("raw_reference_max")

    return df


def _preprocess_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract comparison operators and preprocess raw values.

    Extracts < and > operators, cleans numeric strings, and converts to float.
    Adds is_below_limit, is_above_limit, and value_primary columns.
    """

    # Initialize limit indicator columns
    df["is_below_limit"] = False
    df["is_above_limit"] = False

    # Extract comparison operators and preprocess values
    comparison_results = df["raw_value"].apply(extract_comparison_value)
    df["_preprocessed_value"] = comparison_results.apply(lambda x: x[0])
    df["is_below_limit"] = comparison_results.apply(lambda x: x[1])
    df["is_above_limit"] = comparison_results.apply(lambda x: x[2])

    # Apply additional preprocessing (spaces, trailing =, embedded metadata, comma→period)
    df["_preprocessed_value"] = df["_preprocessed_value"].apply(preprocess_numeric_value)

    # Derive midpoint values for interval-style observations before numeric coercion.
    interval_midpoints = df["_preprocessed_value"].apply(parse_interval_midpoint)
    interval_mask = interval_midpoints.notna()

    # Replace only recognized interval rows so scalar values keep their original parsing path.
    if interval_mask.any():
        df.loc[interval_mask, "_preprocessed_value"] = interval_midpoints.loc[interval_mask].map(str)

    # Convert preprocessed values to numeric
    df["value_primary"] = pd.to_numeric(df["_preprocessed_value"], errors="coerce")

    # Clean up temporary column
    df.drop(columns=["_preprocessed_value"], inplace=True)

    # Log preprocessing results
    limit_count = df["is_below_limit"].sum() + df["is_above_limit"].sum()
    if limit_count > 0:
        logger.info(f"[normalization] Extracted {limit_count} comparison operators (</>)")

    return df


def classify_qualitative_value(text: str) -> int | None:
    """
    Classify qualitative lab result text to boolean value (0/1/None).

    Deterministic pattern matching built from 163-entry production cache.
    Covers Portuguese/English/French/German medical terms.

    Args:
        text: Raw qualitative value from extraction

    Returns:
        0 (negative/normal/absent), 1 (positive/abnormal/present), or None (not classifiable)
    """

    # Guard: empty or missing input
    if not text or not isinstance(text, str):
        return None

    normalized = text.strip().lower()

    # Guard: empty after stripping
    if not normalized or normalized in ("nan", "none", "null", "nr"):
        return None

    # Positive patterns (detected/present/abnormal)
    _POSITIVE_PREFIXES = (
        "positiv",  # positivo, positiva, positive
        "detetad",  # detetado
        "detected",
        "détecté",
        "resultado positivo",
    )
    _POSITIVE_KEYWORDS = {
        "abundantes",
        "levemente turvo",
        "trace",
        "traces",
        "traço",
        "traços",
        "turvo",
        "vestigio",
        "vestígios",
        "vestigios",
        "cultura polimicrobiana",
    }

    # Negative patterns (not detected/absent/normal)
    _NEGATIVE_PREFIXES = (
        "negativ",  # negativo, negativa, negative
        "normal",
        "ausent",  # ausente, ausentes, ausência
        "ausenci",  # ausencia
        "não det",  # não detetado, não det.
        "not detected",
        "nao contem",
        "não contem",
        "não contém",
        "nao contém",
        "nao se ",  # nao se isolaram, nao se observaram
        "não se observaram ovos",
        "não revelou",
        "nao revelou",
        "sem alterações",
        "limpid",  # límpido, límpida, limpido
        "amicrobiano",
        "cultura estéril",
        "por hplc não foram detectadas",
        "não foram observad",
        "exame ecográfico",  # normal findings
        "forma e dimensões normais",
        "de normal morfologia",
        "morfologia celular: normal",
        "o estudo electroforético",
        "num exame químico com ausência",
    )
    _NEGATIVE_KEYWORDS = {
        "amarela",
        "amarelo",
        "amarela clara",
        "citrina",
        "claro",
        "clara",
        "transparente",
    }

    # Check positive patterns
    if any(normalized.startswith(p) for p in _POSITIVE_PREFIXES):
        return 1
    if normalized in _POSITIVE_KEYWORDS:
        return 1
    # "raras plaquetas gigantes" is positive (abnormal finding)
    if normalized.startswith("raras plaquetas"):
        return 1
    # "observa-se mucosa cólica com folículo linfoide hiperplasiado" is positive
    if "hiperplasiado" in normalized:
        return 1
    if "distendida, apresenta paredes finas" in normalized:
        return 0
    if re.fullmatch(r"\++", normalized):
        return 1

    # Check negative patterns
    if any(normalized.startswith(p) for p in _NEGATIVE_PREFIXES):
        return 0
    if normalized in _NEGATIVE_KEYWORDS:
        return 0

    # Not classifiable — return None (value stays in raw_value for review)
    return None


def _classify_qualitative_batch(values: list[str]) -> dict[str, int | None]:
    """
    Classify a batch of qualitative values using deterministic lookup.

    Args:
        values: List of raw qualitative values

    Returns:
        Dictionary mapping raw_value -> 0, 1, or None
    """

    result = {}
    for v in values:
        if v is not None:
            result[v] = classify_qualitative_value(v)
    return result


def _remap_qualitative_variant_rows(
    df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """Remap text-only urine strip analytes onto boolean qualitative variants."""

    # Guard: Missing lab mappings or empty inputs leave nothing to remap.
    if df.empty or "lab_name_standardized" not in df.columns:
        return df

    # Evaluate raw_value first, then raw_comments only when raw_value is absent.
    for source_col, needs_raw_value_isna in [
        ("raw_value", False),
        ("raw_comments", True),
    ]:
        remap_indices: list[int] = []

        # Inspect each row independently so only confidently qualitative rows are remapped.
        for idx in df.index:
            standardized_name = df.at[idx, "lab_name_standardized"]

            # Skip labs that do not have a dedicated qualitative variant.
            if standardized_name not in QUALITATIVE_VARIANT_MAP:
                continue

            # Skip rows that already parsed as numeric values.
            if pd.notna(df.at[idx, "value_primary"]):
                continue

            source_value = df.at[idx, source_col]

            # Skip rows without a qualitative source value to classify.
            if pd.isna(source_value):
                continue

            # Only inspect comments when the raw_value itself is missing.
            if needs_raw_value_isna and pd.notna(df.at[idx, "raw_value"]):
                continue

            # Skip text that the deterministic classifier cannot interpret.
            if classify_qualitative_value(str(source_value)) is None:
                continue

            variant_name = QUALITATIVE_VARIANT_MAP[standardized_name]

            # Skip variants missing from lab_specs so normalization stays internally consistent.
            if variant_name not in lab_specs.specs:
                continue

            remap_indices.append(idx)

        # Guard: No rows qualified for this source column.
        if not remap_indices:
            continue

        # Remap the standardized lab and unit so downstream review/export logic sees boolean rows.
        df.loc[remap_indices, "lab_name_standardized"] = df.loc[remap_indices, "lab_name_standardized"].map(QUALITATIVE_VARIANT_MAP)
        df.loc[remap_indices, "lab_unit_standardized"] = "boolean"
        logger.info(f"[normalization] Remapped {len(remap_indices)} qualitative rows from {source_col} to boolean urine variants")

    return df


def _convert_qualitative_values(
    df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """
    Convert qualitative text values to 0/1 for boolean labs.

    Uses deterministic pattern matching to convert values like "negativo"/"positivo"
    to numeric 0/1 values. Handles both boolean labs and non-boolean labs
    with negative-like values.
    """

    # Find labs with boolean primary unit
    boolean_labs = [lab_name for lab_name in df["lab_name_standardized"].unique() if pd.notna(lab_name) and lab_name != UNKNOWN_VALUE and lab_specs.get_primary_unit(lab_name) == "boolean"]

    # Boolean labs: convert qualitative text to 0/1 from raw_value, then comments
    if boolean_labs:
        for source_col, needs_raw_value_isna in [
            ("raw_value", False),
            ("raw_comments", True),
        ]:
            mask = df["lab_name_standardized"].isin(boolean_labs) & df["value_primary"].isna() & df[source_col].notna()

            # For raw_comments source, only use when raw_value is also missing
            if needs_raw_value_isna:
                mask &= df["raw_value"].isna()

            # No qualifying rows for this source
            if not mask.any():
                continue

            qual_map = _classify_qualitative_batch(df.loc[mask, source_col].tolist())
            df.loc[mask, "value_primary"] = df.loc[mask, source_col].map(lambda v: qual_map.get(v))
            df.loc[mask, "lab_unit_standardized"] = "boolean"
            df.loc[mask, "lab_unit_primary"] = "boolean"
            logger.info(f"[normalization] Converted {mask.sum()} qualitative values from {source_col} (boolean labs)")

    # Non-boolean labs: convert only negative-like values to 0 from raw_value, then comments
    non_boolean_exclude = boolean_labs if boolean_labs else []

    for source_col, needs_raw_value_isna in [
        ("raw_value", False),
        ("raw_comments", True),
    ]:
        mask = df["value_primary"].isna() & df[source_col].notna() & ~df["lab_name_standardized"].isin(non_boolean_exclude)

        # For raw_comments source, only use when raw_value is also missing
        if needs_raw_value_isna:
            mask &= df["raw_value"].isna()

        # No qualifying rows for this source
        if not mask.any():
            continue

        qual_map = _classify_qualitative_batch(df.loc[mask, source_col].tolist())
        converted_count = 0
        for idx in df.loc[mask].index:
            if qual_map.get(df.at[idx, source_col]) == 0:
                df.at[idx, "value_primary"] = 0.0
                converted_count += 1
        if converted_count > 0:
            logger.info(f"[normalization] Converted {converted_count} negative-like {source_col} to 0 (non-boolean labs)")

    return df


def _apply_unit_conversions_for_lab(
    df: pd.DataFrame,
    lab_name_standardized: str,
    lab_specs: LabSpecsConfig,
) -> None:
    """
    Apply unit conversion factors for a single lab type.

    Converts values and reference ranges from their current units to the
    primary unit specified in lab_specs.
    """

    primary_unit = lab_specs.get_primary_unit(lab_name_standardized)

    # No primary unit configured for this lab
    if not primary_unit:
        return

    # Get rows for this lab
    mask = df["lab_name_standardized"] == lab_name_standardized

    # For each unique unit in this lab
    for unit in df.loc[mask, "lab_unit_standardized"].unique():
        # Skip missing units
        if pd.isna(unit):
            continue

        unit_mask = mask & (df["lab_unit_standardized"] == unit)

        # If already in primary unit, just update lab_unit_primary
        if unit == primary_unit:
            df.loc[unit_mask, "lab_unit_primary"] = primary_unit
            continue

        # Get conversion factor
        factor = lab_specs.get_conversion_factor(lab_name_standardized, unit)

        # No conversion factor available for this unit
        if factor is None:
            continue

        # Apply conversion to all matching rows (vectorized)
        df.loc[unit_mask, "value_primary"] = df.loc[unit_mask, "value_primary"] * factor
        df.loc[unit_mask, "reference_min_primary"] = df.loc[unit_mask, "raw_reference_min"] * factor
        df.loc[unit_mask, "reference_max_primary"] = df.loc[unit_mask, "raw_reference_max"] * factor
        df.loc[unit_mask, "lab_unit_primary"] = primary_unit


def apply_unit_conversions(
    df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """
    Convert values to primary units defined in lab specs.

    This function applies unit conversions in a vectorized manner where possible.
    Handles boolean tests by converting qualitative text values (e.g., "negativo", "positivo")
    to 0/1 using deterministic pattern matching.
    """

    # Initialize primary unit columns from standardized values
    df["lab_unit_primary"] = df["lab_unit_standardized"]
    df["reference_min_primary"] = df["raw_reference_min"]
    df["reference_max_primary"] = df["raw_reference_max"]

    # Phase 1: Extract comparison operators and preprocess values
    df = _preprocess_values(df)

    # Phase 2: Remap text-only urine strip analytes onto boolean qualitative variants.
    df = _remap_qualitative_variant_rows(df, lab_specs)

    # Phase 3: Convert qualitative text values to 0/1 for boolean-style labs.
    df = _convert_qualitative_values(df, lab_specs)

    # Phase 4: Apply exact unit conversion factors for each lab type.
    for lab_name_standardized in df["lab_name_standardized"].unique():
        # Skip unknown or missing lab names
        if pd.isna(lab_name_standardized) or lab_name_standardized == UNKNOWN_VALUE:
            continue
        _apply_unit_conversions_for_lab(df, lab_name_standardized, lab_specs)

    return df


def flag_duplicate_entries(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where (date, lab_name_standardized) appears more than once.

    Sets review_needed=True and appends DUPLICATE_ENTRY to review_reason for all
    rows in duplicate groups. The surviving row after dedup retains the flag so
    reviewers can verify the correct value was kept.

    Args:
        df: DataFrame with lab results (must have 'date' and 'lab_name_standardized')

    Returns:
        DataFrame with duplicate entries flagged
    """

    # Skip if missing required columns
    if df.empty or "date" not in df.columns or "lab_name_standardized" not in df.columns:
        return df

    # Initialize review columns if not present, ensuring correct dtypes
    if "review_needed" not in df.columns:
        df["review_needed"] = False
    else:
        df["review_needed"] = df["review_needed"].astype(bool)
    if "review_reason" not in df.columns:
        df["review_reason"] = ""
    else:
        df["review_reason"] = df["review_reason"].fillna("").astype(str)

    # Consider only rows with usable standardized lab names for duplicate detection.
    candidate_mask = df["lab_name_standardized"].notna() & (df["lab_name_standardized"] != UNKNOWN_VALUE)

    duplicate_indices: list[int] = []

    # Only conflicting duplicate groups should reach the reviewer.
    for _, group_df in df.loc[candidate_mask].groupby(["date", "lab_name_standardized"], sort=False):
        # Guard: Single rows are never duplicates.
        if len(group_df) < 2:
            continue

        # Equivalent dual-unit rows are expected and should not raise review noise.
        if _duplicate_group_is_equivalent(group_df):
            continue

        duplicate_indices.extend(group_df.index.tolist())

    # Guard: Equivalent duplicate groups should not leave any review flag behind.
    if not duplicate_indices:
        return df

    dup_mask = df.index.isin(duplicate_indices)
    count = int(dup_mask.sum())
    logger.info(f"Flagging {count} rows as DUPLICATE_ENTRY")

    df.loc[dup_mask, "review_needed"] = True
    df.loc[dup_mask, "review_reason"] = df.loc[dup_mask, "review_reason"].fillna("").apply(lambda x: str(x) + "DUPLICATE_ENTRY; " if "DUPLICATE_ENTRY" not in str(x) else str(x))

    return df


def _duplicate_group_is_equivalent(group_df: pd.DataFrame) -> bool:
    """Return True when a duplicate group represents the same result in multiple units."""

    required_columns = {"value_primary", "is_below_limit", "is_above_limit"}

    # Guard: Missing normalized columns means the group cannot be compared safely.
    if not required_columns.issubset(group_df.columns):
        return False

    below_limit_flags = group_df["is_below_limit"].fillna(False).astype(bool)
    above_limit_flags = group_df["is_above_limit"].fillna(False).astype(bool)

    # Distinct comparison operators represent genuinely different observations.
    if below_limit_flags.nunique() > 1 or above_limit_flags.nunique() > 1:
        return False

    normalized_values = pd.to_numeric(group_df["value_primary"], errors="coerce")

    # Keep rounded dual-unit repeats quiet when they converge after normalization.
    if normalized_values.notna().all():
        value_span = float(normalized_values.max() - normalized_values.min())
        largest_value = max(1.0, float(normalized_values.abs().max()))
        tolerance = max(0.1, largest_value * 0.01)
        return value_span <= tolerance

    raw_values = group_df["raw_value"].fillna("").astype(str).str.strip()
    primary_units = group_df["lab_unit_primary"].fillna("").astype(str).str.strip()

    # Fall back to exact text equality for non-numeric duplicate groups.
    return raw_values.nunique() == 1 and primary_units.nunique() <= 1


def deduplicate_results(df: pd.DataFrame, lab_specs: LabSpecsConfig) -> pd.DataFrame:
    """
    Deduplicate by (date, lab_name_standardized), keeping best match (prefer primary unit).

    Args:
        df: DataFrame with lab results
        lab_specs: Lab specifications configuration

    Returns:
        Deduplicated DataFrame
    """

    # Skip if missing required columns
    if df.empty or "date" not in df.columns or "lab_name_standardized" not in df.columns:
        return df

    # Build priority column: prefer rows already in primary unit (priority 0)
    def _dedup_priority(row):
        lab_name = row["lab_name_standardized"]
        if pd.isna(lab_name) or lab_name == UNKNOWN_VALUE:
            return 1
        primary_unit = lab_specs.get_primary_unit(lab_name) if lab_specs.exists else None
        if primary_unit and row.get("lab_unit_standardized") == primary_unit:
            return 0
        return 1

    df["_dedup_priority"] = df.apply(_dedup_priority, axis=1)
    df = df.sort_values(["date", "lab_name_standardized", "_dedup_priority"])
    df = df.drop_duplicates(subset=["date", "lab_name_standardized"], keep="first")
    df = df.drop(columns=["_dedup_priority"])
    return df.reset_index(drop=True)


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
        # Skip columns not present in DataFrame
        if col not in df.columns:
            continue

        try:
            # Convert datetime columns
            if target_dtype == "datetime64[ns]":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            # Convert string columns
            elif target_dtype == "str":
                df[col] = df[col].astype("string")
            # Convert boolean columns
            elif target_dtype == "boolean":
                df[col] = df[col].astype("boolean")
            # Convert nullable integer columns
            elif target_dtype == "Int64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            # Convert float columns
            elif "float" in target_dtype:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        except (ValueError, TypeError) as e:
            logger.warning(f"Dtype conversion failed for {col} to {target_dtype}: {e}")

    return df
