"""DataFrame normalization and transformation logic."""

import logging
import re

import pandas as pd

from labs_parser.config import UNKNOWN_VALUE, LabSpecsConfig
from labs_parser.utils import ensure_columns

logger = logging.getLogger(__name__)


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


def _infer_unit_by_plausibility(
    lab_name_standardized: str,
    value: float,
    lab_specs: LabSpecsConfig,
    percentage_lab_name: str,
) -> str | None:
    """Check if value is implausible for primary unit but fits percentage range.

    Returns '%' if value clearly belongs to the percentage variant, None otherwise.
    """

    # Look up expected range for the primary unit
    expected_default = lab_specs._specs.get(lab_name_standardized, {}).get("ranges", {}).get("default", [])

    # Not enough range data to evaluate
    if len(expected_default) < 2:
        return None

    expected_min, expected_max = expected_default[0], expected_default[1]

    # Value is plausible for primary unit
    if not expected_max or value <= expected_max * 5:
        return None

    # Check if value fits percentage range instead
    pct_default = lab_specs._specs.get(percentage_lab_name, {}).get("ranges", {}).get("default", [])

    # Not enough percentage range data to evaluate
    if len(pct_default) < 2:
        return None

    pct_min, pct_max = pct_default[0], pct_default[1]

    # Value doesn't fit percentage range either
    if not pct_min or not pct_max or not ((pct_min * 0.5) <= value <= (pct_max * 2)):
        return None

    primary_unit = lab_specs.get_primary_unit(lab_name_standardized)
    logger.debug(f"[unit_inference] Value {value} implausible for {primary_unit} (expected {expected_min}-{expected_max}), assigning '%' instead")
    return "%"


def infer_missing_unit(
    lab_name_standardized: str,
    value: float,
    ref_min: float | None,
    ref_max: float | None,
    lab_specs: LabSpecsConfig,
) -> str | None:
    """
    Infer missing unit when lab_unit_raw is null using generic heuristics.

    Generic approach (no test-specific hardcoding):
    1. Reference range magnitude heuristic - percentages typically have ranges > 10
    2. Value-to-range ratio - value should be within or near the reference range
    3. Lab specs lookup - use primary unit from configuration
    4. Check for percentage variant - if lab has both absolute and % forms

    Args:
        lab_name_standardized: Standardized lab name
        value: Numeric value
        ref_min: Minimum reference value
        ref_max: Maximum reference value
        lab_specs: Lab specifications configuration

    Returns:
        Inferred unit string or None if cannot infer
    """

    # Skip unknown or missing lab names
    if pd.isna(lab_name_standardized) or lab_name_standardized == UNKNOWN_VALUE:
        return None

    # Strategy 1: Reference range magnitude heuristic
    # Percentages typically have reference ranges > 10 (e.g., Albumin: 50-66%)
    # Absolute values typically have smaller ranges (e.g., Albumin: 3.5-5.5 g/dL)
    has_large_range = False
    if ref_min is not None and ref_max is not None:
        has_large_range = max(abs(ref_min), abs(ref_max)) > 10

    # Strategy 2: Check if lab has a percentage variant in lab_specs
    # Many protein tests have both "Blood - Albumin" and "Blood - Albumin (%)"
    has_percentage_variant = False
    percentage_lab_name = None

    if lab_specs.exists and not lab_name_standardized.endswith("(%)"):
        # Check if "{lab_name} (%)" exists in lab_specs
        potential_pct_name = f"{lab_name_standardized} (%)"
        if potential_pct_name in lab_specs._specs:
            has_percentage_variant = True
            percentage_lab_name = potential_pct_name

    # Strategy 3: Value-to-range ratio
    # If value is close to reference range magnitude, they're likely in same unit
    value_matches_range = False
    if value is not None and ref_max is not None and ref_max > 0:
        ratio = value / ref_max
        # Value is within reasonable bounds of reference range (0.3x to 3x)
        value_matches_range = 0.3 <= ratio <= 3.0

    # Decision logic: strong evidence this is a percentage
    if has_large_range and has_percentage_variant and value_matches_range:
        logger.debug(f"[unit_inference] Inferring '%' for {lab_name_standardized}: value={value}, ref_range=({ref_min}, {ref_max})")
        return "%"

    # Strategy 3b: Value implausible for primary unit but fits percentage range
    if lab_specs.exists and has_percentage_variant and value is not None:
        inferred = _infer_unit_by_plausibility(lab_name_standardized, value, lab_specs, percentage_lab_name)
        if inferred:
            return inferred

    # Strategy 4: Fallback to lab specs primary unit
    if lab_specs.exists:
        primary_unit = lab_specs.get_primary_unit(lab_name_standardized)
        if primary_unit:
            logger.debug(f"[unit_inference] Using primary unit '{primary_unit}' from lab_specs for {lab_name_standardized}")
            return primary_unit

    # Could not infer a unit
    return None


def validate_reference_range(
    lab_name_standardized: str,
    value: float,
    unit: str,
    ref_min: float | None,
    ref_max: float | None,
    lab_specs: LabSpecsConfig,
) -> tuple[float | None, float | None]:
    """
    Validate reference range against expected ranges from lab_specs.

    Detects when PDF reference ranges are in wrong units (common when PDFs show
    both percentage and absolute values, and the reference range gets misassigned).

    When a mismatch is detected, attempts to convert the range using known
    conversion factors before nullifying.

    Args:
        lab_name_standardized: Standardized lab name
        value: Numeric value
        unit: Unit of measurement
        ref_min: Minimum reference value from PDF
        ref_max: Maximum reference value from PDF
        lab_specs: Lab specifications configuration

    Returns:
        Tuple of (validated_ref_min, validated_ref_max) - may be converted or nullified
    """

    # Skip if no lab specs available
    if not lab_specs.exists:
        return ref_min, ref_max

    # Skip unknown or missing lab names
    if pd.isna(lab_name_standardized) or lab_name_standardized == UNKNOWN_VALUE:
        return ref_min, ref_max

    lab_config = lab_specs._specs.get(lab_name_standardized, {})

    # Skip validation for labs where ranges legitimately vary (e.g., by menstrual cycle phase)
    if lab_config.get("ranges_vary_with_cycle", False):
        return ref_min, ref_max

    # Skip validation for labs with inherently messy reference data
    # (e.g., WBC differentials where PDFs show both % and absolute counts with shared ref ranges)
    if lab_config.get("skip_range_validation", False):
        return ref_min, ref_max

    # Skip if either reference bound is missing
    if ref_min is None or ref_max is None:
        return ref_min, ref_max

    # Get expected range for this lab from lab_specs
    ranges = lab_config.get("ranges", {})

    # No expected ranges configured
    if not ranges:
        return ref_min, ref_max

    # Try to get expected range for this unit or default
    expected_range = None

    # Prefer default range
    if "default" in ranges:
        expected_range = ranges["default"]
    # Fall back to unit-specific range
    elif unit and unit in ranges:
        expected_range = ranges[unit]

    # No usable expected range found
    if not expected_range or len(expected_range) < 2:
        return ref_min, ref_max

    expected_min, expected_max = expected_range[0], expected_range[1]

    # Convert expected range when current unit differs from primary unit
    # This handles cases where PDF reference ranges are in a different unit than lab_specs
    primary_unit = lab_config.get("primary_unit")

    if unit and primary_unit and unit != primary_unit:
        alternatives = lab_config.get("alternatives", [])
        for alt in alternatives:
            if alt.get("unit") == unit:
                factor = alt.get("factor", 1.0)
                # Convert expected range from primary unit to the current unit
                # factor converts FROM alternative unit TO primary unit
                # So to convert FROM primary TO alternative, we divide by factor
                if factor and factor != 0:
                    expected_min = expected_min / factor
                    expected_max = expected_max / factor
                break

    # Cannot validate against zero or missing expected values
    if expected_min is None or expected_max is None or expected_min == 0 or expected_max == 0:
        return ref_min, ref_max

    # Calculate ratios to detect unit mismatches
    ratio_min = abs(ref_min / expected_min) if expected_min != 0 else 0
    ratio_max = abs(ref_max / expected_max) if expected_max != 0 else 0

    # If ranges differ by >10x or <0.1x, likely wrong unit assigned
    THRESHOLD_HIGH = 10.0
    THRESHOLD_LOW = 0.1

    # Require BOTH ratios to be suspicious to avoid false positives
    # Example false positive: Eosinophils PDF=(0, 5) vs expected=(1, 6)
    #   ratio_min = 0/1 = 0.0 (suspicious), but ratio_max = 5/6 = 0.83 (fine)
    # Example true positive: Neutrophils PDF=(2000, 7500) vs expected=(40, 70)
    #   ratio_min = 50x, ratio_max = 107x (BOTH suspicious)
    is_suspicious = (ratio_min > THRESHOLD_HIGH or ratio_min < THRESHOLD_LOW) and (ratio_max > THRESHOLD_HIGH or ratio_max < THRESHOLD_LOW)

    # Attempt to fix or nullify suspicious ranges
    if is_suspicious:
        # Try to find a conversion factor that would fix the range
        converted_ref_min, converted_ref_max, was_explained = _try_convert_mismatched_range(
            lab_name_standardized,
            ref_min,
            ref_max,
            expected_min,
            expected_max,
            ratio_min,
            ratio_max,
            lab_config,
            lab_specs,
        )

        # Conversion succeeded
        if converted_ref_min is not None and converted_ref_max is not None:
            return converted_ref_min, converted_ref_max

        # Only log warning if the issue wasn't already explained
        if not was_explained:
            logger.error(
                f"[range_validation] Suspicious reference range for {lab_name_standardized} ({unit}): "
                f"PDF=({ref_min:.2f}, {ref_max:.2f}), expected≈({expected_min}, {expected_max}), "
                f"ratios=({ratio_min:.2f}x, {ratio_max:.2f}x) - nullifying range"
            )

        # Nullify suspicious range
        return None, None

    # Range is within acceptable bounds
    return ref_min, ref_max


def _try_match_cross_variant_range(
    lab_name: str,
    ref_min: float,
    ref_max: float,
    variant_lab: str,
    lab_specs: LabSpecsConfig,
) -> bool:
    """Check if PDF reference range matches a cross-variant lab (% vs absolute).

    Returns True and logs if the range matches the variant, False otherwise.
    """

    # Variant lab not in specs
    if variant_lab not in lab_specs._specs:
        return False

    variant_ranges = lab_specs._specs[variant_lab].get("ranges", {}).get("default", [])

    # Not enough range data for variant
    if len(variant_ranges) < 2:
        return False

    variant_min, variant_max = variant_ranges[0], variant_ranges[1]

    # Variant range must have positive bounds
    if not (variant_min > 0 and variant_max > 0):
        return False

    # Check if ratios are close to 1 (0.3-3.0 tolerance for lab-to-lab variations)
    ratio_min = abs(ref_min / variant_min) if variant_min != 0 else 0
    ratio_max = abs(ref_max / variant_max) if variant_max != 0 else 0

    # Ratios too far from 1 to be a match
    if not (0.3 < ratio_min < 3.0 and 0.3 < ratio_max < 3.0):
        return False

    logger.info(f"[range_validation] Detected cross-variant range for {lab_name}: PDF=({ref_min:.2f}, {ref_max:.2f}) matches {variant_lab} range ({variant_min}, {variant_max}) - nullifying (cannot convert between variants)")
    return True


def _try_convert_mismatched_range(
    lab_name_standardized: str,
    ref_min: float,
    ref_max: float,
    expected_min: float,
    expected_max: float,
    ratio_min: float,
    ratio_max: float,
    lab_config: dict,
    lab_specs: LabSpecsConfig,
) -> tuple[float | None, float | None, bool]:
    """
    Attempt to convert a mismatched reference range using known conversion factors.

    This handles cases where the PDF extracted a reference range in a different unit
    than the value. For example:
    - Platelets value in 10⁹/L but range in /mm³ (1000x)
    - Glucose value in mg/dL but range in mmol/L (~18x)
    - Protein electrophoresis with % and g/dL ranges mixed up

    Args:
        lab_name_standardized: Standardized lab name
        ref_min, ref_max: PDF reference range values
        expected_min, expected_max: Expected range from lab_specs
        ratio_min, ratio_max: Ratios of PDF to expected values
        lab_config: Lab configuration from lab_specs
        lab_specs: Full lab specifications

    Returns:
        Tuple of (converted_ref_min, converted_ref_max, was_explained)
        - If conversion successful: (converted_min, converted_max, False)
        - If detected but can't convert: (None, None, True) - issue was logged
        - If unknown mismatch: (None, None, False) - caller should log warning
    """

    TOLERANCE = 0.3  # Allow 30% tolerance when matching conversion factors

    # Strategy 1: Check if ratio matches any alternative unit's conversion factor
    alternatives = lab_config.get("alternatives", [])
    for alt in alternatives:
        factor = alt.get("factor", 1.0)

        # Skip invalid conversion factors
        if factor is None or factor == 0:
            continue

        # The PDF range might be in this alternative unit
        # Conversion factor converts FROM alternative unit TO primary unit
        # So if PDF range / expected range ≈ 1/factor, the PDF range is in alternative unit
        expected_ratio = 1.0 / factor

        # Check if both ratios match this conversion factor
        if abs(ratio_min - expected_ratio) / expected_ratio < TOLERANCE and abs(ratio_max - expected_ratio) / expected_ratio < TOLERANCE:
            # Apply conversion: multiply by factor to convert to primary unit
            converted_min = ref_min * factor
            converted_max = ref_max * factor
            logger.info(
                f"[range_validation] Converted reference range for {lab_name_standardized}: PDF=({ref_min:.2f}, {ref_max:.2f}) -> ({converted_min:.2f}, {converted_max:.2f}) using factor {factor} (detected {alt.get('unit')} unit)"
            )
            return converted_min, converted_max, False

    # Strategy 2: Check if range belongs to percentage variant (for absolute labs)
    if not lab_name_standardized.endswith("(%)"):
        pct_lab_name = f"{lab_name_standardized} (%)"
        if _try_match_cross_variant_range(lab_name_standardized, ref_min, ref_max, pct_lab_name, lab_specs):
            return None, None, True

    # Strategy 3: Check if range belongs to absolute variant (for percentage labs)
    if lab_name_standardized.endswith("(%)"):
        base_lab_name = lab_name_standardized[:-4].strip()
        if _try_match_cross_variant_range(lab_name_standardized, ref_min, ref_max, base_lab_name, lab_specs):
            return None, None, True

    # No conversion strategy matched
    return None, None, False


def fix_misassigned_percentage_units(df: pd.DataFrame, lab_specs: LabSpecsConfig) -> pd.DataFrame:
    """
    Fix values where unit is g/dL but should be % based on biological plausibility.

    This handles protein electrophoresis cases where the source document shows
    g/dL in the unit column but the value and reference range indicate it's
    actually a percentage (e.g., Albumin 61.5 "g/dL" with ref 55-64 is clearly 61.5%).

    Detection criteria (ALL must be true):
    1. Lab has both absolute (g/dL) and percentage (%) variants in lab_specs
    2. Current unit is g/dL (or similar absolute unit)
    3. Value exceeds biological_max for the absolute variant
    4. Value fits within percentage variant's expected range
    5. Reference range (if present) matches percentage variant's range

    Args:
        df: DataFrame with lab results
        lab_specs: Lab specifications configuration

    Returns:
        DataFrame with corrected units and lab names
    """

    # Skip empty data or missing specs
    if df.empty or not lab_specs.exists:
        return df

    corrections_made = 0

    for idx in df.index:
        lab_name_std = df.at[idx, "lab_name_standardized"]
        value = df.at[idx, "value_raw"]
        unit = df.at[idx, "lab_unit_standardized"]
        ref_min = df.at[idx, "reference_min_raw"]
        ref_max = df.at[idx, "reference_max_raw"]

        # Skip if missing key data
        if pd.isna(lab_name_std) or pd.isna(value) or pd.isna(unit):
            continue

        # Skip if already a percentage variant
        if lab_name_std.endswith("(%)"):
            continue

        # Skip if no percentage variant exists
        pct_lab_name = f"{lab_name_std} (%)"
        if pct_lab_name not in lab_specs._specs:
            continue

        # Get configs for both variants
        abs_config = lab_specs._specs.get(lab_name_std, {})
        pct_config = lab_specs._specs.get(pct_lab_name, {})

        # Skip if current unit is not an absolute unit (g/dL, g/L, etc.)
        abs_primary_unit = abs_config.get("primary_unit", "")
        if unit not in [abs_primary_unit, "g/dL", "g/L"]:
            continue

        # Convert value to float for comparison
        try:
            value_float = float(str(value).replace(",", "."))
        except (ValueError, TypeError):
            continue

        # Check 1: Value exceeds biological_max for absolute variant
        bio_max = abs_config.get("biological_max")

        # Fall back to expected range max * 3 when biological_max not configured
        if bio_max is None:
            abs_ranges = abs_config.get("ranges", {}).get("default", [])
            if len(abs_ranges) >= 2:
                bio_max = abs_ranges[1] * 3

        # Value is plausible for absolute unit
        if bio_max is None or value_float <= bio_max:
            continue

        # Check 2: Value fits percentage range
        pct_ranges = pct_config.get("ranges", {}).get("default", [])

        # Not enough percentage range data
        if len(pct_ranges) < 2:
            continue

        pct_min, pct_max = pct_ranges[0], pct_ranges[1]
        # Allow some margin (0.5x to 1.5x) for percentage range
        if not (pct_min * 0.5 <= value_float <= pct_max * 1.5):
            continue  # Value doesn't fit percentage range either

        # Check 3: Reference range matches percentage range (if present)
        # When reference values exist, verify they match the percentage variant
        if pd.notna(ref_min) and pd.notna(ref_max):
            # Reference range should be close to percentage expected range
            ref_matches_pct = abs(ref_min - pct_min) / max(pct_min, 1) < 0.5 and abs(ref_max - pct_max) / max(pct_max, 1) < 0.5
            if not ref_matches_pct:
                continue  # Reference range doesn't match percentage variant

        # All checks passed - fix the unit and lab name
        logger.info(f"[unit_fix] Correcting {lab_name_std}: value {value_float} with unit '{unit}' exceeds biological max ({bio_max}), reassigning to {pct_lab_name} with unit '%'")

        df.at[idx, "lab_unit_standardized"] = "%"
        df.at[idx, "lab_name_standardized"] = pct_lab_name
        corrections_made += 1

    # Log summary of corrections
    if corrections_made > 0:
        logger.info(f"[unit_fix] Corrected {corrections_made} misassigned percentage units")

    return df


def correct_percentage_lab_names(results: list[dict], lab_specs: LabSpecsConfig) -> list[dict]:
    """Correct lab names based on unit: add (%) when unit is %, remove (%) when unit is not %.

    This handles cases where:
    1. Unit is "%" but name doesn't end with "(%) " -> add "(%) "
    2. Unit is NOT "%" but name ends with "(%) " -> remove "(%) " (for absolute counts)
    """

    corrected_to_pct = 0
    corrected_to_abs = 0

    for result in results:
        std_name = result.get("lab_name_standardized")
        std_unit = result.get("lab_unit_standardized")

        # Skip results with missing lab name
        if not std_name:
            continue

        # Case 1: Unit is % but name doesn't have (%) -> add it
        if std_unit == "%" and not std_name.endswith("(%)"):
            percentage_variant = lab_specs.get_percentage_variant(std_name)
            if percentage_variant:
                logger.debug(f"Correcting lab name '{std_name}' -> '{percentage_variant}' (unit is %)")
                result["lab_name_standardized"] = percentage_variant
                corrected_to_pct += 1

        # Case 2: Unit is NOT % but name has (%) -> remove it (for absolute counts)
        elif std_unit != "%" and std_name.endswith("(%)"):
            non_percentage_variant = lab_specs.get_non_percentage_variant(std_name)
            if non_percentage_variant:
                logger.debug(f"Correcting lab name '{std_name}' -> '{non_percentage_variant}' (unit is {std_unit})")
                result["lab_name_standardized"] = non_percentage_variant
                corrected_to_abs += 1

    # Log summary of corrections
    if corrected_to_pct > 0 or corrected_to_abs > 0:
        logger.info(f"Corrected {corrected_to_pct} to percentage, {corrected_to_abs} to absolute lab names")

    return results


def _correct_percentage_names_in_df(df: pd.DataFrame, lab_specs: LabSpecsConfig) -> pd.DataFrame:
    """Correct lab names in a DataFrame based on unit: add (%) when unit is %, remove (%) when not.

    DataFrame equivalent of correct_percentage_lab_names (which operates on list[dict]).
    """

    corrected_to_pct = 0
    corrected_to_abs = 0

    for idx in df.index:
        std_name = df.at[idx, "lab_name_standardized"]
        std_unit = df.at[idx, "lab_unit_standardized"]

        # Skip rows with missing lab name
        if not std_name or pd.isna(std_name):
            continue

        # Unit is % but name doesn't have (%) → add it
        if std_unit == "%" and not std_name.endswith("(%)"):
            pct_variant = lab_specs.get_percentage_variant(std_name)
            if pct_variant:
                df.at[idx, "lab_name_standardized"] = pct_variant
                corrected_to_pct += 1

        # Unit is NOT % but name has (%) → remove it
        elif std_unit != "%" and not pd.isna(std_unit) and std_name.endswith("(%)"):
            abs_variant = lab_specs.get_non_percentage_variant(std_name)
            if abs_variant:
                df.at[idx, "lab_name_standardized"] = abs_variant
                corrected_to_abs += 1

    # Log summary of corrections
    if corrected_to_pct > 0 or corrected_to_abs > 0:
        logger.info(f"[normalization] Corrected {corrected_to_pct} to percentage, {corrected_to_abs} to absolute lab names")

    return df


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
        ["lab_name_standardized", "lab_unit_standardized", "lab_name_raw"],
        default=None,
    )

    # Correct percentage lab names based on unit
    # unit=% → name must end with (%), unit≠% → strip (%) from name
    if lab_specs.exists and "lab_name_standardized" in df.columns and "lab_unit_standardized" in df.columns:
        df = _correct_percentage_names_in_df(df, lab_specs)

    # Fix misassigned percentage units BEFORE other normalizations
    # This catches cases like Albumin 61.5 "g/dL" that should be 61.5 "%"
    if lab_specs.exists:
        df = fix_misassigned_percentage_units(df, lab_specs)

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
        df["value_primary"] = df.get("value_raw")
        df["lab_unit_primary"] = df.get("lab_unit_standardized")
        df["reference_min_primary"] = df.get("reference_min_raw")
        df["reference_max_primary"] = df.get("reference_max_raw")

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
    comparison_results = df["value_raw"].apply(extract_comparison_value)
    df["_preprocessed_value"] = comparison_results.apply(lambda x: x[0])
    df["is_below_limit"] = comparison_results.apply(lambda x: x[1])
    df["is_above_limit"] = comparison_results.apply(lambda x: x[2])

    # Apply additional preprocessing (spaces, trailing =, embedded metadata, comma→period)
    df["_preprocessed_value"] = df["_preprocessed_value"].apply(preprocess_numeric_value)

    # Convert preprocessed values to numeric
    df["value_primary"] = pd.to_numeric(df["_preprocessed_value"], errors="coerce")

    # Clean up temporary column
    df.drop(columns=["_preprocessed_value"], inplace=True)

    # Log preprocessing results
    limit_count = df["is_below_limit"].sum() + df["is_above_limit"].sum()
    if limit_count > 0:
        logger.info(f"[normalization] Extracted {limit_count} comparison operators (</>)")

    return df


def _infer_missing_units(df: pd.DataFrame, lab_specs: LabSpecsConfig) -> pd.DataFrame:
    """
    Infer missing unit values from context and lab specifications.

    Uses value magnitude, reference ranges, and lab_specs configuration
    to determine appropriate units for rows with missing lab_unit_standardized.
    """

    missing_unit_mask = df["lab_unit_standardized"].isna()

    # No missing units to infer
    if not missing_unit_mask.any():
        return df

    logger.info(f"[normalization] Attempting to infer {missing_unit_mask.sum()} missing units")
    inferred_count = 0
    percentage_remap_count = 0

    for idx in df[missing_unit_mask].index:
        lab_name_std = df.at[idx, "lab_name_standardized"]
        value = df.at[idx, "value_primary"]
        ref_min = df.at[idx, "reference_min_raw"]
        ref_max = df.at[idx, "reference_max_raw"]

        inferred_unit = infer_missing_unit(lab_name_std, value, ref_min, ref_max, lab_specs)

        # Could not infer a unit for this row
        if not inferred_unit:
            continue

        df.at[idx, "lab_unit_standardized"] = inferred_unit
        inferred_count += 1

        # If we inferred percentage and percentage variant exists, update lab name
        if inferred_unit == "%" and not lab_name_std.endswith("(%)"):
            potential_pct_name = f"{lab_name_std} (%)"
            if lab_specs.exists and potential_pct_name in lab_specs._specs:
                df.at[idx, "lab_name_standardized"] = potential_pct_name
                percentage_remap_count += 1

    # Log summary of inference results
    if inferred_count > 0:
        logger.info(f"[normalization] Successfully inferred {inferred_count} units")

    if percentage_remap_count > 0:
        logger.info(f"[normalization] Remapped {percentage_remap_count} tests to percentage variants")

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
        "turvo",
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

    # Check negative patterns
    if any(normalized.startswith(p) for p in _NEGATIVE_PREFIXES):
        return 0
    if normalized in _NEGATIVE_KEYWORDS:
        return 0

    # Not classifiable — return None (value stays in value_raw for review)
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

    # Boolean labs: convert qualitative text to 0/1 from value_raw, then comments
    if boolean_labs:
        for source_col, needs_val_raw_isna in [
            ("value_raw", False),
            ("comments", True),
        ]:
            mask = df["lab_name_standardized"].isin(boolean_labs) & df["value_primary"].isna() & df[source_col].notna()

            # For comments source, only use when value_raw is also missing
            if needs_val_raw_isna:
                mask &= df["value_raw"].isna()

            # No qualifying rows for this source
            if not mask.any():
                continue

            qual_map = _classify_qualitative_batch(df.loc[mask, source_col].tolist())
            df.loc[mask, "value_primary"] = df.loc[mask, source_col].map(lambda v: qual_map.get(v))
            df.loc[mask, "lab_unit_primary"] = "boolean"
            logger.info(f"[normalization] Converted {mask.sum()} qualitative values from {source_col} (boolean labs)")

    # Non-boolean labs: convert only negative-like values to 0 from value_raw, then comments
    non_boolean_exclude = boolean_labs if boolean_labs else []

    for source_col, needs_val_raw_isna in [
        ("value_raw", False),
        ("comments", True),
    ]:
        mask = df["value_primary"].isna() & df[source_col].notna() & ~df["lab_name_standardized"].isin(non_boolean_exclude)

        # For comments source, only use when value_raw is also missing
        if needs_val_raw_isna:
            mask &= df["value_raw"].isna()

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
        df.loc[unit_mask, "reference_min_primary"] = df.loc[unit_mask, "reference_min_raw"] * factor
        df.loc[unit_mask, "reference_max_primary"] = df.loc[unit_mask, "reference_max_raw"] * factor
        df.loc[unit_mask, "lab_unit_primary"] = primary_unit


def _fix_specific_gravity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix specific gravity values missing decimal point (1012 → 1.012).

    Some labs report specific gravity as integer (1000-1040) instead of decimal.
    Detects and corrects these values by dividing by 1000.
    """

    specific_gravity_labs = [
        "Urine Type II - Specific Gravity",
        "Urine - Specific Gravity",
    ]
    sg_mask = df["lab_name_standardized"].isin(specific_gravity_labs) & df["value_primary"].notna() & (df["value_primary"] > 100)

    # Divide integer-format values by 1000 to restore decimal form
    if sg_mask.any():
        df.loc[sg_mask, "value_primary"] = df.loc[sg_mask, "value_primary"] / 1000
        logger.info(f"[normalization] Fixed {sg_mask.sum()} specific gravity values (divided by 1000)")

    return df


def _validate_reference_ranges(df: pd.DataFrame, lab_specs: LabSpecsConfig) -> pd.DataFrame:
    """
    Validate and correct reference ranges against lab_specs.

    Detects suspicious reference ranges (wrong units, impossible values)
    and either converts them using known factors or nullifies them.
    """

    # Skip if no lab specs available
    if not lab_specs.exists:
        return df

    validation_count = 0

    for idx in df.index:
        lab_name_std = df.at[idx, "lab_name_standardized"]
        value = df.at[idx, "value_primary"]
        unit = df.at[idx, "lab_unit_primary"]
        ref_min = df.at[idx, "reference_min_primary"]
        ref_max = df.at[idx, "reference_max_primary"]

        # Skip rows with no reference range
        if pd.isna(ref_min) and pd.isna(ref_max):
            continue

        validated_min, validated_max = validate_reference_range(lab_name_std, value, unit, ref_min, ref_max, lab_specs)

        # Update if validation changed the range
        if validated_min != ref_min or validated_max != ref_max:
            df.at[idx, "reference_min_primary"] = validated_min
            df.at[idx, "reference_max_primary"] = validated_max
            validation_count += 1

    # Log summary of validation results
    if validation_count > 0:
        logger.info(f"[normalization] Nullified {validation_count} suspicious reference ranges")

    return df


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
    df["reference_min_primary"] = df["reference_min_raw"]
    df["reference_max_primary"] = df["reference_max_raw"]

    # Phase 1: Extract comparison operators and preprocess values
    df = _preprocess_values(df)

    # Phase 2: Infer missing units from context
    df = _infer_missing_units(df, lab_specs)

    # Phase 3: Convert qualitative text values to 0/1
    df = _convert_qualitative_values(df, lab_specs)

    # Phase 4: Apply unit conversion factors for each lab type
    for lab_name_standardized in df["lab_name_standardized"].unique():
        # Skip unknown or missing lab names
        if pd.isna(lab_name_standardized) or lab_name_standardized == UNKNOWN_VALUE:
            continue
        _apply_unit_conversions_for_lab(df, lab_name_standardized, lab_specs)

    # Phase 5: Fix specific gravity values missing decimal point
    df = _fix_specific_gravity(df)

    # Phase 6: Validate reference ranges against lab_specs
    df = _validate_reference_ranges(df, lab_specs)

    # Phase 7: Sanitize percentage reference ranges
    df = sanitize_percentage_reference_ranges(df)

    return df


def sanitize_percentage_reference_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Discard reference ranges that are clearly in the wrong unit for percentage labs.

    For labs with unit '%', detects when extracted reference ranges appear to be
    in absolute count units (common for leucocyte fractions where PDFs show
    absolute count ranges alongside percentage values).

    Detection heuristic:
    - If value > ref_max * 5: range is clearly wrong (e.g., 77% vs range 2.1-7.6)
    - If ref_max > 100: invalid percentage range

    Args:
        df: DataFrame with lab results containing reference range columns

    Returns:
        DataFrame with suspicious reference ranges nullified
    """

    # Nothing to sanitize
    if df.empty:
        return df

    # Determine column names based on what's available
    unit_col = "lab_unit_primary" if "lab_unit_primary" in df.columns else "lab_unit_standardized"
    value_col = "value_primary" if "value_primary" in df.columns else "value_raw"
    ref_min_col = "reference_min_primary" if "reference_min_primary" in df.columns else "reference_min_raw"
    ref_max_col = "reference_max_primary" if "reference_max_primary" in df.columns else "reference_max_raw"

    # Required columns not present
    if unit_col not in df.columns or value_col not in df.columns:
        return df

    # Find percentage labs
    percentage_mask = df[unit_col] == "%"

    # No percentage labs to check
    if not percentage_mask.any():
        return df

    # Condition 1: value > ref_max * 5 (clear unit mismatch)
    # Condition 2: ref_max > 100 (invalid percentage)
    suspicious_mask = percentage_mask & (df[ref_max_col].notna() & ((df[value_col].notna() & (df[value_col] > df[ref_max_col] * 5)) | (df[ref_max_col] > 100)))

    # Nullify suspicious percentage reference ranges
    if suspicious_mask.any():
        count = suspicious_mask.sum()
        logger.info(f"[normalization] Discarding {count} suspicious reference ranges for percentage labs (likely wrong unit)")

        # Nullify the reference ranges
        df.loc[suspicious_mask, ref_min_col] = pd.NA
        df.loc[suspicious_mask, ref_max_col] = pd.NA

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

    # Find all rows that are part of a duplicate group
    dup_mask = df.duplicated(subset=["date", "lab_name_standardized"], keep=False)

    if not dup_mask.any():
        return df

    count = int(dup_mask.sum())
    logger.info(f"Flagging {count} rows as DUPLICATE_ENTRY")

    df.loc[dup_mask, "review_needed"] = True
    df.loc[dup_mask, "review_reason"] = df.loc[dup_mask, "review_reason"].fillna("").apply(lambda x: str(x) + "DUPLICATE_ENTRY; " if "DUPLICATE_ENTRY" not in str(x) else str(x))

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

    # Skip if missing required columns
    if df.empty or "date" not in df.columns or "lab_name_standardized" not in df.columns:
        return df

    def pick_best_dupe(group):
        """Pick best duplicate: prefer primary unit if multiple entries exist."""

        lab_name_standardized = group.iloc[0]["lab_name_standardized"]
        primary_unit = lab_specs.get_primary_unit(lab_name_standardized) if lab_specs.exists else None

        # Prefer the row already in primary unit
        if primary_unit and "lab_unit_standardized" in group.columns and (group["lab_unit_standardized"] == primary_unit).any():
            return group[group["lab_unit_standardized"] == primary_unit].iloc[0]
        # Fall back to first row
        else:
            return group.iloc[0]

    # Group and apply deduplication
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        deduplicated = (
            df.groupby(
                ["date", "lab_name_standardized"],
                dropna=False,
                as_index=False,
                group_keys=False,
            )
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
        # Skip columns not present in DataFrame
        if col not in df.columns:
            continue

        try:
            # Convert datetime columns
            if target_dtype == "datetime64[ns]":
                df[col] = pd.to_datetime(df[col], errors="coerce")
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
