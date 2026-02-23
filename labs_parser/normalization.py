"""DataFrame normalization and transformation logic."""

import logging
import re
import pandas as pd

from openai import OpenAI

from labs_parser.config import LabSpecsConfig, UNKNOWN_VALUE
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
    if pd.isna(value):
        return str(value), False, False

    s = str(value).strip()
    is_below = s.startswith("<")
    is_above = s.startswith(">")

    if is_below or is_above:
        # Extract numeric part: "<0.05" → "0.05", "< 10" → "10"
        numeric = re.sub(r"^[<>]\s*", "", s)
        # Handle cases like "<20=NR" → "20"
        numeric = numeric.split("=")[0].strip()
        return numeric, is_below, is_above

    return s, False, False


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

    # Decision logic
    if has_large_range and has_percentage_variant and value_matches_range:
        # Strong evidence this is a percentage
        logger.debug(
            f"[unit_inference] Inferring '%' for {lab_name_standardized}: "
            f"value={value}, ref_range=({ref_min}, {ref_max})"
        )
        return "%"

    # Strategy 3b: Biological plausibility check
    # Before using primary unit, verify value is plausible for that unit
    # This catches cases where ref ranges are null but value is clearly a percentage
    if lab_specs.exists and has_percentage_variant and value is not None:
        primary_unit = lab_specs.get_primary_unit(lab_name_standardized)
        expected_ranges = lab_specs._specs.get(lab_name_standardized, {}).get(
            "ranges", {}
        )
        expected_default = expected_ranges.get("default", [])

        if len(expected_default) >= 2:
            expected_min, expected_max = expected_default[0], expected_default[1]

            # Check if value is impossibly high for primary unit (>5x max)
            if expected_max and value > expected_max * 5:
                # Check if value fits percentage range instead
                pct_ranges = lab_specs._specs.get(percentage_lab_name, {}).get(
                    "ranges", {}
                )
                pct_default = pct_ranges.get("default", [])

                if len(pct_default) >= 2:
                    pct_min, pct_max = pct_default[0], pct_default[1]
                    # Allow some margin (0.5x to 2x) for percentage range
                    if (
                        pct_min
                        and pct_max
                        and (pct_min * 0.5) <= value <= (pct_max * 2)
                    ):
                        logger.debug(
                            f"[unit_inference] Value {value} implausible for {primary_unit} "
                            f"(expected {expected_min}-{expected_max}), assigning '%' instead"
                        )
                        return "%"

    # Strategy 4: Fallback to lab specs primary unit
    if lab_specs.exists:
        primary_unit = lab_specs.get_primary_unit(lab_name_standardized)
        if primary_unit:
            logger.debug(
                f"[unit_inference] Using primary unit '{primary_unit}' from lab_specs for {lab_name_standardized}"
            )
            return primary_unit

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
    if not lab_specs.exists:
        return ref_min, ref_max

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

    if ref_min is None or ref_max is None:
        return ref_min, ref_max

    # Get expected range for this lab from lab_specs
    ranges = lab_config.get("ranges", {})
    if not ranges:
        return ref_min, ref_max

    # Try to get expected range for this unit or default
    expected_range = None
    if "default" in ranges:
        expected_range = ranges["default"]
    elif unit and unit in ranges:
        expected_range = ranges[unit]

    if not expected_range or len(expected_range) < 2:
        return ref_min, ref_max

    expected_min, expected_max = expected_range[0], expected_range[1]

    # If unit differs from primary unit, convert expected range using conversion factor
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

    if (
        expected_min is None
        or expected_max is None
        or expected_min == 0
        or expected_max == 0
    ):
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
    is_suspicious = (ratio_min > THRESHOLD_HIGH or ratio_min < THRESHOLD_LOW) and (
        ratio_max > THRESHOLD_HIGH or ratio_max < THRESHOLD_LOW
    )

    if is_suspicious:
        # Try to find a conversion factor that would fix the range
        converted_ref_min, converted_ref_max, was_explained = (
            _try_convert_mismatched_range(
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
        )

        if converted_ref_min is not None and converted_ref_max is not None:
            return converted_ref_min, converted_ref_max

        # Only log warning if the issue wasn't already explained
        if not was_explained:
            logger.error(
                f"[range_validation] Suspicious reference range for {lab_name_standardized} ({unit}): "
                f"PDF=({ref_min:.2f}, {ref_max:.2f}), expected≈({expected_min}, {expected_max}), "
                f"ratios=({ratio_min:.2f}x, {ratio_max:.2f}x) - nullifying range"
            )
        return None, None

    return ref_min, ref_max


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
        if factor is None or factor == 0:
            continue

        # The PDF range might be in this alternative unit
        # Conversion factor converts FROM alternative unit TO primary unit
        # So if PDF range / expected range ≈ 1/factor, the PDF range is in alternative unit
        expected_ratio = 1.0 / factor

        # Check if both ratios match this conversion factor
        if (
            abs(ratio_min - expected_ratio) / expected_ratio < TOLERANCE
            and abs(ratio_max - expected_ratio) / expected_ratio < TOLERANCE
        ):
            # Apply conversion: multiply by factor to convert to primary unit
            converted_min = ref_min * factor
            converted_max = ref_max * factor
            logger.info(
                f"[range_validation] Converted reference range for {lab_name_standardized}: "
                f"PDF=({ref_min:.2f}, {ref_max:.2f}) -> ({converted_min:.2f}, {converted_max:.2f}) "
                f"using factor {factor} (detected {alt.get('unit')} unit)"
            )
            return converted_min, converted_max, False

    # Strategy 2: Check for percentage/absolute value mix-up (protein electrophoresis)
    # If lab is "Blood - X" (g/dL) and there's "Blood - X (%)", check if range fits %
    if not lab_name_standardized.endswith("(%)"):
        pct_lab_name = f"{lab_name_standardized} (%)"
        if pct_lab_name in lab_specs._specs:
            pct_config = lab_specs._specs[pct_lab_name]
            pct_ranges = pct_config.get("ranges", {}).get("default", [])
            if len(pct_ranges) >= 2:
                pct_expected_min, pct_expected_max = pct_ranges[0], pct_ranges[1]
                # Check if PDF range matches percentage expected range
                if pct_expected_min > 0 and pct_expected_max > 0:
                    pct_ratio_min = (
                        abs(ref_min / pct_expected_min) if pct_expected_min != 0 else 0
                    )
                    pct_ratio_max = (
                        abs(ref_max / pct_expected_max) if pct_expected_max != 0 else 0
                    )
                    # If ratios are close to 1, the range is from the percentage variant
                    # Use 0.3-3.0 tolerance to handle slight lab-to-lab variations
                    if 0.3 < pct_ratio_min < 3.0 and 0.3 < pct_ratio_max < 3.0:
                        logger.info(
                            f"[range_validation] Detected percentage range for {lab_name_standardized}: "
                            f"PDF=({ref_min:.2f}, {ref_max:.2f}) matches {pct_lab_name} range "
                            f"({pct_expected_min}, {pct_expected_max}) - nullifying (cannot convert % to g/dL)"
                        )
                        return None, None, True

    # Strategy 3: Check reverse - if lab is "Blood - X (%)" and range fits g/dL variant
    if lab_name_standardized.endswith("(%)"):
        base_lab_name = lab_name_standardized[:-4].strip()  # Remove " (%)"
        if base_lab_name in lab_specs._specs:
            base_config = lab_specs._specs[base_lab_name]
            base_ranges = base_config.get("ranges", {}).get("default", [])
            if len(base_ranges) >= 2:
                base_expected_min, base_expected_max = base_ranges[0], base_ranges[1]
                # Check if PDF range matches absolute expected range
                if base_expected_min > 0 and base_expected_max > 0:
                    base_ratio_min = (
                        abs(ref_min / base_expected_min)
                        if base_expected_min != 0
                        else 0
                    )
                    base_ratio_max = (
                        abs(ref_max / base_expected_max)
                        if base_expected_max != 0
                        else 0
                    )
                    # If ratios are close to 1, the range is from the absolute variant
                    # Use 0.3-3.0 tolerance to handle slight lab-to-lab variations
                    if 0.3 < base_ratio_min < 3.0 and 0.3 < base_ratio_max < 3.0:
                        logger.info(
                            f"[range_validation] Detected g/dL range for {lab_name_standardized}: "
                            f"PDF=({ref_min:.2f}, {ref_max:.2f}) matches {base_lab_name} range "
                            f"({base_expected_min}, {base_expected_max}) - nullifying (cannot convert g/dL to %)"
                        )
                        return None, None, True

    return None, None, False


def fix_misassigned_percentage_units(
    df: pd.DataFrame, lab_specs: LabSpecsConfig
) -> pd.DataFrame:
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

        # Check if percentage variant exists
        pct_lab_name = f"{lab_name_std} (%)"
        if pct_lab_name not in lab_specs._specs:
            continue

        # Get configs for both variants
        abs_config = lab_specs._specs.get(lab_name_std, {})
        pct_config = lab_specs._specs.get(pct_lab_name, {})

        # Check if current unit is an absolute unit (g/dL, g/L, etc.)
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
        if bio_max is None:
            # Fall back to expected range max * 3
            abs_ranges = abs_config.get("ranges", {}).get("default", [])
            if len(abs_ranges) >= 2:
                bio_max = abs_ranges[1] * 3

        if bio_max is None or value_float <= bio_max:
            continue  # Value is plausible for absolute unit

        # Check 2: Value fits percentage range
        pct_ranges = pct_config.get("ranges", {}).get("default", [])
        if len(pct_ranges) < 2:
            continue

        pct_min, pct_max = pct_ranges[0], pct_ranges[1]
        # Allow some margin (0.5x to 1.5x) for percentage range
        if not (pct_min * 0.5 <= value_float <= pct_max * 1.5):
            continue  # Value doesn't fit percentage range either

        # Check 3: Reference range matches percentage range (if present)
        if pd.notna(ref_min) and pd.notna(ref_max):
            # Reference range should be close to percentage expected range
            ref_matches_pct = (
                abs(ref_min - pct_min) / max(pct_min, 1) < 0.5
                and abs(ref_max - pct_max) / max(pct_max, 1) < 0.5
            )
            if not ref_matches_pct:
                continue  # Reference range doesn't match percentage variant

        # All checks passed - fix the unit and lab name
        logger.info(
            f"[unit_fix] Correcting {lab_name_std}: value {value_float} with unit '{unit}' "
            f"exceeds biological max ({bio_max}), reassigning to {pct_lab_name} with unit '%'"
        )

        df.at[idx, "lab_unit_standardized"] = "%"
        df.at[idx, "lab_name_standardized"] = pct_lab_name
        corrections_made += 1

    if corrections_made > 0:
        logger.info(
            f"[unit_fix] Corrected {corrections_made} misassigned percentage units"
        )

    return df


def apply_normalizations(
    df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
    client: OpenAI | None = None,
    model_id: str | None = None,
) -> pd.DataFrame:
    """
    Apply normalization transformations to the DataFrame.

    This focuses on unit conversions to enable cross-provider comparison.
    Health status calculations have been moved to the review tool.

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
    ensure_columns(
        df,
        ["lab_name_standardized", "lab_unit_standardized", "lab_name_raw"],
        default=None,
    )

    # Fix misassigned percentage units BEFORE other normalizations
    # This catches cases like Albumin 61.5 "g/dL" that should be 61.5 "%"
    if lab_specs.exists:
        df = fix_misassigned_percentage_units(df, lab_specs)

    # Look up lab_type from config (vectorized)
    if lab_specs.exists:
        df["lab_type"] = df["lab_name_standardized"].apply(
            lambda name: lab_specs.get_lab_type(name)
            if pd.notna(name) and name != UNKNOWN_VALUE
            else "blood"
        )
    else:
        df["lab_type"] = "blood"

    # Convert to primary units (batched)
    if lab_specs.exists:
        df = apply_unit_conversions(df, lab_specs, client, model_id)
    else:
        # No conversions, just copy values
        df["value_primary"] = df.get("value_raw")
        df["lab_unit_primary"] = df.get("lab_unit_standardized")
        df["reference_min_primary"] = df.get("reference_min_raw")
        df["reference_max_primary"] = df.get("reference_max_raw")

    return df


def apply_unit_conversions(
    df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
    client: OpenAI | None = None,
    model_id: str | None = None,
) -> pd.DataFrame:
    """
    Convert values to primary units defined in lab specs.

    This function applies unit conversions in a vectorized manner where possible.
    Handles boolean tests by converting qualitative text values (e.g., "negativo", "positivo")
    to 0/1 using LLM-based classification with caching.
    """
    # Initialize limit indicator columns
    df["is_below_limit"] = False
    df["is_above_limit"] = False

    # Extract comparison operators and preprocess values
    # Apply extract_comparison_value to get numeric part and limit flags
    comparison_results = df["value_raw"].apply(extract_comparison_value)
    df["_preprocessed_value"] = comparison_results.apply(lambda x: x[0])
    df["is_below_limit"] = comparison_results.apply(lambda x: x[1])
    df["is_above_limit"] = comparison_results.apply(lambda x: x[2])

    # Apply additional preprocessing (spaces, trailing =, embedded metadata, comma→period)
    df["_preprocessed_value"] = df["_preprocessed_value"].apply(
        preprocess_numeric_value
    )

    # Convert preprocessed values to numeric
    df["value_primary"] = pd.to_numeric(df["_preprocessed_value"], errors="coerce")

    # Clean up temporary column
    df.drop(columns=["_preprocessed_value"], inplace=True)

    # Log preprocessing results
    limit_count = df["is_below_limit"].sum() + df["is_above_limit"].sum()
    if limit_count > 0:
        logger.info(
            f"[normalization] Extracted {limit_count} comparison operators (</>)"
        )

    # Infer missing units before unit conversion
    missing_unit_mask = df["lab_unit_standardized"].isna()
    if missing_unit_mask.any():
        logger.info(
            f"[normalization] Attempting to infer {missing_unit_mask.sum()} missing units"
        )
        inferred_count = 0
        percentage_remap_count = 0

        for idx in df[missing_unit_mask].index:
            lab_name_std = df.at[idx, "lab_name_standardized"]
            value = df.at[idx, "value_primary"]
            ref_min = df.at[idx, "reference_min_raw"]
            ref_max = df.at[idx, "reference_max_raw"]

            inferred_unit = infer_missing_unit(
                lab_name_std, value, ref_min, ref_max, lab_specs
            )
            if inferred_unit:
                df.at[idx, "lab_unit_standardized"] = inferred_unit
                inferred_count += 1

                # If we inferred percentage and percentage variant exists, update lab name
                if inferred_unit == "%" and not lab_name_std.endswith("(%)"):
                    potential_pct_name = f"{lab_name_std} (%)"
                    if lab_specs.exists and potential_pct_name in lab_specs._specs:
                        df.at[idx, "lab_name_standardized"] = potential_pct_name
                        percentage_remap_count += 1

        if inferred_count > 0:
            logger.info(f"[normalization] Successfully inferred {inferred_count} units")
        if percentage_remap_count > 0:
            logger.info(
                f"[normalization] Remapped {percentage_remap_count} tests to percentage variants"
            )

    df["lab_unit_primary"] = df["lab_unit_standardized"]
    df["reference_min_primary"] = df["reference_min_raw"]
    df["reference_max_primary"] = df["reference_max_raw"]

    # Handle boolean tests: convert qualitative values to 0/1
    if client and model_id:
        # Import here to avoid circular imports
        from standardization import standardize_qualitative_values

        # Find labs with boolean primary unit
        boolean_labs = [
            lab_name
            for lab_name in df["lab_name_standardized"].unique()
            if pd.notna(lab_name)
            and lab_name != UNKNOWN_VALUE
            and lab_specs.get_primary_unit(lab_name) == "boolean"
        ]

        if boolean_labs:
            # Boolean labs: convert qualitative text to 0/1 from value_raw then comments
            for source_col, needs_val_raw_isna in [
                ("value_raw", False),
                ("comments", True),
            ]:
                mask = (
                    df["lab_name_standardized"].isin(boolean_labs)
                    & df["value_primary"].isna()
                    & df[source_col].notna()
                )
                if needs_val_raw_isna:
                    mask &= df["value_raw"].isna()
                if mask.any():
                    qual_map = standardize_qualitative_values(
                        df.loc[mask, source_col].tolist(), model_id, client
                    )
                    df.loc[mask, "value_primary"] = df.loc[mask, source_col].map(
                        lambda v: qual_map.get(v)
                    )
                    df.loc[mask, "lab_unit_primary"] = "boolean"
                    logger.info(
                        f"[normalization] Converted {mask.sum()} qualitative values from {source_col} (boolean labs)"
                    )

        # Non-boolean labs: convert only negative-like values to 0 from value_raw then comments
        non_boolean_exclude = boolean_labs if boolean_labs else []
        for source_col, needs_val_raw_isna in [
            ("value_raw", False),
            ("comments", True),
        ]:
            mask = (
                df["value_primary"].isna()
                & df[source_col].notna()
                & ~df["lab_name_standardized"].isin(non_boolean_exclude)
            )
            if needs_val_raw_isna:
                mask &= df["value_raw"].isna()
            if mask.any():
                qual_map = standardize_qualitative_values(
                    df.loc[mask, source_col].tolist(), model_id, client
                )
                converted_count = 0
                for idx in df.loc[mask].index:
                    if qual_map.get(df.at[idx, source_col]) == 0:
                        df.at[idx, "value_primary"] = 0.0
                        converted_count += 1
                if converted_count > 0:
                    logger.info(
                        f"[normalization] Converted {converted_count} negative-like {source_col} to 0 (non-boolean labs)"
                    )

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
            df.loc[unit_mask, "value_primary"] = (
                df.loc[unit_mask, "value_primary"] * factor
            )
            df.loc[unit_mask, "reference_min_primary"] = (
                df.loc[unit_mask, "reference_min_raw"] * factor
            )
            df.loc[unit_mask, "reference_max_primary"] = (
                df.loc[unit_mask, "reference_max_raw"] * factor
            )
            df.loc[unit_mask, "lab_unit_primary"] = primary_unit

    # Fix specific gravity values that are missing decimal point (1012 → 1.012)
    # Some labs report specific gravity as integer (1000-1040) instead of decimal (1.000-1.040)
    specific_gravity_labs = [
        "Urine Type II - Specific Gravity",
        "Urine - Specific Gravity",
    ]
    sg_mask = (
        df["lab_name_standardized"].isin(specific_gravity_labs)
        & df["value_primary"].notna()
        & (df["value_primary"] > 100)  # Values like 1012, 1020 need fixing
    )
    if sg_mask.any():
        df.loc[sg_mask, "value_primary"] = df.loc[sg_mask, "value_primary"] / 1000
        logger.info(
            f"[normalization] Fixed {sg_mask.sum()} specific gravity values (divided by 1000)"
        )

    # Validate reference ranges against lab_specs (detect wrong-unit assignments)
    if lab_specs.exists:
        validation_count = 0
        for idx in df.index:
            lab_name_std = df.at[idx, "lab_name_standardized"]
            value = df.at[idx, "value_primary"]
            unit = df.at[idx, "lab_unit_primary"]
            ref_min = df.at[idx, "reference_min_primary"]
            ref_max = df.at[idx, "reference_max_primary"]

            if pd.notna(ref_min) or pd.notna(ref_max):
                validated_min, validated_max = validate_reference_range(
                    lab_name_std, value, unit, ref_min, ref_max, lab_specs
                )

                if validated_min != ref_min or validated_max != ref_max:
                    df.at[idx, "reference_min_primary"] = validated_min
                    df.at[idx, "reference_max_primary"] = validated_max
                    validation_count += 1

        if validation_count > 0:
            logger.info(
                f"[normalization] Nullified {validation_count} suspicious reference ranges"
            )

    # Sanitize percentage reference ranges (discard wrong-unit ranges)
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
    if df.empty:
        return df

    unit_col = (
        "lab_unit_primary"
        if "lab_unit_primary" in df.columns
        else "lab_unit_standardized"
    )
    value_col = "value_primary" if "value_primary" in df.columns else "value_raw"
    ref_min_col = (
        "reference_min_primary"
        if "reference_min_primary" in df.columns
        else "reference_min_raw"
    )
    ref_max_col = (
        "reference_max_primary"
        if "reference_max_primary" in df.columns
        else "reference_max_raw"
    )

    if unit_col not in df.columns or value_col not in df.columns:
        return df

    # Find percentage labs
    percentage_mask = df[unit_col] == "%"

    if not percentage_mask.any():
        return df

    # Condition 1: value > ref_max * 5 (clear unit mismatch)
    # Condition 2: ref_max > 100 (invalid percentage)
    suspicious_mask = percentage_mask & (
        df[ref_max_col].notna()
        & (
            (df[value_col].notna() & (df[value_col] > df[ref_max_col] * 5))
            | (df[ref_max_col] > 100)
        )
    )

    if suspicious_mask.any():
        count = suspicious_mask.sum()
        logger.info(
            f"[normalization] Discarding {count} suspicious reference ranges for percentage labs (likely wrong unit)"
        )

        # Nullify the reference ranges
        df.loc[suspicious_mask, ref_min_col] = pd.NA
        df.loc[suspicious_mask, ref_max_col] = pd.NA

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
    if (
        df.empty
        or "date" not in df.columns
        or "lab_name_standardized" not in df.columns
    ):
        return df

    def pick_best_dupe(group):
        """Pick best duplicate: prefer primary unit if multiple entries exist."""
        lab_name_standardized = group.iloc[0]["lab_name_standardized"]
        primary_unit = (
            lab_specs.get_primary_unit(lab_name_standardized)
            if lab_specs.exists
            else None
        )

        if (
            primary_unit
            and "lab_unit_standardized" in group.columns
            and (group["lab_unit_standardized"] == primary_unit).any()
        ):
            return group[group["lab_unit_standardized"] == primary_unit].iloc[0]
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
        except Exception as e:
            logger.warning(f"Dtype conversion failed for {col} to {target_dtype}: {e}")

    return df
