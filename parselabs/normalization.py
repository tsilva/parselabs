"""DataFrame normalization and transformation logic."""

import logging
import re
import unicodedata

import pandas as pd

from parselabs.config import UNKNOWN_VALUE, LabSpecsConfig
from parselabs.review_flags import append_review_reason_code, ensure_review_flag_columns
from parselabs.utils import ensure_columns

logger = logging.getLogger(__name__)

INTERVAL_VALUE_PATTERN = re.compile(
    r"^\s*(?P<min>\d+(?:\.\d+)?)\s*(?:-|a)\s*(?P<max>\d+(?:\.\d+)?)(?:\s*/.*)?\s*$",
    re.IGNORECASE,
)
TITER_VALUE_PATTERN = re.compile(r"^\s*(?:t[ií]tulo\s*)?(?P<operator>[<>])?\s*1\s*[:/]\s*(?P<denominator>\d+)\s*$", re.IGNORECASE)

QUALITATIVE_VARIANT_MAP = {
    "Blood - Anti-Tissue Transglutaminase Antibody IgA (Anti-tTG IgA)": "Blood - Anti-Tissue Transglutaminase Antibody IgA (Anti-tTG IgA) (Qualitative)",
    "Urine Type II - Blood": "Urine Type II - Blood (Qualitative)",
    "Urine Type II - Bilirubin": "Urine Type II - Bilirubin (Qualitative)",
    "Urine Type II - Glucose": "Urine Type II - Glucose (Qualitative)",
    "Urine Type II - Ketones": "Urine Type II - Ketones (Qualitative)",
    "Urine Type II - Proteins": "Urine Type II - Proteins (Qualitative)",
    "Urine Type II - Urobilinogen": "Urine Type II - Urobilinogen (Qualitative)",
}

SEDIMENT_FIELD_STYLE_LABS = {
    "Urine Type II - Sediment - Crystals",
    "Urine Type II - Sediment - Epithelial Cells",
    "Urine Type II - Sediment - Erythrocytes",
    "Urine Type II - Sediment - Leukocytes",
    "Urine Type II - Sediment - Small Round Cells",
    "Urine Type II - Sediment - Transitional epithelial cells",
    "Urine Type II - Sediment - Tubular epithelial cells",
}
SEDIMENT_ABSOLUTE_VARIANT_MAP = {
    "Urine Type II - Sediment - Crystals": "Urine Type II - Sediment - Crystals (Absolute)",
    "Urine Type II - Sediment - Epithelial Cells": "Urine Type II - Sediment - Epithelial Cells (Absolute)",
    "Urine Type II - Sediment - Erythrocytes": "Urine Type II - Sediment - Erythrocytes (Absolute)",
    "Urine Type II - Sediment - Leukocytes": "Urine Type II - Sediment - Leukocytes (Absolute)",
    "Urine Type II - Sediment - Small Round Cells": "Urine Type II - Sediment - Small Round Cells (Absolute)",
}
SEDIMENT_FIELD_STYLE_VALUE_PATTERN = re.compile(r"/\s*(?:campo|field)\b", re.IGNORECASE)


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

    # Strip approximate prefixes while preserving the numeric measurement.
    # Examples: "Ca. 15" -> "15", "aprox. 12" -> "12"
    s = re.sub(r"^(?:ca\.?|cerca de|aprox\.?|approximately)\s*", "", s, flags=re.IGNORECASE)

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

    # A reported interval is evidence, not a scalar measurement or an inferred midpoint.
    interval_mask = df["_preprocessed_value"].astype(str).str.match(INTERVAL_VALUE_PATTERN)
    df.loc[interval_mask, "_preprocessed_value"] = None

    # Convert preprocessed values to numeric
    df["value_primary"] = pd.to_numeric(df["_preprocessed_value"], errors="coerce").astype(float)

    # Preserve reported text/intervals as an explicit result, even when a boolean interpretation is available.
    numeric_literal = df["raw_value"].astype(str).str.fullmatch(
        r"\s*[<>=≤≥]?\s*[+-]?(?:\d+(?:\s+\d{3})*(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?\s*=?\s*"
    )
    text_mask = (df["value_primary"].isna() | interval_mask | ~numeric_literal) & df["raw_value"].notna()
    df["value_text"] = df["raw_value"].where(text_mask, None)

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
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))

    # Guard: empty after stripping
    if not normalized or normalized in ("nan", "none", "null"):
        return None

    # Positive patterns (detected/present/abnormal)
    _POSITIVE_PREFIXES = (
        "positiv",  # positivo, positiva, positive
        "detetad",  # detetado
        "detect",
        "reagent",
        "reactiv",
        "resultado positivo",
        "rara",
        "raras",
        "raro",
        "raros",
    )
    _POSITIVE_KEYWORDS = {
        "abundantes",
        "anisocitose",
        "anisocitose e anisocromia.",
        "anisocromia",
        "levemente turvo",
        "r",
        "trace",
        "traces",
        "traço",
        "traços",
        "turvo",
        "vestigio",
        "vestígios",
        "vestigios",
        "cultura polimicrobiana",
        "imune",
    }

    # Negative patterns (not detected/absent/normal)
    _NEGATIVE_PREFIXES = (
        "negativ",  # negativo, negativa, negative
        "nr",
        "ausent",  # ausente, ausentes, ausência
        "ausenci",  # ausencia
        "nao reagente",
        "nao reativ",  # não reativo, não reativa
        "nao det",  # nao detetado, nao det.
        "not detected",
        "non reactive",
        "nao contem",
        "nao se ",  # nao se isolaram, nao se observaram
        "nao revelou",
        "sem alterações",
        "sem alteracoes",
        "limpid",  # límpido, límpida, limpido
        "amicrobiano",
        "cultura esteril",
        "esteril",
        "sterile",
        "por hplc não foram detectadas",
        "por hplc nao foram detectadas",
        "nao foram observad",
        "exame ecográfico",  # normal findings
        "exame ecografico",
        "forma e dimensões normais",
        "forma e dimensoes normais",
        "de normal morfologia",
        "morfologia celular: normal",
        "o estudo electroforético",
        "o estudo electroforetico",
        "num exame químico com ausência",
        "num exame quimico com ausencia",
        "amarel",
        "clar",
        "transparen",
    )
    _NEGATIVE_KEYWORDS = {
        "citrina",
    }

    # Check positive patterns
    if any(normalized.startswith(p) for p in _POSITIVE_PREFIXES):
        return 1
    if normalized in _POSITIVE_KEYWORDS:
        return 1
    if "gram positivo" in normalized or "gram negativo" in normalized:
        return 1
    if normalized in {"(*)", "*"}:
        return 1
    # "raras plaquetas gigantes" is positive (abnormal finding)
    if normalized.startswith("raras plaquetas"):
        return 1
    # "observa-se mucosa cólica com folículo linfoide hiperplasiado" is positive
    if "hiperplasiado" in normalized:
        return 1
    if "distendida, apresenta paredes finas" in normalized:
        return 0
    if re.fullmatch(r"\++", normalized) or re.fullmatch(r"\d+\+", normalized):
        return 1

    # A urine-strip presence marker is qualitative; its grading must remain in value_text.
    if re.fullmatch(r"contem\s*\(\+[-+]*\)", normalized):
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
    """Route textual interpretations to configured boolean siblings of the same assay."""

    # Guard: Missing lab mappings or empty inputs leave nothing to remap.
    if df.empty or "lab_name_standardized" not in df.columns:
        return df

    # Evaluate raw_value first, then raw_comments only when raw_value is absent.
    for source_col, needs_raw_value_isna in [
        ("raw_value", False),
        ("raw_comments", True),
    ]:
        remapped_names: dict[int, str] = {}

        # Inspect each row independently so only confidently qualitative rows are remapped.
        for idx in df.index:
            standardized_name = df.at[idx, "lab_name_standardized"]

            # Skip unknown labs and analytes without a configured qualitative sibling.
            if pd.isna(standardized_name) or standardized_name == UNKNOWN_VALUE:
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

            # A printed "Normal" is a qualitative category, without asserting a zero concentration.
            is_normal_category = str(source_value).strip().casefold() == "normal"
            if classify_qualitative_value(str(source_value)) is None and not is_normal_category:
                continue

            variant_name = lab_specs.get_qualitative_variant(standardized_name)

            # Skip variants missing from lab_specs so normalization stays internally consistent.
            if variant_name is None or variant_name == standardized_name or lab_specs.get_primary_unit(variant_name) != "boolean":
                continue

            remapped_names[idx] = variant_name

        # Guard: No rows qualified for this source column.
        if not remapped_names:
            continue

        # Remap the standardized lab and unit so downstream review/export logic sees boolean rows.
        remap_indices = list(remapped_names)
        df.loc[remap_indices, "lab_name_standardized"] = pd.Series(remapped_names)
        df.loc[remap_indices, "lab_unit_standardized"] = "boolean"
        logger.info(f"[normalization] Remapped {len(remap_indices)} qualitative rows from {source_col} to configured boolean variants")

    return df


def _is_urine_strip_section(section_name: object) -> bool:
    """Return whether a raw section label indicates a urine strip context."""

    if pd.isna(section_name):
        return False

    normalized = str(section_name).strip().lower()
    if not normalized:
        return False

    return any(
        marker in normalized
        for marker in (
            "sumária da urina",
            "sumaria da urina",
            "exame sumário",
            "exame sumario",
            "tipo ii",
            "elementos anormais",
            "avaliação bioquímica",
            "avaliacao bioquimica",
        )
    )


def _is_urine_sediment_section(section_name: object) -> bool:
    """Return whether a raw section label indicates a urine sediment context."""

    if pd.isna(section_name):
        return False

    normalized = str(section_name).strip().lower()
    if not normalized:
        return False

    return any(
        marker in normalized
        for marker in (
            "sedimento urin",
            "sedimento",
        )
    )


def _normalize_field_style_sediment_units(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mislabeled sediment `/µl` units back onto the field-count normalization path."""

    required_columns = {"lab_name_standardized", "lab_unit_standardized", "raw_value"}
    if df.empty or not required_columns.issubset(df.columns):
        return df

    remap_indices: list[int] = []

    for idx in df.index:
        standardized_name = df.at[idx, "lab_name_standardized"]
        standardized_unit = df.at[idx, "lab_unit_standardized"]

        if standardized_name not in SEDIMENT_FIELD_STYLE_LABS:
            continue

        raw_unit = str(df.at[idx, "raw_lab_unit"]).strip().lower() if "raw_lab_unit" in df.columns else ""
        # A literal per-volume unit outranks stale cache entries and field-style result wording.
        if raw_unit in {"/ul", "/µl", "/μl"}:
            df.at[idx, "lab_unit_standardized"] = "/µl"
            continue

        if str(standardized_unit).strip().lower() not in {"/ul", "/µl"}:
            continue

        if "raw_section_name" in df.columns and not _is_urine_sediment_section(df.at[idx, "raw_section_name"]):
            continue

        raw_value = "" if pd.isna(df.at[idx, "raw_value"]) else str(df.at[idx, "raw_value"]).strip()
        looks_field_style = bool(SEDIMENT_FIELD_STYLE_VALUE_PATTERN.search(raw_value))
        looks_qualitative = pd.isna(df.at[idx, "value_primary"]) and bool(raw_value)
        raw_unit = str(df.at[idx, "raw_lab_unit"]).strip().lower() if "raw_lab_unit" in df.columns else ""

        # Rarity wording alone cannot contradict an explicitly reported per-volume unit.
        if looks_qualitative and not looks_field_style and raw_unit in {"/ul", "/µl"}:
            continue

        if not looks_field_style and not looks_qualitative:
            continue

        df.at[idx, "lab_unit_standardized"] = "/campo"
        remap_indices.append(idx)

    if remap_indices:
        logger.info(
            "[normalization] Remapped %s urine sediment row(s) from /µl to /campo based on field-style values",
            len(remap_indices),
        )

    return df


def _remap_absolute_sediment_variant_names(
    df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """Split true `/µL` sediment counts away from microscopy-per-field analytes."""

    required_columns = {"lab_name_standardized", "lab_unit_standardized"}
    if df.empty or not required_columns.issubset(df.columns):
        return df

    remap_indices: list[int] = []

    for idx in df.index:
        standardized_name = df.at[idx, "lab_name_standardized"]
        standardized_unit = str(df.at[idx, "lab_unit_standardized"]).strip().lower()

        if standardized_name not in SEDIMENT_ABSOLUTE_VARIANT_MAP:
            continue

        if standardized_unit not in {"/ul", "/µl", "/μl"}:
            continue

        absolute_variant = SEDIMENT_ABSOLUTE_VARIANT_MAP[standardized_name]
        if lab_specs.resolve_lab_name(absolute_variant) is None:
            continue

        df.at[idx, "lab_name_standardized"] = absolute_variant
        df.at[idx, "lab_unit_standardized"] = lab_specs.get_primary_unit(absolute_variant)
        remap_indices.append(idx)

    if remap_indices:
        logger.info(
            "[normalization] Remapped %s urine sediment row(s) to absolute sibling analytes",
            len(remap_indices),
        )

    return df


def _remap_numeric_urine_strip_variants(
    df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """Interpret unscaled strip grades while preserving reported numeric assay units."""

    if df.empty or "lab_name_standardized" not in df.columns or "raw_section_name" not in df.columns:
        return df

    remap_indices: list[int] = []

    for idx in df.index:
        standardized_name = df.at[idx, "lab_name_standardized"]

        if standardized_name not in QUALITATIVE_VARIANT_MAP:
            continue

        if not _is_urine_strip_section(df.at[idx, "raw_section_name"]):
            continue

        numeric_value = df.at[idx, "value_primary"]
        if pd.isna(numeric_value):
            continue

        raw_unit = str(df.at[idx, "raw_lab_unit"]).strip().lower() if "raw_lab_unit" in df.columns else ""

        # A printed concentration or count remains quantitative even with a one-sided or absent interval.
        if lab_specs.get_primary_unit(standardized_name) != "boolean" and raw_unit not in {"", "nan", "none", "null", "boolean", "bool", "unitless", "-"}:
            continue

        # A bounded numeric reference interval indicates a quantitative assay.
        # Do not destroy that information merely because section inference points
        # at a urine-strip context.
        reference_values = pd.to_numeric(
            pd.Series(
                [
                    df.at[idx, "raw_reference_min"] if "raw_reference_min" in df.columns else None,
                    df.at[idx, "raw_reference_max"] if "raw_reference_max" in df.columns else None,
                ]
            ),
            errors="coerce",
        )
        if reference_values.notna().all():
            continue

        variant_name = QUALITATIVE_VARIANT_MAP[standardized_name]
        if lab_specs.get_primary_unit(variant_name) != "boolean":
            continue

        df.at[idx, "value_primary"] = 0 if float(numeric_value) <= 0 else 1
        df.at[idx, "lab_name_standardized"] = variant_name
        df.at[idx, "lab_unit_standardized"] = "boolean"
        remap_indices.append(idx)

    if remap_indices:
        logger.info(
            "[normalization] Remapped %s numeric urine-strip row(s) to qualitative boolean variants",
            len(remap_indices),
        )

    return df


def _canonicalize_qualitative_lab_names(
    df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """Normalize qualitative lab names onto the canonical ``(Qualitative)`` suffix."""

    if df.empty or "lab_name_standardized" not in df.columns:
        return df

    canonical_names = df["lab_name_standardized"].map(
        lambda name: lab_specs.get_canonical_lab_name(name)
        if pd.notna(name) and name != UNKNOWN_VALUE
        else name
    )
    df["lab_name_standardized"] = canonical_names
    return df


def _convert_qualitative_values(
    df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """
    Convert qualitative text values to 0/1 for boolean labs.

    Uses deterministic pattern matching to convert values like "negativo"/"positivo"
    to numeric 0/1 values only for boolean labs. Text on quantitative assays
    remains in value_text unless a configured qualitative sibling was selected.
    """

    # Find labs with boolean primary unit.
    boolean_labs = [lab_name for lab_name in df["lab_name_standardized"].unique() if pd.notna(lab_name) and lab_name != UNKNOWN_VALUE and lab_specs.get_primary_unit(lab_name) == "boolean"]

    # Boolean labs: convert qualitative text to 0/1 from raw_value, then comments.
    if boolean_labs:
        for source_col, needs_raw_value_isna in [
            ("raw_value", False),
            ("raw_comments", True),
        ]:
            mask = df["lab_name_standardized"].isin(boolean_labs) & df["value_primary"].isna() & df[source_col].notna()

            # For raw_comments source, only use when raw_value is also missing.
            if needs_raw_value_isna:
                mask &= df["raw_value"].isna()

            # Guard: Skip sources with no rows left to classify.
            if not mask.any():
                continue

            qual_map = _classify_qualitative_batch(df.loc[mask, source_col].tolist())
            mapped_values = df.loc[mask, source_col].map(lambda v: qual_map.get(v))
            matched_mask = mapped_values.notna()

            # Guard: Leave unrecognized qualitative strings untouched for later review.
            if not matched_mask.any():
                continue

            matched_index = mapped_values.index[matched_mask]
            df.loc[matched_index, "value_primary"] = mapped_values.loc[matched_mask].astype(float)
            df.loc[matched_index, "lab_unit_standardized"] = "boolean"
            df.loc[matched_index, "lab_unit_primary"] = "boolean"
            logger.info(f"[normalization] Converted {len(matched_index)} qualitative values from {source_col} (boolean labs)")

        # Old serology reports often pair a numeric screen value with the actual
        # interpretation in comments (for example ``NR`` / ``R``). Prefer that
        # qualitative interpretation when it is available.
        if "raw_comments" in df.columns:
            override_mask = df["lab_name_standardized"].isin(boolean_labs) & df["raw_comments"].notna()
            override_map = _classify_qualitative_batch(df.loc[override_mask, "raw_comments"].tolist())
            override_values = df.loc[override_mask, "raw_comments"].map(lambda value: override_map.get(value))
            matched_override_mask = override_values.notna()

            if matched_override_mask.any():
                matched_override_index = override_values.index[matched_override_mask]
                df.loc[matched_override_index, "value_primary"] = override_values.loc[matched_override_mask].astype(float)
                df.loc[matched_override_index, "lab_unit_standardized"] = "boolean"
                df.loc[matched_override_index, "lab_unit_primary"] = "boolean"
                logger.info(
                    "[normalization] Overrode %s boolean value(s) from raw_comments interpretations",
                    len(matched_override_index),
                )

    # Numeric assay thresholds cannot become reference intervals for 0/1 interpretations.
    boolean_mask = df["lab_name_standardized"].isin(boolean_labs)
    for column in ("reference_min_primary", "reference_max_primary"):
        invalid_bounds = boolean_mask & df[column].notna() & ~df[column].between(0, 1)
        df.loc[invalid_bounds, column] = float("nan")

    return df


def parse_titer_value(value: object) -> tuple[float, bool, bool] | None:
    """Read a printed 1:N dilution as its denominator, preserving comparison direction."""

    match = TITER_VALUE_PATTERN.fullmatch(str(value))

    # Guard: Only explicit, nonzero dilution formats establish a titer result.
    if match is None or int(match.group("denominator")) <= 0:
        return None

    return float(match.group("denominator")), match.group("operator") == "<", match.group("operator") == ">"


def _normalize_titer_values(df: pd.DataFrame, lab_specs: LabSpecsConfig) -> pd.DataFrame:
    """Keep dilution measurements separate from index assays and boolean interpretations."""

    for idx in df.index:
        name = df.at[idx, "lab_name_standardized"]

        # Guard: Fractions in arbitrary results must not be interpreted as assay titers.
        if pd.isna(name) or lab_specs.get_primary_unit(name) != "titer":
            continue

        parsed = parse_titer_value(df.at[idx, "raw_value"])

        # Guard: Missing or textual results retain value_text without an invented number.
        if parsed is None:
            continue

        df.at[idx, "value_primary"], df.at[idx, "is_below_limit"], df.at[idx, "is_above_limit"] = parsed
        df.at[idx, "lab_unit_standardized"] = "titer"
        df.at[idx, "lab_unit_primary"] = "titer"
        reference = str(df.at[idx, "raw_reference_range"]) if "raw_reference_range" in df.columns else ""

        # Interpretive threshold prose is retained as text, without inventing a normal interval.
        if not re.search(r"1\s*[:/]\s*\d+", reference):
            continue

        df.at[idx, "reference_min_primary"] = float("nan")
        df.at[idx, "reference_max_primary"] = float("nan")
        reference_titer = parse_titer_value(reference)

        # Only a standalone printed comparison establishes a one-sided reference bound.
        if reference_titer is None or not (reference_titer[1] or reference_titer[2]):
            continue

        column = "reference_max_primary" if reference_titer[1] else "reference_min_primary"
        df.at[idx, column] = reference_titer[0]

    return df


def _preserve_nominal_results(df: pd.DataFrame, lab_specs: LabSpecsConfig) -> pd.DataFrame:
    """Keep categories such as color and appearance without inferring normality codes."""

    nominal_mask = df["lab_name_standardized"].map(lab_specs.get_primary_unit).eq("text")

    # Guard: Numeric and boolean assays retain their existing measurement paths.
    if not nominal_mask.any():
        return df

    df.loc[nominal_mask, "value_text"] = df.loc[nominal_mask, "raw_value"]
    df.loc[nominal_mask, "value_primary"] = float("nan")
    df.loc[nominal_mask, ["lab_unit_standardized", "lab_unit_primary"]] = "text"
    df.loc[nominal_mask, ["reference_min_primary", "reference_max_primary"]] = float("nan")
    return df


def _normalize_two_hour_sedimentation_units(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve distance readings over two hours despite stale hourly unit caches."""

    mask = df["lab_name_standardized"].eq("Blood - Erythrocyte Sedimentation Rate (ESR) - 2h")

    # Only an explicitly printed distance unit establishes a two-hour distance reading.
    if not mask.any() or "raw_lab_unit" not in df.columns:
        return df

    raw_units = df["raw_lab_unit"].fillna("").astype(str).str.strip().str.lower()
    distance_mask = mask & raw_units.isin(["mm", "mn"])
    df.loc[distance_mask, ["lab_unit_standardized", "lab_unit_primary"]] = "mm/(2.h)"
    return df


def _normalize_specific_gravity_notation(df: pd.DataFrame) -> pd.DataFrame:
    """Convert conventional four-digit relative-density notation to a ratio."""

    mask = df["lab_name_standardized"].eq("Urine Type II - Specific Gravity")
    values = pd.to_numeric(df["value_primary"], errors="coerce")
    mask &= values.between(1000, 1100) & values.eq(values.round())

    # Decimal ratios and values outside the supported notation remain unchanged.
    if not mask.any():
        return df

    df.loc[mask, "value_primary"] = values.loc[mask] / 1000
    for column in ["reference_min_primary", "reference_max_primary"]:
        bounds = pd.to_numeric(df[column], errors="coerce")
        bound_mask = mask & bounds.between(1000, 1100)
        df.loc[bound_mask, column] = bounds.loc[bound_mask] / 1000
    return df


def _preserve_unclassified_qualitative_text(df: pd.DataFrame, lab_specs: LabSpecsConfig) -> pd.DataFrame:
    """Keep indeterminate interpretations as text without inventing absence."""

    mask = df["lab_name_standardized"].map(lab_specs.get_primary_unit).eq("boolean")
    mask &= df["value_primary"].isna() & df["value_text"].notna()

    # Classified positive/negative observations keep their boolean measurement.
    if not mask.any():
        return df

    df.loc[mask, ["lab_unit_standardized", "lab_unit_primary"]] = "text"
    df.loc[mask, ["reference_min_primary", "reference_max_primary"]] = float("nan")
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
        df.loc[unit_mask, "reference_min_primary"] = pd.to_numeric(df.loc[unit_mask, "raw_reference_min"], errors="coerce").astype(float) * factor
        df.loc[unit_mask, "reference_max_primary"] = pd.to_numeric(df.loc[unit_mask, "raw_reference_max"], errors="coerce").astype(float) * factor
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
    df["reference_min_primary"] = pd.to_numeric(df["raw_reference_min"], errors="coerce").astype(float)
    df["reference_max_primary"] = pd.to_numeric(df["raw_reference_max"], errors="coerce").astype(float)

    # Phase 1: Extract comparison operators and preprocess values
    df = _preprocess_values(df)
    df = _normalize_titer_values(df, lab_specs)
    df = _preserve_nominal_results(df, lab_specs)
    df = _normalize_two_hour_sedimentation_units(df)
    df = _normalize_specific_gravity_notation(df)

    # Phase 2: Recover field-style urine sediment units when the report value contradicts /µl.
    df = _normalize_field_style_sediment_units(df)

    # Phase 2b: Remaining /µL sediment rows are distinct absolute-count analytes.
    df = _remap_absolute_sediment_variant_names(df, lab_specs)

    # Phase 3: Route textual assay interpretations onto configured boolean siblings.
    df = _remap_qualitative_variant_rows(df, lab_specs)

    # Phase 4: Interpret unit-free strip grades without erasing printed quantitative units.
    df = _remap_numeric_urine_strip_variants(df, lab_specs)

    # Phase 5: Convert qualitative text values to 0/1 for boolean-style labs.
    df = _convert_qualitative_values(df, lab_specs)
    df = _preserve_unclassified_qualitative_text(df, lab_specs)

    # Phase 6: Apply exact unit conversion factors for each lab type.
    for lab_name_standardized in df["lab_name_standardized"].unique():
        # Skip unknown or missing lab names
        if pd.isna(lab_name_standardized) or lab_name_standardized == UNKNOWN_VALUE:
            continue
        _apply_unit_conversions_for_lab(df, lab_name_standardized, lab_specs)

    # Phase 7: Expose canonical qualitative names after conversions and remaps settle.
    df = _canonicalize_qualitative_lab_names(df, lab_specs)

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

    # Initialize review columns before duplicate flags are added.
    df = ensure_review_flag_columns(df)

    # Preserve the existing boolean/string coercion behavior for duplicate review rows.
    df["review_needed"] = df["review_needed"].astype(bool)
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

    return append_review_reason_code(df, dup_mask, "DUPLICATE_ENTRY")


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


def _explicit_sample_number(raw_name: object) -> str:
    """Identify separately numbered specimens without inventing sample identities."""

    match = re.search(r"\b(\d+)\s*[ªº]?\s*(?:amostra|sample)\b|\b(?:amostra|sample)\s*(\d+)\b", str(raw_name), re.IGNORECASE)

    # Unnumbered observations retain the existing deduplication identity.
    if match is None:
        return ""

    return str(int(match.group(1) or match.group(2)))


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
    # Distinct textual findings from the same assay must survive numeric-unit deduplication.
    dedup_columns = ["date", "lab_name_standardized"]
    if "value_text" in df.columns:
        dedup_columns.append("value_text")

    # Separately labeled specimens are independent observations even when results agree.
    if "raw_lab_name" in df.columns:
        df["_dedup_specimen"] = df["raw_lab_name"].map(_explicit_sample_number)
        dedup_columns.append("_dedup_specimen")

    df = df.drop_duplicates(subset=dedup_columns, keep="first")
    df = df.drop(columns=["_dedup_priority", "_dedup_specimen"], errors="ignore")
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
