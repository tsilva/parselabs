"""
Value-Based Accuracy Validation for Lab Results

Detects extraction errors from the data itself (without re-checking source images),
flagging suspicious values for review. Integrates with existing review workflow
using review_needed and review_reason fields.
"""

import logging
import re

import pandas as pd

from labs_parser.config import LabSpecsConfig

logger = logging.getLogger(__name__)


class ValueValidator:
    """Validates extracted lab values without source images.

    Detects biologically implausible values, inter-lab relationship mismatches,
    temporal anomalies, format artifacts, and reference range inconsistencies.
    """

    # Valid reason codes for flagging suspicious values
    REASON_CODES = {
        "IMPOSSIBLE_VALUE",
        "RELATIONSHIP_MISMATCH",
        "COMPONENT_EXCEEDS_TOTAL",
        "TEMPORAL_ANOMALY",
        "FORMAT_ARTIFACT",
        "RANGE_INCONSISTENCY",
        "PERCENTAGE_BOUNDS",
        "NEGATIVE_VALUE",
        "EXTREME_DEVIATION",
        "DUPLICATE_ENTRY",
    }

    # Component-total constraints: component value must not exceed total value
    # Format: (component_lab, total_lab)
    COMPONENT_TOTAL_CONSTRAINTS = [
        ("Blood - Albumin", "Blood - Total Protein"),
        ("Blood - Bilirubin Direct", "Blood - Bilirubin Total"),
        ("Blood - Bilirubin Indirect", "Blood - Bilirubin Total"),
        (
            "Blood - High-Density Lipoprotein Cholesterol (HDL Cholesterol)",
            "Blood - Total Cholesterol",
        ),
        (
            "Blood - Low-Density Lipoprotein Cholesterol (LDL Cholesterol)",
            "Blood - Total Cholesterol",
        ),
    ]

    # Column name mappings: (preferred_name, fallback_name)
    COLUMN_MAPPINGS = {
        "value": ("value", "value_primary"),
        "lab_name": ("lab_name", "lab_name_standardized"),
        "unit": ("lab_unit", "lab_unit_primary"),
        "ref_min": ("reference_min", "reference_min_primary"),
        "ref_max": ("reference_max", "reference_max_primary"),
    }

    def __init__(self, lab_specs: LabSpecsConfig):
        """Initialize validator with lab specifications.

        Args:
            lab_specs: LabSpecsConfig containing biological limits and relationships
        """

        self.lab_specs = lab_specs
        self._validation_stats = {
            "total_rows": 0,
            "rows_flagged": 0,
            "flags_by_reason": {},
        }
        # Column name mappings (resolved once per validate() call)
        self._value_col: str = ""
        self._lab_name_col: str = ""
        self._unit_col: str = ""
        self._date_col: str = "date"
        self._ref_min_col: str = ""
        self._ref_max_col: str = ""
        self._value_raw_col: str = "raw_value"

    def _resolve_column(self, df: pd.DataFrame, key: str) -> str:
        """Resolve column name using preferred or fallback naming convention.

        Args:
            df: DataFrame to check columns against
            key: Key from COLUMN_MAPPINGS

        Returns:
            Actual column name to use (preferred if exists, else fallback)
        """

        preferred, fallback = self.COLUMN_MAPPINGS[key]
        return preferred if preferred in df.columns else fallback

    @property
    def validation_stats(self) -> dict:
        """Get statistics from the last validation run."""

        return self._validation_stats

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all validations on DataFrame, setting review flags.

        Args:
            df: DataFrame with lab results (must have 'value', 'lab_name', 'unit' columns)

        Returns:
            DataFrame with review_needed and review_reason updated
        """

        # Empty DataFrame — nothing to validate
        if df.empty:
            return df

        # Work on a copy to avoid mutating the original
        df = df.copy()

        # Initialize review columns if not present
        if "review_needed" not in df.columns:
            df["review_needed"] = False
        if "review_reason" not in df.columns:
            df["review_reason"] = ""
        # Resolve column names once (supports both naming conventions)
        self._value_col = self._resolve_column(df, "value")
        self._lab_name_col = self._resolve_column(df, "lab_name")
        self._unit_col = self._resolve_column(df, "unit")
        self._ref_min_col = self._resolve_column(df, "ref_min")
        self._ref_max_col = self._resolve_column(df, "ref_max")

        # Reset stats
        self._validation_stats = {
            "total_rows": len(df),
            "rows_flagged": 0,
            "flags_by_reason": {},
        }

        # Run validation checks in order
        df = self._check_biological_plausibility(df)
        df = self._check_inter_lab_relationships(df)
        df = self._check_component_total_constraints(df)
        df = self._check_temporal_consistency(df)
        df = self._check_format_artifacts(df)
        df = self._check_reference_ranges(df)

        # Update stats
        self._validation_stats["rows_flagged"] = int(df["review_needed"].sum())

        logger.info(f"Validation complete: {self._validation_stats['rows_flagged']}/{len(df)} rows flagged for review")

        return df

    def _flag_row(self, df: pd.DataFrame, mask: pd.Series, reason_code: str) -> pd.DataFrame:
        """Flag rows matching mask with given reason code.

        Args:
            df: DataFrame to modify
            mask: Boolean series indicating which rows to flag
            reason_code: Reason code from REASON_CODES

        Returns:
            Modified DataFrame
        """

        # No rows to flag
        if not mask.any():
            return df

        # Set review flag and append reason code to review_reason
        df.loc[mask, "review_needed"] = True
        # Handle NaN values in review_reason by converting to empty string first
        df.loc[mask, "review_reason"] = df.loc[mask, "review_reason"].fillna("").apply(lambda x: str(x) + f"{reason_code}; " if reason_code not in str(x) else str(x))

        # Update stats
        count = int(mask.sum())
        self._validation_stats["flags_by_reason"][reason_code] = self._validation_stats["flags_by_reason"].get(reason_code, 0) + count

        logger.debug(f"Flagged {count} rows with {reason_code}")

        return df

    def _batch_flag_by_indices(self, df: pd.DataFrame, flags: dict[str, list]) -> pd.DataFrame:
        """Batch flag rows by collected indices for each reason code.

        Args:
            df: DataFrame to modify
            flags: Dict mapping reason codes to lists of indices to flag

        Returns:
            Modified DataFrame
        """

        for reason_code, indices in flags.items():
            # Skip reason codes with no flagged indices
            if indices:
                mask = df.index.isin(indices)
                df = self._flag_row(df, mask, reason_code)
        return df

    # =========================================================================
    # 1. Biological Plausibility Checks
    # =========================================================================

    def _check_biological_plausibility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for biologically impossible values.

        Validates:
        - Negative concentrations (most labs can't be negative)
        - Values exceeding biological maximums
        - Percentage bounds (0-100%)
        - Boolean constraints (0/1 for positive/negative tests)
        """

        # Required columns missing — skip check
        if self._value_col not in df.columns or self._lab_name_col not in df.columns:
            return df

        # Labs allowed to have negative values
        negative_allowed = {"Blood - Anion Gap"}

        # Labs where percentage values can legitimately exceed 100%
        # (e.g., prothrombin activity is measured relative to normal, not as fraction of total)
        percentage_exceeds_100_allowed = {
            "Blood - Prothrombin Time (PT) (%)",
            "Blood - Factor II Activity (%)",
            "Blood - Factor V Activity (%)",
            "Blood - Factor VII Activity (%)",
            "Blood - Factor VIII Activity (%)",
            "Blood - Factor IX Activity (%)",
            "Blood - Factor X Activity (%)",
            "Blood - Factor XI Activity (%)",
            "Blood - Factor XII Activity (%)",
            "Blood - Protein C Activity (%)",
            "Blood - Protein S Activity (%)",
            "Blood - Antithrombin III Activity (%)",
        }

        flags: dict[str, list] = {
            "NEGATIVE_VALUE": [],
            "PERCENTAGE_BOUNDS": [],
            "IMPOSSIBLE_VALUE": [],
        }

        has_unit_col = self._unit_col in df.columns

        # Use itertuples for efficient row iteration
        for row in df.itertuples():
            idx = row.Index
            value = getattr(row, self._value_col, None)
            lab_name = getattr(row, self._lab_name_col, None)

            # Skip rows with missing value or lab name
            if pd.isna(value) or pd.isna(lab_name):
                continue

            # Check for negative concentrations
            if value < 0 and lab_name not in negative_allowed:
                flags["NEGATIVE_VALUE"].append(idx)
                continue

            # Check percentage bounds
            unit = getattr(row, self._unit_col, None) if has_unit_col else None
            if unit == "%":
                # Some labs (coagulation factors) can legitimately exceed 100%
                max_pct = 200 if lab_name in percentage_exceeds_100_allowed else 100
                # Value outside allowed percentage range
                if value < 0 or value > max_pct:
                    flags["PERCENTAGE_BOUNDS"].append(idx)
                    continue

            # Check biological limits from lab_specs
            spec = self.lab_specs.specs.get(lab_name, {})
            bio_min = spec.get("biological_min")
            bio_max = spec.get("biological_max")

            # Check value against biological min/max
            if (bio_min is not None and value < bio_min) or (bio_max is not None and value > bio_max):
                flags["IMPOSSIBLE_VALUE"].append(idx)

        return self._batch_flag_by_indices(df, flags)

    # =========================================================================
    # 2. Inter-Lab Relationship Checks
    # =========================================================================

    def _evaluate_single_relationship(self, rel: dict, lab_values: dict) -> tuple[float | None, float]:
        """Evaluate one relationship definition against lab values for a single date.

        Returns:
            (pct_diff, tolerance_pct) if relationship can be evaluated, (None, 0) otherwise
        """

        target = rel.get("target")
        formula = rel.get("formula")
        tolerance_pct = rel.get("tolerance_percent", 15)

        # Missing definition — cannot evaluate
        if not target or not formula:
            return None, 0

        # Target value not available for this date
        target_value = lab_values.get(target)
        if target_value is None:
            return None, 0

        # Calculate expected value from formula
        calculated = self._evaluate_formula(formula, lab_values)

        # Formula missing components — cannot evaluate
        if calculated is None:
            return None, 0

        # Compute percentage difference
        # Non-zero target — use relative percentage difference
        if target_value != 0:
            pct_diff = abs(calculated - target_value) / abs(target_value) * 100
        # Zero target — use absolute diff scaled to percentage
        else:
            pct_diff = abs(calculated - target_value) * 100

        return pct_diff, tolerance_pct

    def _check_inter_lab_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate mathematical relationships between labs.

        Checks relationships like:
        - LDL = Total Chol - HDL - (Trig/5) [Friedewald formula]
        - A/G Ratio = Albumin / Globulin
        - Total Bilirubin = Direct + Indirect
        - MCHC = MCH/MCV * 100
        """

        relationships = self.lab_specs.specs.get("_relationships", [])

        # No relationships defined — skip check
        if not relationships:
            return df

        # Date column missing — cannot group by date
        if self._date_col not in df.columns:
            return df

        # Group by date to check relationships within same test date
        for date_val, date_group in df.groupby(self._date_col):
            # Skip rows with missing date
            if pd.isna(date_val):
                continue

            # Build lookup of lab values for this date
            lab_values = {}
            for _, row in date_group.iterrows():
                lab_name = row.get(self._lab_name_col)
                value = row.get(self._value_col)
                if pd.notna(lab_name) and pd.notna(value):
                    lab_values[lab_name] = value

            # Evaluate each relationship against this date's values
            for rel in relationships:
                try:
                    pct_diff, tolerance_pct = self._evaluate_single_relationship(rel, lab_values)
                # Evaluation error — log and skip
                except Exception as e:
                    logger.debug(f"Error evaluating relationship {rel.get('name')}: {e}")
                    continue

                # Relationship couldn't be evaluated (missing components)
                if pct_diff is None:
                    continue

                # Within tolerance — no flag needed
                if pct_diff <= tolerance_pct:
                    continue

                # Exceeds tolerance — flag the target row
                target = rel.get("target")
                mask = (df[self._date_col] == date_val) & (df[self._lab_name_col] == target)
                df = self._flag_row(df, mask, "RELATIONSHIP_MISMATCH")
                logger.debug(f"Relationship mismatch for {target} on {date_val}: {pct_diff:.1f}% diff (tolerance: {tolerance_pct}%)")

        return df

    def _check_component_total_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check that component values don't exceed their totals.

        For example: Albumin should not exceed Total Protein,
        Direct Bilirubin should not exceed Total Bilirubin.
        """

        # Date column missing — cannot group by date
        if self._date_col not in df.columns:
            return df

        component_exceeds_indices: list = []

        # Group by date to check constraints within same test date
        for date_val, date_group in df.groupby(self._date_col):
            # Skip rows with missing date
            if pd.isna(date_val):
                continue

            # Build lookup of lab values and indices for this date
            lab_values: dict[str, float] = {}
            lab_indices: dict[str, int] = {}
            for idx, row in date_group.iterrows():
                lab_name = row.get(self._lab_name_col)
                value = row.get(self._value_col)
                if pd.notna(lab_name) and pd.notna(value):
                    lab_values[lab_name] = value
                    lab_indices[lab_name] = idx

            # Check each constraint
            for component_lab, total_lab in self.COMPONENT_TOTAL_CONSTRAINTS:
                component_value = lab_values.get(component_lab)
                total_value = lab_values.get(total_lab)

                # Either component or total missing — skip constraint
                if component_value is None or total_value is None:
                    continue

                # Component should not exceed total (with small tolerance for rounding)
                if component_value > total_value * 1.05:  # 5% tolerance
                    component_idx = lab_indices.get(component_lab)
                    # Record index for batch flagging
                    if component_idx is not None:
                        component_exceeds_indices.append(component_idx)
                        logger.debug(f"Component exceeds total on {date_val}: {component_lab}={component_value} > {total_lab}={total_value}")

        # Batch flag all collected indices
        if component_exceeds_indices:
            mask = df.index.isin(component_exceeds_indices)
            df = self._flag_row(df, mask, "COMPONENT_EXCEEDS_TOTAL")

        return df

    def _evaluate_formula(self, formula: str, lab_values: dict) -> float | None:
        """Evaluate a formula string with lab values.

        Args:
            formula: Formula like "Blood - Cholesterol Total - Blood - HDL Cholesterol - (Blood - Triglycerides / 5)"
            lab_values: Dict mapping lab names to values

        Returns:
            Calculated value or None if missing components
        """

        # Extract lab names from formula (they're the parts that start with lab type prefix)
        lab_prefixes = ("Blood - ", "Urine - ", "Feces - ")

        # Replace lab names with their values
        result_formula = formula
        for prefix in lab_prefixes:
            # Find all lab names with this prefix
            pattern = re.escape(prefix) + r"[^+\-*/()]+?(?=[+\-*/()]|$)"
            matches = re.findall(pattern, formula)

            for match in matches:
                lab_name = match.strip()
                value = lab_values.get(lab_name)
                # Missing component — formula cannot be evaluated
                if value is None:
                    return None
                result_formula = result_formula.replace(lab_name, str(value))

        # Safely evaluate the arithmetic expression
        try:
            # Only allow safe arithmetic operations
            allowed = set("0123456789.+-*/() ")

            # Unsafe characters present — reject formula
            if not all(c in allowed for c in result_formula):
                return None
            return eval(result_formula)
        # Evaluation failed (division by zero, syntax error, etc.)
        except Exception:
            return None

    # =========================================================================
    # 3. Temporal Consistency Checks
    # =========================================================================

    def _check_temporal_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag implausible changes between consecutive tests.

        Uses max_daily_change from lab_specs to determine if a value
        changed too rapidly between tests.
        """

        # Required columns missing — skip check
        if self._date_col not in df.columns or self._value_col not in df.columns:
            return df

        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self._date_col]):
            df[self._date_col] = pd.to_datetime(df[self._date_col], errors="coerce")

        # Collect all indices to flag
        temporal_anomaly_indices: list = []

        # Group by lab name and check temporal consistency
        for lab_name, group in df.groupby(self._lab_name_col):
            # Skip rows with missing lab name
            if pd.isna(lab_name):
                continue

            spec = self.lab_specs.specs.get(lab_name, {})
            max_daily_change = spec.get("max_daily_change")

            # No max daily change defined — skip this lab
            if max_daily_change is None:
                continue

            # Sort by date
            group_sorted = group.sort_values(self._date_col)

            prev_date = None
            prev_value = None

            for idx in group_sorted.index:
                curr_date = group_sorted.loc[idx, self._date_col]
                curr_value = group_sorted.loc[idx, self._value_col]

                # Skip rows with missing data
                if pd.isna(curr_date) or pd.isna(curr_value):
                    continue

                # First valid row — nothing to compare against
                if prev_date is None or prev_value is None:
                    prev_date = curr_date
                    prev_value = curr_value
                    continue

                # Same-day tests — skip temporal comparison
                days_diff = (curr_date - prev_date).days
                if days_diff <= 0:
                    prev_date = curr_date
                    prev_value = curr_value
                    continue

                # Check if daily change rate exceeds maximum
                daily_rate = abs(curr_value - prev_value) / days_diff
                if daily_rate > max_daily_change:
                    temporal_anomaly_indices.append(idx)
                    logger.debug(f"Temporal anomaly for {lab_name}: {prev_value:.1f} -> {curr_value:.1f} over {days_diff} days (rate: {daily_rate:.2f}/day, max: {max_daily_change})")

                prev_date = curr_date
                prev_value = curr_value

        # Batch flag all collected indices
        if temporal_anomaly_indices:
            mask = df.index.isin(temporal_anomaly_indices)
            df = self._flag_row(df, mask, "TEMPORAL_ANOMALY")

        return df

    # =========================================================================
    # 4. Format Validation Checks
    # =========================================================================

    def _check_format_artifacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect extraction artifacts from formatting issues.

        Checks:
        - Excessive decimal places (likely concatenation error)
        - Magnitude mismatch (wrong order of magnitude for unit)
        - Embedded text in numeric fields
        """

        # Raw value column missing — skip check
        if self._value_raw_col not in df.columns:
            return df

        flags: dict[str, list] = {
            "FORMAT_ARTIFACT": [],
        }

        has_value_col = self._value_col in df.columns
        has_lab_name_col = self._lab_name_col in df.columns

        for row in df.itertuples():
            idx = row.Index
            value_raw = getattr(row, self._value_raw_col, None)
            value = getattr(row, self._value_col, None) if has_value_col else None
            lab_name = getattr(row, self._lab_name_col, None) if has_lab_name_col else None

            # Skip rows with missing raw value
            if pd.isna(value_raw):
                continue

            value_raw_str = str(value_raw)

            # Check for excessive decimals (more than 4 decimal places is suspicious)
            if "." in value_raw_str:
                decimal_part = value_raw_str.split(".")[-1]
                decimal_digits = re.match(r"\d+", decimal_part)
                # More than 4 decimal digits — likely a concatenation error
                if decimal_digits and len(decimal_digits.group()) > 4:
                    flags["FORMAT_ARTIFACT"].append(idx)
                    logger.debug(f"Excessive decimals in {lab_name}: {value_raw_str}")
                    continue

            # Check for concatenation errors (e.g., "52.6=1946" where reference got appended)
            if pd.notna(value) and isinstance(value_raw_str, str):
                cleaned = value_raw_str.replace(",", ".").strip()
                has_digits = bool(re.search(r"\d", cleaned))
                starts_with_number = bool(re.match(r"^[\d\.\-\+<>≤≥]", cleaned))

                # Numeric-looking string — check for "value=reference" concatenation
                if has_digits and starts_with_number:
                    concat_pattern = re.match(r"^[\d\.\-\+<>≤≥]+\s*=\s*\d+", cleaned)
                    # Pattern matched — flag as concatenation artifact
                    if concat_pattern:
                        flags["FORMAT_ARTIFACT"].append(idx)
                        logger.debug(f"Concatenation error in {lab_name}: {value_raw_str}")
                        continue

        return self._batch_flag_by_indices(df, flags)

    # =========================================================================
    # 5. Reference Range Consistency Checks
    # =========================================================================

    def _check_reference_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-validate values against extracted reference ranges.

        Checks:
        - ref_min > ref_max (invalid range)
        - Value extremely far outside extracted range (100x)
        """

        # Reference range columns missing — skip check
        if self._ref_min_col not in df.columns or self._ref_max_col not in df.columns:
            return df

        flags: dict[str, list] = {
            "RANGE_INCONSISTENCY": [],
            "EXTREME_DEVIATION": [],
        }

        has_value_col = self._value_col in df.columns
        has_lab_name_col = self._lab_name_col in df.columns

        for row in df.itertuples():
            idx = row.Index
            value = getattr(row, self._value_col, None) if has_value_col else None
            ref_min = getattr(row, self._ref_min_col, None)
            ref_max = getattr(row, self._ref_max_col, None)
            lab_name = getattr(row, self._lab_name_col, None) if has_lab_name_col else None

            # Check ref_min > ref_max (inverted range)
            if pd.notna(ref_min) and pd.notna(ref_max):
                # Inverted range — min exceeds max
                if ref_min > ref_max:
                    flags["RANGE_INCONSISTENCY"].append(idx)
                    logger.debug(f"Inverted range for {lab_name}: {ref_min} > {ref_max}")
                    continue

            # Check for extreme deviation from reference range
            if pd.notna(value) and pd.notna(ref_min) and pd.notna(ref_max):
                range_size = ref_max - ref_min

                # Valid range size — check deviation
                if range_size > 0:
                    # Value below reference minimum
                    if value < ref_min:
                        deviation = ref_min - value
                    # Value above reference maximum
                    elif value > ref_max:
                        deviation = value - ref_max
                    # Value within reference range
                    else:
                        deviation = 0

                    # Flag if deviation is more than 10x the range size
                    if deviation > range_size * 10:
                        flags["EXTREME_DEVIATION"].append(idx)
                        logger.debug(f"Extreme deviation for {lab_name}: {value} (range: {ref_min}-{ref_max}, deviation: {deviation:.1f})")

        return self._batch_flag_by_indices(df, flags)
