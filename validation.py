"""
Value-Based Accuracy Validation for Lab Results

Detects extraction errors from the data itself (without re-checking source images),
flagging suspicious values for review. Integrates with existing review workflow
using review_needed, review_reason, and review_confidence fields.
"""

import re
import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import LabSpecsConfig

logger = logging.getLogger(__name__)


class ValueValidator:
    """Validates extracted lab values without source images.

    Detects biologically implausible values, inter-lab relationship mismatches,
    temporal anomalies, format artifacts, and reference range inconsistencies.
    """

    # Reason codes with confidence multipliers (lower = more suspicious)
    REASON_CODES = {
        "IMPOSSIBLE_VALUE": 0.3,       # Biologically impossible
        "RELATIONSHIP_MISMATCH": 0.5,  # Calculated vs extracted mismatch
        "TEMPORAL_ANOMALY": 0.6,       # Implausible change rate
        "FORMAT_ARTIFACT": 0.7,        # Parsing/format issue
        "RANGE_INCONSISTENCY": 0.7,    # Reference range issues
        "PERCENTAGE_BOUNDS": 0.4,      # Percentage outside 0-100
        "NEGATIVE_VALUE": 0.3,         # Negative concentration
        "EXTREME_DEVIATION": 0.5,      # Value far outside reference range
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

    @property
    def validation_stats(self) -> dict:
        """Get statistics from the last validation run."""
        return self._validation_stats

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all validations on DataFrame, setting review flags.

        Args:
            df: DataFrame with lab results (must have 'value', 'lab_name', 'unit' columns)

        Returns:
            DataFrame with review_needed, review_reason, review_confidence updated
        """
        if df.empty:
            return df

        df = df.copy()

        # Initialize review columns if not present
        if 'review_needed' not in df.columns:
            df['review_needed'] = False
        if 'review_reason' not in df.columns:
            df['review_reason'] = ''
        if 'review_confidence' not in df.columns:
            df['review_confidence'] = 1.0

        # Reset stats
        self._validation_stats = {
            "total_rows": len(df),
            "rows_flagged": 0,
            "flags_by_reason": {},
        }

        # Run validation checks in order
        df = self._check_biological_plausibility(df)
        df = self._check_inter_lab_relationships(df)
        df = self._check_temporal_consistency(df)
        df = self._check_format_artifacts(df)
        df = self._check_reference_ranges(df)

        # Update stats
        self._validation_stats["rows_flagged"] = int(df['review_needed'].sum())

        logger.info(
            f"Validation complete: {self._validation_stats['rows_flagged']}/{len(df)} "
            f"rows flagged for review"
        )

        return df

    def _flag_row(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        reason_code: str
    ) -> pd.DataFrame:
        """Flag rows matching mask with given reason code.

        Args:
            df: DataFrame to modify
            mask: Boolean series indicating which rows to flag
            reason_code: Reason code from REASON_CODES

        Returns:
            Modified DataFrame
        """
        if not mask.any():
            return df

        confidence_mult = self.REASON_CODES.get(reason_code, 0.8)

        df.loc[mask, 'review_needed'] = True
        # Handle NaN values in review_reason by converting to empty string first
        df.loc[mask, 'review_reason'] = df.loc[mask, 'review_reason'].fillna('').apply(
            lambda x: str(x) + f'{reason_code}; ' if reason_code not in str(x) else str(x)
        )
        df.loc[mask, 'review_confidence'] = df.loc[mask, 'review_confidence'] * confidence_mult

        # Update stats
        count = int(mask.sum())
        self._validation_stats["flags_by_reason"][reason_code] = (
            self._validation_stats["flags_by_reason"].get(reason_code, 0) + count
        )

        logger.debug(f"Flagged {count} rows with {reason_code}")

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
        value_col = 'value' if 'value' in df.columns else 'value_primary'
        lab_name_col = 'lab_name' if 'lab_name' in df.columns else 'lab_name_standardized'
        unit_col = 'unit' if 'unit' in df.columns else 'lab_unit_primary'

        if value_col not in df.columns or lab_name_col not in df.columns:
            return df

        # Check negative values (excluding labs that can be negative like temperature change)
        negative_allowed = {
            'Blood - Anion Gap',  # Can be negative in some conditions
        }

        for idx, row in df.iterrows():
            value = row.get(value_col)
            lab_name = row.get(lab_name_col)
            unit = row.get(unit_col)

            if pd.isna(value) or pd.isna(lab_name):
                continue

            # Check for negative concentrations
            if value < 0 and lab_name not in negative_allowed:
                df = self._flag_row(
                    df,
                    pd.Series([True if i == idx else False for i in df.index], index=df.index),
                    "NEGATIVE_VALUE"
                )
                continue

            # Check percentage bounds
            if unit == '%':
                if value < 0 or value > 100:
                    df = self._flag_row(
                        df,
                        pd.Series([True if i == idx else False for i in df.index], index=df.index),
                        "PERCENTAGE_BOUNDS"
                    )
                    continue

            # Check biological limits from lab_specs
            spec = self.lab_specs.specs.get(lab_name, {})
            bio_min = spec.get('biological_min')
            bio_max = spec.get('biological_max')

            if bio_min is not None and value < bio_min:
                df = self._flag_row(
                    df,
                    pd.Series([True if i == idx else False for i in df.index], index=df.index),
                    "IMPOSSIBLE_VALUE"
                )
            elif bio_max is not None and value > bio_max:
                df = self._flag_row(
                    df,
                    pd.Series([True if i == idx else False for i in df.index], index=df.index),
                    "IMPOSSIBLE_VALUE"
                )

        return df

    # =========================================================================
    # 2. Inter-Lab Relationship Checks
    # =========================================================================

    def _check_inter_lab_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate mathematical relationships between labs.

        Checks relationships like:
        - LDL = Total Chol - HDL - (Trig/5) [Friedewald formula]
        - A/G Ratio = Albumin / Globulin
        - Total Bilirubin = Direct + Indirect
        - MCHC = MCH/MCV * 100
        """
        relationships = self.lab_specs.specs.get('_relationships', [])
        if not relationships:
            return df

        value_col = 'value' if 'value' in df.columns else 'value_primary'
        lab_name_col = 'lab_name' if 'lab_name' in df.columns else 'lab_name_standardized'
        date_col = 'date'

        if date_col not in df.columns:
            return df

        # Group by date to check relationships within same test date
        for date_val, date_group in df.groupby(date_col):
            if pd.isna(date_val):
                continue

            # Build lookup of lab values for this date
            lab_values = {}
            for _, row in date_group.iterrows():
                lab_name = row.get(lab_name_col)
                value = row.get(value_col)
                if pd.notna(lab_name) and pd.notna(value):
                    lab_values[lab_name] = value

            # Check each relationship
            for rel in relationships:
                target = rel.get('target')
                formula = rel.get('formula')
                tolerance_pct = rel.get('tolerance_percent', 15)

                if not target or not formula:
                    continue

                # Get target value
                target_value = lab_values.get(target)
                if target_value is None:
                    continue

                # Calculate expected value from formula
                try:
                    calculated = self._evaluate_formula(formula, lab_values)
                    if calculated is None:
                        continue

                    # Check if within tolerance
                    if target_value != 0:
                        pct_diff = abs(calculated - target_value) / abs(target_value) * 100
                    else:
                        pct_diff = abs(calculated - target_value) * 100  # Use absolute diff for zero

                    if pct_diff > tolerance_pct:
                        # Flag the target row
                        mask = (
                            (df[date_col] == date_val) &
                            (df[lab_name_col] == target)
                        )
                        df = self._flag_row(df, mask, "RELATIONSHIP_MISMATCH")
                        logger.debug(
                            f"Relationship mismatch for {target} on {date_val}: "
                            f"expected ~{calculated:.1f}, got {target_value:.1f} "
                            f"({pct_diff:.1f}% diff)"
                        )

                except Exception as e:
                    logger.debug(f"Error evaluating relationship {rel.get('name')}: {e}")

        return df

    def _evaluate_formula(self, formula: str, lab_values: dict) -> Optional[float]:
        """Evaluate a formula string with lab values.

        Args:
            formula: Formula like "Blood - Cholesterol Total - Blood - HDL Cholesterol - (Blood - Triglycerides / 5)"
            lab_values: Dict mapping lab names to values

        Returns:
            Calculated value or None if missing components
        """
        # Extract lab names from formula (they're the parts that start with lab type prefix)
        lab_prefixes = ('Blood - ', 'Urine - ', 'Feces - ')

        # Replace lab names with their values
        result_formula = formula
        for prefix in lab_prefixes:
            # Find all lab names with this prefix
            pattern = re.escape(prefix) + r'[^+\-*/()]+?(?=[+\-*/()]|$)'
            matches = re.findall(pattern, formula)

            for match in matches:
                lab_name = match.strip()
                value = lab_values.get(lab_name)
                if value is None:
                    return None  # Missing component
                result_formula = result_formula.replace(lab_name, str(value))

        # Safely evaluate the arithmetic expression
        try:
            # Only allow safe arithmetic operations
            allowed = set('0123456789.+-*/() ')
            if not all(c in allowed for c in result_formula):
                return None
            return eval(result_formula)
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
        value_col = 'value' if 'value' in df.columns else 'value_primary'
        lab_name_col = 'lab_name' if 'lab_name' in df.columns else 'lab_name_standardized'
        date_col = 'date'

        if date_col not in df.columns or value_col not in df.columns:
            return df

        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Group by lab name and check temporal consistency
        for lab_name, group in df.groupby(lab_name_col):
            if pd.isna(lab_name):
                continue

            spec = self.lab_specs.specs.get(lab_name, {})
            max_daily_change = spec.get('max_daily_change')

            if max_daily_change is None:
                continue

            # Sort by date
            group_sorted = group.sort_values(date_col)

            prev_date = None
            prev_value = None

            for idx, row in group_sorted.iterrows():
                curr_date = row.get(date_col)
                curr_value = row.get(value_col)

                if pd.isna(curr_date) or pd.isna(curr_value):
                    continue

                if prev_date is not None and prev_value is not None:
                    # Calculate days between tests
                    days_diff = (curr_date - prev_date).days
                    if days_diff > 0:
                        # Calculate actual daily change rate
                        value_change = abs(curr_value - prev_value)
                        daily_rate = value_change / days_diff

                        # Check if exceeds max allowed daily change
                        if daily_rate > max_daily_change:
                            df = self._flag_row(
                                df,
                                pd.Series([True if i == idx else False for i in df.index], index=df.index),
                                "TEMPORAL_ANOMALY"
                            )
                            logger.debug(
                                f"Temporal anomaly for {lab_name}: {prev_value:.1f} -> {curr_value:.1f} "
                                f"over {days_diff} days (rate: {daily_rate:.2f}/day, max: {max_daily_change})"
                            )

                prev_date = curr_date
                prev_value = curr_value

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
        value_col = 'value' if 'value' in df.columns else 'value_primary'
        value_raw_col = 'value_raw'
        lab_name_col = 'lab_name' if 'lab_name' in df.columns else 'lab_name_standardized'

        if value_raw_col not in df.columns:
            return df

        for idx, row in df.iterrows():
            value_raw = row.get(value_raw_col)
            value = row.get(value_col)
            lab_name = row.get(lab_name_col)

            if pd.isna(value_raw):
                continue

            value_raw_str = str(value_raw)

            # Check for excessive decimals (more than 4 decimal places is suspicious)
            if '.' in value_raw_str:
                decimal_part = value_raw_str.split('.')[-1]
                # Remove any trailing non-digits
                decimal_digits = re.match(r'\d+', decimal_part)
                if decimal_digits and len(decimal_digits.group()) > 4:
                    df = self._flag_row(
                        df,
                        pd.Series([True if i == idx else False for i in df.index], index=df.index),
                        "FORMAT_ARTIFACT"
                    )
                    logger.debug(f"Excessive decimals in {lab_name}: {value_raw_str}")
                    continue

            # Check for concatenation errors (e.g., "52.6=1946" where reference got appended)
            # Skip simple trailing characters like "=" or "= NR"
            if pd.notna(value) and isinstance(value_raw_str, str):
                cleaned = value_raw_str.replace(',', '.').strip()

                # Skip if value_raw is purely qualitative (no digits, or doesn't start with digit/sign)
                has_digits = bool(re.search(r'\d', cleaned))
                starts_with_number = bool(re.match(r'^[\d\.\-\+<>≤≥]', cleaned))

                if has_digits and starts_with_number:
                    # Check for concatenation pattern: number followed by = and more numbers
                    # e.g., "52.6=1946", "0.8= 30", "0.4= 15"
                    concat_pattern = re.match(r'^[\d\.\-\+<>≤≥]+\s*=\s*\d+', cleaned)
                    if concat_pattern:
                        df = self._flag_row(
                            df,
                            pd.Series([True if i == idx else False for i in df.index], index=df.index),
                            "FORMAT_ARTIFACT"
                        )
                        logger.debug(f"Concatenation error in {lab_name}: {value_raw_str}")
                        continue

            # Check for magnitude mismatch using defined biological limits only
            # Skip qualitative values (value_raw doesn't start with a digit)
            if pd.notna(value) and pd.notna(lab_name) and value > 0:
                # Skip if value_raw is qualitative (doesn't start with digit or sign)
                if not re.match(r'^[\d\.\-\+<>≤≥]', value_raw_str):
                    continue

                spec = self.lab_specs.specs.get(lab_name, {})
                bio_max = spec.get('biological_max')

                # Only flag if we have explicit biological_max defined and value exceeds it
                if bio_max is not None and value > bio_max:
                    df = self._flag_row(
                        df,
                        pd.Series([True if i == idx else False for i in df.index], index=df.index),
                        "IMPOSSIBLE_VALUE"
                    )
                    logger.debug(
                        f"Value exceeds biological max for {lab_name}: {value} > {bio_max}"
                    )

        return df

    # =========================================================================
    # 5. Reference Range Consistency Checks
    # =========================================================================

    def _check_reference_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-validate values against extracted reference ranges.

        Checks:
        - ref_min > ref_max (invalid range)
        - Value extremely far outside extracted range (100x)
        """
        value_col = 'value' if 'value' in df.columns else 'value_primary'
        ref_min_col = 'reference_min' if 'reference_min' in df.columns else 'reference_min_primary'
        ref_max_col = 'reference_max' if 'reference_max' in df.columns else 'reference_max_primary'
        lab_name_col = 'lab_name' if 'lab_name' in df.columns else 'lab_name_standardized'

        if ref_min_col not in df.columns or ref_max_col not in df.columns:
            return df

        for idx, row in df.iterrows():
            value = row.get(value_col)
            ref_min = row.get(ref_min_col)
            ref_max = row.get(ref_max_col)
            lab_name = row.get(lab_name_col)

            # Check ref_min > ref_max (inverted range)
            if pd.notna(ref_min) and pd.notna(ref_max):
                if ref_min > ref_max:
                    df = self._flag_row(
                        df,
                        pd.Series([True if i == idx else False for i in df.index], index=df.index),
                        "RANGE_INCONSISTENCY"
                    )
                    logger.debug(f"Inverted range for {lab_name}: {ref_min} > {ref_max}")
                    continue

            # Check for extreme deviation from reference range
            if pd.notna(value) and pd.notna(ref_min) and pd.notna(ref_max):
                range_size = ref_max - ref_min
                if range_size > 0:
                    # Calculate how far outside the range the value is
                    if value < ref_min:
                        deviation = ref_min - value
                    elif value > ref_max:
                        deviation = value - ref_max
                    else:
                        deviation = 0

                    # Flag if deviation is more than 10x the range size
                    if deviation > range_size * 10:
                        df = self._flag_row(
                            df,
                            pd.Series([True if i == idx else False for i in df.index], index=df.index),
                            "EXTREME_DEVIATION"
                        )
                        logger.debug(
                            f"Extreme deviation for {lab_name}: {value} "
                            f"(range: {ref_min}-{ref_max}, deviation: {deviation:.1f})"
                        )

        return df


def validate_lab_results(df: pd.DataFrame, lab_specs: LabSpecsConfig) -> pd.DataFrame:
    """Convenience function to validate lab results DataFrame.

    Args:
        df: DataFrame with lab results
        lab_specs: LabSpecsConfig instance

    Returns:
        DataFrame with validation flags applied
    """
    validator = ValueValidator(lab_specs)
    return validator.validate(df)
