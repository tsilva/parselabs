"""
Edge Case Detection for Lab Results

Identifies extraction results that may need human review based on
various quality indicators and assigns confidence scores.
"""

import re

import pandas as pd


class EdgeCaseDetector:
    """Identifies extraction results that need human review."""

    def __init__(self):
        self.edge_case_rules = [
            self._check_null_value_with_source_text,
            self._check_text_in_comments_not_value,
            self._check_missing_unit_for_numeric_value,
            self._check_unusual_value_patterns,
            self._check_complex_reference_ranges,
            self._check_multiple_values_same_test,
        ]

    def identify_edge_cases(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify rows that need human review.

        Returns DataFrame with additional columns:
        - review_needed: bool
        - review_reason: str
        - review_confidence: float (0-1)
        """
        df = df.copy()
        df['review_needed'] = False
        df['review_reason'] = ''
        df['review_confidence'] = 1.0

        # Apply each edge case detection rule
        for rule in self.edge_case_rules:
            df = rule(df)

        return df

    def _check_null_value_with_source_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag cases where value is null but source_text suggests there's a value."""
        mask = (
            df['value_raw'].isna() &
            df['source_text'].notna() &
            df['source_text'].str.len() > 10  # Has meaningful source text
        )

        df.loc[mask, 'review_needed'] = True
        df.loc[mask, 'review_reason'] += 'NULL_VALUE_WITH_SOURCE; '
        df.loc[mask, 'review_confidence'] = df.loc[mask, 'review_confidence'] * 0.5

        return df

    def _check_text_in_comments_not_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-populate value_raw from comments for qualitative results.

        Qualitative tests (NEGATIVO, POSITIVO, NORMAL, etc.) often have their
        results in the comments field. Instead of flagging for review, we
        auto-populate value_raw from comments since this is valid extraction.
        """
        # Skip if comments column doesn't exist
        if 'comments' not in df.columns:
            return df

        qualitative_terms = [
            'NEGATIVO', 'POSITIVO', 'NORMAL', 'ANORMAL',
            'NEGATIVA', 'POSITIVA', 'AUSENTE', 'PRESENTE',
            'RAROS', 'RARAS', 'ABUNDANTES', 'AMARELA', 'AMARELO',
            'REAGENTE', 'NAO REAGENTE', 'NÃO REAGENTE',
        ]

        pattern = '|'.join(qualitative_terms)

        # Convert comments to string type for string operations
        comments_str = df['comments'].astype(str)

        mask = (
            df['value_raw'].isna() &
            df['comments'].notna() &
            comments_str.str.contains(pattern, case=False, na=False)
        )

        # Auto-populate value_raw from comments (this is valid extraction, not an error)
        # Cast to object type first to avoid dtype incompatibility warning
        if mask.any():
            df['value_raw'] = df['value_raw'].astype(object)
            df.loc[mask, 'value_raw'] = df.loc[mask, 'comments']

        # Don't flag for review - this is correctly extracted qualitative data

        return df

    def _check_missing_unit_for_numeric_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag numeric values without units (only if not resolved during standardization)."""
        # Convert to numeric to check if it's a number
        numeric_mask = pd.to_numeric(df['value_raw'], errors='coerce').notna()

        # Check if lab_unit_standardized column exists and was populated
        has_standardized_unit = (
            ('lab_unit_standardized' in df.columns) &
            df['lab_unit_standardized'].notna() &
            (df['lab_unit_standardized'] != '') &
            (df['lab_unit_standardized'] != 'null')
        )

        mask = (
            numeric_mask &
            df['lab_unit_raw'].isna() &
            ~has_standardized_unit &  # Only flag if standardization didn't resolve it
            ~df['lab_name_raw'].str.contains('pH|ratio|index|score', case=False, na=False)  # Exclude unitless tests
        )

        df.loc[mask, 'review_needed'] = True
        df.loc[mask, 'review_reason'] += 'NUMERIC_NO_UNIT; '
        df.loc[mask, 'review_confidence'] = df.loc[mask, 'review_confidence'] * 0.8

        return df

    def _check_unusual_value_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag unusual patterns in values (like '<175' appearing as a standalone value)."""
        # Check for inequality operators in value_raw
        # Extra safety: also check string representation isn't "nan" or "None"
        value_raw_str = df['value_raw'].astype(str)
        is_actual_value = (
            df['value_raw'].notna() &
            ~value_raw_str.isin(['nan', 'None', ''])
        )
        mask = (
            is_actual_value &
            value_raw_str.str.contains(r'^[<>≤≥]', na=False, regex=True)
        )

        df.loc[mask, 'review_needed'] = True
        df.loc[mask, 'review_reason'] += 'INEQUALITY_IN_VALUE; '
        df.loc[mask, 'review_confidence'] = df.loc[mask, 'review_confidence'] * 0.6

        return df

    def _check_complex_reference_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse complex multi-tier reference ranges (like Vitamin D).

        For ranges with Deficiency/Insufficiency/Sufficiency/Toxicity tiers,
        extract the Sufficiency range as the normal reference range.
        """

        # Convert to string for string operations
        ref_range_str = df['reference_range'].astype(str)

        # Find rows with multi-tier reference ranges that haven't been parsed
        mask = (
            df['reference_range'].notna() &
            ref_range_str.str.contains('suficiência|sufficiency', case=False, na=False) &
            df['reference_min_raw'].isna() &
            df['reference_max_raw'].isna()
        )

        if not mask.any():
            return df

        # Parse the Sufficiency range for each matching row
        # Pattern matches: "Suficiência: 30 - 100" or "Suficiência: 30-100" or "Sufficiency: 30 - 100"
        # Use negative lookbehind to avoid matching "Insuficiência"
        sufficiency_pattern = re.compile(
            r'(?<!in)sufici[êe]ncia\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*[-–a]\s*(\d+(?:[.,]\d+)?)',
            re.IGNORECASE
        )

        for idx in df[mask].index:
            ref_text = str(df.loc[idx, 'reference_range'])
            match = sufficiency_pattern.search(ref_text)
            if match:
                # Extract min and max, handling decimal comma
                min_val = float(match.group(1).replace(',', '.'))
                max_val = float(match.group(2).replace(',', '.'))
                df.loc[idx, 'reference_min_raw'] = min_val
                df.loc[idx, 'reference_max_raw'] = max_val

        # Don't flag for review - the Sufficiency range has been extracted

        return df

    def _check_multiple_values_same_test(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag cases where same STANDARDIZED test name appears multiple times on same page."""
        # Group by document, page, and STANDARDIZED lab name (not raw)
        # This avoids false positives where the same raw name maps to different standardized names
        # (e.g., "Basophils" absolute count vs "Basophils (%)" percentage)
        if 'page_number' in df.columns and 'source_file' in df.columns and 'lab_name_standardized' in df.columns:
            duplicates = df.groupby(['source_file', 'page_number', 'lab_name_standardized']).size()
            duplicate_tests = duplicates[duplicates > 1].index

            for source_file, page_num, lab_name in duplicate_tests:
                mask = (
                    (df['source_file'] == source_file) &
                    (df['page_number'] == page_num) &
                    (df['lab_name_standardized'] == lab_name)
                )
                df.loc[mask, 'review_needed'] = True
                df.loc[mask, 'review_reason'] += 'DUPLICATE_TEST_NAME; '
                df.loc[mask, 'review_confidence'] = df.loc[mask, 'review_confidence'] * 0.7

        return df
