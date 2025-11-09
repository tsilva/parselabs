"""
Edge Case Detection for Lab Results

Identifies extraction results that may need human review based on
various quality indicators and assigns confidence scores.
"""

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
        - needs_review: bool
        - review_reason: str
        - confidence_score: float (0-1)
        """
        df = df.copy()
        df['needs_review'] = False
        df['review_reason'] = ''
        df['confidence_score'] = 1.0

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

        df.loc[mask, 'needs_review'] = True
        df.loc[mask, 'review_reason'] += 'NULL_VALUE_WITH_SOURCE; '
        df.loc[mask, 'confidence_score'] = df.loc[mask, 'confidence_score'] * 0.5

        return df

    def _check_text_in_comments_not_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag cases where comments contains qualitative results but value_raw is null."""
        # Skip if comments column doesn't exist
        if 'comments' not in df.columns:
            return df

        qualitative_terms = [
            'NEGATIVO', 'POSITIVO', 'NORMAL', 'ANORMAL',
            'NEGATIVA', 'POSITIVA', 'AUSENTE', 'PRESENTE',
            'RAROS', 'RARAS', 'ABUNDANTES', 'AMARELA'
        ]

        pattern = '|'.join(qualitative_terms)

        # Convert comments to string type for string operations
        comments_str = df['comments'].astype(str)

        mask = (
            df['value_raw'].isna() &
            df['comments'].notna() &
            comments_str.str.contains(pattern, case=False, na=False)
        )

        df.loc[mask, 'needs_review'] = True
        df.loc[mask, 'review_reason'] += 'QUALITATIVE_IN_COMMENTS; '
        df.loc[mask, 'confidence_score'] = df.loc[mask, 'confidence_score'] * 0.3

        return df

    def _check_missing_unit_for_numeric_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag numeric values without units (might be missing)."""
        # Convert to numeric to check if it's a number
        numeric_mask = pd.to_numeric(df['value_raw'], errors='coerce').notna()

        mask = (
            numeric_mask &
            df['lab_unit_raw'].isna() &
            ~df['lab_name_raw'].str.contains('pH|ratio|index|score', case=False, na=False)  # Exclude unitless tests
        )

        df.loc[mask, 'needs_review'] = True
        df.loc[mask, 'review_reason'] += 'NUMERIC_NO_UNIT; '
        df.loc[mask, 'confidence_score'] = df.loc[mask, 'confidence_score'] * 0.8

        return df

    def _check_unusual_value_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag unusual patterns in values (like '<175' appearing as a standalone value)."""
        # Check for inequality operators in value_raw
        value_raw_str = df['value_raw'].astype(str)
        mask = (
            df['value_raw'].notna() &
            value_raw_str.str.contains(r'^[<>≤≥]', na=False, regex=True)
        )

        df.loc[mask, 'needs_review'] = True
        df.loc[mask, 'review_reason'] += 'INEQUALITY_IN_VALUE; '
        df.loc[mask, 'confidence_score'] = df.loc[mask, 'confidence_score'] * 0.6

        return df

    def _check_complex_reference_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag complex multi-condition reference ranges."""
        # Convert to string for string operations
        ref_range_str = df['reference_range'].astype(str)

        mask = (
            df['reference_range'].notna() &
            (
                ref_range_str.str.contains(';', na=False) |
                ref_range_str.str.contains('deficiência|insuficiência|suficiência', case=False, na=False)
            ) &
            df['reference_min_raw'].isna() &
            df['reference_max_raw'].isna()
        )

        df.loc[mask, 'needs_review'] = True
        df.loc[mask, 'review_reason'] += 'COMPLEX_REFERENCE_RANGE; '
        df.loc[mask, 'confidence_score'] = df.loc[mask, 'confidence_score'] * 0.7

        return df

    def _check_multiple_values_same_test(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag cases where same test name appears multiple times on same page (might need disambiguation)."""
        # Group by document, page, and lab name
        if 'page_number' in df.columns and 'source_file' in df.columns:
            duplicates = df.groupby(['source_file', 'page_number', 'lab_name_raw']).size()
            duplicate_tests = duplicates[duplicates > 1].index

            for source_file, page_num, lab_name in duplicate_tests:
                mask = (
                    (df['source_file'] == source_file) &
                    (df['page_number'] == page_num) &
                    (df['lab_name_raw'] == lab_name)
                )
                df.loc[mask, 'needs_review'] = True
                df.loc[mask, 'review_reason'] += 'DUPLICATE_TEST_NAME; '
                df.loc[mask, 'confidence_score'] = df.loc[mask, 'confidence_score'] * 0.7

        return df
