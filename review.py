"""
Human-in-the-Loop Review System for Edge Cases

Identifies low-confidence extractions and provides an interactive interface
for human review and correction.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys
from datetime import datetime


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


class ReviewInterface:
    """Interactive interface for reviewing edge cases."""

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.reviews_file = self.output_path / "human_reviews.json"
        self.load_existing_reviews()

    def load_existing_reviews(self):
        """Load previously completed reviews."""
        if self.reviews_file.exists():
            with open(self.reviews_file, 'r') as f:
                self.reviews = json.load(f)
        else:
            self.reviews = {}

    def save_review(self, review_id: str, review_data: dict):
        """Save a review decision."""
        self.reviews[review_id] = {
            **review_data,
            'reviewed_at': datetime.now().isoformat(),
            'reviewer': 'human'
        }

        with open(self.reviews_file, 'w') as f:
            json.dump(self.reviews, f, indent=2)

    def generate_review_id(self, row: pd.Series) -> str:
        """Generate unique ID for a review item."""
        return f"{row.get('source_file', 'unknown')}_{row.get('page_number', 0)}_{row.name}"

    def review_batch(self, df: pd.DataFrame, max_items: Optional[int] = None) -> pd.DataFrame:
        """
        Present edge cases for review in an interactive session.

        Args:
            df: DataFrame with needs_review column
            max_items: Maximum number of items to review (None = all)

        Returns:
            DataFrame with review decisions applied
        """
        review_items = df[df['needs_review'] == True].copy()

        if len(review_items) == 0:
            print("✅ No items need review!")
            return df

        if max_items:
            review_items = review_items.head(max_items)

        print(f"\n{'='*80}")
        print(f"HUMAN REVIEW SESSION")
        print(f"{'='*80}")
        print(f"Found {len(review_items)} items that need review")
        print(f"Reviewing up to {max_items if max_items else 'all'} items\n")

        reviewed_df = df.copy()

        for idx, row in review_items.iterrows():
            review_id = self.generate_review_id(row)

            # Skip if already reviewed
            if review_id in self.reviews:
                print(f"⏭️  Skipping already reviewed item {idx}")
                continue

            print(f"\n{'-'*80}")
            print(f"Item {idx + 1}/{len(review_items)}")
            print(f"Confidence: {row.get('confidence_score', 1.0):.2f}")
            print(f"Reason: {row.get('review_reason', 'Unknown')}")
            print(f"{'-'*80}")

            self._display_item(row)

            # Get user decision
            decision = self._get_user_decision(row)

            if decision['action'] == 'skip_session':
                print("\n🛑 Review session ended by user")
                break

            # Apply decision
            if decision['action'] == 'accept':
                reviewed_df.loc[idx, 'needs_review'] = False
                reviewed_df.loc[idx, 'human_verified'] = True
            elif decision['action'] == 'correct':
                # Apply corrections
                for field, value in decision['corrections'].items():
                    reviewed_df.loc[idx, field] = value
                reviewed_df.loc[idx, 'needs_review'] = False
                reviewed_df.loc[idx, 'human_verified'] = True
                reviewed_df.loc[idx, 'human_corrected'] = True
            elif decision['action'] == 'delete':
                reviewed_df.loc[idx, 'should_delete'] = True
                reviewed_df.loc[idx, 'needs_review'] = False

            # Save review
            self.save_review(review_id, decision)

            print(f"✅ Review saved\n")

        return reviewed_df

    def _display_item(self, row: pd.Series):
        """Display a review item's details."""
        print(f"\n📄 Source: {row.get('source_file', 'Unknown')}")
        print(f"📄 Page: {row.get('page_number', 'Unknown')}")
        print(f"\n🧪 Lab Test:")
        print(f"   Name: {row.get('lab_name_raw', 'N/A')}")
        print(f"   Value: {row.get('value_raw', 'NULL')}")
        print(f"   Unit: {row.get('lab_unit_raw', 'NULL')}")
        print(f"   Reference Range: {row.get('reference_range', 'NULL')}")
        if pd.notna(row.get('comments')):
            print(f"   Comments: {row.get('comments')}")
        print(f"\n📝 Source Text: {row.get('source_text', 'N/A')}")

        # Show image path if available
        if pd.notna(row.get('source_file')):
            doc_name = Path(row['source_file']).stem
            page_num = str(row.get('page_number', '')).zfill(3)
            image_path = self.output_path / doc_name / f"{doc_name}.{page_num}.jpg"
            if image_path.exists():
                print(f"\n🖼️  Image: {image_path}")

    def _get_user_decision(self, row: pd.Series) -> dict:
        """Get user's review decision."""
        print(f"\nWhat would you like to do?")
        print(f"  [a] Accept as-is")
        print(f"  [c] Correct values")
        print(f"  [d] Delete (false positive)")
        print(f"  [s] Skip this item")
        print(f"  [q] Quit review session")

        while True:
            choice = input("\nYour choice: ").strip().lower()

            if choice == 'a':
                return {'action': 'accept'}
            elif choice == 'c':
                return self._get_corrections(row)
            elif choice == 'd':
                return {'action': 'delete', 'reason': input("Reason for deletion: ")}
            elif choice == 's':
                return {'action': 'skip'}
            elif choice == 'q':
                return {'action': 'skip_session'}
            else:
                print("Invalid choice. Please try again.")

    def _get_corrections(self, row: pd.Series) -> dict:
        """Get field corrections from user."""
        print("\nEnter corrections (press Enter to keep current value):")

        corrections = {}

        fields_to_correct = [
            ('value_raw', 'Value'),
            ('lab_unit_raw', 'Unit'),
            ('reference_range', 'Reference Range'),
            ('reference_min_raw', 'Reference Min'),
            ('reference_max_raw', 'Reference Max'),
        ]

        for field, label in fields_to_correct:
            current = row.get(field, 'NULL')
            new_value = input(f"  {label} [{current}]: ").strip()

            if new_value:
                # Handle special values
                if new_value.lower() == 'null':
                    corrections[field] = None
                elif field in ['reference_min_raw', 'reference_max_raw']:
                    try:
                        corrections[field] = float(new_value)
                    except ValueError:
                        print(f"    Warning: Could not convert '{new_value}' to number, skipping")
                else:
                    corrections[field] = new_value

        return {
            'action': 'correct',
            'corrections': corrections,
            'note': input("Optional note about this correction: ").strip()
        }


def run_review_session(csv_path: str, output_path: str, max_items: int = 20, report_only: bool = False):
    """
    Run an interactive review session for a processed document.

    Args:
        csv_path: Path to the CSV file with extracted results
        output_path: Path to output directory
        max_items: Maximum number of items to review in this session
        report_only: If True, only generate report without interactive review
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} lab results")

    # Detect edge cases
    print("\nIdentifying edge cases...")
    detector = EdgeCaseDetector()
    df = detector.identify_edge_cases(df)

    needs_review = df['needs_review'].sum()
    print(f"Found {needs_review} items that need review")

    if needs_review == 0:
        print("✅ No edge cases detected! Extraction quality is excellent.")
        return

    # Show summary of edge case types
    print("\nEdge case breakdown:")
    reasons = df[df['needs_review']]['review_reason'].value_counts()
    for reason, count in reasons.items():
        print(f"  {reason}: {count}")

    # Show low confidence items
    low_conf = df[df['confidence_score'] < 0.7].sort_values('confidence_score')
    if len(low_conf) > 0:
        print(f"\n⚠️  {len(low_conf)} items with confidence < 0.7:")
        for idx, row in low_conf.head(5).iterrows():
            print(f"   • {row['lab_name_raw']}: {row['value_raw']} (confidence: {row['confidence_score']:.2f})")

    # Save report with edge case flags
    report_path = Path(csv_path).parent / f"{Path(csv_path).stem}_with_review_flags.csv"
    df.to_csv(report_path, index=False)
    print(f"\n💾 Report with review flags saved to: {report_path}")

    if report_only:
        print("\n📋 Report-only mode: Use the interactive mode to review and correct edge cases")
        return

    # Run interactive review
    interface = ReviewInterface(Path(output_path))
    reviewed_df = interface.review_batch(df, max_items=max_items)

    # Save reviewed data
    reviewed_path = Path(csv_path).parent / f"{Path(csv_path).stem}_reviewed.csv"
    reviewed_df.to_csv(reviewed_path, index=False)
    print(f"\n💾 Reviewed data saved to: {reviewed_path}")

    # Show statistics
    verified_count = reviewed_df.get('human_verified', pd.Series([False])).sum()
    corrected_count = reviewed_df.get('human_corrected', pd.Series([False])).sum()
    deleted_count = reviewed_df.get('should_delete', pd.Series([False])).sum()

    print(f"\n📊 Review Statistics:")
    print(f"   Verified as correct: {verified_count}")
    print(f"   Corrected: {corrected_count}")
    print(f"   Marked for deletion: {deleted_count}")
    print(f"   Still need review: {reviewed_df['needs_review'].sum()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python review.py <csv_path> [output_path] [max_items] [--report-only]")
        print("\nModes:")
        print("  Interactive (default): Review and correct edge cases interactively")
        print("  Report-only (--report-only): Only identify and report edge cases")
        print("\nExamples:")
        print("  # Interactive review (up to 20 items)")
        print("  python review.py output/2024-11-20/2024-11-20.csv output 20")
        print("\n  # Report-only mode (no interactive review)")
        print("  python review.py output/2024-11-20/2024-11-20.csv output 0 --report-only")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else str(Path(csv_path).parent.parent)
    max_items = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != '--report-only' else 20
    report_only = '--report-only' in sys.argv

    run_review_session(csv_path, output_path, max_items, report_only)
