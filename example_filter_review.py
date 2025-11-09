"""
Example script showing how to filter results using review flags.

After running main.py, use this to identify items that need human review.
"""

import pandas as pd
from pathlib import Path

def show_review_summary(csv_path: str):
    """Show summary of items needing review."""
    df = pd.read_csv(csv_path)

    print(f"Total results: {len(df)}")
    print()

    # Check if review columns exist
    if 'needs_review' not in df.columns:
        print("⚠️  This CSV doesn't have review flags yet.")
        print()
        print("The review flags are automatically added when you run main.py.")
        print("If this is an old CSV, you can add the flags by running:")
        print()
        print("  from edge_case_detection import EdgeCaseDetector")
        print("  detector = EdgeCaseDetector()")
        print("  df = detector.identify_edge_cases(df)")
        print("  df.to_csv('output_with_flags.csv', index=False)")
        print()
        print("Or simply re-run main.py to regenerate all.csv with the flags.")
        return

    # Overall review statistics
    needs_review = df[df['needs_review'] == True]
    low_confidence = df[df['confidence_score'] < 0.7]

    print("REVIEW STATISTICS")
    print("=" * 60)
    print(f"Items needing review: {len(needs_review)} ({len(needs_review)/len(df)*100:.1f}%)")
    print(f"High-priority items (confidence < 0.7): {len(low_confidence)} ({len(low_confidence)/len(df)*100:.1f}%)")
    print()

    if len(needs_review) > 0:
        # Breakdown by category
        print("EDGE CASE BREAKDOWN")
        print("=" * 60)

        # Count occurrences of each reason
        reason_counts = {}
        for reasons in needs_review['review_reason'].dropna():
            for reason in reasons.split(';'):
                reason = reason.strip()
                if reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{reason}: {count}")
        print()

        # Show high-priority examples
        if len(low_confidence) > 0:
            print("HIGH-PRIORITY EXAMPLES (Confidence < 0.7)")
            print("=" * 60)
            for idx, row in low_confidence.head(5).iterrows():
                print(f"Lab: {row['lab_name_raw']}")
                print(f"Value: {row['value_raw']}")
                print(f"Confidence: {row['confidence_score']:.2f}")
                print(f"Reasons: {row['review_reason']}")
                print()

    # Show how to filter
    print("FILTERING EXAMPLES")
    print("=" * 60)
    print(f"# Get all items needing review:")
    print(f"needs_review = df[df['needs_review'] == True]  # {len(needs_review)} items")
    print()
    print(f"# Get high-priority items:")
    print(f"high_priority = df[df['confidence_score'] < 0.7]  # {len(low_confidence)} items")
    print()
    print(f"# Get clean data (exclude flagged items):")
    print(f"clean_data = df[df['needs_review'] == False]  # {len(df[df['needs_review'] == False])} items")
    print()

    # Filter by specific issues
    if len(needs_review) > 0:
        inequality = df[df['review_reason'].str.contains('INEQUALITY_IN_VALUE', na=False)]
        duplicates = df[df['review_reason'].str.contains('DUPLICATE_TEST_NAME', na=False)]

        print(f"# Filter by specific issues:")
        print(f"inequality_issues = df[df['review_reason'].str.contains('INEQUALITY_IN_VALUE', na=False)]  # {len(inequality)} items")
        print(f"duplicate_tests = df[df['review_reason'].str.contains('DUPLICATE_TEST_NAME', na=False)]  # {len(duplicates)} items")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default to all.csv in output directory
        csv_path = "output/all.csv"

    if not Path(csv_path).exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        print()
        print("Usage:")
        print("  python example_filter_review.py [path/to/all.csv]")
        print()
        print("Example:")
        print('  python example_filter_review.py "output/all.csv"')
        sys.exit(1)

    show_review_summary(csv_path)
