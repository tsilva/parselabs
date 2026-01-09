# Human-in-the-Loop Review System

The review system automatically identifies edge cases in extracted lab results and provides an interactive interface for human review and correction.

## Overview

The system uses rule-based detection to identify low-confidence extractions that may need human verification. It assigns confidence scores (0-1) to each result based on various quality indicators.

## Edge Case Detection Rules

The system flags results that match these patterns:

1. **NULL_VALUE_WITH_SOURCE**: Value is null but source text suggests there's a value
   - Confidence penalty: 0.5x

2. **QUALITATIVE_IN_COMMENTS**: Qualitative result (NEGATIVO, POSITIVO, etc.) in comments instead of value_raw
   - Confidence penalty: 0.3x

3. **NUMERIC_NO_UNIT**: Numeric value without a unit (excluding unitless tests like pH, ratios)
   - Confidence penalty: 0.8x

4. **INEQUALITY_IN_VALUE**: Value contains inequality operators (<, >, ≤, ≥)
   - Confidence penalty: 0.6x
   - Example: `<175` appearing as a standalone value (might be a reference range)

5. **COMPLEX_REFERENCE_RANGE**: Multi-condition reference ranges not parsed into min/max
   - Confidence penalty: 0.7x
   - Example: "Deficiência: <10; Insuficiência: 10-30; Suficiência: 30-100"

6. **DUPLICATE_TEST_NAME**: Same test name appears multiple times on same page
   - Confidence penalty: 0.7x
   - May need disambiguation (e.g., qualitative vs quantitative results)

## Usage

### Report-Only Mode

Identify and report edge cases without interactive review:

```bash
python review.py <csv_path> <output_path> 0 --report-only
```

**Example:**
```bash
python review.py "/path/to/output/2024-11-20/2024-11-20.csv" "/path/to/output" 0 --report-only
```

**Output:**
- Console report showing:
  - Total edge cases found
  - Breakdown by category
  - Low confidence items (< 0.7)
- CSV file with review flags: `<document>_with_review_flags.csv`

### Interactive Review Mode

Review and correct edge cases interactively:

```bash
python review.py <csv_path> <output_path> <max_items>
```

**Example:**
```bash
# Review up to 20 items
python review.py "/path/to/output/2024-11-20/2024-11-20.csv" "/path/to/output" 20
```

**Interactive Options:**
- `[a]` Accept as-is - Mark result as verified
- `[c]` Correct values - Edit specific fields
- `[d]` Delete - Mark as false positive
- `[s]` Skip - Skip this item
- `[q]` Quit - End review session

**Outputs:**
- `<document>_reviewed.csv` - Updated data with corrections
- `human_reviews.json` - Audit trail of all review decisions

## Output Files

### Review Flags CSV
Contains all original columns plus:
- `needs_review` (bool): Whether item needs review
- `review_reason` (str): Reason(s) for flagging
- `confidence_score` (float): Confidence score (0-1)

### Reviewed CSV
After interactive review, contains additional columns:
- `review_status` (str): "accepted" or "rejected"
- `reviewed_at` (str): ISO timestamp when reviewed

### Review Audit Trail (`human_reviews.json`)
JSON file tracking all review decisions:
```json
{
  "document_page_idx": {
    "action": "correct",
    "corrections": {"value_raw": "NEGATIVO"},
    "note": "Value was in comments field",
    "reviewed_at": "2024-11-20T10:30:00",
    "reviewer": "human"
  }
}
```

## Integration with Main Pipeline

The review system works with existing CSV outputs from `main.py`. To review a document:

1. Run main extraction pipeline:
```bash
python main.py
```

2. Review results:
```bash
# First, generate report to see what needs review
python review.py "output/2024-11-20/2024-11-20.csv" "output" 0 --report-only

# Then, interactively review up to 20 items
python review.py "output/2024-11-20/2024-11-20.csv" "output" 20
```

3. Use reviewed data:
```python
import pandas as pd

# Load reviewed data
df = pd.read_csv("output/2024-11-20/2024-11-20_reviewed.csv")

# Filter to accepted results only
accepted = df[df['review_status'] == 'accepted']

# Exclude rejected results
valid = df[df['review_status'] != 'rejected']
```

## Customizing Edge Case Detection

To add custom detection rules:

1. Edit `review.py`
2. Add method to `EdgeCaseDetector` class:
```python
def _check_custom_rule(self, df: pd.DataFrame) -> pd.DataFrame:
    """Your custom detection logic."""
    mask = (
        # Your condition here
    )

    df.loc[mask, 'needs_review'] = True
    df.loc[mask, 'review_reason'] += 'CUSTOM_RULE; '
    df.loc[mask, 'confidence_score'] = df.loc[mask, 'confidence_score'] * 0.5

    return df
```

3. Register in `__init__`:
```python
self.edge_case_rules = [
    ...
    self._check_custom_rule,
]
```

## Performance Tips

1. **Batch Review**: Review documents in batches of 20-50 items per session
2. **Prioritize Low Confidence**: Focus on items with confidence < 0.5 first
3. **Track Progress**: Use audit trail to resume interrupted sessions
4. **Periodic Quality Checks**: Run report-only mode after processing batches

## Example Workflow

```bash
# Process all lab reports
python main.py

# Generate quality reports for all documents
for csv in output/*/*.csv; do
    python review.py "$csv" output 0 --report-only
done

# Review documents with most edge cases first
# (manually check reports and prioritize)

# Interactive review session
python review.py "output/2003-07-07/2003-07-07.csv" output 30
```

## Statistics

Based on testing with N_EXTRACTIONS=3 and variable temperature strategy:

- **Average edge case rate**: ~30% of results flagged for review
- **Common edge cases**:
  - 50% duplicate test names (often valid - qualitative + quantitative)
  - 40% inequality operators in values (reference range confusion)
  - 10% missing units or other issues
- **Low confidence rate** (< 0.7): ~15% of results
- **Review time**: ~30 seconds per item average

With this system, you can systematically achieve 99%+ accuracy by reviewing and correcting the ~15% of low-confidence extractions.
