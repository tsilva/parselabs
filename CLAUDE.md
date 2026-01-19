# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Labs Parser is a Python tool that uses AI (via OpenRouter API) to extract laboratory test results from medical documents (PDFs/images). It converts unstructured lab reports into structured CSV/Excel data with standardized test names and units.

**Design Principle:** Extraction is objective (what's on the page). Analysis is subjective (health status, custom ranges) and belongs in a separate review tool.

## Key Commands

### Running the Parser
```bash
# Using a profile (required):
python extract.py --profile tsilva

# List available profiles:
python extract.py --list-profiles

# Override settings:
python extract.py --profile tsilva --model google/gemini-2.5-pro

# Data integrity validation:
python test.py

# Unified viewer (browse + review):
python viewer.py --profile tsilva

# Legacy review tool:
python review.py --profile tsilva

# Legacy browse tool:
python browse.py --profile tsilva
```

### Development
The `utils/` directory contains helper scripts for building and maintaining configuration:
- `build_lab_specs_conversions.py` - Generate unit conversion factors for lab_specs.json
- `build_lab_specs_ranges.py` - Generate healthy ranges for lab_specs.json
- `sort_lab_specs.py` - Sort lab specifications alphabetically
- `analyze_unknowns.py` - Analyze $UNKNOWN$ values in extracted results

## Architecture

### Core Pipeline (extract.py)

The processing pipeline has 3 main stages:

1. **PDF Processing** (`process_single_pdf`)
   - Converts each PDF page to preprocessed JPG images (grayscale, contrast-enhanced)
   - Each page is processed independently

2. **Extraction** (`extract_labs_from_page_image`)
   - Extracts structured lab data directly from page images using vision models
   - Returns `HealthLabReport` with nested `LabResult` objects validated by Pydantic

3. **Normalization & Mapping**
   - Maps raw lab names/units to standardized enums via config files
   - Converts values to primary units using conversion factors
   - Deduplicates by (date, lab_name) pairs

4. **Value Validation** (`validation.py`)
   - Detects extraction errors from the data itself (no source image re-check)
   - Flags suspicious values for review in `review.py`

### Value Validation (validation.py)

The `ValueValidator` class detects extraction errors by analyzing the data itself:

| Category | Reason Codes | Description |
|----------|--------------|-------------|
| Biological Plausibility | `NEGATIVE_VALUE`, `IMPOSSIBLE_VALUE`, `PERCENTAGE_BOUNDS` | Values outside biological limits |
| Inter-Lab Relationships | `RELATIONSHIP_MISMATCH` | Calculated values don't match (e.g., LDL Friedewald formula) |
| Temporal Consistency | `TEMPORAL_ANOMALY` | Implausible change rate between tests |
| Format Artifacts | `FORMAT_ARTIFACT` | Concatenation errors (e.g., "52.6=1946") |
| Reference Ranges | `RANGE_INCONSISTENCY`, `EXTREME_DEVIATION` | Reference range issues or value far outside range |

**Configuration in `lab_specs.json`:**
```json
{
  "Blood - Hemoglobin (Hgb)": {
    "biological_min": 0,
    "biological_max": 25,
    "max_daily_change": 2.0
  },
  "_relationships": [
    {
      "name": "LDL_FRIEDEWALD",
      "formula": "Blood - Total Cholesterol - Blood - HDL Cholesterol - (Blood - Triglycerides / 5)",
      "target": "Blood - LDL Cholesterol",
      "tolerance_percent": 15
    }
  ]
}
```

Flagged rows appear in `review.py` under "Needs Review" filter with `review_needed=True`, `review_reason`, and `review_confidence` columns.

### Unified Viewer (viewer.py)

The unified viewer combines browse and review functionality into a single tool:

**Layout:**
- Left: Data table with all results (click to select)
- Right: Tabbed panel with Plot, Source image, and Details tabs
- Bottom-right: Review actions (Accept/Reject/Skip)

**Features:**
- **From browse.py**: Time-series plots with dual reference ranges (lab_specs + PDF), multi-lab selection, CSV export
- **From review.py**: Accept/Reject workflow, keyboard shortcuts (Y/N/S), review persistence to JSON files

**Filters (all always visible):**
- Lab name multi-select
- Abnormal only / Latest only checkboxes
- Review status dropdown (All, Unreviewed, Needs Review, Low Confidence, Accepted, Rejected)

**Keyboard shortcuts:**
- `Y` = Accept, `N` = Reject, `S` = Skip
- Arrow keys or `j`/`k` = Navigate

**Ports:**
- `viewer.py` runs on port 7862
- `review.py` (legacy) runs on port 7860
- `browse.py` (legacy) runs on port 7861

### Configuration System

**Profiles** (`profiles/*.yaml` or `profiles/*.json`):
```yaml
# profiles/john.yaml
name: "John Doe"
input_path: "/path/to/labs"
output_path: "/path/to/output"
input_file_regex: "*.pdf"  # optional

# Optional overrides:
workers: 4
```

Note: Model IDs are configured via `.env` only (not in profiles). Paths are configured via profiles only (not in `.env`).

**Lab Specs** (`config/lab_specs.json`):
- 335 standardized lab test names
- Primary units and conversion factors
- Reference ranges (for review tool)

Example entry:
```json
{
  "Blood - Glucose": {
    "primary_unit": "mg/dL",
    "alternatives": [
      {"unit": "mmol/L", "factor": 18.0}
    ],
    "ranges": {
      "default": [70, 100]
    }
  }
}
```

### Output Schema (17 columns)

```csv
# Core identification
date                # Report/collection date
source_file         # Original PDF filename
page_number         # Page number

# Extracted values (standardized)
lab_name            # Standardized name (e.g., "Blood - Glucose")
value               # Numeric value in primary unit
unit                # Primary unit (e.g., "mg/dL")

# Reference ranges from PDF
reference_min       # Min reference from report
reference_max       # Max reference from report

# Raw values (for audit)
lab_name_raw        # Original name from PDF
value_raw           # Original value (before conversion)
unit_raw            # Original unit

# Quality
confidence          # Defaults to 1.0

# Review flags (from validation.py)
review_needed       # Boolean: needs human review?
review_reason       # Reason codes (e.g., "FORMAT_ARTIFACT; EXTREME_DEVIATION;")
review_confidence   # Adjusted confidence after validation

# Internal
lab_type            # blood/urine/feces (hidden in Excel)
result_index        # Index within page (hidden in Excel)
```

### Pydantic Models

- `LabResult`: Single test result with raw fields (`_raw` suffix) and metadata
- `HealthLabReport`: Document-level metadata + list of LabResult objects
- `LabType`: Enum for test types (blood, urine, saliva, feces, unknown)

### Output Files

For each PDF `{doc_stem}.pdf`:
- `{doc_stem}/` directory containing:
  - `{doc_stem}.{page}.jpg` - Preprocessed page images
  - `{doc_stem}.{page}.json` - Extracted structured data
  - `{doc_stem}.csv` - Combined results for the document

Final outputs in `OUTPUT_PATH`:
- `all.csv` - Merged results from all documents
- `all.xlsx` - Excel with formatted data

## Environment Configuration

Required:
- `OPENROUTER_API_KEY` - API key for OpenRouter

Optional (with smart defaults):
- `EXTRACT_MODEL_ID` - Vision model (default: `google/gemini-3-flash-preview`)
- `N_EXTRACTIONS` - Self-consistency extractions (default: 1)
- `MAX_WORKERS` - Parallel workers (default: CPU count)

Note: Input and output paths must be specified via profiles. See `profiles/_template.yaml`.

## Validation (test.py)

The test suite validates data integrity:
- All rows have dates
- No duplicate rows (by hash)
- No duplicate (date, lab_name) pairs
- Lab name prefixes match lab types
- Percentage units have correct naming
- Lab unit consistency per lab_name
- Outlier detection (>3 std from mean)

Run with `python test.py` - prints report to console.

## Important Conventions

### Package Management
Always use `uv` for package management, never `pip` directly:
```bash
uv sync              # Install dependencies from pyproject.toml
uv add <package>     # Add a new dependency
uv remove <package>  # Remove a dependency
uv pip install <pkg> # If you must install directly
```

### Documentation Maintenance
When modifying the extraction pipeline in `extract.py` or related modules:
- **Always update `docs/pipeline.md`** to reflect the changes
- The pipeline diagram and step descriptions must stay in sync with the code

### Lab Name Prefixes
Lab names MUST start with lab type prefix:
- Blood labs: "Blood - {name}"
- Urine labs: "Urine - {name}"
- Feces labs: "Feces - {name}"

### Percentage Units
- Labs with unit "%" must have standardized names ending in "(%)"

### Reviewing Extracted Data

Use the unified viewer for interactive review:
```bash
python viewer.py --profile tsilva
```

Or filter programmatically:
```python
import pandas as pd
df = pd.read_csv("output/all.csv")

# Items flagged by validation
needs_review = df[df['review_needed'] == True]

# Low confidence items
low_conf = df[df['confidence'] < 0.7]

# Filter by specific validation reason
format_errors = df[df['review_reason'].str.contains('FORMAT_ARTIFACT', na=False)]
```
