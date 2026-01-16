# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Labs Parser is a Python tool that uses AI (via OpenRouter API) to extract laboratory test results from medical documents (PDFs/images). It converts unstructured lab reports into structured CSV/Excel data with standardized test names and units.

**Design Principle:** Extraction is objective (what's on the page). Analysis is subjective (health status, custom ranges) and belongs in a separate review tool.

## Key Commands

### Running the Parser
```bash
# Using a profile (required):
python main.py --profile tsilva

# List available profiles:
python main.py --list-profiles

# Override settings:
python main.py --profile tsilva --model google/gemini-2.5-pro --no-verify

# Post-extraction verification (run verification on cached data):
python main.py --profile tsilva --verify-only                  # Verify all pages
python main.py --profile tsilva --verify-only --unverified-only  # Only unverified
python main.py --profile tsilva --verify-only --document "2024-01-15-labs"  # Specific doc
python main.py --profile tsilva --verify-only --date-from 2024-01-01 --date-to 2024-06-30

# Data integrity validation:
python test.py

# Review extracted results:
python review_ui.py --profile tsilva
```

### Development
The `utils/` directory contains helper scripts for building and maintaining configuration:
- `build_lab_specs_conversions.py` - Generate unit conversion factors for lab_specs.json
- `build_lab_specs_ranges.py` - Generate healthy ranges for lab_specs.json
- `sort_lab_specs.py` - Sort lab specifications alphabetically
- `analyze_unknowns.py` - Analyze $UNKNOWN$ values in extracted results

## Architecture

### Core Pipeline (main.py)

The processing pipeline has 3 main stages:

1. **PDF Processing** (`process_single_pdf`)
   - Converts each PDF page to preprocessed JPG images (grayscale, contrast-enhanced)
   - Each page is processed independently

2. **Extraction** (`extract_labs_from_page_image`)
   - Extracts structured lab data directly from page images using vision models
   - Returns `HealthLabReport` with nested `LabResult` objects validated by Pydantic
   - Optional cross-model verification for accuracy

3. **Normalization & Mapping**
   - Maps raw lab names/units to standardized enums via config files
   - Converts values to primary units using conversion factors
   - Deduplicates by (date, lab_name) pairs

### Simplified Verification (verification.py)

Cross-model verification validates extracted values against the source image:

1. **Cross-Model Extraction** - Re-extract with a different model family
2. **Batch Verification** - For disagreements, verify each value

**Configuration:**
```bash
--no-verify        # Disable verification during extraction
--verify-only      # Run verification on already-extracted data (skip extraction)
--unverified-only  # Only verify pages that haven't been verified yet
--document         # Filter to specific document stem
--date-from        # Filter results >= date (YYYY-MM-DD)
--date-to          # Filter results <= date (YYYY-MM-DD)
```

### Configuration System

**Profiles** (`profiles/*.yaml` or `profiles/*.json`):
```yaml
# profiles/john.yaml
name: "John Doe"
input_path: "/path/to/labs"
output_path: "/path/to/output"
input_file_regex: "*.pdf"  # optional

# Optional overrides:
model: "google/gemini-2.5-flash"
verify: true
workers: 4
```

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

### Output Schema (15 columns)

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
confidence          # 0-1 score
verified            # Boolean: was cross-model verification done?

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
- `EXTRACT_MODEL_ID` - Vision model (default: `google/gemini-2.5-flash`)
- `N_EXTRACTIONS` - Self-consistency extractions (default: 1)
- `MAX_WORKERS` - Parallel workers (default: CPU count)
- `ENABLE_VERIFICATION` - Cross-model verification (default: false)

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

### Documentation Maintenance
When modifying the extraction pipeline in `main.py` or related modules:
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

Filter for unmapped values:
```python
import pandas as pd
df = pd.read_csv("output/all.csv")

# Low confidence items
low_conf = df[df['confidence'] < 0.7]
```
