# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Overview

Labs Parser is a Python tool that uses AI (via OpenRouter API) to extract laboratory test results from medical documents (PDFs/images). It converts unstructured lab reports into structured CSV/Excel data with standardized test names and units.

**Design Principle:** Extraction is objective (what's on the page). Analysis is subjective (health status, optimal targets, custom ranges) and belongs in a separate review tool.

## Key Commands

### Running the Parser
```bash
# Run all profiles:
parselabs

# Run specific profile:
parselabs --profile tsilva

# List available profiles:
parselabs --list-profiles

# Override settings:
parselabs --profile tsilva --model google/gemini-2.5-pro

# Data integrity validation:
uv run python test.py

# View and review extracted results:
parselabs review --profile tsilva
parselabs review --profile tsilva --tab review

# Deterministic row-by-row MCP review server:
parselabs-review-mcp
```

### Development
`parselabs admin` is the preferred entry point for maintenance and migration utilities. Legacy `utils/*.py` wrappers still exist, but new documentation should use the consolidated admin commands:
- `parselabs admin validate-lab-specs` - Validate lab_specs.json schema and LOINC code presence
- `parselabs admin lab-specs sort` - Sort lab specifications alphabetically
- `parselabs admin lab-specs fix-encoding` - Normalize lab_specs.json encoding
- `parselabs admin lab-specs build-conversions --input output/all.csv` - Generate unit conversion factors for lab_specs.json
- `parselabs admin lab-specs build-ranges --user-stats user_stats.json` - Generate evidence-based optimal ranges for lab_specs.json
- `parselabs admin analyze-unknowns` - Analyze $UNKNOWN$ values in extracted results
- `parselabs admin update-standardization-caches` - Batch-update name/unit standardization caches via LLM
- `parselabs admin regression ...` - Sync/report approved-document regression fixtures
- `parselabs admin review-artifacts ...` - Fetch deterministic review artifacts and persist row decisions
- `parselabs admin migrate-output-dirs` - Batch-rename legacy output directories to include file hash suffix
- `parselabs admin migrate-raw-columns` - Rename legacy raw-column fields to the current schema
- `parselabs admin cleanup-removed-fields` - Remove fields that are no longer persisted

See `utils/README.md` for detailed usage instructions.

### LLM Prompts
Prompt templates live in `prompts/` as `.md` files and are loaded at module level:
- `extraction_system.md`, `extraction_user.md` - vision extraction prompts
- `name_standardization.md`, `unit_standardization.md` - standardization prompts (used by `parselabs admin update-standardization-caches`)
- `conversion_factor_system.md`, `conversion_factor_user.md` - unit conversion factor prompts (template: `{lab_name}`, `{from_unit}`, `{to_unit}`)
- `health_range_system.md`, `health_range_user.md` - optimal range prompts (template: `{lab_name}`, `{primary_unit}`, `{user_stats_json}`)

## Architecture

### Core Pipeline (`parselabs/pipeline.py`)

The processing pipeline has 5 main stages:

1. **PDF Processing** (`parselabs.pipeline.process_single_pdf`)
   - Converts each PDF page to preprocessed JPG images (grayscale, contrast-enhanced)
   - Each page is processed independently

2. **Extraction** (`parselabs.extraction.extract_labs_from_page_image`)
   - Extracts structured lab data directly from page images using vision models
   - Returns `HealthLabReport` with nested `LabResult` objects validated by Pydantic

3. **Normalization & Mapping**
   - Maps raw lab names/units to standardized lab specs through cached standardization helpers
   - Converts values to primary units using conversion factors
   - Deduplicates publishable export rows after normalization

4. **Value Validation** (`parselabs/validation.py`)
   - Detects extraction errors from the data itself (no source image re-check)
   - Flags suspicious values for the combined review UI

5. **Review/Export Row Building** (`parselabs/rows.py`)
   - Builds per-document review CSVs from page JSON
   - Builds accepted-row final exports from reviewed JSON when strict rebuilds are requested

### Value Validation (`parselabs/validation.py`)

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

Flagged rows appear in the combined review UI under filters that use `review_needed=True` and `review_reason` columns.

### Review UI (`parselabs/ui.py`)

Interactive UI for browsing and reviewing extracted lab results:

**Layout:**
- Results Explorer tab: filtered table, summary cards, plots, source-page inspection, and details
- Review Queue tab: document/page queue, large source page, compact row inspector, and review actions

**Features:**
- Time-series plots with dual reference ranges (lab_specs + PDF)
- Multi-lab selection with stacked subplots
- Accept/Reject workflow with JSON persistence
- CSV export of filtered data

**Filters (all always visible):**
- Lab name multi-select
- Abnormal only / Latest only checkboxes
- Review status dropdown (All, Needs Review, Abnormal, Suboptimal, Unreviewed)

**Keyboard shortcuts:**
- `Y` = Approve, `N` = Reject, `M` = Missing Row, `U` = Undo
- Arrow keys or `j`/`k` = Navigate

**Ports:** 7862 for the Results Explorer launch mode, 7863 for the Review Queue launch mode

### Configuration System

**Profiles** (`~/.config/parselabs/profiles/*.yaml` or `~/.config/parselabs/profiles/*.json`):
```yaml
# ~/.config/parselabs/profiles/john.yaml
name: "John Doe"
paths:
  input_path: "/path/to/labs"
  output_path: "/path/to/output"
  input_file_regex: "*.pdf"  # optional

processing:
  workers: 4
```

**Shared runtime settings** (`~/.config/parselabs/.env`):
```bash
OPENROUTER_API_KEY=your_api_key
# Optional:
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
EXTRACT_MODEL_ID=google/gemini-2.5-pro
```

**Lab Specs** (`config/lab_specs.json`):
- 328 standardized lab test names
- Primary units and conversion factors
- Reference ranges (for review tool)
- LOINC codes for lab test identification and interoperability (required for all labs)

Example entry:
```json
{
  "Blood - Glucose": {
    "primary_unit": "mg/dL",
    "loinc_code": "2345-7",
    "alternatives": [
      {"unit": "mmol/L", "factor": 18.0}
    ],
    "ranges": {
      "default": [70, 100]
    }
  }
}
```

### Output Schema

```csv
# Core identification
date                # Report/collection date

# Extracted values (standardized)
lab_name            # Standardized name (e.g., "Blood - Glucose")
value               # Numeric value in primary unit
lab_unit            # Primary unit (e.g., "mg/dL")

# Source identification
source_file         # Original PDF filename
page_number         # Page number

# Reference ranges from PDF
reference_min       # Min reference from report
reference_max       # Max reference from report

# Raw values (for audit)
raw_lab_name        # Original name from PDF
raw_value           # Original value (before conversion)
raw_unit            # Original unit

# Review flags (from validation.py)
review_needed       # Boolean: needs human review?
review_reason       # Reason codes (e.g., "FORMAT_ARTIFACT; EXTREME_DEVIATION;")

# Limit indicators
is_below_limit      # Value reported as below limit (e.g., "<0.05")
is_above_limit      # Value reported as above limit (e.g., ">738")

# Internal
lab_type            # blood/urine/feces (hidden in Excel)
result_index        # Index within page (hidden in Excel)
```

Current review CSVs also include page/result identity, review status, bbox metadata, raw section names, and additional normalized/export alias columns used by the review UI.

### Pydantic Models

- `LabResult`: Single test result with raw fields (`raw_` prefix) and metadata
- `HealthLabReport`: Document-level metadata + list of LabResult objects
- `LabType`: Enum for test types (blood, urine, saliva, feces, unknown)

### Output Files

For each PDF `{doc_stem}.pdf`:
- `{doc_stem}_{hash}/` directory (hash = first 8 chars of SHA-256) containing:
  - `{doc_stem}.{page}.jpg` - Preprocessed page images
  - `{doc_stem}.{page}.json` - Extracted structured data
  - `{doc_stem}.csv` - Combined results for the document

Final outputs in the profile `output_path`:
- `all.csv` - Merged results from all documents
- `all.xlsx` - Excel with formatted data
- `lab_specs.json` - Copy of lab specifications used for this extraction (for reproducibility)

## Profile Configuration

Required in each extraction profile:
- `paths.input_path`
- `paths.output_path`

Optional:
- `processing.workers` (or top-level `workers`) - Parallel workers (default: CPU count)

Required in shared config (`~/.config/parselabs/.env` or shell environment):
- `OPENROUTER_API_KEY` - API key for OpenRouter
- `EXTRACT_MODEL_ID` - Vision model for extraction

Optional in shared config:
- `OPENROUTER_BASE_URL` - Alternate OpenRouter-compatible endpoint

## Validation (test.py)

The test suite validates both configuration and data integrity:

### Configuration Validation
- **Schema validation** (via `parselabs admin validate-lab-specs`):
  - JSON structure and syntax
  - Required fields for all labs (lab_type, primary_unit, loinc_code)
  - Data types and value ranges
  - LOINC code presence (all labs must have LOINC codes except known exceptions)
  - Relationship configurations
  - Lab name prefixes match lab_type
  - Unit conversion factors
  - Reference range consistency
- **LOINC-specific checks**:
  - Critical LOINC codes are correct (AAT, ALP, Bilirubin, etc.)
  - No duplicate LOINC codes across different test types

### Data Integrity Validation
- All rows have dates
- No duplicate rows (by hash)
- No duplicate (date, lab_name) pairs
- Lab name prefixes match lab types
- Percentage units have correct naming
- Lab unit consistency per lab_name
- Outlier detection (>3 std from mean)

Run with `uv run python test.py` - prints report to console.

You can also run the schema validator standalone:
```bash
parselabs admin validate-lab-specs
```

## Important Conventions

### Package Management
Always use `uv` for package management, never `pip` directly:
```bash
uv sync              # Install dependencies from pyproject.toml
uv add <package>     # Add a new dependency
uv remove <package>  # Remove a dependency
uv pip install <pkg> # If you must install directly
```
Note: `uv sync` requires `dangerouslyDisableSandbox: true` in Claude Code sandbox mode (UV cache writes are blocked).

### Documentation Maintenance
When modifying the extraction pipeline in `parselabs/pipeline.py` or related modules:
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
parselabs review --profile tsilva
parselabs review --profile tsilva --tab review
```

Or filter programmatically:
```python
import pandas as pd
df = pd.read_csv("output/all.csv")

# Items flagged by validation
needs_review = df[df['review_needed'] == True]

# Filter by specific validation reason
format_errors = df[df['review_reason'].str.contains('FORMAT_ARTIFACT', na=False)]
```
