# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Labs Parser is a Python tool that uses AI (via OpenRouter API) to extract laboratory test results from medical documents (PDFs/images). It converts unstructured lab reports into structured CSV/Excel data with standardized test names and units.

## Key Commands

### Running the Parser
```bash
# Main entry point - processes all lab reports
python main.py

# Data integrity validation
python test.py
```

### Development
The `utils/` directory contains helper scripts for building and maintaining configuration:
- `build_labs_specs.py` / `build_labs_specs_2.py` - Generate lab specifications
- `sort_lab_specs.py` - Sort lab specifications alphabetically

## Architecture

> **Maintenance Note:** When modifying the pipeline, update `docs/pipeline.md` to reflect the changes.

### Core Pipeline (main.py)

The processing pipeline has 3 main stages:

1. **PDF Processing** (`process_single_pdf`)
   - Converts each PDF page to preprocessed JPG images (grayscale, contrast-enhanced)
   - Each page is processed independently

2. **Extraction** (`extract_labs_from_page_image`)
   - Extracts structured lab data directly from page images using vision models with function calling
   - Returns `HealthLabReport` with nested `LabResult` objects validated by Pydantic
   - No intermediate text transcription step - extraction happens directly from images

3. **Normalization & Mapping**
   - Maps raw lab names/units to standardized enums via config files
   - Converts values to primary units using conversion factors
   - Deduplicates by (date, lab_name_enum) pairs

### Self-Consistency Pattern

The `self_consistency` function is critical for accuracy:
- Runs extraction N times (configurable via N_EXTRACTIONS)
- If outputs differ, uses an LLM to vote on the most consistent result
- Applied to the extraction step to ensure accuracy

### Post-Extraction Verification (verification.py)

After extraction, an optional multi-model verification pipeline validates extracted values against the source image:

**6-Stage Verification Pipeline:**

1. **Cross-Model Extraction** - Re-extract with a different model family (e.g., if primary is Gemini, verify with Claude)
2. **Comparison** - Compare primary vs verification extractions, identify matches and disagreements
3. **Batch Verification** - For disagreements, ask verification model to confirm each value
4. **Character-Level Verification** - For uncertain values, read digit-by-digit with arbitration model
5. **Arbitration** - For unresolved disagreements, use third model to determine correct value
6. **Completeness Check** - Detect any results missed by primary extraction

**Model Selection Strategy:**
- Automatically selects verification model from different provider than primary
- Arbitration model selected from third provider when possible
- Cross-provider verification catches provider-specific biases

**Supported Models (January 2026):**
| Provider | Models | Notes |
|----------|--------|-------|
| Anthropic | `claude-opus-4.5`, `claude-sonnet-4`, `claude-haiku-4.5` | Opus 4.5 is frontier model |
| Google | `gemini-3-flash-preview`, `gemini-2.5-flash`, `gemini-2.5-pro` | Gemini 3 is latest |
| OpenAI | `gpt-5.2`, `gpt-4.1`, `gpt-4.1-mini` | GPT-5.2 is latest (400K context) |
| Qwen | `qwen3-max`, `qwen3-vl-32b-instruct`, `qwen3-vl-8b-instruct` | Qwen3-Max is SOTA (256K context) |

**Verification Columns Added:**
- `verification_status`: "verified", "corrected", "uncertain", "not_verified", "recovered"
- `verification_confidence`: 0-1 confidence score from verification
- `verification_method`: Method used (cross_model_match, batch_verification, character_level, arbitration)
- `cross_model_verified`: Boolean - whether cross-model extraction agreed
- `verification_corrected`: Boolean - whether value was corrected
- `value_raw_original`: Original value if corrected

**Configuration:**
```env
ENABLE_VERIFICATION=true
VERIFICATION_MODEL_ID=anthropic/claude-sonnet-4  # Optional, auto-selected if not set
ARBITRATION_MODEL_ID=openai/gpt-4o              # Optional, auto-selected if not set
ENABLE_COMPLETENESS_CHECK=true
ENABLE_CHARACTER_VERIFICATION=true
```

### Configuration System

One JSON config file in `config/` drives the standardization and normalization:

**`lab_specs.json`**
- **Keys**: Standardized lab test names (e.g., "Blood - Hemoglobin A1c", "Blood - Glucose")
- **Values**: Lab specifications including:
  - `primary_unit`: The preferred unit for this test (e.g., "mg/dL", "%")
  - `alternatives`: List of alternative units with conversion factors
  - `ranges`: Healthy reference ranges (min/max values)

This single file serves three purposes:
1. **Standardized Names**: Keys provide the canonical list of standardized lab names for LLM-based name mapping
2. **Standardized Units**: Values provide the canonical list of standardized units for LLM-based unit mapping
3. **Normalization**: Alternative units and conversion factors enable value normalization to primary units

Example entry:
```json
{
  "Blood - Glucose": {
    "primary_unit": "mg/dL",
    "alternatives": [
      {"unit": "mmol/L", "factor": 18.0}
    ],
    "ranges": {
      "healthy": {"min": 70, "max": 100}
    }
  }
}
```

### Data Schema (COLUMN_SCHEMA)

The centralized `COLUMN_SCHEMA` dictionary defines:
- Column names and pandas dtypes
- Excel export formatting (widths, hidden columns)
- Plotting roles (date, value, group, unit)
- Derivation logic for computed columns

Key column categories:
- **Raw extraction** (suffix: `_raw`):
  - `lab_name_raw`: Raw test name from PDF (e.g., "HEMATOLOGIA - HEMOGRAMA - Eritrocitos")
  - `value_raw`: Numeric or text value from PDF
  - `lab_unit_raw`: Raw unit from PDF (e.g., "mg/dl", "x10^9/L")
  - `reference_range`: Reference range text (not converted)
  - `reference_min_raw`, `reference_max_raw`: Reference range bounds from PDF
- **Standardized** (via LLM, cleans up spelling/format):
  - `lab_name_standardized`: Standardized test name (e.g., "Blood - Erythrocytes")
  - `lab_unit_standardized`: Standardized unit with proper capitalization (e.g., "mg/dl" → "mg/dL", "x10^9/L" → "10⁹/L")
  - Unknown/unmappable values are stored as `$UNKNOWN$` for manual review
- **Primary** (suffix: `_primary`, after unit conversion):
  - `value_primary`: Value converted to primary unit from lab_specs.json
  - `lab_unit_primary`: Primary unit for this test (e.g., all glucose in "mg/dL" regardless of source unit)
  - `reference_min_primary`, `reference_max_primary`: Reference ranges converted to primary unit
- **Data quality**:
  - Filter by `lab_name_standardized == '$UNKNOWN$'` to find tests needing config updates
  - Filter by `lab_unit_standardized == '$UNKNOWN$'` to find units needing config updates
- **Review/quality flags** (automatically added by edge case detection):
  - `needs_review`: Boolean flag indicating result needs human review
  - `review_reason`: Semicolon-separated reasons (e.g., "INEQUALITY_IN_VALUE; DUPLICATE_TEST_NAME;")
  - `confidence_score`: 0-1 score (values < 0.7 are high priority for review)

### Pydantic Models

- `LabResult`: Single test result with metadata including:
  - Raw fields (suffix `_raw`): `lab_name_raw`, `value_raw`, `lab_unit_raw`, `reference_min_raw`, `reference_max_raw`
  - Standardized fields (added post-extraction): `lab_name_standardized`, `lab_unit_standardized`
- `HealthLabReport`: Document-level metadata + list of LabResult objects
- `LabType`: Enum for test types (blood, urine, saliva, feces, unknown)

### Standardization Approach

The extraction process has two distinct phases:

1. **Raw Extraction** (during image extraction):
   - Vision model extracts test names and units EXACTLY as they appear in the image
   - No standardization or normalization at this stage
   - Goal: Perfect accuracy and traceability

2. **Standardization** (post-extraction):
   - After all raw data is extracted, a separate LLM call maps raw values to standardized enums
   - Uses `lab_specs.json` as the source of truth for valid standardized names and units
   - Raw test names → Standardized names (e.g., "GLICOSE -jejum-" → "Blood - Glucose (Fasting)")
   - Raw units → Standardized units (e.g., "mg/dl" → "mg/dL", "x10^9/L" → "10⁹/L")
   - Unknown values marked as `$UNKNOWN$` for manual review

### Output Files

For each PDF `{doc_stem}.pdf`:
- `{doc_stem}/` directory containing:
  - `{doc_stem}.{page}.jpg` - Preprocessed page images
  - `{doc_stem}.{page}.json` - Extracted structured data (directly from images)
  - `{doc_stem}.csv` - Combined results for the document

Final outputs in `OUTPUT_PATH`:
- `all.csv` - Merged results from all documents
- `all.xlsx` - Excel with two sheets:
  - "AllData" - All lab results
  - "MostRecentByEnum" - Most recent value per lab_name_enum
- `plots/` - Time-series plots for each lab_name_enum (with reference ranges)

### Caching & Idempotency

The system is designed to be resumable:
- Files are only regenerated if they don't exist
- PDF processing is parallelized via multiprocessing.Pool (MAX_WORKERS)
- Extraction uses ThreadPoolExecutor for concurrent self-consistency checks

### Date Resolution Logic

Dates are resolved with this priority:
1. `collection_date` from first page extraction
2. `report_date` from first page extraction
3. Date pattern (YYYY-MM-DD) in PDF filename
4. None if no date found (warning logged)

## Important Conventions

### Lab Name Prefixes
Lab name enums MUST start with lab type prefix:
- Blood labs: "Blood - {name}"
- Urine labs: "Urine - {name}"
- Feces labs: "Feces - {name}"

### Percentage Units
- Labs with unit "%" must have standardized names ending in "(%)"
- This ensures consistency between unit representation and lab name

### Slugification
The `slugify` function normalizes text for mapping keys:
- Lowercase, strip whitespace
- Replace µ/μ with "micro", % with "percent"
- Remove non-alphanumeric except hyphens
- Collapse spaces/underscores to hyphens, then remove hyphens

### Reviewing Extracted Data

The pipeline automatically detects edge cases and adds review flags to the output CSV. You can filter results that need attention:

```python
import pandas as pd

# Load CSV
df = pd.read_csv("output/all.csv")

# Get all items flagged for review
needs_review = df[df['needs_review'] == True]
print(f"Found {len(needs_review)} items needing review")

# Get high-priority items (low confidence)
high_priority = df[df['confidence_score'] < 0.7]
print(f"Found {len(high_priority)} high-priority items")

# Filter by specific issues
inequality_issues = df[df['review_reason'].str.contains('INEQUALITY_IN_VALUE', na=False)]
duplicate_tests = df[df['review_reason'].str.contains('DUPLICATE_TEST_NAME', na=False)]

# Exclude flagged items for analysis
clean_data = df[df['needs_review'] == False]
```

**Edge Case Categories:**
- `NULL_VALUE_WITH_SOURCE` - Value is null but source text suggests data exists
- `QUALITATIVE_IN_COMMENTS` - Text result in comments instead of value_raw
- `NUMERIC_NO_UNIT` - Numeric value without unit (excluding pH, ratios)
- `INEQUALITY_IN_VALUE` - Value contains <, >, ≤, ≥ (might be reference range)
- `COMPLEX_REFERENCE_RANGE` - Multi-condition ranges not parsed into min/max
- `DUPLICATE_TEST_NAME` - Same test appears multiple times on page

You can also identify unmapped values by checking for $UNKNOWN$ in the CSV:

```python
# Check for unknown standardized lab names
unknown_names = df[df['lab_name_standardized'] == '$UNKNOWN$']
if len(unknown_names) > 0:
    print(f"❌ {len(unknown_names)} tests need standardization:")
    for _, row in unknown_names.iterrows():
        print(f"  - {row['lab_name_raw']}")

# Check for unknown standardized units
unknown_units = df[df['lab_unit_standardized'] == '$UNKNOWN$']
if len(unknown_units) > 0:
    print(f"❌ {len(unknown_units)} units need standardization:")
    for _, row in unknown_units[['lab_name_raw', 'lab_unit_raw']].drop_duplicates().iterrows():
        print(f"  - {row['lab_name_raw']}: {row['lab_unit_raw']}")
```

This is useful for:
- Finding labs that need to be added to `lab_specs.json`
- Identifying units that need to be added to `lab_specs.json`
- Auditing data quality after extraction
- Improving standardization prompts to better guide the LLM

## Environment Configuration

Required `.env` variables:
- `SELF_CONSISTENCY_MODEL_ID` - Model for voting on self-consistency results
- `EXTRACT_MODEL_ID` - Vision model for extraction (e.g., google/gemini-2.5-flash, anthropic/claude-3.5-sonnet)
- `INPUT_PATH` - Directory containing PDF files
- `INPUT_FILE_REGEX` - Regex pattern to match input files (e.g., `.*\.pdf`)
- `OUTPUT_PATH` - Directory for output files
- `OPENROUTER_API_KEY` - API key for OpenRouter
- `N_EXTRACTIONS` - Number of extraction attempts for self-consistency (default: 1)
- `MAX_WORKERS` - Parallel workers for PDF processing (default: 1)

Optional verification `.env` variables:
- `ENABLE_VERIFICATION` - Enable post-extraction verification (default: true)
- `VERIFICATION_MODEL_ID` - Model for cross-model verification (auto-selected from different provider if not set)
- `ARBITRATION_MODEL_ID` - Model for resolving disagreements (auto-selected if not set)
- `ENABLE_COMPLETENESS_CHECK` - Check for missed results (default: true)
- `ENABLE_CHARACTER_VERIFICATION` - Do character-level verification for uncertain values (default: true)

## Validation (test.py)

The test suite validates data integrity:
- All rows have dates
- No duplicate rows (by hash)
- No duplicate (date, lab_name_enum) pairs
- Lab name prefixes match lab types
- Percentage units have correct naming
- Lab unit consistency per lab_name_enum
- Outlier detection (>3 std from mean)
- Config file consistency (specs ↔ mappings)

Run with `python test.py` - prints report to console.

## Known Issues & TODOs

See TODO.md for current development priorities. Key items:
- Performance optimization for existence checks
- Better handling of non-blood test types
- Automatic validation of page CSV vs source image
- Logging of unmapped lab names/units
- Merging mappings into labspecs
