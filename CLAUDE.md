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
- `map.py` - Build lab name and unit mappings
- `fix_lab_name_percent.py` / `fix_lab_names_mapping_percent_suffix.py` - Fix percentage-based lab names
- `sort_lab_specs.py` - Sort lab specifications alphabetically

## Architecture

### Core Pipeline (main.py)

The processing pipeline has 4 main stages:

1. **PDF Processing** (`process_single_pdf`)
   - Converts each PDF page to preprocessed JPG images (grayscale, contrast-enhanced)
   - Each page is processed independently

2. **Transcription** (`transcription_from_page_image`)
   - Uses vision models (default: Gemini Flash) to transcribe page images to text
   - Preserves layout and formatting using markdown tables

3. **Extraction** (`extract_labs_from_page_transcription`)
   - Extracts structured lab data from transcriptions using function calling
   - Returns `HealthLabReport` with nested `LabResult` objects validated by Pydantic

4. **Normalization & Mapping**
   - Maps raw lab names/units to standardized enums via config files
   - Converts values to primary units using conversion factors
   - Deduplicates by (date, lab_name_enum) pairs

### Self-Consistency Pattern

The `self_consistency` function is critical for accuracy:
- Runs transcription/extraction N times (configurable via N_TRANSCRIPTIONS, N_EXTRACTIONS)
- If outputs differ, uses an LLM to vote on the most consistent result
- Applied to both transcription and extraction steps

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
- **Raw extraction**:
  - `test_name`: Raw test name from PDF (e.g., "HEMATOLOGIA - HEMOGRAMA - Eritrocitos")
  - `value`: Numeric or text value
  - `unit`: Raw unit from PDF (e.g., "mg/dl", "x10^9/L")
  - `reference_range`, `reference_min`, `reference_max`: Reference ranges from PDF
- **Standardized** (added by post-extraction LLM standardization):
  - `lab_name_standardized`: Standardized test name (e.g., "Blood - Erythrocytes")
  - `lab_unit_standardized`: Standardized unit (e.g., "mg/dL", "10⁹/L")
  - Unknown/unmappable values are stored as `$UNKNOWN$` for manual review
- **Data quality**:
  - Filter by `lab_name_standardized == '$UNKNOWN$'` to find tests needing config updates
  - Filter by `lab_unit_standardized == '$UNKNOWN$'` to find units needing config updates

### Pydantic Models

- `LabResult`: Single test result with metadata including:
  - Raw fields: `test_name`, `value`, `unit`, `reference_range`, etc.
  - Standardized fields: `lab_name_standardized`, `lab_unit_standardized` (added post-extraction)
- `HealthLabReport`: Document-level metadata + list of LabResult objects
- `LabType`: Enum for test types (blood, urine, saliva, feces, unknown)

### Standardization Approach

The extraction process has two distinct phases:

1. **Raw Extraction** (during transcription/extraction):
   - LLM extracts test names and units EXACTLY as they appear in the PDF
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
  - `{doc_stem}.{page}.txt` - Transcribed text
  - `{doc_stem}.{page}.json` - Extracted structured data
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
- Extraction and transcription use ThreadPoolExecutor for concurrent self-consistency checks

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
- Labs with unit "%" must have lab_name_enum ending in "(%)"
- Keys in lab_names_mappings.json containing "percent" must map to values ending with "(%)"

### Slugification
The `slugify` function normalizes text for mapping keys:
- Lowercase, strip whitespace
- Replace µ/μ with "micro", % with "percent"
- Remove non-alphanumeric except hyphens
- Collapse spaces/underscores to hyphens, then remove hyphens

### Reviewing Extracted Data

You can identify unmapped values by checking for $UNKNOWN$ in the CSV:

```python
import pandas as pd

# Load CSV
df = pd.read_csv("output/doc_name/doc_name.csv")

# Check for unknown standardized lab names
unknown_names = df[df['lab_name_standardized'] == '$UNKNOWN$']
if len(unknown_names) > 0:
    print(f"❌ {len(unknown_names)} tests need standardization:")
    for _, row in unknown_names.iterrows():
        print(f"  - {row['test_name']}")

# Check for unknown standardized units
unknown_units = df[df['lab_unit_standardized'] == '$UNKNOWN$']
if len(unknown_units) > 0:
    print(f"❌ {len(unknown_units)} units need standardization:")
    for _, row in unknown_units[['test_name', 'unit']].drop_duplicates().iterrows():
        print(f"  - {row['test_name']}: {row['unit']}")
```

This is useful for:
- Finding labs that need to be added to `lab_specs.json`
- Identifying units that need to be added to `lab_specs.json`
- Auditing data quality after extraction
- Improving standardization prompts to better guide the LLM

## Environment Configuration

Required `.env` variables:
- `SELF_CONSISTENCY_MODEL_ID` - Model for voting on self-consistency results
- `TRANSCRIBE_MODEL_ID` - Vision model for OCR (e.g., google/gemini-2.5-flash)
- `EXTRACT_MODEL_ID` - Model for structured extraction
- `INPUT_PATH` - Directory containing PDF files
- `INPUT_FILE_REGEX` - Regex pattern to match input files (e.g., `.*\.pdf`)
- `OUTPUT_PATH` - Directory for output files
- `OPENROUTER_API_KEY` - API key for OpenRouter
- `N_TRANSCRIPTIONS` - Number of transcription attempts for self-consistency (default: 1)
- `N_EXTRACTIONS` - Number of extraction attempts for self-consistency (default: 1)
- `MAX_WORKERS` - Parallel workers for PDF processing (default: 1)

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
