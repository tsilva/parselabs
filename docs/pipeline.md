# Pipeline Architecture

This document describes the data processing pipeline in Labs Parser.

## Data Flow Overview

```
PDF Files
    |
    v
[1. PDF Discovery & Filtering]
    |
    v
[2. Parallel PDF Processing] ─────────────────────────────────┐
    |                                                         |
    v                                                         |
[3. Image Conversion & Preprocessing]                         |
    |                                                         |
    v                                                         |
[4. Vision Model Extraction + Self-Consistency]               |
    |                                                         |
    v                                                         |
[5. Post-Extraction Verification] (optional)                  |
    |                                                         |
    v                                                         |
[6. Page JSON Cache]                                          |
    |                                                         |
    v                                                         |
[7. Date Resolution]                                          |
    |                                                         |
    v                                                         |
[8. Lab Name & Unit Standardization]                          |
    |                                                         |
    v                                                         |
[Document CSV] <──────────────────────────────────────────────┘
    |
    v
[9. Merge All CSVs]
    |
    v
[10. Normalization & Unit Conversion]
    |
    v
[11. Edge Case Detection]
    |
    v
[12. Deduplication]
    |
    v
[13. Type Conversion & Column Ordering]
    |
    v
[all.csv] + [all.xlsx] + [plots/] + [Run Report]
```

## Pipeline Stages

### Stage 1: PDF Discovery & Filtering
**Location:** `main.py` - `main()`

- Scans `INPUT_PATH` for PDFs matching `INPUT_FILE_REGEX`
- Filters out already-processed PDFs (checks for existing CSV)
- Prompts user about reprocessing PDFs with empty extraction pages

### Stage 2: Parallel PDF Processing
**Location:** `main.py` - `main()`, `process_single_pdf()`

- Uses `multiprocessing.Pool` with `MAX_WORKERS` processes
- Each PDF processed independently with progress tracking
- Returns CSV path for successfully processed PDFs

### Stage 3: Image Conversion & Preprocessing
**Location:** `main.py` - `process_single_pdf()`, `utils.py` - `preprocess_page_image()`

1. Copies PDF to output directory
2. Converts each page to PIL Image via `pdf2image.convert_from_path()`
3. Preprocesses each image:
   - Converts to grayscale
   - Resizes to max 1200px width (maintains aspect ratio)
   - Enhances contrast by 2x
4. Saves as `{page_name}.jpg`

### Stage 4: Vision Model Extraction
**Location:** `extraction.py` - `extract_labs_from_page_image()`, `self_consistency()`

1. Encodes preprocessed image to base64
2. Sends to vision model with function calling (OpenAI tools format)
3. Model extracts structured data directly from image (no OCR step)
4. Returns `HealthLabReport` validated by Pydantic

**Self-Consistency Pattern** (when `N_EXTRACTIONS > 1`):
- Runs extraction N times in parallel via `ThreadPoolExecutor`
- If results differ, uses LLM voting to select most consistent result
- Temperature fixed at 0.5 for diversity

### Stage 5: Post-Extraction Verification (Optional)
**Location:** `verification.py`

6-stage verification pipeline when `ENABLE_VERIFICATION=true`:

1. **Cross-Model Extraction** - Re-extract with different provider
2. **Comparison** - Identify matches vs disagreements
3. **Batch Verification** - Confirm disagreed values
4. **Character-Level Verification** - Digit-by-digit reading for uncertain values
5. **Arbitration** - Third model resolves unresolved disagreements
6. **Completeness Check** - Detect missed results

Adds columns: `verification_status`, `verification_confidence`, `verification_method`, etc.

### Stage 6: Page JSON Cache
**Location:** `main.py` - `process_single_pdf()`

- Saves extraction results to `{page_name}.json`
- Enables resumability - skips re-extraction on subsequent runs

### Stage 7: Date Resolution
**Location:** `main.py` - `process_single_pdf()`

Priority order:
1. `collection_date` from first page extraction
2. `report_date` from first page extraction
3. Date pattern (YYYY-MM-DD) in PDF filename
4. None (warning logged)

### Stage 8: Lab Name & Unit Standardization
**Location:** `normalization.py` - `standardize_lab_names()`, `standardize_lab_units()`

1. **Lab Name Standardization**
   - Collects unique raw lab names
   - LLM maps raw names to standardized names from `lab_specs.json` keys
   - Cached in `config/cache/lab_names.json`
   - Unknown values marked as `$UNKNOWN$`

2. **Lab Unit Standardization**
   - Normalizes null/empty units to "null" string
   - LLM maps (raw_unit, lab_name) tuples to standardized units
   - Cached in `config/cache/lab_units.json`
   - Unknown values marked as `$UNKNOWN$`

3. **Percentage Correction**
   - If unit is "%" but name doesn't end with "(%)", finds percentage variant

### Stage 9: Merge All CSVs
**Location:** `main.py` - `main()`

- Concatenates all document CSVs into single DataFrame
- Tracks source file for each row

### Stage 10: Normalization & Unit Conversion
**Location:** `normalization.py`

- **Lab Type Lookup** - Maps `lab_name_standardized` to lab type (blood/urine/feces)
- **Unit Conversion** - Converts values to primary unit using factors from `lab_specs.json`
- **Reference Range Conversion** - Converts min/max to primary unit
- **Qualitative Handling** - Converts text values (e.g., "NEGATIVO" to 0/1)
- **Health Status** - Computes `is_out_of_reference` and `is_in_healthy_range`

### Stage 11: Edge Case Detection
**Location:** `edge_case_detection.py`

Flags results needing human review:
- `NULL_VALUE_WITH_SOURCE` - Value null but source text exists
- `QUALITATIVE_IN_COMMENTS` - Text result in comments
- `NUMERIC_NO_UNIT` - Numeric value without unit
- `INEQUALITY_IN_VALUE` - Value contains <, >, etc.
- `COMPLEX_REFERENCE_RANGE` - Multi-condition ranges
- `DUPLICATE_TEST_NAME` - Same test multiple times on page

Sets: `needs_review`, `review_reason`, `confidence_score`

### Stage 12: Deduplication
**Location:** `main.py` - `main()`

- Groups by (date, `lab_name_standardized`)
- Keeps single result per group
- Prefers primary unit if multiple units exist

### Stage 13: Output Generation
**Location:** `main.py` - `main()`

1. **CSV Export** - `output/all.csv`
2. **Excel Export** - `output/all.xlsx`
   - "AllData" sheet - All results sorted by date desc
   - "MostRecentByEnum" sheet - Latest value per lab test
3. **Time-Series Plots** - `output/plots/` via `LabPlotter`
4. **Run Report** - Summary to console

## Data Models

### LabResult
Single test result with three field groups:

| Group | Fields | Description |
|-------|--------|-------------|
| Raw | `lab_name_raw`, `value_raw`, `lab_unit_raw`, `reference_min_raw`, `reference_max_raw` | Extracted exactly as shown in image |
| Standardized | `lab_name_standardized`, `lab_unit_standardized` | Mapped to canonical names/units |
| Primary | `value_primary`, `lab_unit_primary`, `reference_min_primary`, `reference_max_primary` | Converted to primary unit |

### HealthLabReport
Document-level container:
- `collection_date` - Specimen collection date
- `report_date` - Report issue date
- `lab_facility` - Laboratory name
- `page_has_lab_data` - Boolean flag
- `lab_results` - List of `LabResult`

## Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `main()` | `main.py` | Pipeline orchestration |
| `process_single_pdf()` | `main.py` | Per-PDF processing |
| `extract_labs_from_page_image()` | `extraction.py` | Vision model extraction |
| `self_consistency()` | `extraction.py` | Multi-extraction voting |
| `standardize_lab_names()` | `normalization.py` | Name standardization |
| `standardize_lab_units()` | `normalization.py` | Unit standardization |
| `preprocess_page_image()` | `utils.py` | Image preprocessing |

## Configuration

| File | Purpose |
|------|---------|
| `.env` | Runtime configuration (models, paths, API keys) |
| `config/lab_specs.json` | Standardized names, units, conversion factors, healthy ranges |
| `config/cache/lab_names.json` | Cached name mappings (user-editable) |
| `config/cache/lab_units.json` | Cached unit mappings (user-editable) |
