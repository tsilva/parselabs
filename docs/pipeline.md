# Extraction Pipeline

This document describes the data flow through the labs-parser extraction pipeline, from PDF input to validated CSV/Excel output.

## Pipeline Overview

```
PDF Files
  │
  ├─ 1. PDF Processing & Hashing
  ├─ 2. Extraction (text-first, vision fallback)
  ├─ 3. Standardization (cache-based mapping)
  ├─ 4. Per-PDF CSV Export
  ├─ 5. Merge All CSVs
  ├─ 6. Normalization (unit conversion, dedup)
  ├─ 7. Validation (flag suspicious values)
  └─ 8. Final Export (CSV + Excel)
```

## Stage 1: PDF Processing & Hashing

**Module:** `main.py` — `_setup_pdf_processing()`, `_compute_file_hash()`

Each PDF is processed independently (parallelized via `multiprocessing.Pool`):

1. Compute SHA-256 hash of the file (first 8 hex chars)
2. Create output directory: `{pdf_stem}_{hash}/` — hash prevents collisions when different files share the same name
3. Copy original PDF into the output directory for archival

## Stage 2: Extraction

**Modules:** `main.py` — `process_single_pdf()`, `labs_parser/extraction.py`

The pipeline uses a **text-first strategy** with vision fallback to optimize API costs.

### Path A: Text Extraction (Cheap)

```
PDF → pdftotext (layout mode) → raw text
  → Check if ≥200 non-whitespace chars
  → extract_labs_from_text() → LLM function calling
  → HealthLabReport with LabResult objects
```

- Uses `pdftotext` (poppler) with `-layout` flag
- Threshold: 200 non-whitespace characters minimum
- If text is sufficient, sends to LLM as text (no image tokens)

### Path B: Vision Extraction (Fallback)

```
PDF → pdf2image (page-by-page)
  → preprocess_page_image() (grayscale, resize ≤1200px, contrast 2x)
  → extract_labs_from_page_image() → LLM vision function calling
  → HealthLabReport with LabResult objects
```

- Each page converted to JPG independently
- Image preprocessing: grayscale → downscale to max 1200px width → 2x contrast enhancement
- Results cached as `{stem}.{page}.json` — skips re-extraction on rerun

### LLM Extraction Details

Both paths use OpenRouter API with **function calling** (structured output):

- **LLM-facing schema** (`LabResultExtraction`): Smaller, token-efficient — excludes internal fields (`review_*`, `page_number`, `source_file`, `result_index`)
- **Internal schema** (`HealthLabReport` / `LabResult`): Full schema with all metadata

**Retry logic** with temperature escalation on malformed output:
- Attempt 1: temperature = 0.0 (deterministic)
- Attempt 2: temperature = 0.2
- Attempt 3: temperature = 0.4
- Attempt 4: temperature = 0.6
- After all retries fail: returns empty report with `_extraction_failed=True`

**Prompt templates** (loaded from `prompts/`):
- `extraction_system.md` + `extraction_user.md` — vision extraction
- `text_extraction_user.md` — text-based extraction (templates: `{text}`, `{std_reminder}`)

The system prompt includes a list of standardized lab names from `lab_specs.json` to guide inline standardization.

### Extraction Output

Each `LabResult` contains:

| Field | Description |
|-------|-------------|
| `raw_lab_name` | Test name as printed on the PDF |
| `raw_value` | Numeric value as string |
| `raw_lab_unit` | Unit symbol |
| `raw_reference_range` | Full range text |
| `raw_reference_min`, `raw_reference_max` | Parsed range bounds |
| `raw_is_abnormal` | Abnormal flag from PDF |
| `lab_name_standardized` | Inline LLM mapping (may be `$UNKNOWN$`) |
| `lab_unit_standardized` | Inline LLM mapping (may be `$UNKNOWN$`) |

## Stage 3: Standardization

**Modules:** `main.py` — `_apply_standardization_fallbacks()`, `labs_parser/standardization.py`

Maps raw lab names and units to standardized enum values using **persistent JSON caches**. No LLM calls at runtime.

```
For each LabResult where lab_name_standardized == $UNKNOWN$ or missing:
  → standardize_lab_names(): cache lookup raw_name → standardized_name
  → standardize_lab_units(): cache lookup (raw_unit, lab_name) → standardized_unit
```

### Cache Files

```
config/cache/name_standardization.json   # raw_name → standardized_name
config/cache/unit_standardization.json   # raw_unit|lab_name → standardized_unit
```

- **Cache hit:** return standardized value
- **Cache miss:** return `$UNKNOWN$` + log warning
- **Updating caches:** run `utils/update_standardization_caches.py` (batch LLM processing)

## Stage 4: Per-PDF CSV Export

**Module:** `main.py` — `_save_results_to_csv()`

Each PDF produces a CSV in its output directory: `{stem}_{hash}/{stem}_{hash}.csv`

Contains all LabResult fields mapped to the output column schema.

## Stage 5: Merge

**Module:** `main.py` — `merge_csv_files()`

All per-PDF CSVs are concatenated into a single `all.csv` in the profile's output directory.

## Stage 6: Normalization

**Module:** `labs_parser/normalization.py`

Transforms the merged DataFrame through several steps:

### 6a. Numeric Preprocessing — `preprocess_numeric_value()`

Cleans raw value strings before numeric conversion:
- Strip trailing `=` (e.g., `"0.9="` → `"0.9"`)
- Extract first number from concatenation artifacts (e.g., `"52.6=1946"` → `"52.6"`)
- Remove space thousands separators (e.g., `"256 000"` → `"256000"`)
- European decimal format (comma → period)

### 6b. Comparison Operators — `extract_comparison_value()`

Parses limit indicators and sets boolean flags:

| Raw Value | Parsed Value | `is_below_limit` | `is_above_limit` |
|-----------|-------------|-------------------|-------------------|
| `<100` | 100 | True | False |
| `>200` | 200 | False | True |
| `≤50` | 50 | True | False |
| `≥30` | 30 | False | True |

### 6c. Unit Conversion — `apply_normalizations()`

Converts values to primary units using factors from `lab_specs.json`:

```
raw_value=5.0, raw_unit="mmol/L", lab="Blood - Glucose"
  → primary_unit = "mg/dL"
  → factor = 18.0
  → value = 5.0 × 18.0 = 90.0 mg/dL
```

### 6d. Deduplication — `deduplicate_results()`

- Groups by `(date, lab_name)`
- Keeps row with latest `result_index` (most complete data)
- Logs deduplication stats

### 6e. Type Conversion — `apply_dtype_conversions()`

Forces proper column types: `date` → datetime, `value` → float64, booleans, etc.

## Stage 7: Validation

**Module:** `labs_parser/validation.py` — `ValueValidator`

Detects extraction errors by analyzing the data itself (no source image re-check). Sets `review_needed=True` and appends reason codes to `review_reason`.

| Category | Reason Codes | Description |
|----------|--------------|-------------|
| Biological Plausibility | `NEGATIVE_VALUE`, `IMPOSSIBLE_VALUE`, `PERCENTAGE_BOUNDS` | Values outside biological limits |
| Inter-Lab Relationships | `RELATIONSHIP_MISMATCH`, `COMPONENT_EXCEEDS_TOTAL` | Calculated values don't match (e.g., LDL Friedewald formula) |
| Temporal Consistency | `TEMPORAL_ANOMALY` | Implausible change rate between tests |
| Format Artifacts | `FORMAT_ARTIFACT` | Excessive decimals, concatenation patterns |
| Reference Ranges | `RANGE_INCONSISTENCY`, `EXTREME_DEVIATION` | Inverted ranges or value 100x outside range |
| Duplicates | `DUPLICATE_ENTRY` | Flagged by deduplication |

### Configuration

Validation parameters are defined in `config/lab_specs.json` per lab:

```json
{
  "Blood - Hemoglobin (Hgb)": {
    "biological_min": 0,
    "biological_max": 25,
    "max_daily_change": 2.0
  }
}
```

Inter-lab relationships use formulas from the `_relationships` key:

```json
{
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

## Stage 8: Final Export

**Module:** `main.py` — `export_excel()`

### Output Files

| File | Description |
|------|-------------|
| `all.csv` | Merged, normalized, deduplicated, validated results |
| `all.xlsx` | Excel with formatted columns, frozen header, hidden internal columns |
| `lab_specs.json` | Copy of lab specifications used (for reproducibility) |

### Output Schema (17 columns)

```
date              # Collection/report date
source_file       # Original PDF filename
page_number       # Page number in PDF
lab_name          # Standardized name (e.g., "Blood - Glucose")
value             # Numeric value in primary unit
unit              # Primary unit (e.g., "mg/dL")
reference_min     # Min reference from report
reference_max     # Max reference from report
raw_lab_name      # Original name from PDF
raw_value         # Original value (before conversion)
raw_unit          # Original unit
review_needed     # Boolean: needs human review?
review_reason     # Reason codes (e.g., "FORMAT_ARTIFACT; EXTREME_DEVIATION;")
is_below_limit    # Value was reported as below limit (e.g., "<0.05")
is_above_limit    # Value was reported as above limit (e.g., ">738")
lab_type          # blood/urine/feces (hidden in Excel)
result_index      # Index within page (hidden in Excel)
```

## Full Data Flow Diagram

```
INPUT: PDF files (per profile)
    │
    ▼
[process_single_pdf] ──── parallelized via multiprocessing.Pool
    │
    ├── _compute_file_hash() ── SHA-256, first 8 hex chars
    ├── _setup_pdf_processing() ── create {stem}_{hash}/ directory
    ├── _copy_pdf_to_output()
    │
    ├── TEXT PATH:
    │   ├── extract_text_from_pdf() ── pdftotext -layout
    │   ├── _text_has_enough_content() ── ≥200 chars?
    │   └── extract_labs_from_text() ── LLM function calling
    │
    ├── VISION PATH (fallback):
    │   ├── _convert_pdf_to_images() ── pdf2image
    │   └── per page:
    │       ├── preprocess_page_image() ── grayscale, resize, contrast
    │       ├── extract_labs_from_page_image() ── LLM vision + retry
    │       └── cache JPG + JSON
    │
    ├── _apply_standardization_fallbacks()
    │   ├── standardize_lab_names() ── cache lookup
    │   └── standardize_lab_units() ── cache lookup
    │
    └── _save_results_to_csv() ── per-PDF CSV
         │
         ▼
[merge_csv_files] ── concatenate all per-PDF CSVs
         │
         ▼
[apply_normalizations]
    ├── preprocess_numeric_value() ── clean raw strings
    ├── extract_comparison_value() ── parse <, >, ≤, ≥
    ├── unit conversion ── value × factor → primary unit
    └── date parsing
         │
         ▼
[deduplicate_results] ── group by (date, lab_name), keep latest
         │
         ▼
[apply_dtype_conversions] ── force column types
         │
         ▼
[ValueValidator.validate()]
    ├── biological plausibility
    ├── inter-lab relationships
    ├── temporal consistency
    ├── format artifact detection
    └── reference range checks
         │
         ▼
OUTPUT: all.csv + all.xlsx + lab_specs.json
```
