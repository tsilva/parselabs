# Extraction Pipeline

This document describes the data flow through the parselabs extraction pipeline, from PDF input to validated CSV/Excel output.

Before any PDF discovery starts, the CLI validates the configured API key and `extract_model_id` by sending a minimal chat completion request. Runs fail fast if authentication, authorization, model access, or base URL configuration is invalid.

## Pipeline Overview

```
PDF Files
  │
  ├─ 1. PDF Discovery
  ├─ 2. Preflight Cache Check & Hashing
  ├─ 3. PDF Processing
  ├─ 4. Extraction (per-page text/vision hybrid)
  ├─ 5. Standardization (cache-based mapping)
  ├─ 6. Per-PDF CSV Export
  ├─ 7. Merge All CSVs
  ├─ 8. Normalization (unit conversion, dedup)
  ├─ 9. Validation (flag suspicious values)
  └─ 10. Final Export (CSV + Excel)
```

## Stage 1: PDF Discovery

**Module:** `main.py` — `_discover_pdf_files()`, `run_for_profile()`

Before processing starts, the runner enumerates the input directory and matches only top-level files against `input_file_regex`.

1. Call `Path.iterdir()` on the configured input directory
2. Surface missing-path and permission failures immediately
3. Match filenames case-insensitively, so `*.pdf` also catches `.PDF`

Immediately before this stage, `run_for_profile()` sends a tiny prompt through `chat.completions.create()` using the configured extraction model. This confirms the API key can perform the same class of request the extraction pipeline depends on.

This avoids the misleading "0 PDFs found" result that `Path.glob()` can produce on some cloud-backed macOS folders when directory access is denied.

## Stage 2: Preflight Cache Check & Hashing

**Module:** `main.py` — `_prepare_pdf_run()`, `_load_pdf_inventory()`, `_compute_file_hash()`

Explicit runs perform a stat-first preflight before any extraction workers start:

1. Collect `resolved path`, `size`, and `mtime_ns` for each input PDF
2. Load `output/logs/pdf_inventory.json` if it exists
3. Treat manifest matches with valid per-PDF CSVs as warm-cache hits
4. Hash only the remaining PDFs with SHA-256 (first 8 hex chars)
5. Deduplicate exact-content matches across both cached and newly-hashed PDFs

The manifest is only a warm-cache shortcut. Exact identity still comes from SHA-256, and files are re-hashed whenever size or modification time changes.

## Stage 3: PDF Processing

**Module:** `main.py` — `_setup_pdf_processing()`, `process_single_pdf()`

Each unique PDF that survives preflight is processed independently (parallelized via `multiprocessing.Pool`):

1. Reuse the precomputed SHA-256 hash from preflight
2. Create output directory: `{pdf_stem}_{hash}/` — hash prevents collisions when different files share the same name
3. Copy original PDF into the output directory for archival

## Stage 4: Extraction

**Modules:** `main.py` — `process_single_pdf()`, `parselabs/extraction.py`

The pipeline uses a **per-page hybrid strategy** to improve recall on mixed-quality PDFs.

### Path A: Page Text Extraction (Preferred When Strong)

```
PDF → pdftotext (layout mode) → split into per-page text
  → Check each page for sufficient text content
  → extract_labs_from_text() per page → LLM function calling
  → Keep text result if page-level quality looks strong
```

- Uses `pdftotext` (poppler) with `-layout` flag
- Threshold: 80 non-whitespace characters minimum per page
- Text extraction is accepted only when the page result looks complete enough (non-empty rows, reasonable value coverage)

### Path B: Vision Extraction (Fallback + Re-read)

```
PDF → pdf2image (page-by-page)
  → create_page_image_variants()
      → primary: color, autocontrast, sharpen, resize ≤1800px
      → fallback: grayscale, autocontrast, sharpen, contrast boost, resize ≤1800px
  → extract_labs_from_page_image() on primary image
  → If page output looks weak: retry on fallback image
  → Keep the strongest page candidate
```

- Each page converted to JPG independently
- Primary image preserves color/table structure; fallback image exaggerates contrast for hard-to-read scans
- Weak pages trigger a second pass on the alternate image variant instead of trusting the first pass
- Results cached as `{stem}.{page}.json` — skips re-extraction on rerun

### LLM Extraction Details

Both paths use OpenRouter API with **function calling** (structured output):

- **LLM-facing schema** (`LabResultExtraction`): Smaller, token-efficient — only raw extraction fields (excludes internal fields like `review_*`, `page_number`, `source_file`, `result_index`, `lab_name_standardized`, `lab_unit_standardized`)
- **Internal schema** (`HealthLabReport` / `LabResult`): Full schema with all metadata

**Retry logic** with temperature escalation on malformed output:
- Attempt 1: temperature = 0.0 (deterministic)
- Attempt 2: temperature = 0.2
- Attempt 3: temperature = 0.4
- Attempt 4: temperature = 0.6
- After all retries fail: returns empty report with `_extraction_failed=True`

**Prompt templates** (loaded from `prompts/`):
- `extraction_system.md` + `extraction_user.md` — vision extraction
- `text_extraction_user.md` — text-based extraction (template: `{text}`)

### Page-Level Candidate Selection

For each page, the pipeline chooses the strongest available candidate:

1. Try text extraction if the page has enough embedded text
2. If the text result is weak, run vision on the primary image
3. If the primary vision result is weak, run vision again on the fallback image
4. Score candidates heuristically (result count, null-value rate, empty names) and keep the best one

### Extraction Output

Each `LabResult` contains:

| Field | Description |
|-------|-------------|
| `raw_lab_name` | Test name as printed on the PDF |
| `raw_value` | Numeric value as string |
| `raw_lab_unit` | Unit symbol |
| `raw_reference_range` | Full range text |
| `raw_reference_min`, `raw_reference_max` | Parsed range bounds |

## Stage 5: Standardization

**Modules:** `main.py` — `_apply_standardization()`, `parselabs/standardization.py`

Maps raw lab names and units to standardized enum values using **persistent JSON caches**. No LLM calls at runtime. This is the sole standardization path — the LLM extracts only raw data.

```
For each LabResult:
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

## Stage 6: Per-PDF CSV Export

**Module:** `main.py` — `_save_results_to_csv()`

Each PDF produces a CSV in its output directory: `{stem}_{hash}/{stem}.csv`

Contains one row per extracted result with:

- document/date metadata
- page and result indexes for round-tripping review actions
- raw extracted fields
- cached standardized name/unit mappings
- review status columns (initially empty until reviewed)

Each page JSON remains the authoritative editable source of truth for review:

- per-result `review_status`
- per-result `review_completed_at`
- root-level `review_missing_rows` markers for omitted source rows the reviewer will repair manually

## Stage 7: Merge

**Module:** `main.py` — `merge_csv_files()`

All per-PDF CSVs are concatenated into a single `all.csv` in the profile's output directory.

## Stage 8: Normalization

**Module:** `parselabs/normalization.py`

Transforms the merged DataFrame through several steps:

### 8a. Numeric Preprocessing — `preprocess_numeric_value()`

Cleans raw value strings before numeric conversion:
- Strip trailing `=` (e.g., `"0.9="` → `"0.9"`)
- Extract first number from concatenation artifacts (e.g., `"52.6=1946"` → `"52.6"`)
- Remove space thousands separators (e.g., `"256 000"` → `"256000"`)
- European decimal format (comma → period)

### 8b. Comparison Operators — `extract_comparison_value()`

Parses limit indicators and sets boolean flags:

| Raw Value | Parsed Value | `is_below_limit` | `is_above_limit` |
|-----------|-------------|-------------------|-------------------|
| `<100` | 100 | True | False |
| `>200` | 200 | False | True |
| `≤50` | 50 | True | False |
| `≥30` | 30 | False | True |

### 8c. Unit Conversion — `apply_normalizations()`

Converts values to primary units using factors from `lab_specs.json`:

```
raw_value=5.0, raw_unit="mmol/L", lab="Blood - Glucose"
  → primary_unit = "mg/dL"
  → factor = 18.0
  → value = 5.0 × 18.0 = 90.0 mg/dL
```

### 8d. Deduplication — `deduplicate_results()`

- Groups by `(date, lab_name)`
- Keeps row with latest `result_index` (most complete data)
- Logs deduplication stats

### 8e. Type Conversion — `apply_dtype_conversions()`

Forces proper column types: `date` → datetime, `value` → float64, booleans, etc.

## Stage 9: Validation

**Module:** `parselabs/validation.py` — `ValueValidator`

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

## Stage 10: Final Export

**Module:** `main.py` — `run_pipeline_for_pdf_files()`, `build_final_output_dataframe()`, `export_excel()`

### Output Files

| File | Description |
|------|-------------|
| `{stem}_{hash}/{stem}.csv` | Per-document review CSV rebuilt from processed page JSON state |
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
lab_unit          # Primary unit (e.g., "mg/dL")
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
[run_pipeline_for_pdf_files]
    │
    ├── _prepare_pdf_run()
    │   ├── _load_pdf_inventory()
    │   ├── stat each PDF (path, size, mtime_ns)
    │   ├── warm-cache check against valid per-PDF CSVs
    │   └── _compute_file_hash() only for uncached/changed PDFs
    │
    └── [process_single_pdf] ─ parallelized via multiprocessing.Pool
    │
        │
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
        ├── _apply_standardization()
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

## Approved Document Regression

Approved-document regressions rerun a private PDF corpus and compare the final CSV output instead of page-level JSON.

### Shared Pipeline Entry Point

`main.py` now exposes:

- `run_pipeline_for_pdf_files(pdf_files, config, lab_specs)` — runs the same extraction, merge, normalization, deduplication, validation, and export-shaping flow used by the CLI, and returns the final DataFrame plus run metadata.
- `build_final_output_dataframe(pdf_files, config, lab_specs)` — convenience wrapper that returns only the final `all.csv`-shape DataFrame.

These are used by both the CLI profile flow and the regression tooling so there is only one implementation of the pipeline logic.

### Fixture Layout

Private approved fixtures live under `tests/fixtures/approved/`:

```text
tests/fixtures/approved/
  <stem>_<hash>/
    document.pdf
    expected.csv
    case.json
```

`case.json` records:

- `case_id`
- `original_filename`
- `stem`
- `file_hash`
- `profile`
- `approved_at`
- `reviewed_at`

### Canonical CSV Comparison

Regression comparisons canonicalize the final export before comparing:

- keep the final `all.csv` column set
- sort rows by `date`, `lab_name`, `page_number`, `result_index`, `raw_lab_name`, `raw_value`, `raw_unit`
- normalize dates to `YYYY-MM-DD`
- normalize floats with `.15g`
- normalize nullable ints to decimal strings
- normalize booleans to `true` / `false` / empty string
- trim leading/trailing whitespace in string cells

The comparison remains exact after canonicalization; there is no numeric tolerance and no ignored final-output columns.

### Approval Workflow

Review processed outputs first:

```bash
parselabs-review-docs --profile myname
```

Then rebuild reviewed outputs and sync fixture-ready documents into the private fixture corpus:

```bash
parselabs --profile myname --rebuild-from-json
uv run python utils/regression_cases.py sync-reviewed --profile myname
```

The sync helper:

- scans the profile `output_path`
- rebuilds each document CSV from page JSON review state
- blocks final rebuilds by default when pending rows or `review_missing_rows` markers remain
- exports only rows with `review_status == accepted`
- copies only fixture-ready documents: every extracted row reviewed, no unresolved missing-row markers
- rewrites each valid case `expected.csv` from reviewed JSON truth instead of a fresh extraction run
- removes stale fixture cases for the same profile when a document is no longer fixture-ready

For corpus-level quality analysis:

```bash
uv run python utils/regression_cases.py report --profile myname
```

This prints counts for rejected rows, unresolved missing-row markers, unknown mappings, validation reasons, and the top raw names/units involved in rejected rows.

### Running the Suite

The pytest suite is intentionally opt-in because it uses real PDFs and live extraction calls:

```bash
RUN_APPROVED_DOCS=1 uv run pytest -m approved_docs
```

Behavior:

- skips cleanly when `RUN_APPROVED_DOCS` is not set
- fails setup if the flag is set but no approved fixtures exist
- fails setup if the selected profile is missing `openrouter_api_key` or `extract_model_id`
