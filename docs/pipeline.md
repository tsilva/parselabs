# Extraction Pipeline

This document describes the data flow through the parselabs extraction pipeline, from PDF input to validated CSV/Excel output.

Before any PDF discovery starts, the CLI validates the configured API key and `extract_model_id` by sending a minimal chat completion request. Runs fail fast if authentication, authorization, model access, or base URL configuration is invalid.

Processed document directories now use the canonical `{stem}_{hash}/` layout only, and stored extraction rows use the current `raw_*` field names only. Migrate older outputs with `utils/migrate_output_dirs.py` and `utils/migrate_raw_columns.py` before running the current runtime against historical data.

## Pipeline Overview

```
PDF Files
  │
  ├─ 1. PDF Discovery
  ├─ 2. Preflight Cache Check & Hashing
  ├─ 3. PDF Processing
  ├─ 4. Extraction To Canonical Page JSON
  ├─ 5. Review Dataset Build
  ├─ 6. Human Review
  ├─ 7. Reviewed-Truth Transform
  └─ 8. Final Export (CSV + Excel)
```

The pipeline is intentionally review-first:

`page extraction JSON -> review dataframe/view -> document review CSVs -> merged all.csv/all.xlsx`

Per-page JSON is the only canonical persisted intermediate state. Per-document CSVs and final `all.csv` / `all.xlsx` outputs are always derived from that JSON state.

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

## Stage 4: Extraction To Canonical Page JSON

**Modules:** `main.py` — `process_single_pdf()`, `parselabs/extraction.py`

The pipeline uses a **deterministic per-page routing strategy**:

### Path A: Text First When Embedded Text Is Strong

```
PDF → pdftotext (layout mode) → split into per-page text
  → Check each page for sufficient text content
  → extract_labs_from_text() per page → LLM function calling
  → Keep the text result unless it hard-fails or returns an empty likely-lab page
```

- Uses `pdftotext` (poppler) with `-layout` flag
- Threshold: 80 non-whitespace characters minimum per page
- Text extraction is preferred because it is cheaper and more reproducible on digitally-generated PDFs

### Path B: Vision Fallback

```
PDF → pdf2image (page-by-page)
  → create_page_image_variants()
      → primary: color, autocontrast, sharpen, resize ≤1800px
      → fallback: grayscale, autocontrast, sharpen, contrast boost, resize ≤1800px
  → extract_labs_from_page_image() on primary image
  → If the primary image hard-fails or returns an empty likely-lab page: retry once on fallback image
```

- Each page converted to JPG independently
- Primary image preserves color/table structure; fallback image exaggerates contrast for hard-to-read scans
- The runtime does not score or compare multiple candidates; it follows a single deterministic fallback chain
- Results are cached as `{stem}.{page}.json` and reused on rerun

### LLM Extraction Details

Both paths use OpenRouter API with **function calling** (structured output):

- **LLM-facing schema** (`LabResultExtraction`): Smaller, token-efficient — only raw extraction fields (excludes internal fields like `review_*`, `page_number`, `source_file`, `result_index`, `lab_name_standardized`, `lab_unit_standardized`)
- **Internal schema** (`HealthLabReport` / `LabResult`): Full schema with all metadata

**Retry logic** with limited temperature escalation on malformed output:
- Attempt 1: temperature = 0.0 (deterministic)
- Attempt 2: temperature = 0.2
- After retries fail: returns a failure-marked page JSON payload with `_extraction_failed=True`

**Prompt templates** (loaded from `prompts/`):
- `extraction_system.md` + `extraction_user.md` — vision extraction
- `text_extraction_user.md` — text-based extraction (template: `{text}`)

### Extraction Output

Each `LabResult` contains:

| Field | Description |
|-------|-------------|
| `raw_lab_name` | Test name as printed on the PDF |
| `raw_value` | Numeric value as string |
| `raw_lab_unit` | Unit symbol |
| `raw_reference_range` | Full range text |
| `raw_reference_min`, `raw_reference_max` | Parsed range bounds |
| `bbox_left`, `bbox_top`, `bbox_right`, `bbox_bottom` | Optional per-result bounding box, normalized to a 0-1000 page coordinate system |

Bounding boxes are requested in the same vision extraction call as the lab values, so the reviewer can highlight the selected result on the source page without a second OCR/layout pass. Text-only extraction keeps these fields null because plain text does not preserve page geometry.

Failed pages are also persisted in page JSON with `_extraction_failed` metadata so review tooling can surface extraction failures directly.

## Stage 5: Review Dataset Build

**Modules:** `parselabs/review_sync.py`, `parselabs/standardization.py`, `parselabs/normalization.py`

Per-document review CSVs are rebuilt from canonical page JSON, not treated as persisted truth.

```
For each extracted review row:
  → standardize_lab_names(): cache lookup raw_name → standardized_name
  → standardize_lab_units(): cache lookup (raw_unit, lab_name) → standardized_unit
  → when raw_unit is blank, fall back to the spec primary unit only for safe implied-unit labs (`pH`, `unitless`, `boolean`)
  → deterministically remap `%` vs absolute sibling lab names from the resolved standardized unit
  → apply deterministic normalization only
  → attach reviewer-facing ambiguity reasons
  → keep every extracted row visible before deduplication
```

### Cache Files

```
config/cache/name_standardization.json   # raw_name → standardized_name
config/cache/unit_standardization.json   # raw_unit|lab_name → standardized_unit
```

- **Cache hit:** return standardized value
- **Cache miss:** return `$UNKNOWN$` + log warning
- **Updating caches:** run `utils/update_standardization_caches.py` (batch LLM processing)

### Deterministic Normalization Only

The always-on normalization path now does only mechanically justified transforms:

### 5a. Numeric Preprocessing

- Strip trailing `=` (e.g., `"0.9="` → `"0.9"`)
- Extract first number from concatenation artifacts (e.g., `"52.6=1946"` → `"52.6"`)
- Remove space thousands separators (e.g., `"256 000"` → `"256000"`)
- European decimal format (comma → period)

### 5b. Comparison Operators

Parses limit indicators and sets boolean flags:

| Raw Value | Parsed Value | `is_below_limit` | `is_above_limit` |
|-----------|-------------|-------------------|-------------------|
| `<100` | 100 | True | False |
| `>200` | 200 | False | True |
| `≤50` | 50 | True | False |
| `≥30` | 30 | False | True |

### 5c. Qualitative Boolean Remap

Text-only urine strip analytes that initially map to numeric labs are remapped onto dedicated qualitative boolean variants before unit conversion.

- numeric urine rows stay on their original `mg/dL` standardized lab
- text urine rows normalize to `0/1` on qualitative boolean variants
- this avoids mixing numeric and boolean units under the same standardized lab name

### 5d. Percentage Variant Canonicalization

The review/export pipeline now uses the resolved standardized unit as the tie-breaker when a raw label can represent both an absolute analyte and a percentage analyte.

Examples:

- `raw_lab_name="Monocitos"` + standardized unit `%` → `Blood - Monocytes (%)`
- `raw_lab_name="Monocitos (%)"` + standardized unit `10⁹/L` → `Blood - Monocytes`

This remap is generic for any sibling pair present in `lab_specs.json`. It does not change extraction JSON or the name-standardization cache model; it only corrects the derived review/export rows before duplicate detection and export aliasing.

### 5e. Exact Unit Conversion

Converts values to primary units only when the standardized lab and unit are explicit and a conversion factor exists:

```
raw_value=5.0, raw_unit="mmol/L", lab="Blood - Glucose"
  → primary_unit = "mg/dL"
  → factor = 18.0
  → value = 5.0 × 18.0 = 90.0 mg/dL
```

The runtime still avoids subjective auto-corrections such as nullifying suspicious reference ranges. The only missing-unit fallback is the narrow implied-unit case for labs whose spec primary unit is intrinsically safe to infer (`pH`, `unitless`, `boolean`).

### 5f. Review Reasons

Review rows now surface ambiguity directly in `review_reason`, including:

- `EXTRACTION_FAILED`
- `UNKNOWN_LAB_MAPPING`
- `UNKNOWN_UNIT_MAPPING`
- `AMBIGUOUS_PERCENTAGE_VARIANT`
- `SUSPICIOUS_REFERENCE_RANGE`
- `DUPLICATE_ENTRY`

## Stage 6: Human Review

**Modules:** `viewer.py`, `review_documents.py`

The viewer reads derived review rows from canonical page JSON and lets the reviewer:

- inspect every extracted row before deduplication
- accept or reject rows directly in the backing page JSON
- highlight the selected row on the source page when bounding boxes are available
- see validation and ambiguity reasons together
- rebuild merged outputs later from the current reviewed JSON state

## Stage 7: Reviewed-Truth Transform

**Modules:** `main.py`, `parselabs/review_sync.py`, `parselabs/validation.py`

The reviewed-truth helpers used by fixture sync and approved-document regression start from rows with `review_status == accepted` only. They then:

- reapplies the shared deterministic normalization path
- drops rows that still have unresolved lab or unit mappings
- flags duplicates
- deduplicates only among accepted rows
- runs validation as a QA layer, not a self-healing layer

Validation still appends reason codes to `review_reason`:

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

**Module:** `main.py` — `run_pipeline_for_pdf_files()`, `build_final_output_dataframe()`, `export_excel()`

### Output Files

| File | Description |
|------|-------------|
| `{stem}_{hash}/{stem}.csv` | Per-document review CSV rebuilt from processed page JSON state |
| `all.csv` | Merge of all per-document review CSVs, including pending and rejected rows |
| `all.xlsx` | Excel export of the merged review dataset |
| `lab_specs.json` | Copy of lab specifications used (for reproducibility) |

### Output Schema

`all.csv` uses the same review-oriented schema as each per-document review CSV, including extracted raw fields, mapped lab/unit/value fields, validation flags, and review status columns such as `review_status` and `review_completed_at`.

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
        │       ├── create_page_image_variants()
        │       ├── extract_labs_from_page_image() ── primary image
        │       ├── extract_labs_from_page_image() ── fallback image only if needed
        │       └── cache canonical page JSON
        │
        └── rebuild_document_csv() ── derived review CSV from page JSON
             │
             ▼
[build_document_review_dataframe]
         │
         ▼
[human review in viewer]
         │
         ▼
[accepted reviewed rows only]
         │
         ▼
[apply_normalizations]
    ├── preprocess_numeric_value()
    ├── extract_comparison_value()
    ├── qualitative boolean conversion
    └── exact unit conversion only
         │
         ▼
[drop unresolved mappings]
         │
         ▼
[deduplicate_results] ── accepted rows only
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
OUTPUT: merged-review all.csv + all.xlsx + lab_specs.json
```

## Approved Document Regression

Approved-document regressions rerun a private PDF corpus and compare the final CSV output instead of page-level JSON.

### Shared Pipeline Entry Point

`main.py` now exposes:

- `run_pipeline_for_pdf_files(pdf_files, config, lab_specs)` — runs extraction, rebuilds the per-document review CSVs, computes the accepted-row reviewed-truth DataFrame for regression helpers, and returns that final DataFrame plus run metadata.
- `build_final_output_dataframe(pdf_files, config, lab_specs)` — convenience wrapper that returns the accepted-row reviewed-truth DataFrame used by the regression tooling.

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
