# Labs Extraction Pipeline

> **Keep this doc updated when modifying the pipeline.**

## Flow Diagram

```
INPUT_PATH/*.pdf
       │
       ▼
┌─────────────────────────────────────────────────────┐
│ 1. PDF Discovery                                     │
│    glob(INPUT_FILE_REGEX) → skip if CSV valid        │
│    prompt to reprocess empty extraction pages        │
└────────────────────┬────────────────────────────────┘
                     │  one PDF at a time (multiprocessing Pool)
                     ▼
┌─────────────────────────────────────────────────────┐
│ 2. Extraction (raw data only)                         │
│    Text-first: pdftotext → ≥200 non-ws chars?        │
│      → extract_labs_from_text() → cache .json + .txt  │
│    Vision fallback (per page):                       │
│      → pdf2image → preprocess_page_image()           │
│      → extract_labs_from_page_image()               │
│        └─ function calling (1 LLM call/page)         │
│        └─ temperature-escalation retry (up to 3x)   │
│    cache → {doc}/{doc}.{page}.jpg + .json            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 3. Normalization                                     │
│    a. % name correction (unit=% ↔ name ends with (%))│
│    b. Fix misassigned % units (bio plausibility)     │
│    c. Convert values to primary units                │
│       - extract comparison operators (< >)           │
│       - preprocess: trim =, spaces, comma→period     │
│       - infer missing units (4-strategy heuristic)   │
│       - apply conversion factors from lab_specs      │
│    d. Qualitative → 0/1                              │
│       deterministic pattern matching (no LLM)        │
│    e. Validate reference ranges (detect wrong units) │
│    f. Sanitize % reference ranges                    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 4. Merge + Filter + Deduplicate                      │
│    concat all CSVs → drop rows with $UNKNOWN$ names  │
│    group by (date, lab_name) → keep primary unit     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 5. Validation (ValueValidator.validate)              │
│    Biological plausibility, inter-lab relationships, │
│    temporal consistency, format artifacts,           │
│    reference range consistency                       │
│    → sets review_needed / review_reason columns      │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 6. Export                                            │
│    all.csv + all.xlsx (hidden cols, frozen header)   │
│    lab_specs.json copied to output                   │
└─────────────────────────────────────────────────────┘
```

---

## Step Details

### 1. PDF Discovery (`run_for_profile`)
- Glob `INPUT_PATH` with `input_file_regex` (default `*.pdf`)
- Skip PDFs whose output CSV already has all `REQUIRED_CSV_COLS`
- Prompt interactively to delete and reprocess pages with empty JSON extractions
- Remaining PDFs dispatched via `multiprocessing.Pool` with `max_workers`

### 2. Extraction (`process_single_pdf`)

**Text-First path (cost optimization):**
1. `pdftotext -layout` extracts text; fails gracefully if poppler not installed
2. **Content check** (`_text_has_enough_content`): skip if < 200 non-whitespace chars
3. `extract_labs_from_text()` uses the same `TOOLS` schema and system prompt as vision; no retries
4. If text extraction returns zero results → fall through to vision
5. Successful result cached as `{doc}/{doc}.json` + `{doc}/{doc}.txt`

**Vision path (per-page, fallback):**

**Function calling:** The model is forced to invoke `extract_lab_results` (JSON schema derived from `HealthLabReportExtraction.model_json_schema()`), eliminating free-text responses. The LLM extracts only raw data — no standardization is performed during extraction.

**Post-extraction standardization** (`_apply_standardization`) maps raw names and units to standardized values via cache-based lookup:
1. **Name standardization**: calls `standardize_lab_names()` (cache-only) for all results → sets `lab_name_standardized`
2. **Unit standardization**: calls `standardize_lab_units()` (cache-only) for all `(raw_unit, std_name)` pairs → sets `lab_unit_standardized`
3. Cache misses return `$UNKNOWN$` and log a warning. Use `utils/update_standardization_caches.py` to batch-update caches via LLM.
4. Caches stored at: `config/cache/name_standardization.json`, `config/cache/unit_standardization.json`

**Retry strategy (temperature escalation):**
- Attempt 0: `temperature=0.0`
- Retries: `temperature += 0.2` per attempt, max 3 retries
- Only retries on `MALFORMED OUTPUT` Pydantic errors
- On exhaustion: returns `{_extraction_failed: true, _failure_reason: ...}`

**Pre-validation repair (`_fix_lab_results_format`):**
1. Normalize date formats (DD/MM/YYYY → YYYY-MM-DD)
2. Strip embedded metadata from numeric reference fields
3. Filter out non-dict items

**Cache:** preprocessed image saved as `{page}.jpg`; extracted JSON saved as `{page}.json` — both skipped if already present.

**Date Resolution (`_extract_document_date`):**
Priority order: `collection_date` → `report_date` → `YYYY-MM-DD` regex in filename → `None`

### 3. Normalization (`apply_normalizations` / `apply_unit_conversions`)

**a. % name correction:**
- `unit="%" but name lacks "(%) "` → swap to `{name} (%)` variant
- `unit≠"%" but name ends with "(%) "` → swap to absolute-count variant

**b. Fix misassigned % units (`fix_misassigned_percentage_units`)**
Detects protein-electrophoresis-style errors (e.g., Albumin `61.5 g/dL` that should be `61.5 %`):
- Value exceeds `biological_max` for the absolute variant
- Value fits within the percentage variant's expected range
- Reference range (if present) matches the percentage variant

**c. Unit conversion:**
- Extract `<`/`>` operators → `is_below_limit` / `is_above_limit` columns
- Preprocess: strip trailing `=`, embedded metadata (`52.6=1946` → `52.6`), spaces in thousands separators, comma → period
- Infer missing units (4-strategy heuristic): reference range magnitude, percentage variant check, value-to-range ratio, lab_specs primary unit
- Apply `conversion_factor` from `lab_specs.alternatives[]`
- Convert reference ranges by same factor
- Special case: specific gravity `> 100` → divide by 1000

**d. Qualitative values (`classify_qualitative_value`):**
- Deterministic pattern matching (~40 Portuguese/English/French/German medical terms)
- No LLM call — pure string matching with prefix/exact patterns
- Boolean labs: convert `raw_value` or `raw_comments` to `0`/`1`
- Non-boolean labs: only `0` (negative) values converted
- Unknown values return `None` (safe default — stays in `raw_value` for review)

**e. Reference range validation (`validate_reference_range`):**
- Compare extracted range against `lab_specs.ranges.default`
- If ratio (PDF ÷ expected) > 10× or < 0.1× for both min and max → suspicious
- Attempt auto-conversion using alternative unit factors before nullifying

**f. Percentage range sanitization (`sanitize_percentage_reference_ranges`):**
- For `unit="%"`: if `value > ref_max × 5` or `ref_max > 100` → nullify range

### 4. Merge + Filter + Deduplicate

- Concatenate per-document CSVs
- Drop rows where `lab_name_standardized == "$UNKNOWN$"`
- Group by `(date, lab_name_standardized)`. Within duplicates, prefer the row whose `lab_unit_standardized` matches the primary unit; otherwise keep the first.

### 5. Value Validation (`ValueValidator.validate`)
Five sequential checks; each can set `review_needed=True` and append to `review_reason`:

| Check | Reason Codes |
|-------|-------------|
| Biological plausibility | `NEGATIVE_VALUE`, `PERCENTAGE_BOUNDS`, `IMPOSSIBLE_VALUE` |
| Inter-lab relationships | `RELATIONSHIP_MISMATCH`, `COMPONENT_EXCEEDS_TOTAL` |
| Temporal consistency | `TEMPORAL_ANOMALY` |
| Format artifacts | `FORMAT_ARTIFACT` |
| Reference range cross-check | `RANGE_INCONSISTENCY`, `EXTREME_DEVIATION` |

Relationship formulas (e.g., LDL Friedewald) are defined in `lab_specs.json` under `_relationships[]` and evaluated per (date, lab) group.

### 6. Export
- `all.csv` — UTF-8, 17 columns (see schema below)
- `all.xlsx` — frozen header, per-column widths, internal columns hidden (`lab_type`, `result_index`)
- `lab_specs.json` copied to output for reproducibility

---

## LLM Calls per Pipeline Run

| Stage | LLM Calls |
|-------|-----------|
| Extraction | 1/page (function calling, raw data only) |
| Name standardization | 0 (cache-only; misses return `$UNKNOWN$`) |
| Unit standardization | 0 (cache-only; misses return `$UNKNOWN$`) |
| Qualitative→boolean | 0 (deterministic pattern matching) |
| **Total (3-page PDF)** | **3** (extraction only) |

---

## Output Schema (17 columns)

| Column | Type | Notes |
|--------|------|-------|
| `date` | datetime | Report/collection date |
| `source_file` | str | Original PDF stem + page |
| `page_number` | Int64 | 1-based |
| `lab_name` | str | Standardized (e.g., `"Blood - Glucose"`) |
| `value` | float64 | In primary unit |
| `unit` | str | Primary unit |
| `reference_min` | float64 | From PDF, in primary unit |
| `reference_max` | float64 | From PDF, in primary unit |
| `raw_lab_name` | str | Exactly as extracted |
| `raw_value` | str | Exactly as extracted |
| `raw_unit` | str | Exactly as extracted |
| `review_needed` | boolean | Set by ValueValidator |
| `review_reason` | str | Semicolon-separated reason codes |
| `is_below_limit` | boolean | Value was `< X` |
| `is_above_limit` | boolean | Value was `> X` |
| `lab_type` | str | `blood`/`urine`/`feces` (hidden in Excel) |
| `result_index` | Int64 | Position in source JSON (hidden in Excel) |

---

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Pipeline orchestration (`run_for_profile`, `process_single_pdf`) |
| `labs_parser/extraction.py` | Vision/text LLM extraction, `HealthLabReport`/`LabResult` Pydantic models, LLM-facing `HealthLabReportExtraction` schema |
| `labs_parser/standardization.py` | Lab name and unit standardization via cache-only lookup (no LLM at runtime) |
| `labs_parser/normalization.py` | Unit conversion, value preprocessing, unit inference, range validation, % name correction |
| `labs_parser/validation.py` | `ValueValidator` — data-driven extraction error detection |
| `labs_parser/config.py` | `ExtractionConfig`, `ProfileConfig`, `LabSpecsConfig` |
| `labs_parser/utils.py` | `preprocess_page_image`, logging setup |
| `config/lab_specs.json` | 328 standardized lab names, units, conversion factors, ranges, relationships |
| `config/cache/` | Persistent LLM decision caches (user-editable JSON files) |
| `utils/update_standardization_caches.py` | Batch-update name/unit standardization caches via LLM |
| `prompts/` | Prompt templates: `extraction_system`, `extraction_user`, `name_standardization`, `unit_standardization` |
| `viewer.py` | Interactive Gradio UI for reviewing extracted results |
| `test.py` | Data integrity and schema validation tests |
