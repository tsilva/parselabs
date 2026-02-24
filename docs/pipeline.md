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
│ 2. Extraction (with inline standardization)          │
│    Text-first: pdftotext → ≥200 non-ws chars?        │
│      → extract_labs_from_text() → cache .text.json  │
│    Vision fallback (per page):                       │
│      → pdf2image → preprocess_page_image()           │
│      → extract_labs_from_page_image()               │
│        └─ function calling (1 LLM call/page)         │
│        └─ inline: lab_name + unit populated by LLM  │
│        └─ temperature-escalation retry (up to 3x)   │
│      → self_consistency(N) → LLM vote if N > 1      │
│    cache → {doc}/{doc}.{page}.jpg + .json            │
│    Standardized names list appended to system prompt │
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
│       cache → config/cache/qualitative_to_boolean.json│
│       LLM fallback on cache miss                     │
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
│    → sets review_needed / review_reason /            │
│      review_confidence columns                       │
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
5. Successful result cached as `{doc}/{doc}.text.json`

**Vision path (per-page, fallback):**

**Function calling:** The model is forced to invoke `extract_lab_results` (JSON schema derived from `HealthLabReportExtraction.model_json_schema()`), eliminating free-text responses.

**Inline standardization:** The system prompt includes a dynamically built `STANDARDIZED LAB NAMES AND UNITS` section (generated from `lab_specs.json`) so the LLM populates:
- `lab_name` → mapped to `lab_name_standardized` (standardized name from the list)
- `unit` → mapped to `lab_unit_standardized` (format-normalized unit)

A cache-based fallback (`config/cache/name_standardization.json`) is applied for any results where `lab_name_standardized` is still `$UNKNOWN$` after extraction.

**Retry strategy (temperature escalation):**
- Attempt 0: `temperature=0.0`
- Retries: `temperature += 0.2` per attempt, max 3 retries
- Only retries on `MALFORMED OUTPUT` Pydantic errors
- On exhaustion: returns `{_extraction_failed: true, _failure_reason: ...}`

**Pre-validation repair (`_fix_lab_results_format`):**
1. Normalize date formats (DD/MM/YYYY → YYYY-MM-DD)
2. Map `lab_name` → `lab_name_standardized`, `unit` → `lab_unit_standardized`
3. Reassemble flattened key-value strings into dicts
4. Strip embedded metadata from numeric reference fields
5. Parse string results via LLM fallback (`_parse_string_results_with_llm`)

**Self-consistency (`self_consistency`):**
- `N_EXTRACTIONS=1` (default): single call, no voting
- `N > 1`: parallel calls at `temperature=0.5`; if all identical → return first; else LLM votes on best

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

**d. Qualitative values (`standardize_qualitative_values`):**
- Cache-first lookup (`config/cache/qualitative_to_boolean.json`)
- LLM fallback only on cache miss
- Boolean labs: convert `value_raw` or `comments` to `0`/`1`
- Non-boolean labs: only `0` (negative) values converted

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

| Check | Reason Codes | Confidence Mult |
|-------|-------------|-----------------|
| Biological plausibility | `NEGATIVE_VALUE`, `PERCENTAGE_BOUNDS`, `IMPOSSIBLE_VALUE` | 0.3–0.4 |
| Inter-lab relationships | `RELATIONSHIP_MISMATCH`, `COMPONENT_EXCEEDS_TOTAL` | 0.3–0.5 |
| Temporal consistency | `TEMPORAL_ANOMALY` | 0.6 |
| Format artifacts | `FORMAT_ARTIFACT` | 0.7 |
| Reference range cross-check | `RANGE_INCONSISTENCY`, `EXTREME_DEVIATION` | 0.5–0.7 |

Relationship formulas (e.g., LDL Friedewald) are defined in `lab_specs.json` under `_relationships[]` and evaluated per (date, lab) group.

### 6. Export
- `all.csv` — UTF-8, 19 columns (see schema below)
- `all.xlsx` — frozen header, per-column widths, internal columns hidden (`lab_type`, `result_index`)
- `lab_specs.json` copied to output for reproducibility

---

## LLM Calls: Before vs After

| Stage | Before | After |
|-------|--------|-------|
| Viability check | 1 (LLM) | 0 (simple char count) |
| Extraction | 1/page | 1/page (includes inline standardization) |
| Name standardization | 1/PDF | 0 (done in extraction) |
| Unit standardization | 1/PDF | 0 (done in extraction) |
| Qualitative→boolean | 0-2/PDF | 0 on cache hit, LLM on miss |
| **Total (3-page PDF)** | **~5-7** | **~1-3** (extraction only, warm cache) |

---

## Output Schema (19 columns)

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
| `lab_name_raw` | str | Exactly as extracted |
| `value_raw` | str | Exactly as extracted |
| `unit_raw` | str | Exactly as extracted |
| `confidence` | float64 | Default `1.0` |
| `review_needed` | boolean | Set by ValueValidator |
| `review_reason` | str | Semicolon-separated reason codes |
| `review_confidence` | float64 | Product of confidence multipliers |
| `is_below_limit` | boolean | Value was `< X` |
| `is_above_limit` | boolean | Value was `> X` |
| `lab_type` | str | `blood`/`urine`/`feces` (hidden in Excel) |
| `result_index` | Int64 | Position in source JSON (hidden in Excel) |

---

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Pipeline orchestration (`run_for_profile`, `process_single_pdf`) |
| `labs_parser/extraction.py` | Vision/text LLM extraction, `HealthLabReport`/`LabResult` Pydantic models, LLM-facing `HealthLabReportExtraction` schema, self-consistency, inline standardization |
| `labs_parser/standardization.py` | Qualitative-value standardization with LLM + cache |
| `labs_parser/normalization.py` | Unit conversion, value preprocessing, unit inference, range validation, % name correction |
| `labs_parser/validation.py` | `ValueValidator` — data-driven extraction error detection |
| `labs_parser/config.py` | `ExtractionConfig`, `ProfileConfig`, `LabSpecsConfig` |
| `labs_parser/utils.py` | `preprocess_page_image`, logging setup |
| `config/lab_specs.json` | 328 standardized lab names, units, conversion factors, ranges, relationships |
| `config/cache/` | Persistent LLM decision caches (user-editable JSON files) |
| `prompts/` | Prompt templates: `extraction_system`, `extraction_user`, `name_standardization`, `unit_standardization` |
| `viewer.py` | Interactive Gradio UI for reviewing extracted results |
| `test.py` | Data integrity and schema validation tests |
