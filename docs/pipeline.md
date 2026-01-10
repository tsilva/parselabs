# Labs Extraction Pipeline

> **Keep this doc updated when modifying the pipeline.**

## Quick Overview

```
PDF → Images → Vision LLM → Verify → Standardize → Normalize → Dedupe → Export
```

## Pipeline Steps

### 1. PDF Discovery
- Scan `INPUT_PATH` for PDFs matching `INPUT_FILE_REGEX`
- Skip already-processed PDFs (CSV exists)

### 2. PDF → Images
- Convert each page to JPG via `pdf2image`
- Preprocess: grayscale, resize (max 1200px), contrast 2x
- Save as `{doc}/{doc}.{page}.jpg`

### 3. Vision Model Extraction
- Send image to vision LLM with function calling schema
- Returns `HealthLabReport` with `LabResult[]` validated by Pydantic
- Self-consistency: run N times, LLM votes on best if results differ
- Cache result as `{doc}/{doc}.{page}.json`

### 4. Post-Extraction Verification (optional)
- Re-extract with different model provider
- Compare results, batch verify disagreements
- Character-level verification for uncertain values
- Arbitration via third model if needed
- Adds: `verification_status`, `verification_confidence`

### 5. Date Resolution
Priority: `collection_date` → `report_date` → filename pattern → None

### 6. Lab Name Standardization
- LLM maps raw names → standardized names from `lab_specs.json`
- Unknown → `$UNKNOWN$`

### 7. Lab Unit Standardization
- LLM maps (raw_unit, lab_name) → standardized units
- Correct percentage names: unit "%" → name ends with "(%)"

### 8. Merge All Documents
- Concatenate all document CSVs into single DataFrame

### 9. Normalization
- Convert values to primary units using factors from `lab_specs.json`
- Convert reference ranges to primary units
- Map qualitative values (e.g., "NEGATIVO" → 0)
- Compute `is_out_of_reference`, `is_in_healthy_range`

### 10. Edge Case Detection
Flags for review: null value with source text, qualitative in comments, numeric without unit, inequality in value, complex reference ranges, duplicate tests

### 11. Deduplication
- Group by (date, lab_name_standardized)
- Keep one result per group (prefer primary unit)

### 12. Export
- `all.csv` - merged results
- `all.xlsx` - AllData + MostRecentByEnum sheets
- `plots/` - time-series plots per lab test
- Run report - console summary

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Pipeline orchestration |
| `extraction.py` | Vision LLM extraction |
| `verification.py` | Multi-model verification |
| `standardization.py` | Name/unit standardization |
| `normalization.py` | Unit conversion, value normalization |
| `edge_case_detection.py` | Review flag detection |
| `config/lab_specs.json` | Standardized names, units, conversion factors |
