# Labs Extraction Pipeline

> **Keep this doc updated when modifying the pipeline.**

## Quick Overview

```
PDF → [Text LLM (if viable)] OR [Images → Vision LLM] → Standardize → Normalize → Dedupe → Export
```

## Pipeline Steps

### 1. PDF Discovery
- Scan `INPUT_PATH` for PDFs matching `INPUT_FILE_REGEX`
- Skip already-processed PDFs (CSV exists)

### 2. Text-First Extraction (cost optimization)
- Extract text from PDF using `pdftotext`
- Check viability (`text_extraction_is_viable`):
  - Minimum 200 characters (excluding whitespace)
  - LLM classification: sends first 1000 chars to cheap model (`google/gemini-3-flash-preview`)
  - Classifies whether text contains structured lab data (test names, values, units, ranges)
  - Results cached in `config/cache/viability_cache.json` (keyed by MD5 hash of first 500 chars)
- If viable, use text-based LLM extraction (cheaper than vision)
- Cache result as `{doc}/{doc}.text.json`
- Fall back to vision extraction if text extraction fails or returns no results

### 3. PDF → Images (fallback)
- Used when text extraction isn't viable
- Convert each page to JPG via `pdf2image`
- Preprocess: grayscale, resize (max 1200px), contrast 2x
- Save as `{doc}/{doc}.{page}.jpg`

### 4. Vision Model Extraction
- Send image to vision LLM with function calling schema
- Returns `HealthLabReport` with `LabResult[]` validated by Pydantic
- Self-consistency: run N times, LLM votes on best if results differ
- Cache result as `{doc}/{doc}.{page}.json`

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

### 10. Deduplication
- Group by (date, lab_name_standardized)
- Keep one result per group (prefer primary unit)

### 11. Export
- `all.csv` - merged results
- `all.xlsx` - AllData + MostRecentByEnum sheets

## Key Files

| File | Purpose |
|------|---------|
| `extract.py` | Pipeline orchestration |
| `extraction.py` | Vision LLM extraction |
| `standardization.py` | Name/unit standardization |
| `normalization.py` | Unit conversion, value normalization |
| `config.py` | Configuration classes (ExtractionConfig, ProfileConfig, LabSpecsConfig) |
| `utils.py` | Utility functions (image preprocessing, logging) |
| `config/lab_specs.json` | Standardized names, units, conversion factors |
| `plotting.py` | Time-series plot generation (standalone) |
| `reporting.py` | Run reports with edge case detection (standalone) |
| `review.py` | Interactive review UI |
| `browse.py` | Results browser UI |
| `test.py` | Data validation tests |
