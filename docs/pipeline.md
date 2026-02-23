# Labs Extraction Pipeline

> **Keep this doc updated when modifying the pipeline.**

## Quick Overview

```
PDF → [Text LLM (if viable)] OR [Images → Vision LLM] → Standardize → Normalize → Validate → Dedupe → Export
```

## Pipeline Steps

### 1. PDF Discovery
- Scan `INPUT_PATH` for PDFs matching `INPUT_FILE_REGEX`
- Skip already-processed PDFs (CSV exists with required columns)
- Prompt to reprocess PDFs with empty extraction pages

### 2. Text-First Extraction (cost optimization)
- Extract text from PDF using `pdftotext`
- Check viability (`text_extraction_is_viable`):
  - Minimum 200 characters (excluding whitespace)
  - LLM classification: sends first 1000 chars to model
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
- Tracks extraction failures with reason codes

### 5. Date Resolution
Priority: `collection_date` → `report_date` → filename pattern → None

### 6. Lab Name Standardization
- LLM maps raw names → standardized names from `lab_specs.json`
- Unknown → `$UNKNOWN$`

### 7. Lab Unit Standardization
- LLM maps (raw_unit, lab_name) → standardized units
- Correct percentage names: unit "%" → name ends with "(%)"
- Falls back to primary unit from lab specs for null units

### 8. Merge All Documents
- Concatenate all document CSVs into single DataFrame
- Filter out rows with `$UNKNOWN$` lab names

### 9. Normalization
- Convert values to primary units using factors from `lab_specs.json`
- Convert reference ranges to primary units
- Map qualitative values (e.g., "NEGATIVO" → 0)

### 10. Value Validation
- Detect extraction errors by analyzing data patterns
- Flag suspicious values with reason codes:
  - `NEGATIVE_VALUE`, `IMPOSSIBLE_VALUE`, `PERCENTAGE_BOUNDS` - biological plausibility
  - `RELATIONSHIP_MISMATCH` - inter-lab relationships (e.g., LDL Friedewald formula)
  - `TEMPORAL_ANOMALY` - implausible change rates
  - `FORMAT_ARTIFACT` - concatenation errors
  - `RANGE_INCONSISTENCY`, `EXTREME_DEVIATION` - reference range issues
- Sets `review_needed`, `review_reason`, `review_confidence` columns

### 11. Deduplication
- Group by (date, lab_name_standardized)
- Keep one result per group (prefer primary unit)

### 12. Export
- `all.csv` - merged results
- `all.xlsx` - formatted Excel workbook

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Pipeline orchestration |
| `extraction.py` | Vision/text LLM extraction |
| `standardization.py` | Name/unit standardization |
| `normalization.py` | Unit conversion, value normalization |
| `validation.py` | Value-based validation and error detection |
| `config.py` | Configuration classes (ExtractionConfig, ProfileConfig, LabSpecsConfig) |
| `utils.py` | Utility functions (image preprocessing, logging) |
| `config/lab_specs.json` | Standardized names, units, conversion factors, ranges |
| `review.py` | Gradio-based extraction review UI |
| `browse.py` | Results browser UI |
| `plotting.py` | Time-series plot generation |
| `reporting.py` | Run reports with edge case detection |
| `edge_case_detection.py` | Edge case analysis utilities |
| `test.py` | Data integrity validation tests |
