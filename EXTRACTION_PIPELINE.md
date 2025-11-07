# Lab Results Extraction Pipeline

This document provides a step-by-step breakdown of the lab results extraction pipeline.

## Overview

The pipeline transforms PDF lab reports into structured, normalized CSV/Excel data through 8 main stages: configuration, PDF processing, transcription, extraction, normalization, deduplication, output generation, and visualization.

## Pipeline Stages

### 1. Configuration & Setup

- **Load Environment Variables** (via `load_env_config()`)
  - `INPUT_PATH` - Directory containing PDF files
  - `INPUT_FILE_REGEX` - Pattern to match input PDFs
  - `OUTPUT_PATH` - Output directory
  - `SELF_CONSISTENCY_MODEL_ID` - Model for voting on inconsistent results
  - `TRANSCRIBE_MODEL_ID` - Vision model for OCR (e.g., Gemini Flash)
  - `EXTRACT_MODEL_ID` - Model for structured extraction
  - `N_TRANSCRIPTIONS` - Number of transcription attempts per page (default: 1)
  - `N_EXTRACTIONS` - Number of extraction attempts per page (default: 1)
  - `MAX_WORKERS` - Parallel workers for PDF processing (default: 1)
  - `OPENROUTER_API_KEY` - API key

- **Initialize Logging**
  - Clear previous logs (optional)
  - Configure file handlers: `logs/info.log` and `logs/error.log`

- **Load Configuration Files**
  - `config/lab_names_mappings.json` - Maps slugified lab names to standardized enums
  - `config/lab_specs.json` - Defines primary units, conversion factors, and healthy ranges
  - Note: Lab units are enforced via `LabUnit` enum in Pydantic model

- **Create Output Directory Structure**
  - Ensure output directory exists
  - Create subdirectory for each PDF document

### 2. PDF Discovery & Parallel Processing

- **Find PDF Files**
  - Scan `INPUT_PATH` for files matching `INPUT_FILE_REGEX`
  - Sort files alphabetically

- **Initialize Worker Pool**
  - Create multiprocessing pool with `MAX_WORKERS` workers
  - Distribute PDFs across workers for parallel processing

### 3. Single PDF Processing (`process_single_pdf`)

For each PDF document, the following steps are executed:

#### 3.1 Setup
- Create output subdirectory: `OUTPUT_PATH/{pdf_stem}/`
- Copy PDF to output directory (if not already present)
- Convert PDF to PIL image objects (one per page)

#### 3.2 Page-by-Page Processing

For each page in the PDF:

##### 3.2.1 Image Preprocessing (`preprocess_page_image`)
- **Check if preprocessed image exists**: `{pdf_stem}/{pdf_stem}.{page_num}.jpg`
- **If not, preprocess**:
  - Convert to grayscale
  - Resize to max width of 1200px (maintain aspect ratio)
  - Enhance contrast (factor: 2.0)
  - Save as JPG (quality: 95)

##### 3.2.2 Transcription (`transcription_from_page_image`)
- **Check if transcription exists**: `{pdf_stem}/{pdf_stem}.{page_num}.txt`
- **If not, transcribe**:
  - Use vision model (e.g., Gemini Flash) to OCR the image
  - Apply **self-consistency pattern** (run N_TRANSCRIPTIONS times):
    - If N=1: Return single result
    - If N>1 and outputs differ:
      - Submit all outputs to voting model
      - Select most consistent transcription
  - Save transcription as `.txt` file
  - **Prompt**: Preserve layout, exact text, numbers, units, reference ranges

##### 3.2.3 Structured Extraction (`extract_labs_from_page_transcription`)
- **Check if extraction exists**: `{pdf_stem}/{pdf_stem}.{page_num}.json`
- **If not, extract**:
  - Use extraction model with function calling
  - Apply **self-consistency pattern** (run N_EXTRACTIONS times):
    - If N=1: Return single result
    - If N>1 and outputs differ:
      - Submit all outputs to voting model
      - **Fallback**: If voting fails, select result with highest average confidence
  - **Function schema**: `HealthLabReport` (Pydantic model)
    - Document metadata: `report_date`, `collection_date`, `lab_facility`, `patient_name`, etc.
    - Nested list of `LabResult` objects with fields:
      - `lab_type`, `lab_name`, `lab_code`, `lab_value`, `lab_unit`, `lab_method`
      - `lab_range_min`, `lab_range_max`, `reference_range_text`, `is_flagged`
      - `confidence`, `lack_of_confidence_reason`, `source_text`, `page_number`
  - **Preprocessing & Error Handling**:
    - Fix common LLM errors (e.g., `test_name` → `lab_name`, `unit` → `lab_unit`)
    - Parse malformed `page_count` strings (e.g., "2/2" → 2)
    - Salvage valid results if validation fails
  - **Validation**: Pydantic validates against schema
  - Normalize empty strings to `null`
  - Save as `.json` file

##### 3.2.4 Metadata Enrichment
- Add `page_number` (1-indexed) to each `LabResult`
- Add `source_file` (page-specific filename: `{pdf_stem}.{page_num}`)

#### 3.3 Date Resolution
- **Priority order**:
  1. `collection_date` from first page extraction
  2. `report_date` from first page extraction
  3. YYYY-MM-DD pattern in PDF filename (regex match)
  4. `None` (with warning logged)
- Apply resolved date to all results in the PDF

#### 3.4 DataFrame Assembly
- Combine all page results into single DataFrame
- Add `date` column with resolved date
- Ensure all core `LabResult` columns exist (fill missing with `None`)
- Save PDF-level CSV: `{pdf_stem}/{pdf_stem}.csv`

### 4. Cross-PDF Aggregation

- **Merge PDF DataFrames**
  - Read all PDF-level CSVs
  - Add `source_file` column (PDF CSV filename)
  - Concatenate into single DataFrame

- **Result**: `merged_df` with all lab results from all PDFs

### 5. Normalization & Mapping

#### 5.1 Slugification
- **`lab_name_slug`**: `{lab_type}-{slugified_name}`
  - Example: "blood-hemoglobina1c"
  - Lab type defaults to "blood" if missing/unknown
  - Slugify: lowercase, µ→"micro", %→"percent", remove non-alphanumeric except hyphens, collapse spaces/underscores, remove hyphens

#### 5.2 Enum Mapping
- **`lab_name_enum`**: Map `lab_name_slug` → standardized enum via `lab_names_mappings.json`
  - Example: "blood-hemoglobina1c" → "Blood - Hemoglobin A1c"
  - Log error if unmapped

- **`lab_unit_enum`**: Alias of `lab_unit` (already standardized via LabUnit enum during extraction)
  - Example: "mg/dL" (directly from extraction, no mapping needed)

#### 5.3 Unit Conversion (`convert_to_primary_unit`)
For each row:
- Look up `lab_specs[lab_name_enum]`
- Get `primary_unit` and `alternatives` with conversion factors
- **If unit matches primary unit**: No conversion needed
- **If unit is in alternatives**: Multiply value/ranges by conversion factor
- **Output**:
  - `lab_value_final` - Converted value
  - `lab_unit_final` - Primary unit
  - `lab_range_min_final` / `lab_range_max_final` - Converted ranges

#### 5.4 Health Status Computation

- **`is_flagged_final`** (`compute_is_flagged_final`)
  - `True` if value is outside `[lab_range_min_final, lab_range_max_final]`
  - Based on document's own reference ranges

- **`healthy_range_min` / `healthy_range_max`** (`get_healthy_range`)
  - Look up from `lab_specs[lab_name_enum].ranges.healthy`
  - Universal healthy ranges (not document-specific)

- **`is_in_healthy_range`** (`compute_is_in_healthy_range`)
  - `True` if value is within `[healthy_range_min, healthy_range_max]`
  - `None` if no healthy range defined

### 6. Deduplication (`pick_best_dupe`)

- **Group by**: `(date, lab_name_enum)`
- **Selection logic**:
  - If multiple results for same (date, lab) pair:
    - Prefer result with `lab_unit_enum == primary_unit`
    - Otherwise: Keep first result
- **Result**: One result per (date, lab_name_enum) combination

### 7. Output Generation

#### 7.1 Column Ordering & Type Conversion
- **Reorder columns** per `COLUMN_SCHEMA`
- **Convert dtypes**:
  - `date` → `datetime64[ns]`
  - Boolean columns → `boolean`
  - Numeric columns → `float64` or `Int64`
- **Sort by**: `date` (descending), `lab_name_enum` (ascending)

#### 7.2 CSV Export
- **`OUTPUT_PATH/all.csv`** - All lab results (merged, normalized, deduplicated)

#### 7.3 Excel Export
- **`OUTPUT_PATH/all.xlsx`** - Two-sheet workbook:
  1. **AllData** sheet:
     - All lab results
     - Hidden columns per `COLUMN_SCHEMA.excel_hidden`
     - Column widths per `COLUMN_SCHEMA.excel_width`
     - Freeze top row
  2. **MostRecentByEnum** sheet:
     - Drop duplicates by `lab_name_enum` (keep most recent by date)
     - Same formatting as AllData sheet
     - Shows latest value for each lab test type

### 8. Visualization (`plot_lab_enum`)

- **Create plot directories**:
  - `plots/` (project root)
  - `OUTPUT_PATH/plots/`
  - Clear previous plots

- **For each `lab_name_enum` with ≥2 data points**:
  - Filter data for that lab
  - Sort by date (ascending)
  - Create time-series line plot:
    - X-axis: Date (formatted as years)
    - Y-axis: `lab_value_final` with unit label
    - Plot line with markers
    - **Reference ranges**:
      - Light green band: `[healthy_range_min, healthy_range_max]`
      - Light red bands: Below/above healthy range
      - Gray dashed lines: Mode of range boundaries
    - Legend if ranges present
  - Save as: `{lab_name_enum}.png` (with percentage replaced as "percentage" in filename)
  - Save to both plot directories

- **Parallel processing**: Use multiprocessing pool (CPU count - 1 workers)

## Key Design Patterns

### Self-Consistency Pattern
Used in transcription and extraction steps:
1. Run operation N times (N=1 for single pass, N>1 for consensus)
2. If outputs differ, use voting model to select most consistent result
3. Fallback: For extraction, use result with highest confidence

### Idempotency & Caching
- All intermediate files (JPG, TXT, JSON) are cached
- Processing skips existing files (no regeneration unless missing)
- Enables resumable pipeline execution

### Error Handling
- Individual page/PDF failures are logged but don't halt pipeline
- Salvage logic attempts to recover partial results from validation errors
- Fallback values prevent downstream KeyErrors

## Data Flow Summary

```
PDF → Pages → Images (.jpg)
            → Transcriptions (.txt)
            → Extractions (.json)
            → Per-PDF CSV
→ Merged CSV
→ Normalized/Mapped/Deduplicated DataFrame
→ all.csv + all.xlsx (AllData + MostRecentByEnum)
→ Time-series plots (plots/*.png)
```

## Key Files & Locations

### Inputs
- `INPUT_PATH/*.pdf` - Source lab reports

### Configuration
- `.env` - Environment variables
- `config/lab_names_mappings.json` - Name standardization
- `config/lab_specs.json` - Primary units, conversions, healthy ranges

### Intermediate Outputs (per PDF)
- `OUTPUT_PATH/{pdf_stem}/` - Document directory
  - `{pdf_stem}.pdf` - Copied source PDF
  - `{pdf_stem}.{page}.jpg` - Preprocessed page images
  - `{pdf_stem}.{page}.txt` - Transcriptions
  - `{pdf_stem}.{page}.json` - Extracted structured data
  - `{pdf_stem}.csv` - Combined results for document

### Final Outputs
- `OUTPUT_PATH/all.csv` - All results (merged, normalized, deduplicated)
- `OUTPUT_PATH/all.xlsx` - Excel with AllData + MostRecentByEnum sheets
- `OUTPUT_PATH/plots/*.png` - Time-series plots per lab test
- `plots/*.png` - Plots also saved to project root

### Logs
- `logs/info.log` - Info-level logs
- `logs/error.log` - Error-level logs
