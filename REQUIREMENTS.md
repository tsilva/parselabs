# GOAL

Extract lab results from PDF documents with 100% accuracy, standardize units, and produce a single clean CSV for data analysis. Priority: accuracy > cost > speed.

# REQUIREMENTS

## Extraction

- Accepts PDF lab reports as input; processes page by page for accuracy and easy retry/resume.
- Text-first approach: extracts text from PDF; falls back to vision if extractable text is under 1000 characters.
- Image preprocessing: converts pages to grayscale, applies 2x contrast enhancement, downscales to max 1200px width.
- Retry with temperature escalation on extraction failures: starts at 0.0, increments by 0.2, up to 3 retries (max 0.6).
- Server health check before starting extraction to fail fast on unreachable endpoints.
- Resume behavior: re-running only processes missing or failed pages; already-extracted pages are skipped.
- Preserves raw values alongside standardized values for audit.
- Document date is resolved as: sample collection date > issue date > first date found. All pages share the same date.

## Standardization

- Lab names map to a canonical vocabulary prefixed by type: "Blood - ", "Urine - ", or "Feces - ".
- Percentage-unit labs end in "(%)" to distinguish from non-fraction counterparts.
- Values are converted to primary units using configured conversion factors.
- Cache-only at runtime: name and unit standardization use pre-built caches with no LLM calls during extraction. On cache miss, the value is marked `$UNKNOWN$` for later resolution.
- Qualitative-to-boolean conversion: qualitative results (e.g., POSITIVO/NEGATIVO, REACTIVE/NON-REACTIVE) are converted to numeric values (1/0).

## Data Cleaning

- European decimal format: commas used as decimal separators are converted to periods.
- Comparison value extraction: strips comparison operators from values like `<0.05` or `>100`, preserving the numeric part.
- Trailing `=` characters are stripped from values (artifact of OCR/extraction errors).
- Space-separated thousands (e.g., `1 234`) are joined into a single number.
- Unit inference when missing: infers units from reference range heuristics and lab_specs lookup when the extraction returns no unit.

## Validation

Flagged results include one or more reason codes. Checks include:

- **Biological plausibility**: negative values, values exceeding known limits, percentages out of bounds.
- **Inter-lab relationships**: calculated values (e.g., LDL via Friedewald) compared against extracted values within tolerance.
- **Component-total constraints**: component values must not exceed their parent total.
- **Temporal consistency**: implausible change rates between consecutive results per configured daily thresholds.
- **Format artifacts**: extraction errors like value-reference concatenation (e.g., "52.6=1946") or excessive decimals.
- **Reference range consistency**: min ≤ max, extreme deviations flagged.

## Deduplication

- Duplicate (date, lab_name) pairs are flagged as likely extraction errors.
- Duplicate documents detected by file hash; duplicates are skipped with a warning.

## Viewer

Interactive web UI for browsing, plotting, and reviewing results:

- Data table filterable by lab name, review status (All, Needs Review, Abnormal, Unhealthy, Unreviewed, Accepted, Rejected), and latest-only toggle.
- Abnormal = value outside the PDF-reported reference range. Unhealthy = value outside the lab_specs healthy range. These are distinct concepts.
- Summary statistics: total results, unique tests, date range, review/abnormal/unhealthy/reviewed counts.
- Time-series plots with bands for PDF-reported and configured healthy ranges; multi-lab stacked subplots.
- Source page image display for visual verification; raw vs. standardized value comparison.
- Accept/Reject/Skip review workflow persisted across sessions; keyboard shortcuts for actions and navigation.
- Healthy ranges adjusted by user demographics (age, gender) when provided.
- CSV export of filtered data.

## Output

- Primary output: merged CSV and formatted Excel (frozen header, optimized widths, internal columns hidden).
- 15-column schema: date, source file, page number, standardized name/value/unit, reference min/max, raw name/value/unit, review flags (needed/reason), lab type, result index.
- Per-document folder (named by document stem + file hash) containing: original PDF, preprocessed page images, per-page JSON, and per-document CSV.

## Configuration

- **Profiles** (YAML/JSON): name, input path, output path, optional file pattern, worker count, and demographics (gender, DOB, height, weight).
- **Environment variables**: AI model selection and API credentials (separate from profiles).
- **Lab specs** (`lab_specs.json`): canonical lab names, types, primary units, alternative units with conversion factors, healthy ranges (with age/gender variants), biological limits, max daily change rates, inter-lab relationships, and LOINC codes (required for every lab).

## CLI

Supports running all or a specific profile, listing profiles, and overriding model, worker count, file pattern, and env file.

## Testing

Validates configuration integrity (schema, LOINC codes, naming conventions, conversion factors, range consistency) and data integrity (no missing dates, no duplicates, prefix/unit consistency, outlier detection).

## Logging

Logs processing details and reports failed pages with reasons.
