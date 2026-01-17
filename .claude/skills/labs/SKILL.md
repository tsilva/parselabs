---
name: labs
description: Data diagnostics for labs-parser. Use when investigating extraction issues, validating data integrity, analyzing unknowns, or checking statistics. Knows how to find and read source document images to verify extractions.
user_invocable: true
---

# Labs Diagnostics Skill

Data diagnostics for labs-parser extraction results.

## Core Principle

**Diagnostics = Detection + Investigation**

When you find an issue (outlier, unknown, validation error), you must **look at the source document** to determine:
- Is the extraction wrong? → Fix extraction or re-run
- Is the data actually unusual? → Keep it, maybe update config

## Project Structure

### Key Paths (from active profile)

```python
# Load profile to get paths
import yaml
profile_path = "profiles/{profile_name}.yaml"  # or .json
profile = yaml.safe_load(open(profile_path))

OUTPUT_PATH = profile['output_path']
INPUT_PATH = profile['input_path']
```

### File Organization

For each PDF `{doc_stem}.pdf`:
```
{OUTPUT_PATH}/
├── {doc_stem}/
│   ├── {doc_stem}.001.jpg          # Preprocessed page images
│   ├── {doc_stem}.002.jpg
│   ├── {doc_stem}.001.json         # Extraction data per page
│   ├── {doc_stem}.002.json
│   └── {doc_stem}.csv              # Combined results for document
├── all.csv                          # Merged results from all documents
└── all.xlsx                         # Excel with formatted data
```

### Mapping CSV Rows to Source Files

Given a row from `all.csv`:
```python
source_file = row['source_file']      # e.g., "2024-01-15-labs"
page_number = row['page_number']      # e.g., 2

# Source image path (zero-padded to 3 digits):
image_path = f"{OUTPUT_PATH}/{source_file}/{source_file}.{page_number:03d}.jpg"

# Extraction JSON path:
json_path = f"{OUTPUT_PATH}/{source_file}/{source_file}.{page_number:03d}.json"
```

## Diagnostic Workflows

| Trigger Keywords | Workflow |
|------------------|----------|
| "run diagnostics", "validate", "check data" | Run `python test.py`, parse issues, investigate each |
| "investigate", "check", "look at" + source/page/lab | Find result in CSV, read source image, compare |
| "unknowns", "unmapped", "$UNKNOWN$" | Run `python utils/analyze_unknowns.py`, show source context |
| "stats", "summary", "statistics" | Load `all.csv`, compute and show statistics |

## Workflow Details

### 1. Validation Workflow

When asked to validate or run diagnostics:

```bash
# Run the test suite
python test.py
```

Parse the output for issues:
- Outliers (values >3 std from mean)
- Duplicate rows
- Duplicate (date, lab_name) pairs
- Lab name prefix mismatches
- Percentage unit issues
- Unit inconsistencies

For each issue found, **investigate the source**.

### 2. Investigation Workflow

When asked to investigate a specific result:

1. **Find the result in `all.csv`**:
   ```python
   import pandas as pd
   df = pd.read_csv(f"{OUTPUT_PATH}/all.csv")

   # Filter by source_file, page_number, lab_name, etc.
   result = df[(df['source_file'] == source_file) &
               (df['page_number'] == page_number)]
   ```

2. **Reconstruct the image path**:
   ```python
   image_path = f"{OUTPUT_PATH}/{source_file}/{source_file}.{page_number:03d}.jpg"
   ```

3. **Read the source image** using the Read tool (Claude can view images)

4. **Compare** what's visually in the image vs what was extracted

5. **Determine** if it's an extraction error or real unusual data

### 3. Unknowns Workflow

When asked about unknowns or unmapped values:

```bash
# Run the analyzer
python utils/analyze_unknowns.py
```

This shows:
- Lab names that couldn't be mapped to standardized names
- Units that couldn't be converted
- Their frequency and source locations

For each unknown, read the source image to see how it's actually written.

### 4. Statistics Workflow

When asked for stats or summary:

```python
import pandas as pd
df = pd.read_csv(f"{OUTPUT_PATH}/all.csv")

# Key statistics:
# - Total results count
# - Unique lab names
# - Date range covered
# - Confidence score distribution
# - Verification status counts
# - Results per source file
```

## Issue-Specific Guidance

### Outliers (>3 std from mean)

1. Find the flagged value in all.csv
2. Read the source image at the page
3. Check: Is the extracted value visually correct?
4. If wrong extraction: Re-run extraction for that page
5. If correct: The data is genuinely unusual, keep it

### Duplicates

1. Find duplicate rows in all.csv
2. Check if same result appears twice on same page (extraction issue)
3. Check if same test was run multiple times (valid)
4. Remove actual duplicates from extraction JSON if needed

### Unknowns ($UNKNOWN$ values)

1. Read the source image to see exact spelling/formatting
2. Check `config/lab_specs.json` for similar standardized names
3. Either:
   - Add mapping to `normalization.py` if it's a known lab with different spelling
   - Add new lab to `config/lab_specs.json` if it's a genuinely new test

### Percentage Out of Range

1. Verify the value in source image
2. Check if unit is actually "%" or something else
3. Ensure lab name ends with "(%) " if unit is percentage

### Unit Inconsistencies

1. Find all rows for the lab name
2. Check if units differ across sources (legitimate variation)
3. Verify conversion factors are correct in lab_specs.json

## Fix Actions

| Determination | Action |
|---------------|--------|
| Extraction error on single page | Delete the JSON, re-run `python extract.py --profile X --document STEM` |
| Systematic extraction error | Review extraction prompts, re-run affected documents |
| Real unusual data | Keep it, optionally add note |
| Unknown lab name | Add mapping to `normalization.py` KNOWN_LAB_NAMES dict |
| Unknown unit | Add conversion to `config/lab_specs.json` alternatives |
| Missing lab spec | Add new entry to `config/lab_specs.json` |

## Re-running Extraction

To re-extract specific pages:

```bash
# Delete the page's JSON file first (to force re-extraction)
rm "{OUTPUT_PATH}/{doc_stem}/{doc_stem}.{page:03d}.json"

# Re-run extraction for that document
python extract.py --profile {profile} --document {doc_stem}
```

To re-extract with verification:
```bash
python extract.py --profile {profile} --document {doc_stem} --verify
```

## Example Investigation Session

User: "Check the outliers in my data"

1. Run `python test.py`
2. Find outliers section in output
3. For each outlier:
   - Show: lab_name, value, source_file, page_number
   - Read the source image
   - Report: "Value X for {lab} on {source} page {page} is {correct/incorrect} - source shows {actual value}"
4. Recommend fixes for incorrect extractions

## Key Files Reference

| File | Purpose |
|------|---------|
| `test.py` | Data validation suite |
| `utils/analyze_unknowns.py` | Unknown value analyzer |
| `config/lab_specs.json` | Standardized lab definitions |
| `normalization.py` | Name/unit mapping logic |
| `all.csv` | Combined extraction results |
