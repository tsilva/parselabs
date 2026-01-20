---
name: labs
description: Comprehensive diagnostics for labs-parser with reactive investigation and proactive accuracy verification. Use when "investigating extraction issues", "validating data integrity", "analyzing unknowns", "checking statistics", "verifying extraction accuracy", "generating accuracy reports", or "checking data quality". Performs issue investigation, systematic batch verification with intelligent sampling, visual verification against source images, statistical analysis, and generates comprehensive reports.
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
| "verify accuracy", "accuracy report", "quality check", "batch verify" | Systematic accuracy verification with intelligent sampling and vision verification |

## Skill Scripts

This skill includes helper scripts in `.claude/skills/labs/`:

| Script | Purpose | Usage |
|--------|---------|-------|
| `create_sample.py` | Create intelligent sample for verification | `python .claude/skills/labs/create_sample.py <profile> [--sample-size N]` |
| `analyze_stats.py` | Generate statistical analysis report | `python .claude/skills/labs/analyze_stats.py <profile> [--output file.md]` |
| `verify_sample.py` | Track verification workflow | See verification workflow below |

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

### 5. Systematic Accuracy Verification Workflow

When asked to verify accuracy, generate quality reports, or perform batch validation:

**Purpose:** Proactive quality assurance through intelligent sampling, vision verification, and comprehensive reporting.

#### Step 1: Create Intelligent Sample

Use the `create_sample.py` script to generate an intelligent sample for verification:

```bash
cd /Users/tsilva/repos/tsilva/labs-parser
python .claude/skills/labs/create_sample.py <profile_name> --sample-size 50
```

This creates:
- `{OUTPUT_PATH}/sample_for_verification.csv` - Human-readable sample
- `{OUTPUT_PATH}/sample_data.json` - Machine-readable sample data

The script intelligently selects rows based on:

1. **All flagged rows** (review_needed == True) - highest priority
2. **High-value labs** - 2 samples each from: Glucose, Hemoglobin, Cholesterol (Total/LDL/HDL), Creatinine, TSH
3. **Stratified sampling** - 2-3 samples per source file (up to 20 files)
4. **Low confidence** - All rows with confidence < 0.8
5. **Edge cases** - Samples with is_below_limit or is_above_limit flags

#### Step 2: Initialize Verification Tracking

```bash
python .claude/skills/labs/verify_sample.py <profile_name> --init
```

This creates `{OUTPUT_PATH}/verification_results.json` to track verification progress.

#### Step 3: Vision Verification Protocol

Load the sample and verify each row against source images using Claude's vision capabilities.

**Process for each sampled row:**

1. Load sample data:
```python
import json
profile = load_profile(profile_name)
OUTPUT_PATH = profile['output_path']

with open(f"{OUTPUT_PATH}/sample_data.json") as f:
    sample_data = json.load(f)
```

2. For each sample, construct image path and verify:
```python
for sample in sample_data['samples']:
    source_file = sample['source_file']
    page_number = sample['page_number']
    lab_name_raw = sample['lab_name_raw']
    extracted_value = sample['value']
    extracted_unit = sample['unit']

    # Construct image path (zero-padded to 3 digits)
    image_path = f"{OUTPUT_PATH}/{source_file}/{source_file}.{page_number:03d}.jpg"

    # Use Read tool to view the source image
    # Visually locate the lab test and compare against extracted value
```

3. Record the result:
```bash
python .claude/skills/labs/verify_sample.py <profile_name> --record <index> \
  --status <status> \
  --notes "..." \
  [--actual-value "X.X"] \
  [--actual-unit "unit"]
```

**Discrepancy Categories:**

| Category | Description | Example |
|----------|-------------|---------|
| **Match** | Correct extraction | Source shows "14.2", extracted "14.2" |
| **Wrong Digit** | Single digit error | Source shows "14.2", extracted "14.7" |
| **Decimal Error** | Decimal place misplacement | Source shows "5.2", extracted "52" or "0.52" |
| **Unit Mismatch** | Correct number, wrong unit | Source shows "5.2 mmol/L", extracted "5.2 mg/dL" |
| **Missing Value** | Value exists but not extracted | Source shows value, extracted as blank or $UNKNOWN$ |
| **Hallucination** | Extracted value doesn't exist | Source shows "14.2", extracted "142" (not on page) |
| **Ambiguous** | Source unclear/illegible | Cannot determine correct value from image |

**Verification Questions:**
- "What is the value shown for {lab_name_raw} on this lab report?"
- "What unit is shown for this value?"
- "Is the value clearly legible?"

#### Step 4: Statistical Analysis

Run the statistical analysis script to analyze the full dataset:

```bash
python .claude/skills/labs/analyze_stats.py <profile_name>
```

This generates `{OUTPUT_PATH}/stats_report.md` with:
- Distribution analysis per lab (mean, median, std, outliers)
- Precision patterns (detecting suspicious decimal patterns)
- Confidence score distribution
- Per-lab quality metrics
- Per-document quality metrics

#### Step 5: Generate Comprehensive Report

After completing verification, generate the final accuracy report:

```bash
python .claude/skills/labs/verify_sample.py <profile_name> --report
```

This creates `{OUTPUT_PATH}/accuracy_report.md` with the following structure:

```markdown
# Lab Extraction Accuracy Report
Generated: {timestamp}
Profile: {profile_name}
Dataset: {total_rows} rows, {verified_rows} verified

## Executive Summary
- Overall Accuracy: XX.X% ({matches}/{verified} verified rows)
- Critical Issues: {count} requiring immediate action
- Confidence Calibration: {assessment}
- Top Recommendation: {priority_action}

## 1. Verification Results

### Verified Rows: {count}
- ✓ Matches: {count} (XX.X%)
- ✗ Discrepancies: {count} (XX.X%)

### Discrepancies by Type
| Type | Count | % | Example |
|------|-------|---|---------|
| Wrong Digit | X | XX% | Row 123: Expected 14.2, Got 14.7 (source_file.pdf page 2) |
| Decimal Error | X | XX% | Row 456: Expected 5.2, Got 52 (source_file.pdf page 5) |
| Unit Mismatch | X | XX% | Row 789: mg/dL vs mmol/L (source_file.pdf page 1) |
| Missing Value | X | XX% | Value present but not extracted |
| Hallucination | X | XX% | Extracted value not on page |
| Ambiguous | X | XX% | Source illegible/unclear |

### Critical Errors (High Impact)
[List errors in high-value labs with specific row references]

Example:
- **Row 234 - Blood - Glucose**: Extracted 142 mg/dL, source shows 14.2 mg/dL (decimal error)
  - Source: 2024-01-15-labs.pdf, page 3
  - Impact: Critical - glucose values used for diabetes management
  - Action: Delete JSON and re-extract

## 2. Statistical Analysis

### Distribution Analysis
[For each lab with n >= 5, show key statistics]

Example:
**Blood - Glucose** (n=45)
- Mean: 98.2 mg/dL (σ=12.3)
- Median: 96.5 mg/dL
- Range: [72, 145] mg/dL
- Outliers: 3 values >3σ from mean (rows: 123, 456, 789)

### Precision Patterns
[Flag suspicious patterns]

Example:
⚠ **Blood - Creatinine**: 85% of values end in ".0" - possible rounding artifact

### Confidence Calibration
[Error rates by confidence bucket]

| Confidence | Samples | Errors | Error Rate |
|------------|---------|--------|------------|
| High (≥0.9) | 25 | 1 | 4% |
| Medium (0.7-0.9) | 15 | 3 | 20% |
| Low (<0.7) | 10 | 6 | 60% |

Assessment: Confidence scores are well-calibrated / poorly calibrated / [specific finding]

## 3. Quality Metrics

### Per-Lab Accuracy
| Lab Name | Total Rows | Verified | Accuracy | Avg Confidence |
|----------|------------|----------|----------|----------------|
| Blood - Glucose | 45 | 10 | 100% | 0.95 |
| Blood - Hemoglobin (Hgb) | 38 | 8 | 87.5% | 0.88 |
| Blood - Creatinine | 42 | 9 | 100% | 0.92 |

### Per-Document Quality
| Source File | Total Rows | Verified | Accuracy | Avg Confidence |
|-------------|------------|----------|----------|----------------|
| 2024-01-15-labs | 85 | 12 | 91.7% | 0.91 |
| 2024-03-22-labs | 72 | 10 | 100% | 0.94 |

### Temporal Analysis
[If patterns emerge by date]

Example:
📊 Recent extractions (last 3 months) show 95% accuracy vs 88% for older extractions

## 4. Recommendations

### Immediate Actions (Critical)
1. **Re-extract specific files:**
   - 2024-01-15-labs.pdf - 3 critical errors found (decimal errors in glucose)
   - Command: `python extract.py --profile {profile} --document 2024-01-15-labs`

2. **Review specific labs:**
   - Blood - Creatinine - systematic rounding detected
   - Blood - LDL Cholesterol - 2 hallucinations found

### Configuration Updates
1. **Add to lab_specs.json:**
   - {specific missing labs or units found during verification}

2. **Update validation rules:**
   - {specific validation.py improvements suggested}

### Process Improvements
1. {Pattern-based recommendations based on error analysis}
2. {Model/preprocessing guidance if systematic issues found}

Example:
- Consider using higher-resolution page images - 3 ambiguous results due to illegibility
- Review extraction prompt for cholesterol panel - consistent unit confusion
```

#### Workflow Variants

**Full Accuracy Check:**
1. Ask user for profile name
2. Load profile and `all.csv`
3. Intelligent sampling (30-50 rows)
4. Visual verification loop with progress updates
5. Statistical analysis on full dataset
6. Generate comprehensive report

**Targeted Lab Verification:**
1. Ask user for profile and specific lab name
2. Filter to specific lab
3. Select flagged rows + random sample
4. Visual verification
5. Report accuracy with examples

**Document Quality Assessment:**
1. Ask user for profile and specific source_file
2. Filter to document
3. Sample across all pages
4. Verify against source images
5. Calculate document-level metrics and report

#### Progress Updates

Provide progress updates during batch verification:
- "Verifying row 15/50: Blood - Glucose from 2024-01-15-labs page 2..."
- "Found discrepancy: Expected 14.2, source shows 142 (decimal error)"
- "Match: Blood - Hemoglobin 14.5 g/dL verified correct"

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
