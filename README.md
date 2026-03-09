<div align="center">
  <img src="logo.png" alt="parselabs" width="512"/>

  # parselabs

  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-≥3.8-3776ab.svg)](https://python.org)

  **🔬 Extract lab results from medical PDFs using AI vision with self-consistency 📊**

  [Documentation](docs/pipeline.md) · [Issues](https://github.com/tsilva/parselabs/issues)
</div>

---

## Overview

[![CI](https://github.com/tsilva/parselabs/actions/workflows/release.yml/badge.svg)](https://github.com/tsilva/parselabs/actions/workflows/release.yml)

parselabs uses AI vision models to extract laboratory test results from PDF documents and images, converting unstructured medical reports into clean, standardized CSV/Excel data. It automatically normalizes test names, converts units, and validates results for accuracy.

## Features

- **AI-Powered Extraction** — Vision models extract lab names, values, units, and reference ranges directly from PDF pages
- **Smart Validation** — Detects extraction errors across 5 categories: biological plausibility, inter-lab relationships, temporal consistency, format artifacts, and reference range deviations
- **Cost-Optimized** — Text-first extraction uses cheaper LLM calls when PDF text is parseable, falling back to vision only when needed
- **Profile-Based Workflow** — Configure multiple profiles for different users or data sources with simple YAML files in `~/.config/parselabs/profiles/`
- **Gradio Review UI** — Side-by-side comparison of source documents and extracted data with keyboard shortcuts
- **335+ Standardized Labs** — Comprehensive lab specifications with unit conversions and reference ranges

## Quick Start

```bash
# Install the tool
uv tool install . --editable

# Create your profile directory
mkdir -p ~/.config/parselabs/profiles

# Create ~/.config/parselabs/profiles/myname.yaml with your paths and runtime settings
# Example:
# name: "My Labs"
# paths:
#   input_path: "/path/to/lab/pdfs"
#   output_path: "/path/to/output"
# openrouter:
#   api_key: "your_key_here"
# models:
#   extract_model_id: "google/gemini-3-flash-preview"

# Extract lab results
parselabs --profile myname

# Review results
parselabs-viewer --profile myname
```

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [Poppler](https://poppler.freedesktop.org/) for PDF processing

### Setup

```bash
git clone https://github.com/tsilva/parselabs.git
cd parselabs
uv tool install . --editable
```

If `parselabs` is not found after installation, add `~/.local/bin` to your `PATH`.

### macOS (Poppler)

```bash
brew install poppler
```

## Configuration

### Profiles

Profiles define both paths and runtime settings. Store them under `~/.config/parselabs/profiles/`, one file per user or data source:

```yaml
# ~/.config/parselabs/profiles/john.yaml
name: "John Doe"
paths:
  input_path: "/path/to/lab/pdfs"
  output_path: "/path/to/output"
  input_file_regex: "*.pdf"  # Optional filter

openrouter:
  api_key: "your_key_here"
  base_url: "https://openrouter.ai/api/v1"  # Optional

models:
  extract_model_id: "google/gemini-3-flash-preview"

processing:
  workers: 4

# Optional demographics for personalized ranges
demographics:
  gender: "male"
  date_of_birth: "1990-01-15"
```

List available profiles:

```bash
parselabs --list-profiles
```

### Lab Specifications

The `config/lab_specs.json` file contains 335+ standardized lab tests with:
- Primary units and conversion factors
- Reference ranges
- Biological limits for validation
- Inter-lab relationships (e.g., LDL Friedewald formula)

## Usage

### Extract Lab Results

```bash
# Run all profiles (default)
parselabs

# Run specific profile
parselabs --profile myname

# Override model
parselabs --profile myname --model google/gemini-2.5-pro

# Filter files
parselabs --profile myname --pattern "2024-*.pdf"
```

### Review Extracted Data

```bash
parselabs-viewer --profile myname
```

The Gradio-based review UI provides:
- **Side-by-side view** — Source document image alongside extracted data
- **Keyboard shortcuts** — Y=Accept, N=Reject, S=Skip, Arrow keys=Navigate
- **Smart filters** — Unreviewed, Low Confidence, Needs Review, Accepted, Rejected
- **Progress tracking** — Review counts and completion status

### Validate Data Integrity

```bash
uv run python test.py --profile myname
```

Omit `--profile` to validate all configured profiles. The script checks for duplicate rows, missing dates, outliers, and naming conventions.

### Approved Document Regression

Use the private approved-document suite to rerun real PDFs and compare the final CSV output after normalization, deduplication, and validation.

Workflow:

1. Approve or refresh one or more real PDFs from an existing profile.

By pattern:

```bash
uv run python utils/regression_cases.py approve --profile myname --pattern "2024-*.pdf"
```

By explicit filenames:

```bash
uv run python utils/regression_cases.py approve --profile myname --files "2024-01-15 analises.pdf" "2024-03-20 analises.pdf"
```

2. Run the approved-document regression suite:

```bash
RUN_APPROVED_DOCS=1 uv run pytest -m approved_docs
```

Notes:
- The approval command copies the selected PDFs into `tests/fixtures/approved/` and rebuilds `expected.csv` for the full approved corpus.
- The pytest command reruns the full approved corpus together, then compares each document's final CSV output against its approved `expected.csv`.
- Each approved case uses the runtime settings from its recorded profile file.
- Approved fixtures live under `tests/fixtures/approved/` and remain uncommitted/private.
- Each case directory contains `document.pdf`, `expected.csv`, and `case.json`.
- The `approve` command rebuilds baselines for the full approved corpus, not just the newly selected files.

## Output

For each PDF, the tool generates:

| File | Description |
|------|-------------|
| `{doc}/` | Directory with page images and JSON extractions |
| `{doc}.csv` | Combined results for the document |
| `all.csv` | Merged results from all documents |
| `all.xlsx` | Excel workbook with formatted data |

### Output Schema

| Column | Description |
|--------|-------------|
| `date` | Report/collection date |
| `lab_name` | Standardized name (e.g., "Blood - Glucose") |
| `value` | Numeric value in primary unit |
| `unit` | Primary unit (e.g., "mg/dL") |
| `reference_min/max` | Reference range from report |
| `raw_lab_name`, `raw_value`, `raw_unit` | Original values for audit |
| `review_needed` | Boolean flag for items needing review |
| `review_reason` | Validation reason codes |

## Architecture

The extraction pipeline has 5 stages:

1. **PDF Processing** — Text extraction or page-to-image conversion
2. **Extraction** — Vision/text LLM extracts structured `LabResult` objects
3. **Standardization** — Maps to standardized names and units
4. **Normalization** — Converts values to primary units
5. **Validation** — Flags suspicious values for review

For detailed documentation, see [docs/pipeline.md](docs/pipeline.md).

## Validation Categories

| Category | Reason Codes | Description |
|----------|--------------|-------------|
| Biological Plausibility | `NEGATIVE_VALUE`, `IMPOSSIBLE_VALUE`, `PERCENTAGE_BOUNDS` | Values outside biological limits |
| Inter-Lab Relationships | `RELATIONSHIP_MISMATCH` | Calculated values don't match formulas |
| Temporal Consistency | `TEMPORAL_ANOMALY` | Implausible change rate between tests |
| Format Artifacts | `FORMAT_ARTIFACT` | OCR/extraction concatenation errors |
| Reference Ranges | `RANGE_INCONSISTENCY`, `EXTREME_DEVIATION` | Reference range issues |

## License

[MIT](LICENSE)
