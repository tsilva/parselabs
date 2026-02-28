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

parselabs uses AI vision models to extract laboratory test results from PDF documents and images, converting unstructured medical reports into clean, standardized CSV/Excel data. It automatically normalizes test names, converts units, and validates results for accuracy.

## Features

- **AI-Powered Extraction** — Vision models extract lab names, values, units, and reference ranges directly from PDF pages
- **Smart Validation** — Detects extraction errors across 5 categories: biological plausibility, inter-lab relationships, temporal consistency, format artifacts, and reference range deviations
- **Cost-Optimized** — Text-first extraction uses cheaper LLM calls when PDF text is parseable, falling back to vision only when needed
- **Profile-Based Workflow** — Configure multiple profiles for different users or data sources with simple YAML files
- **Gradio Review UI** — Side-by-side comparison of source documents and extracted data with keyboard shortcuts
- **335+ Standardized Labs** — Comprehensive lab specifications with unit conversions and reference ranges

## Quick Start

```bash
# Install dependencies
uv sync

# Create your profile
cp profiles/_template.yaml profiles/myname.yaml
# Edit profiles/myname.yaml with your input/output paths

# Configure environment (copy .env.example and edit)
cp .env.example .env
# Edit .env with your API key and model settings

# Extract lab results
python main.py --profile myname

# Review results
python review.py --profile myname
```

## Installation

### Prerequisites

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager
- [Poppler](https://poppler.freedesktop.org/) for PDF processing

### Setup

```bash
git clone https://github.com/tsilva/parselabs.git
cd parselabs
uv sync
```

### macOS (Poppler)

```bash
brew install poppler
```

### Environment Variables

Create a `.env` file:

```bash
# Required
OPENROUTER_API_KEY=your_key_here
EXTRACT_MODEL_ID=google/gemini-3-flash-preview       # Vision model for extraction
SELF_CONSISTENCY_MODEL_ID=google/gemini-3-flash-preview  # Model for self-consistency

# Optional
N_EXTRACTIONS=1    # Self-consistency extractions
MAX_WORKERS=4      # Parallel workers
```

## Configuration

### Profiles

Profiles define input/output paths and optional settings. Create one per user or data source:

```yaml
# profiles/john.yaml
name: "John Doe"
input_path: "/path/to/lab/pdfs"
output_path: "/path/to/output"
input_file_regex: "*.pdf"  # Optional filter

# Optional demographics for personalized ranges
demographics:
  gender: "male"
  date_of_birth: "1990-01-15"
```

List available profiles:

```bash
python main.py --list-profiles
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
python main.py

# Run specific profile
python main.py --profile myname

# Override model
python main.py --profile myname --model google/gemini-2.5-pro

# Filter files
python main.py --profile myname --pattern "2024-*.pdf"
```

### Review Extracted Data

```bash
python review.py --profile myname
```

The Gradio-based review UI provides:
- **Side-by-side view** — Source document image alongside extracted data
- **Keyboard shortcuts** — Y=Accept, N=Reject, S=Skip, Arrow keys=Navigate
- **Smart filters** — Unreviewed, Low Confidence, Needs Review, Accepted, Rejected
- **Progress tracking** — Review counts and completion status

### Validate Data Integrity

```bash
python test.py
```

Checks for duplicate rows, missing dates, outliers, and naming conventions.

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
