<div align="center">
  <img src="logo.png" alt="labs-parser" width="200"/>

  # labs-parser

  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-≥3.8-3776ab.svg)](https://python.org)

  **Extract structured lab test results from medical documents with AI precision**

  [Documentation](docs/pipeline.md) · [Issues](https://github.com/tsilva/labs-parser/issues)
</div>

---

## Features

- **AI-Powered Extraction** — Vision models extract lab names, values, units, and reference ranges directly from PDF images
- **Smart Validation** — Automatically detects extraction errors across 5 categories: biological plausibility, inter-lab relationships, temporal consistency, format artifacts, and reference range deviations
- **Profile-Based Workflow** — Configure multiple profiles for different users or data sources with simple YAML files
- **Gradio Review UI** — Side-by-side comparison of source documents and extracted data with keyboard shortcuts
- **Time-Series Visualization** — Track lab values over time with auto-generated plots

## Quick Start

```bash
# Install dependencies
uv sync

# Create your profile
cp profiles/_template.yaml profiles/myname.yaml
# Edit profiles/myname.yaml with your input/output paths

# Set your API key
export OPENROUTER_API_KEY=your_key_here

# Extract lab results
python extract.py --profile myname

# Review results in the browser
python review.py --profile myname
```

## Installation

### Prerequisites

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager
- [Poppler](https://poppler.freedesktop.org/) for PDF processing

### Setup

```bash
git clone https://github.com/tsilva/labs-parser.git
cd labs-parser
uv sync
```

### Environment Variables

Create a `.env` file with your API key:

```bash
OPENROUTER_API_KEY=your_key_here

# Optional overrides:
EXTRACT_MODEL_ID=google/gemini-3-flash-preview  # Vision model
N_EXTRACTIONS=1                                  # Self-consistency extractions
MAX_WORKERS=4                                    # Parallel workers
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
python extract.py --list-profiles
```

### Lab Specifications

The `config/lab_specs.json` file contains 335+ standardized lab tests with:
- Primary units and conversion factors
- Reference ranges
- Biological limits for validation

## Usage

### Extract Lab Results

```bash
# Using a profile (recommended)
python extract.py --profile myname

# Override model
python extract.py --profile myname --model google/gemini-2.5-pro
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
| `{doc}/` | Directory containing page images and JSON extractions |
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
| `lab_name_raw`, `value_raw`, `unit_raw` | Original values for audit |
| `review_needed` | Boolean flag for items needing review |
| `review_reason` | Validation reason codes |

## Architecture

The extraction pipeline has 4 stages:

1. **PDF Processing** — Converts pages to preprocessed grayscale images
2. **Extraction** — Vision models extract structured `LabResult` objects
3. **Normalization** — Maps to standardized names/units with conversions
4. **Validation** — Flags suspicious values for review

For detailed documentation, see [docs/pipeline.md](docs/pipeline.md).

## License

[MIT](LICENSE)
