<div align="center">
  <img src="logo.png" alt="parselabs" width="512"/>

  # parselabs

  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-≥3.8-3776ab.svg)](https://python.org)

  **🔬 Extract lab results from medical PDFs with hybrid text/vision extraction and reviewed JSON fixtures 📊**

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
parselabs-review-docs --profile myname

# Rebuild final outputs from reviewed JSON only
parselabs --profile myname --rebuild-from-json
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
- **Keyboard shortcuts** — Y=Approve, N=Reject, M=Missing Row, Arrow keys/J/K=Navigate
- **Fixture-readiness filter** — Focus on documents that still have pending rows or unresolved missing-row markers
- **Progress tracking** — Document/page counters for reviewed, rejected, pending, and missing rows

### Validate Data Integrity

```bash
uv run python test.py --profile myname
```

Omit `--profile` to validate all configured profiles. The script checks for duplicate rows, missing dates, outliers, and naming conventions.

### Approved Document Regression

Use the private approved-document suite to rerun real PDFs and compare the final CSV output after normalization, deduplication, and validation.

Workflow:

1. Review processed documents line by line until every extracted row you want to keep is marked `accepted`:

```bash
parselabs-review-docs --profile myname
```

2. Manually repair any rejected or missing rows directly in the page JSON files, then rebuild the final outputs from reviewed JSON:

```bash
parselabs --profile myname --rebuild-from-json
```

3. Sync all fixture-ready processed documents into the private approved fixture corpus:

```bash
uv run python utils/regression_cases.py sync-reviewed --profile myname
```
4. Run the approved-document regression suite:

```bash
RUN_APPROVED_DOCS=1 uv run pytest -m approved_docs
```

Notes:
- `parselabs --rebuild-from-json` rebuilds each per-document review CSV from page JSON, then rewrites the merged `all.csv` / `all.xlsx` snapshot from those document CSVs.
- The accepted-only reviewed export is still available through the internal reviewed-JSON helpers used by fixture sync and approved-document regression tests.
- The sync command scans the profile `output_path` and copies only fixture-ready documents: every extracted row reviewed and no unresolved `review_missing_rows` markers.
- Expected fixture CSVs are rebuilt from reviewed JSON truth, not from a fresh extraction run.
- The pytest command reruns the full approved corpus together, then compares each document's final CSV output against its approved `expected.csv`.
- Each approved case uses the runtime settings from its recorded profile file.
- Approved fixtures live under `tests/fixtures/approved/` and remain uncommitted/private.
- Each case directory contains `document.pdf`, `expected.csv`, and `case.json`.
- `sync-reviewed` also removes stale fixture cases from the same profile when the processed document is no longer fixture-ready.
- `uv run python utils/regression_cases.py report --profile myname` prints rejected-row, missing-row, unknown-mapping, and validation-reason counts for the reviewed corpus.

## Output

For each PDF, the tool generates:

| File | Description |
|------|-------------|
| `{doc}/` | Directory with page images and JSON extractions |
| `{doc}.csv` | Per-document review CSV with page/result ids, mapped values, validation flags, and review status |
| `all.csv` | Merge of all per-document review CSVs, including pending/rejected rows |
| `all.xlsx` | Excel workbook for the merged review dataset |
| `{stem}.{page}.json` | Page-level extraction JSON plus review fields (`review_status`, `review_completed_at`, `review_missing_rows`) |

### Output Schema

`all.csv` uses the same review-oriented schema as each per-document `{doc}.csv`, including mapped values, validation flags, and review status columns such as `review_status` and `review_completed_at`.
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
