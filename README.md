<div align="center">
  <img src="https://raw.githubusercontent.com/tsilva/parselabs/main/logo.png" alt="parselabs" width="512"/>

  # parselabs

  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-%3E%3D3.10-3776ab.svg)](https://python.org)

  **🔬 Extract lab results from medical PDFs with vision extraction and reviewed JSON fixtures 📊**

  [Documentation](docs/pipeline.md) · [Issues](https://github.com/tsilva/parselabs/issues)
</div>

---

## Overview

[![CI](https://github.com/tsilva/parselabs/actions/workflows/release.yml/badge.svg)](https://github.com/tsilva/parselabs/actions/workflows/release.yml)

parselabs uses AI vision models to extract laboratory test results from PDF documents and images, converting unstructured medical reports into clean, standardized CSV/Excel data. It automatically normalizes test names, converts units, and validates results for accuracy.

## Features

- **AI-Powered Extraction** — Vision models extract lab names, values, units, and reference ranges directly from PDF pages
- **Smart Validation** — Detects extraction errors across 5 categories: biological plausibility, inter-lab relationships, temporal consistency, format artifacts, and reference range deviations
- **BBox-Backed Results** — The pipeline stays vision-only so extracted rows can retain source-page bounding boxes for review
- **Profile-Based Workflow** — Configure multiple profiles for different users or data sources with simple YAML files in `~/.config/parselabs/profiles/`
- **Gradio Review UI** — Side-by-side comparison of source documents and extracted data with keyboard shortcuts
- **335+ Standardized Labs** — Comprehensive lab specifications with unit conversions and configured optimal ranges

## Quick Start

```bash
# Install the tool
uv tool install . --editable

# Create your profile directory
mkdir -p ~/.config/parselabs/profiles

# Create ~/.config/parselabs/.env with shared runtime settings
# Example:
# OPENROUTER_API_KEY="your_key_here"
# OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
# EXTRACT_MODEL_ID="google/gemini-3-flash-preview"
#
# Create ~/.config/parselabs/profiles/myname.yaml with your paths
# Example:
# name: "My Labs"
# paths:
#   input_path: "/path/to/lab/pdfs"
#   output_path: "/path/to/output"

# Extract lab results
parselabs --profile myname

# Review results (same combined app, different default tabs)
parselabs review --profile myname
parselabs review --profile myname --tab review

# Admin / maintenance
parselabs admin --help

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

Profiles define paths and local processing settings. Put shared runtime settings in `~/.config/parselabs/.env`:

```bash
mkdir -p ~/.config/parselabs
cat > ~/.config/parselabs/.env <<'EOF'
OPENROUTER_API_KEY=your_key_here
# Optional:
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
EXTRACT_MODEL_ID=google/gemini-3-flash-preview
EOF
```

Values in `~/.config/parselabs/.env` supply the default runtime settings. Shell environment variables still take highest precedence.

Store profiles under `~/.config/parselabs/profiles/`, one file per user or data source:

```yaml
# ~/.config/parselabs/profiles/john.yaml
name: "John Doe"
paths:
  input_path: "/path/to/lab/pdfs"
  output_path: "/path/to/output"
  input_file_regex: "*.pdf"  # Optional filter

processing:
  workers: 4

# Optional demographics for personalized optimal ranges
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
parselabs review --profile myname
parselabs review --list-profiles
parselabs-review-mcp
```

`parselabs review` requires an explicit `--profile` and launches the combined Gradio app for that profile only. Use the default Results Explorer tab for browsing results, or pass `--tab review` to open the Review Queue directly.

`parselabs-review-mcp` runs a stdio MCP server for deterministic row-by-row review without the browser UI. The server exposes:
- `next_pending_row` — returns the next unresolved row plus the full page image and deterministic bbox crop as MCP image content
- `decide_row` — persists `accept`, `reject`, or `clear` for a previously returned `row_id`

The combined review UI provides:
- **Results Explorer** — Filtered table, summary cards, plots, and source-page inspection
- **3-pane review workspace** — Pending-row queue, large source page, and compact row inspector
- **Keyboard shortcuts** — Y=Approve, N=Reject, M=Missing Row, U=Undo, Arrow keys/J/K=Navigate
- **Show reviewed toggle** — Hide reviewed rows by default and reveal them only when needed
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
parselabs review --profile myname --tab review
```

2. Manually repair any rejected or missing rows directly in the page JSON files, then rebuild the final outputs from reviewed JSON:

```bash
parselabs --profile myname --rebuild-from-json
```

3. Sync all fixture-ready processed documents into the private approved fixture corpus:

```bash
parselabs admin regression sync-reviewed --profile myname
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
- Each approved case uses its recorded profile name plus the current shared runtime settings from `~/.config/parselabs/.env` or the shell environment.
- Approved fixtures live under `tests/fixtures/approved/` and remain uncommitted/private.
- Each case directory contains `document.pdf`, `expected.csv`, `review_state.json`, and `case.json`.
- `sync-reviewed` also removes stale fixture cases from the same profile when the processed document is no longer fixture-ready.
- `parselabs admin regression report --profile myname` prints rejected-row, missing-row, unknown-mapping, and validation-reason counts for the reviewed corpus.

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

1. **PDF Processing** — Page-to-image conversion and preprocessing
2. **Extraction** — Vision LLM extracts structured `LabResult` objects with bounding boxes when available
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
