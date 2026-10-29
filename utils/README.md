# Utility Scripts

`parselabs admin` is the preferred entry point for maintenance and migration commands. Installable implementations live in `parselabs/admin/`; the legacy `utils/*.py` paths are thin compatibility entry points.

## Lab Specifications Manager

### `parselabs admin lab-specs`
Consolidated utility for managing lab_specs.json operations.

The mutating `sort` and `fix-encoding` commands require an editable source checkout so installed package resources remain immutable.

**Commands:**

| Command | Description |
|---------|-------------|
| `sort` | Sort lab_specs.json alphabetically by lab name |
| `fix-encoding` | Convert Unicode escape sequences to UTF-8 characters |
| `build-conversions` | Generate unit conversion factors from extracted CSV data |
| `build-ranges` | Generate evidence-based optimal ranges using LLM |

Usage:
```bash
# Sort lab specifications alphabetically
parselabs admin lab-specs sort

# Fix encoding issues (creates backup by default)
parselabs admin lab-specs fix-encoding

# Build conversion factors from extracted data
parselabs admin lab-specs build-conversions --input output/all.csv

# Build evidence-based optimal ranges using LLM
parselabs admin lab-specs build-ranges --user-stats user_stats.json
```

Options:
- `--workers, -w`: Number of parallel workers (default: auto)
- `--input, -i`: Input CSV file for build-conversions
- `--output, -o`: Output JSON file
- `--user-stats, -u`: User stats JSON file for build-ranges
- `--no-backup`: Skip backup creation for fix-encoding

## Validation

### `parselabs admin validate-lab-specs`
Comprehensive schema validator for lab_specs.json.

Validates:
- JSON structure and syntax
- Required fields (lab_type, primary_unit, loinc_code)
- Data types and value ranges
- LOINC code presence (with known exceptions)
- Relationship configurations
- Lab name prefixes match lab_type
- Unit conversion factors
- Reference range consistency
- Biological limits

Usage:
```bash
# Run standalone
parselabs admin validate-lab-specs

# Or as part of test suite
python test.py
```

Exit codes:
- 0: Validation passed
- 1: Validation failed (errors found)

## Analysis

### `parselabs admin analyze-unknowns`
Analyze $UNKNOWN$ values in extracted results to identify patterns and missing lab mappings.

Usage:
```bash
parselabs admin analyze-unknowns
```

## Review Artifacts

### `parselabs admin review-artifacts`
Return one pending stored extraction row plus deterministic visual artifacts for native row-by-row auditing without the Gradio reviewer.

Usage:
```bash
# Fetch the next pending row, page image path, and bbox crop
parselabs admin review-artifacts next --profile cristina

# Restrict the queue to rows flagged by validation
parselabs admin review-artifacts next --profile cristina --review-needed-only

# Persist a review decision using the returned row_id
parselabs admin review-artifacts decide --profile cristina --row-id <ROW_ID> --decision accept
parselabs admin review-artifacts decide --profile cristina --row-id <ROW_ID> --decision reject
```

Output:
- `next` prints one JSON payload with `row_id`, `stored_result`, `page_image_path`, `bbox_clip_path`, and bbox metadata.
- `decide` prints one JSON payload confirming whether the persisted action succeeded.

## Local Medical Golden Corpus

`parselabs admin regression` promotes only documents whose rows are fully reviewed, whose missing-row markers are resolved, and whose separate source/export audit is current. Previously accepted rows still need this audit. Keep ambiguous or incorrectly normalized documents outside the approved corpus; record their review findings locally.

The default corpus is `~/.local/share/parselabs/approved/<profile>/<case-id>/`. An optional `PARSELABS_APPROVED_FIXTURES_DIR` overrides the root, which must remain outside every Git checkout, including ignored folders and symlinks into a checkout. Choose a local directory outside cloud synchronization. Corpus directories are private and artifact files are owner-only. No real medical artifacts or corpus diagnostics belong in Git or CI uploads.

Each case contains `document.pdf`, `expected.csv`, `case.json`, `review_state.json`, `source_review.json`, and frozen page JSON under `reviewed/`. The source-review record binds all page JSON and the original PDF with SHA-256, records the completed audit, and fingerprints the canonical normalized export. Changes to source data, decisions, dates, or normalized output require another review before promotion. The attestation records the caller's source review; it is not an automated substitute for examining the document.

```bash
# After visually auditing every page and row, verifying omissions and normalized output:
parselabs admin regression attest-reviewed --profile myname \
  --source-file "document.pdf" --review-note "Evidence and outcome of the completed source and export audit"
parselabs admin regression sync-reviewed --profile myname --source-file "document.pdf"

# Local normalization/export regression; no API calls and no profile credentials needed:
RUN_LOCAL_GOLDENS=1 uv run pytest -m golden_local

# Recommended after significant parser/prompt/model/normalization/export changes:
make regression-quick

# Faster deterministic replay of the same curated sample, without model calls:
make regression-quick-offline

# Separately opt into fresh vision extraction using the configured profile/API:
RUN_APPROVED_DOCS=1 uv run pytest -m approved_docs
```

Repeat `--source-file` to promote a curated selection. Omitting it selects every currently audited, fixture-ready document for that profile. Existing cases that are absent or no longer eligible in live outputs remain frozen; they are never silently deleted. Use `--replace` only after deliberately reviewing a changed baseline. Both private tests are opt-in and fail on an empty corpus when enabled. Offline replay checks corpus fingerprints before comparing current normalization/export against frozen expected CSVs; extraction changes require the separate re-extraction test.

Different PDFs may share a filename. Select these with `--document-key "document_01234567"`, the exact hashed processed-directory name, instead of `--source-file` in either review command. Repeat `--document-key` for a curated promotion. Filename selection fails when ambiguous; the corpus preserves separate profile/stem/hash identities and the extraction test stages their PDFs in separate input directories.

The quick commands add `--golden-suite=quick` to the relevant private pytest mode. The default `full` suite continues to select every approved case. The private manifest lives at the approved corpus root's sibling `quick-suite.json` (normally `~/.local/share/parselabs/quick-suite.json`). It contains version `1`, `required_profiles`, positive integer `max_documents` and `max_pages` budgets, and a `cases` list. Each entry specifies `profile`, `case_id`, the full `pdf_sha256`, coverage `tags`, and a curation `rationale`. Keep these patient-specific selectors outside Git; changing a baseline never silently changes the sample. Curate whole documents to preserve continuation dates and cross-page behavior, balancing different people, layout families, scanned and digital sources, historical columns, qualitative/interval results and unit scales. Source reviews are checked before fresh extraction; gateway authentication is optional when the configured AgentBridge endpoint permits it.

Fresh extraction uses up to two workers for the quick sample. Runtime depends on the configured model and gateway. Frozen review decisions transfer only after the same multiset of observations matches on every page; reordering and equivalent numeric/raw formatting are allowed, but new, missing, changed or ambiguously reviewed observations fail. Qualitative sentence-ending periods are ignored. Normalized exports are then compared for dates, assay identity, numeric/qualitative results, units, reference bounds and limit flags. The offline mode retains the stricter full export comparison.

Fresh outputs and logs are retained with owner-only access in the corpus's sibling `regression-runs/` directory. A completed comparison writes `latest-result.json` there, including mismatches and private artifact paths; failed extraction also leaves diagnostics. Keep these files private and delete old run directories locally when no longer needed. No baseline is replaced automatically. Ordinary pytest skips both private modes, and explicit private execution in CI fails before accessing the corpus. No scheduler or Git hook runs these commands automatically; repository guidance calls for the quick test after significant parser changes.

Private replay and extraction refuse CI execution before reading the corpus, even when their opt-in environment flags are set. Ordinary tests and CI skip them. Run the quick extraction locally after significant extraction, prompt, model-routing, normalization, mapping or export changes; use offline replay for a deterministic saved-parse check. No automatic schedule or Git hook is installed. Failures require investigation; do not regenerate expected CSVs to make a run pass.

## Migration

### `parselabs admin migrate-output-dirs`
Batch-rename legacy output directories to include the file hash suffix (`{stem}/` → `{stem}_{hash}/`).

Usage:
```bash
# Preview changes without renaming
parselabs admin migrate-output-dirs --dry-run

# Migrate a single profile
parselabs admin migrate-output-dirs --profile tsilva

# Migrate all profiles
parselabs admin migrate-output-dirs
```

### `parselabs admin migrate-raw-columns`
Rename `_raw` suffix columns to `raw_` prefix in JSON, per-document CSV, and all.csv files.

Usage:
```bash
parselabs admin migrate-raw-columns --profile tsilva
parselabs admin migrate-raw-columns --profile tsilva --dry-run
parselabs admin migrate-raw-columns
```

## Legacy Scripts (Consolidated)

The following scripts have been consolidated into `lab_specs_manager.py`:
- `build_lab_specs_conversions.py` → Use `lab_specs_manager.py build-conversions`
- `build_lab_specs_ranges.py` → Use `lab_specs_manager.py build-ranges`
- `sort_lab_specs.py` → Use `lab_specs_manager.py sort`
- `fix_lab_specs_encoding.py` → Use `lab_specs_manager.py fix-encoding`

## License

[MIT License](../LICENSE).
