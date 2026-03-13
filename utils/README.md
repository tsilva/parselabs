# Utility Scripts

`parselabs-admin` is the preferred entry point for maintenance and migration commands. The legacy `utils/*.py` scripts still work as compatibility wrappers, but new documentation should prefer the consolidated admin CLI.

## Lab Specifications Manager

### lab_specs_manager.py
Consolidated utility for managing lab_specs.json operations.

**Commands:**

| Command | Description |
|---------|-------------|
| `sort` | Sort lab_specs.json alphabetically by lab name |
| `fix-encoding` | Convert Unicode escape sequences to UTF-8 characters |
| `build-conversions` | Generate unit conversion factors from extracted CSV data |
| `build-ranges` | Generate healthy reference ranges using LLM |

Usage:
```bash
# Sort lab specifications alphabetically
parselabs-admin lab-specs sort

# Fix encoding issues (creates backup by default)
parselabs-admin lab-specs fix-encoding

# Build conversion factors from extracted data
parselabs-admin lab-specs build-conversions --input output/all.csv

# Build healthy ranges using LLM
parselabs-admin lab-specs build-ranges --user-stats user_stats.json
```

Options:
- `--workers, -w`: Number of parallel workers (default: auto)
- `--input, -i`: Input CSV file for build-conversions
- `--output, -o`: Output JSON file
- `--user-stats, -u`: User stats JSON file for build-ranges
- `--no-backup`: Skip backup creation for fix-encoding

## Validation

### validate_lab_specs_schema.py
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
parselabs-admin validate-lab-specs

# Or as part of test suite
python test.py
```

Exit codes:
- 0: Validation passed
- 1: Validation failed (errors found)

## Analysis

### analyze_unknowns.py
Analyze $UNKNOWN$ values in extracted results to identify patterns and missing lab mappings.

Usage:
```bash
parselabs-admin analyze-unknowns
```

## Migration

### migrate_output_dirs.py
Batch-rename legacy output directories to include the file hash suffix (`{stem}/` → `{stem}_{hash}/`).

Usage:
```bash
# Preview changes without renaming
parselabs-admin migrate-output-dirs --dry-run

# Migrate a single profile
parselabs-admin migrate-output-dirs --profile tsilva

# Migrate all profiles
parselabs-admin migrate-output-dirs
```

### migrate_raw_columns.py
Rename `_raw` suffix columns to `raw_` prefix in JSON, per-document CSV, and all.csv files.

Usage:
```bash
parselabs-admin migrate-raw-columns --profile tsilva
parselabs-admin migrate-raw-columns --profile tsilva --dry-run
parselabs-admin migrate-raw-columns
```

## Legacy Scripts (Consolidated)

The following scripts have been consolidated into `lab_specs_manager.py`:
- `build_lab_specs_conversions.py` → Use `lab_specs_manager.py build-conversions`
- `build_lab_specs_ranges.py` → Use `lab_specs_manager.py build-ranges`
- `sort_lab_specs.py` → Use `lab_specs_manager.py sort`
- `fix_lab_specs_encoding.py` → Use `lab_specs_manager.py fix-encoding`
