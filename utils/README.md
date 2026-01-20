# Utility Scripts

Helper scripts for building and maintaining the labs-parser configuration and validation.

## Configuration Building

### build_lab_specs_conversions.py
Generate unit conversion factors for lab_specs.json.

Usage:
```bash
python utils/build_lab_specs_conversions.py
```

### build_lab_specs_ranges.py
Generate healthy ranges for lab_specs.json.

Usage:
```bash
python utils/build_lab_specs_ranges.py
```

### sort_lab_specs.py
Sort lab specifications alphabetically in lab_specs.json.

Usage:
```bash
python utils/sort_lab_specs.py
```

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
python utils/validate_lab_specs_schema.py

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
python utils/analyze_unknowns.py
```

### update_loinc_code.py
Update LOINC codes for specific labs in lab_specs.json.

Usage:
```bash
python utils/update_loinc_code.py
```

### verify_loinc_codes.py
Verify LOINC codes are valid and correctly assigned.

Usage:
```bash
python utils/verify_loinc_codes.py
```
