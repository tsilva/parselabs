# Hardcode Removal Summary

## Overview

Removed all hardcoded lab test names and unit inference rules from the codebase. The system is now **fully config-driven** through `lab_specs.json`.

## What Was Changed

### ✅ Before: Hardcoded Inference Rules

**Location:** `standardization.py:218-222`

The LLM prompt contained hardcoded rules:
```python
5. For null/missing units, infer from lab name:
   - "Urine Type II - Color", "Urine Type II - Density" → "unitless"
   - "Urine Type II - pH" → "pH"
   - "Urine Type II - Proteins", "Urine Type II - Glucose", "Urine Type II - Blood", etc. → "boolean"
   - "Blood - Red Blood Cell Morphology" (qualitative) → "{unknown}"
```

**Problems:**
- ❌ Hardcoded specific lab names
- ❌ Manual mapping logic outside of config
- ❌ Not comprehensive - only a few tests
- ❌ Maintenance burden
- ❌ Inconsistent coverage

### ✅ After: Config-Driven with Dynamic Context

**Changes Made:**

1. **Updated `standardize_lab_units()` function signature** (standardization.py:179-185)
   - Added optional `lab_specs_config` parameter
   - Function now accepts LabSpecsConfig instance

2. **Dynamic Primary Units Mapping** (standardization.py:205-228)
   ```python
   # Build primary units mapping for null unit inference
   primary_units_map = {}
   if lab_specs_config and lab_specs_config.exists:
       for _, lab_name in unique_pairs:
           if lab_name and lab_name != UNKNOWN_VALUE:
               primary_unit = lab_specs_config.get_primary_unit(lab_name)
               if primary_unit:
                   primary_units_map[lab_name] = primary_unit

   # Build primary units context for prompt
   primary_units_context = ""
   if primary_units_map:
       primary_units_list = [f'  "{lab}": "{unit}"' for lab, unit in sorted(primary_units_map.items())]
       primary_units_context = f"""
   PRIMARY UNITS MAPPING (use this for null/missing units):
   {{
   {chr(10).join(primary_units_list)}
   }}
   """
   ```

3. **Updated LLM Prompt** (standardization.py:240-241)
   ```python
   5. For null/missing units, look up the lab_name in the PRIMARY UNITS MAPPING (if provided)
   6. If NO good match exists or lab not in mapping, use exactly: "{unknown}"
   ```

4. **Updated Caller in main.py** (main.py:232-238)
   ```python
   unit_mapping = standardize_lab_units(
       unit_contexts,
       config.self_consistency_model_id,
       lab_specs.standardized_units,
       client,
       lab_specs  # <-- Now passes lab_specs
   )
   ```

## How It Works Now

### Three-Layer Approach

1. **LLM with Dynamic Context** (First attempt)
   - LLM receives PRIMARY UNITS MAPPING built from lab_specs.json
   - Only includes labs actually present in the current batch
   - LLM can intelligently map null units using this context

2. **Post-Processing Fallback** (Second attempt - already existed)
   - If LLM returns $UNKNOWN$ for null units, use lab_specs.get_primary_unit()
   - Located in main.py:243-248

3. **Safety Net**
   - If both fail, value remains as $UNKNOWN$ for manual review

### Benefits

✅ **Fully Config-Driven**
- All lab information comes from lab_specs.json
- No hardcoded lab names or units in code

✅ **Dynamic Context**
- LLM gets relevant primary units for current batch
- More efficient than passing all 299 labs every time

✅ **Comprehensive**
- Works for ALL labs in lab_specs.json
- Not limited to a few hardcoded examples

✅ **Maintainable**
- Add new labs to lab_specs.json → automatically works
- No code changes needed for new tests

✅ **Consistent**
- Same logic applies to all lab tests
- No special cases or exceptions

## Verification

### ✅ No Syntax Errors
```bash
python -m py_compile standardization.py  # OK
python -m py_compile main.py             # OK
```

### ✅ No Hardcoded Values Found
Searched all core pipeline files:
- main.py
- standardization.py
- normalization.py
- extraction.py
- config.py

Only hardcoded values found are:
- **Helper scripts** (fix_all_unknown.py, analyze_root_causes.py, etc.) - acceptable
- **Docstring examples** (in prompts) - acceptable for demonstration purposes

### ✅ All Config-Driven
- Lab names: From lab_specs.standardized_names
- Units: From lab_specs.standardized_units
- Primary units: From lab_specs.get_primary_unit()
- Conversions: From lab_specs.get_conversion_factor()
- Ranges: From lab_specs.get_healthy_range()

## Testing

Run the pipeline to test the changes:
```bash
python main.py
```

Expected behavior:
1. ✅ LLM will receive PRIMARY UNITS MAPPING with relevant labs
2. ✅ Null units will be resolved using this mapping
3. ✅ Fallback post-processing will catch any LLM misses
4. ✅ No $UNKNOWN$ values for labs that exist in lab_specs.json

## Files Modified

1. **standardization.py**
   - Updated `standardize_lab_units()` signature (line 179-185)
   - Added primary units mapping logic (line 205-229)
   - Updated LLM prompt to use dynamic context (line 231-252)
   - Removed hardcoded inference rules

2. **main.py**
   - Updated `standardize_lab_units()` call (line 232-238)
   - Now passes `lab_specs` parameter

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Inference rules | Hardcoded in prompt | Dynamic from config |
| Coverage | ~5 specific tests | All 299 tests |
| Maintenance | Update code | Update config |
| Consistency | Special cases | Uniform logic |
| Extensibility | Code change needed | Just add to config |

The system is now **100% config-driven** with no hardcoded lab names or unit mappings in the core pipeline code.
