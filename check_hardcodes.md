# Hardcode Analysis Report

## Summary

Found **hardcoded inference rules** in LLM prompts that should be removed.

## Issues Found

### 1. Hardcoded Unit Inference Rules (standardization.py:218-222)

**Location:** `standardization.py`, `standardize_lab_units()` function

**Issue:**
```python
5. For null/missing units, infer from lab name:
   - "Urine Type II - Color", "Urine Type II - Density" → "unitless"
   - "Urine Type II - pH" → "pH"
   - "Urine Type II - Proteins", "Urine Type II - Glucose", "Urine Type II - Blood", etc. → "boolean"
   - "Blood - Red Blood Cell Morphology" (qualitative) → "{unknown}"
```

**Why it's problematic:**
1. ❌ Hardcodes specific lab names in the prompt
2. ❌ Creates manual mapping logic outside of lab_specs.json
3. ❌ Not comprehensive - only covers a few specific tests
4. ❌ Maintenance burden - must update prompt when adding new tests
5. ❌ Inconsistent - some tests get hardcoded rules, others don't

**Current behavior:**
- LLM tries to infer units using these hardcoded rules
- If LLM returns $UNKNOWN$, main.py post-processing uses lab_specs.get_primary_unit()

## Better Approach

Instead of hardcoded rules, pass lab_specs context to the LLM:

1. **Pass primary units mapping to LLM**
   ```python
   # In standardize_lab_units, add to context:
   lab_primary_units = {
       (lab_name, lab_specs.get_primary_unit(lab_name))
       for lab_name in set(lab_name for _, lab_name in unit_contexts)
       if lab_name in lab_specs.specs
   }
   ```

2. **Update prompt to use this context**
   ```
   5. For null/missing units, look up the lab_name in the PRIMARY_UNITS mapping
   6. If lab_name not found or no good match exists, use: "{unknown}"
   ```

3. **Keep post-processing as safety net**
   - Current main.py logic (lines 243-248) already handles this correctly

## Examples in Prompts

The examples in the prompts (lines 164-167, 230-235) use specific lab names but this is **acceptable**:
- ✅ They're just demonstrating the format
- ✅ They don't create inference logic
- ✅ They help the LLM understand the task

## Code Quality Assessment

### ✅ What's Good
- No hardcoded mappings in main pipeline code
- No hardcoded dictionary literals
- All lab names and units come from lab_specs.json
- Post-processing fallback in main.py is excellent

### ⚠️ What Needs Fixing
- Remove hardcoded inference rules from standardization.py prompt (lines 218-222)
- Pass lab_specs primary units as context to LLM
- Make the system fully config-driven

## Recommendation

**Priority: HIGH**

Remove the hardcoded inference rules and make the LLM consult lab_specs.json through the context instead of hardcoded patterns.
