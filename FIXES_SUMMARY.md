# Fixes Applied for $UNKNOWN$ Values

## Summary

Performed a deep search through all generated CSV files and fixed the root causes of all `$UNKNOWN$` values found.

## Issues Found

### Before Fixes
- **4 unique unknown lab names** (10 total occurrences)
- **20 unique unknown unit combinations** (50+ total occurrences)
- **Missing lab_type column** in all.csv (normalization wasn't applied to current data)

## Root Causes Identified

### 1. Missing Lab Specifications
The following hepatitis markers were not in `lab_specs.json`:

| Raw Test Name | Standardized Name | Primary Unit |
|--------------|-------------------|--------------|
| IMUNOLOGIA - ANTICORPO ANTI-HAV | Blood - Hepatitis A Antibody Total | unitless |
| IMUNOLOGIA - ANTICORPO ANTI-HAV (IgM) | Blood - Hepatitis A Antibody (IgM) | unitless |
| IMUNOLOGIA - MARCADORES VIRICOS DA HEPATITE B - Anticorpo Anti-HBe | Blood - Hepatitis B e Antibody (Anti-HBe) | unitless |

**Note:** Also previously fixed "Blood - Hepatitis B e Antigen (HBeAg)"

### 2. Missing Unit Alternatives
The following units appeared in raw data but weren't recognized:

| Lab Name | Raw Unit | Should Map To | Conversion Factor |
|----------|----------|---------------|-------------------|
| Blood - Erythrocytes | E6/mm3 | 10¹²/L | 0.001 |
| Blood - Erythrocytes | 10⁶/mm³ | 10¹²/L | 0.001 |
| Blood - Basophils | x10¹/L | 10⁹/L | 1.0 (OCR error) |
| Blood - Eosinophils | x10¹/L | 10⁹/L | 1.0 (OCR error) |
| Blood - Neutrophils | x10¹/L | 10⁹/L | 1.0 (OCR error) |
| Blood - Leukocytes | x10¹/L | 10⁹/L | 1.0 (OCR error) |
| Blood - Lymphocytes | x10¹/L | 10⁹/L | 1.0 (OCR error) |
| Blood - Monocytes | x10¹/L | 10⁹/L | 1.0 (OCR error) |
| Blood - Basophils | /mm3 | 10⁹/L | 0.001 |
| Blood - Eosinophils | /mm3 | 10⁹/L | 0.001 |
| Blood - Neutrophils | /mm3 | 10⁹/L | 0.001 |
| Blood - Leukocytes | /mm3 | 10⁹/L | 0.001 |
| Blood - Lymphocytes | /mm3 | 10⁹/L | 0.001 |
| Blood - Monocytes | /mm3 | 10⁹/L | 0.001 |
| Blood - Platelets | /mm3 | 10⁹/L | 0.001 |
| Blood - Platelets | 10³/mm³ | 10⁹/L | 1.0 |
| Blood - TSH | mUI/L | µIU/mL | 1.0 |
| Blood - TSH | mIU/L | µIU/mL | 1.0 |
| Blood - ESR - 1h | mn | mm/h | 1.0 (typo) |
| Blood - ESR - 2h | mn | mm/h | 1.0 (typo) |

### 3. lab_type Column Missing
The current `all.csv` was generated before proper normalization was applied, so it lacks derived columns like `lab_type`, `lab_name`, `lab_unit`, etc.

## Fixes Applied

### Fix 1: Updated lab_specs.json
Added 4 new lab specifications:
```json
{
  "Blood - Hepatitis A Antibody Total": {...},
  "Blood - Hepatitis A Antibody (IgM)": {...},
  "Blood - Hepatitis B e Antibody (Anti-HBe)": {...},
  "Blood - Hepatitis B e Antigen (HBeAg)": {...}
}
```

### Fix 2: Added Unit Alternatives
Added 26 new unit alternatives to handle:
- OCR errors (x10¹/L → 10⁹/L)
- Format variations (/mm3, cells/mm³, E6/mm3, 10⁶/mm³)
- Case variations (mUI/L, mIU/L)
- Typos (mn → mm)

### Fix 3: Enhanced main.py
Added post-processing logic (lines 243-248) to automatically use primary units from lab_specs when raw unit is null and LLM returns $UNKNOWN$.

## Files Modified

1. **config/lab_specs.json**
   - Added 4 new lab specifications
   - Added 26 new unit alternatives
   - Total labs: 295 → 299

2. **main.py** (lines 243-248)
   - Enhanced unit standardization with fallback to primary units

## Expected Results After Re-running Pipeline

| Metric | Before | After |
|--------|--------|-------|
| Unknown lab names | 4 unique (10 occurrences) | 0 |
| Unknown units | 20 unique (50+ occurrences) | 0 |
| Rows with lab_type | 0 | All rows |

## Next Steps

**IMPORTANT:** The fixes won't take effect until you re-run the extraction pipeline:

```bash
python main.py
```

This will:
1. Re-standardize all existing extracted data using the updated lab_specs.json
2. Apply the enhanced unit standardization logic
3. Add all derived columns (lab_type, lab_name, lab_unit, etc.) via normalization
4. Regenerate all.csv and all.xlsx with correct data

After re-running, verify the fixes:
```bash
python deep_search_unknown.py
```

Expected output: ✅ No unknown lab names or units found!

## Scripts Created

1. **deep_search_unknown.py** - Comprehensive search for $UNKNOWN$ values across all CSVs
2. **analyze_root_causes.py** - Detailed analysis of why values are unknown
3. **fix_all_unknown.py** - Automated fixes to lab_specs.json
4. **verify_fixes.py** - Verification that fixes were applied correctly
5. **search_unknown.py** - Quick search in all.csv
6. **analyze_unknown.py** - Detailed breakdown of all.csv unknowns
