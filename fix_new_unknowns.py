#!/usr/bin/env python3
"""Fix newly discovered $UNKNOWN$ values by adding missing specs and units."""

import json
from pathlib import Path

LAB_SPECS_PATH = Path("config/lab_specs.json")

# Load lab specs
with open(LAB_SPECS_PATH, "r") as f:
    lab_specs = json.load(f)

print("="*80)
print("FIXING NEW $UNKNOWN$ VALUES")
print("="*80)

# =============================================================================
# CATEGORY 1: Add missing lab specifications
# =============================================================================
print("\n" + "="*80)
print("CATEGORY 1: ADDING MISSING LAB SPECS")
print("="*80)

new_labs = {
    # Enzyme assays
    "Blood - Glucose-6-Phosphate Dehydrogenase (G6PD)": {
        "lab_type": "blood",
        "primary_unit": "IU/g Hb",
        "alternatives": [
            {"unit": "UI/g Hb", "factor": 1.0},
            {"unit": "U/g Hb", "factor": 1.0}
        ],
        "ranges": {}
    },
    "Blood - Pyruvate Kinase": {
        "lab_type": "blood",
        "primary_unit": "IU/g Hb",
        "alternatives": [
            {"unit": "UI/g Hb", "factor": 1.0},
            {"unit": "U/g Hb", "factor": 1.0}
        ],
        "ranges": {}
    },

    # Serological tests
    "Blood - Anti-Cytomegalovirus IgG": {
        "lab_type": "blood",
        "primary_unit": "IU/mL",
        "alternatives": [
            {"unit": "UI/ml", "factor": 1.0},
            {"unit": "Indice", "factor": 1.0}
        ],
        "ranges": {}
    },
    "Blood - Anti-Cytomegalovirus IgM": {
        "lab_type": "blood",
        "primary_unit": "index",
        "alternatives": [
            {"unit": "Indice", "factor": 1.0}
        ],
        "ranges": {}
    },
    "Blood - Anti-Epstein-Barr Virus (EBV) VCA IgG": {
        "lab_type": "blood",
        "primary_unit": "index",
        "alternatives": [
            {"unit": "Indice", "factor": 1.0}
        ],
        "ranges": {}
    },
    "Blood - Anti-Epstein-Barr Virus (EBV) VCA IgM": {
        "lab_type": "blood",
        "primary_unit": "index",
        "alternatives": [
            {"unit": "Indice", "factor": 1.0}
        ],
        "ranges": {}
    },
    "Blood - Anti-Endomysial Antibody (IgA)": {
        "lab_type": "blood",
        "primary_unit": "unitless",
        "alternatives": [],
        "ranges": {}
    },
    "Blood - Anti-Endomysial Antibody (IgG)": {
        "lab_type": "blood",
        "primary_unit": "unitless",
        "alternatives": [],
        "ranges": {}
    },

    # HIV tests
    "Blood - HIV (Antibody + p24 Antigen)": {
        "lab_type": "blood",
        "primary_unit": "boolean",
        "alternatives": [],
        "ranges": {}
    },

    # Syphilis test
    "Blood - Treponema pallidum Hemagglutination (TPHA)": {
        "lab_type": "blood",
        "primary_unit": "boolean",
        "alternatives": [],
        "ranges": {}
    },

    # Morphology studies
    "Blood - Leukocyte Morphology": {
        "lab_type": "blood",
        "primary_unit": "unitless",
        "alternatives": [],
        "ranges": {}
    },
    "Blood - Platelet Morphology": {
        "lab_type": "blood",
        "primary_unit": "unitless",
        "alternatives": [],
        "ranges": {}
    },

    # Blood bank tests
    "Blood - Direct Antiglobulin Test (DAT)": {
        "lab_type": "blood",
        "primary_unit": "boolean",
        "alternatives": [],
        "ranges": {}
    },

    # Hemoglobin HPLC
    "Blood - Hemoglobin HPLC Comment": {
        "lab_type": "blood",
        "primary_unit": "unitless",
        "alternatives": [],
        "ranges": {}
    },

    # PNH tests - Flow cytometry for Paroxysmal Nocturnal Hemoglobinuria
    "Blood - PNH Monocytes CD14 Negative": {
        "lab_type": "blood",
        "primary_unit": "%",
        "alternatives": [],
        "ranges": {}
    },
    "Blood - PNH Monocytes CD157 Negative": {
        "lab_type": "blood",
        "primary_unit": "%",
        "alternatives": [],
        "ranges": {}
    },
    "Blood - PNH Monocytes FLAER Negative": {
        "lab_type": "blood",
        "primary_unit": "%",
        "alternatives": [],
        "ranges": {}
    },
    "Blood - PNH Monocytes Clone": {
        "lab_type": "blood",
        "primary_unit": "%",
        "alternatives": [],
        "ranges": {}
    },
    "Blood - PNH Neutrophils CD16 Negative": {
        "lab_type": "blood",
        "primary_unit": "%",
        "alternatives": [],
        "ranges": {}
    },
    "Blood - PNH Neutrophils CD66b Negative": {
        "lab_type": "blood",
        "primary_unit": "%",
        "alternatives": [],
        "ranges": {}
    },
    "Blood - PNH Neutrophils CD157 Negative": {
        "lab_type": "blood",
        "primary_unit": "%",
        "alternatives": [],
        "ranges": {}
    },
    "Blood - PNH Neutrophils FLAER Negative": {
        "lab_type": "blood",
        "primary_unit": "%",
        "alternatives": [],
        "ranges": {}
    },
    "Blood - PNH Neutrophils Clone": {
        "lab_type": "blood",
        "primary_unit": "%",
        "alternatives": [],
        "ranges": {}
    },

    # Osmotic resistance tests
    "Blood - Osmotic Resistance Initial (Immediate)": {
        "lab_type": "blood",
        "primary_unit": "g/dL NaCl",
        "alternatives": [
            {"unit": "g/dl NaCl", "factor": 1.0}
        ],
        "ranges": {}
    },
    "Blood - Osmotic Resistance Total (Immediate)": {
        "lab_type": "blood",
        "primary_unit": "g/dL NaCl",
        "alternatives": [
            {"unit": "g/dl NaCl", "factor": 1.0}
        ],
        "ranges": {}
    },
    "Blood - Osmotic Resistance Initial (After Incubation)": {
        "lab_type": "blood",
        "primary_unit": "g/dL NaCl",
        "alternatives": [
            {"unit": "g/dl NaCl", "factor": 1.0}
        ],
        "ranges": {}
    },
    "Blood - Osmotic Resistance Total (After Incubation)": {
        "lab_type": "blood",
        "primary_unit": "g/dL NaCl",
        "alternatives": [
            {"unit": "g/dl NaCl", "factor": 1.0}
        ],
        "ranges": {}
    },

    # Other specialized tests
    "Blood - Lymphoplasmacytic Cells": {
        "lab_type": "blood",
        "primary_unit": "%",
        "alternatives": [],
        "ranges": {}
    },

    # Urine tests
    "Urine - Hemosiderin": {
        "lab_type": "urine",
        "primary_unit": "boolean",
        "alternatives": [],
        "ranges": {}
    },

    # Sample information (metadata)
    "Sample - Product Type": {
        "lab_type": "blood",
        "primary_unit": "unitless",
        "alternatives": [],
        "ranges": {}
    }
}

added_count = 0
for lab_name, spec in new_labs.items():
    if lab_name not in lab_specs:
        print(f"  ✓ Adding: {lab_name}")
        lab_specs[lab_name] = spec
        added_count += 1
    else:
        print(f"  - Already exists: {lab_name}")

print(f"\n✅ Added {added_count} new lab specifications")

# =============================================================================
# CATEGORY 2: Add missing unit alternatives
# =============================================================================
print("\n" + "="*80)
print("CATEGORY 2: ADDING MISSING UNIT ALTERNATIVES")
print("="*80)

def add_alternative(lab_name, unit, factor=1.0):
    if lab_name in lab_specs:
        alternatives = lab_specs[lab_name].get("alternatives", [])
        has_unit = any(alt.get("unit") == unit for alt in alternatives)

        if not has_unit:
            print(f"  ✓ Adding '{unit}' to {lab_name}")
            alternatives.append({"unit": unit, "factor": factor})
            lab_specs[lab_name]["alternatives"] = alternatives
            return True
        else:
            print(f"  - Already has '{unit}': {lab_name}")
            return False
    else:
        print(f"  ⚠ Lab not found: {lab_name}")
        return False

# Add L/L for hematocrit (1 L/L = 100%)
add_alternative("Blood - Hematocrit (HCT)", "L/L", 100.0)
add_alternative("Blood - Haematocrit (HCT) (%)", "L/L", 100.0)

# Add mmol/mol Hg for HbA1c IFCC
add_alternative("Blood - Hemoglobin A1c (IFCC)", "mmol / mol Hg", 1.0)
add_alternative("Blood - Hemoglobin A1c (IFCC)", "mmol/mol Hg", 1.0)
add_alternative("Blood - Hemoglobin A1c (IFCC)", "mmol/mol", 1.0)

# Save updated lab specs
with open(LAB_SPECS_PATH, "w") as f:
    json.dump(lab_specs, f, ensure_ascii=False, indent=2)

print("\n" + "="*80)
print("✅ ALL FIXES APPLIED SUCCESSFULLY!")
print("="*80)
print(f"\nTotal labs in config: {len(lab_specs)}")
print(f"Added: {added_count} new labs")

print("\n" + "="*80)
print("IMPORTANT NOTES")
print("="*80)
print("""
1. REFERENCE RANGE INDICATORS
   These entries appear in some CSVs but are NOT lab tests:
   - "Ferropenia absoluta no adulto"
   - "Vitamina D - Deficiencia", "Insuficiencia", "Suficiencia", "Toxicidade"
   - "Avaliação de Risco - Alto risco"

   These are interpretation guidelines extracted from PDF text.
   They have no numeric values and should be filtered during extraction.

   Action: These will automatically be marked as $UNKNOWN$ and can be
   filtered out in post-processing or ignored.

2. TEST NAME VARIATIONS
   Some tests have slight naming variations that the LLM may not map correctly:
   - "HEMOGRAMA COM FÓRMULA - Fórmula Leucocitária - Monócitos"
     vs
     "HEMOGRAMA - Fórmula Leucocitária - Monócitos"

   The LLM standardization should handle these, but if not, they'll need
   manual mapping or prompt improvement.

3. NEXT STEPS
   Run the pipeline to apply these fixes:
   python main.py

   Then verify:
   python verify_pipeline_fixes.py
""")
