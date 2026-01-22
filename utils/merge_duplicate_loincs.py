#!/usr/bin/env python3
"""
Migration Script: Merge Duplicate LOINC Codes

This script resolves duplicate LOINC codes in lab_specs.json by:
1. Merging duplicate entries (keeping canonical names, merging alternatives)
2. Updating the name_standardization.json cache
3. Finding affected documents that need re-extraction

Usage:
    python utils/merge_duplicate_loincs.py --dry-run  # Preview changes
    python utils/merge_duplicate_loincs.py            # Execute migration
"""

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path


# =============================================================================
# CANONICAL NAME MAPPINGS
# =============================================================================
# Maps old names to canonical names. The canonical names are the ones to keep.
# Key = name to remove, Value = canonical name to keep

CANONICAL_MAPPINGS = {
    # American English spelling over British
    "Blood - Haemoglobin (Hb)": "Blood - Hemoglobin (Hgb)",
    "Blood - Haematocrit (HCT) (%)": "Blood - Hematocrit (HCT) (%)",
    "Urine Type II - Colour": "Urine Type II - Color",
    "Urine Type II - Haemoglobin": "Urine Type II - Blood",  # Also American term

    # Full names over abbreviations
    "Blood - AST (SGOT)": "Blood - Aspartate Aminotransferase (AST)",
    "Blood - BUN (Blood Urea Nitrogen)": "Blood - Urea",

    # Standard terms over regional variants
    "Blood - Quick Time": "Blood - Prothrombin Time (PT)",
    "Blood - Quick Time (%)": "Blood - Prothrombin Time (PT) (%)",
    "Blood - Waaler-Rose Test": "Blood - Rheumatoid Factor (RF)",
    "Blood - Circulating Immune Complexes": "Blood - Rheumatoid Factor (RF)",

    # Simpler/cleaner naming
    "Blood - Erythrocytes (RBC)": "Blood - Erythrocytes",
    "Blood - Reticulocytes (absolute)": "Blood - Reticulocyte Count",
    "Blood - Reticulocytes (%)": "Blood - Reticulocyte Count (%)",

    # Consistent naming conventions
    "Blood - Reticulocyte Hemoglobin Content (CHr)": "Blood - Reticulocyte Hemoglobin Content",
    "Blood - Reticulocyte Hemoglobin Equivalent": "Blood - Reticulocyte Hemoglobin Content",
    "Blood - Immature Granulocytes (IG) (%)": "Blood - Immature Granulocytes (%)",
    "Blood - Red Cell Distribution Width (RDW-CV) (%)": "Blood - Red Cell Distribution Width (RDW) (%)",

    # Reticulocyte RNA content standardization
    "Blood - High mRNA Content (%)": "Blood - Reticulocytes - High RNA (%)",
    "Blood - Medium mRNA Content (%)": "Blood - Reticulocytes - Medium RNA (%)",
    "Blood - Low mRNA Content (%)": "Blood - Reticulocytes - Low RNA (%)",

    # Hypochromic cells
    "Blood - Erythrocytes - Hypochromic (%)": "Blood - Hypochromic Red Blood Cells (%)",

    # Urine standardization
    "Urine Type II - Density": "Urine Type II - Specific Gravity",
    "Urine Type II - Turbidity": "Urine Type II - Appearance",
    "Urine Type II - Sediment - Total Casts": "Urine Type II - Sediment - Casts",

    # Same test, different unit representation - merge to version without (%)
    # These have the same LOINC because they measure the same thing, just expressed differently
    "Blood - Red Cell Distribution Width - Standard Deviation (RDW-SD) (%)":
        "Blood - Red Cell Distribution Width - Standard Deviation (RDW-SD)",
    "Blood - Hemoglobin A1c (IFCC) (%)": "Blood - Hemoglobin A1c (IFCC)",
    "Blood - Mean Corpuscular Hemoglobin Concentration (MCHC) (%)":
        "Blood - Mean Corpuscular Hemoglobin Concentration (MCHC)",

    # Urine sediment - merge related categories
    "Urine Type II - Deposit": "Urine Type II - Sediment - Epithelial Cells",
    "Urine Type II - Sediment - Cells": "Urine Type II - Sediment - Epithelial Cells",
}


# =============================================================================
# LOINC CODE FIXES
# =============================================================================
# Cases where distinct tests incorrectly share the same LOINC code.
# These need new LOINC codes assigned.

LOINC_FIXES = {
    # Hemolysis tests - "total, after incubation" keeps 50795-4, "total, immediate" gets new code
    "Blood - Hemolysis (total, immediate) (%)": "4667-6",

    # Osmotic resistance - need distinct codes
    # Initial Immediate keeps 5911-3, others get new codes
    "Blood - Osmotic Resistance Initial (After Incubation)": "40741-1",
    "Blood - Osmotic Resistance Total (Immediate)": "34966-0",
    "Blood - Osmotic Resistance Total (After Incubation)": "40742-9",

    # PNH Monocytes - CD157- keeps 90736-0, others get distinct codes
    "Blood - PNH Monocytes CD14 Negative (%)": "55474-6",
    "Blood - PNH Monocytes FLAER Negative (%)": "55478-7",
    "Blood - PNH Monocytes Clone (%)": "55479-5",

    # PNH Neutrophils - CD157- keeps 90737-8, others get distinct codes
    "Blood - PNH Neutrophils CD16 Negative (%)": "55481-1",
    "Blood - PNH Neutrophils CD66b Negative (%)": "55482-9",
    "Blood - PNH Neutrophils FLAER Negative (%)": "55483-7",
    "Blood - PNH Neutrophils Clone (%)": "55484-5",

    # Gram bacteria - different sample types need distinct codes
    # 18482-0 = "Bacteria identified in Urine by Gram stain"
    "Urine - Bacteriological - Gram Bacteria": "18482-0",
    "Urine - Bacteriological - Gram Negative Bacteria": "18483-8",
    "Urine - Bacteriological - Gram Positive Bacteria": "18484-6",
    # Feces - Gram Bacteria keeps 664-3

    # Cell morphology vs Platelet morphology - distinct tests
    "Blood - Platelet Morphology": "32206-3",
    # Blood - Cell Morphology keeps 11125-2

    # Treponema vs Wright - distinct tests
    "Blood - Wright Reaction": "5319-8",
    # Blood - Treponema pallidum Hemagglutination (TPHA) keeps 5406-3

    # Urine sediment distinctions - distinct cell types
    "Urine Type II - Sediment - Upper tract cells": "30089-5",
    # Urine Type II - Sediment - Tubular epithelial cells keeps 12248-1

    "Urine Type II - Sediment - Transitional epithelial cells": "30088-7",
    # Urine Type II - Sediment - Small Round Cells keeps 13945-1
}


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    """Save JSON file with consistent formatting."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # Ensure file ends with newline
    with open(path, 'a', encoding='utf-8') as f:
        f.write('\n')


def find_duplicates(lab_specs: dict) -> dict[str, list[str]]:
    """Find all duplicate LOINC codes."""
    loinc_to_labs = defaultdict(list)
    for lab_name, spec in lab_specs.items():
        if lab_name.startswith('_'):
            continue
        loinc = spec.get('loinc_code', '')
        if loinc:
            loinc_to_labs[loinc].append(lab_name)
    return {k: v for k, v in loinc_to_labs.items() if len(v) > 1}


def merge_lab_specs(
    lab_specs: dict,
    canonical_mappings: dict,
    loinc_fixes: dict,
    verbose: bool = True
) -> tuple[dict, list[str], list[str]]:
    """
    Merge duplicate lab entries and fix LOINC codes.

    Always modifies a copy in-memory, regardless of dry_run mode.

    Returns:
        Tuple of (updated lab_specs, list of removed lab names, list of changes)
    """
    removed_labs = []
    changes = []
    # Deep copy to avoid modifying the original
    updated_specs = copy.deepcopy(lab_specs)

    # First, apply LOINC fixes
    for lab_name, new_loinc in loinc_fixes.items():
        if lab_name in updated_specs:
            old_loinc = updated_specs[lab_name].get('loinc_code', 'none')
            if old_loinc != new_loinc:
                change = f"[LOINC FIX] {lab_name}: {old_loinc} -> {new_loinc}"
                changes.append(change)
                if verbose:
                    print(f"  {change}")
                updated_specs[lab_name]['loinc_code'] = new_loinc

    # Then, merge canonical entries
    for old_name, canonical_name in canonical_mappings.items():
        if old_name not in updated_specs:
            if verbose:
                print(f"  [SKIP] {old_name} not found in lab_specs")
            continue
        if canonical_name not in updated_specs:
            if verbose:
                print(f"  [SKIP] Canonical {canonical_name} not found in lab_specs")
            continue

        old_spec = updated_specs[old_name]
        canonical_spec = updated_specs[canonical_name]

        # Merge alternatives from old into canonical
        old_alts = old_spec.get('alternatives', [])
        canonical_alts = canonical_spec.get('alternatives', [])

        # Get existing units in canonical
        existing_units = {alt['unit'] for alt in canonical_alts}
        existing_units.add(canonical_spec.get('primary_unit', ''))

        # Add new alternatives from old spec
        new_alts = []
        for alt in old_alts:
            if alt['unit'] not in existing_units:
                new_alts.append(alt)
                existing_units.add(alt['unit'])

        if new_alts:
            change = f"[MERGE] {old_name} -> {canonical_name} (adding alternatives: {[a['unit'] for a in new_alts]})"
            changes.append(change)
            if verbose:
                print(f"  {change}")
            canonical_spec['alternatives'] = canonical_alts + new_alts

        # Merge ranges if canonical doesn't have them
        old_ranges = old_spec.get('ranges', {})
        canonical_ranges = canonical_spec.get('ranges', {})
        for range_key, range_val in old_ranges.items():
            if range_key not in canonical_ranges:
                change = f"[MERGE RANGE] {old_name} -> {canonical_name}: Adding range '{range_key}': {range_val}"
                changes.append(change)
                if verbose:
                    print(f"  {change}")
                if 'ranges' not in canonical_spec:
                    canonical_spec['ranges'] = {}
                canonical_spec['ranges'][range_key] = range_val

        # Remove old entry
        change = f"[REMOVE] {old_name}"
        changes.append(change)
        if verbose:
            print(f"  {change}")
        del updated_specs[old_name]
        removed_labs.append(old_name)

    return updated_specs, removed_labs, changes


def update_name_cache(
    cache: dict,
    canonical_mappings: dict,
    verbose: bool = True
) -> tuple[dict, list[str]]:
    """
    Update name_standardization.json cache to point old names to canonical names.

    Always modifies a copy in-memory.

    Returns:
        Tuple of (updated cache, list of changes)
    """
    updated_cache = dict(cache)
    changes = []

    for raw_name, standardized_name in cache.items():
        # If the standardized name is one we're removing, update to canonical
        if standardized_name in canonical_mappings:
            canonical = canonical_mappings[standardized_name]
            change = f"[CACHE] '{raw_name}': '{standardized_name}' -> '{canonical}'"
            changes.append(change)
            if verbose:
                print(f"  {change}")
            updated_cache[raw_name] = canonical

    # Also add direct mappings for the old standardized names
    for old_name, canonical_name in canonical_mappings.items():
        old_name_lower = old_name.lower()
        if old_name_lower not in updated_cache:
            change = f"[CACHE ADD] '{old_name_lower}' -> '{canonical_name}'"
            changes.append(change)
            if verbose:
                print(f"  {change}")
            updated_cache[old_name_lower] = canonical_name

    print(f"\n  Total cache updates: {len(changes)}")
    return updated_cache, changes


def find_affected_documents(csv_path: Path, removed_labs: list[str]) -> list[str]:
    """
    Find documents that have entries with removed lab names.

    Returns:
        List of affected document stems
    """
    if not csv_path.exists():
        return []

    import csv
    affected_docs = set()

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lab_name = row.get('lab_name', '')
            if lab_name in removed_labs:
                source_file = row.get('source_file', '')
                if source_file:
                    # Get stem (filename without extension)
                    doc_stem = Path(source_file).stem
                    affected_docs.add(doc_stem)

    return sorted(affected_docs)


def main():
    parser = argparse.ArgumentParser(
        description='Merge duplicate LOINC code entries in lab_specs.json'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    parser.add_argument(
        '--config',
        default='config/lab_specs.json',
        help='Path to lab_specs.json'
    )
    parser.add_argument(
        '--cache',
        default='config/cache/name_standardization.json',
        help='Path to name_standardization.json'
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cache_path = Path(args.cache)

    print("=" * 60)
    print("LOINC Duplicate Merge Migration")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY RUN - No files will be modified ***\n")

    # Load files
    print("\n[1] Loading configuration files...")
    lab_specs = load_json(config_path)
    name_cache = load_json(cache_path)
    print(f"    Loaded {len(lab_specs) - 1} lab specs (excluding _relationships)")
    print(f"    Loaded {len(name_cache)} cache entries")

    # Find current duplicates
    print("\n[2] Current duplicate LOINC codes:")
    duplicates = find_duplicates(lab_specs)
    for loinc, labs in sorted(duplicates.items()):
        print(f"    {loinc}: {labs}")
    print(f"\n    Total: {len(duplicates)} duplicate LOINC codes")

    # Merge lab specs (always apply to in-memory copy)
    print("\n[3] Merging lab specifications...")
    updated_specs, removed_labs, spec_changes = merge_lab_specs(
        lab_specs, CANONICAL_MAPPINGS, LOINC_FIXES, verbose=True
    )

    # Update name cache (always apply to in-memory copy)
    print("\n[4] Updating name standardization cache...")
    updated_cache, cache_changes = update_name_cache(
        name_cache, CANONICAL_MAPPINGS, verbose=True
    )

    # Find affected documents
    print("\n[5] Finding affected documents...")
    profiles_output = {
        'tiago': Path('/Users/tsilva/Google Drive/My Drive/labsparser-tiago/all.csv'),
        'cristina': Path('/Users/tsilva/Google Drive/My Drive/labsparser-cristina/all.csv'),
    }

    all_affected = {}
    for profile, csv_path in profiles_output.items():
        affected = find_affected_documents(csv_path, removed_labs)
        all_affected[profile] = affected
        if affected:
            print(f"    {profile}: {len(affected)} documents affected")
            for doc in affected[:5]:
                print(f"      - {doc}")
            if len(affected) > 5:
                print(f"      ... and {len(affected) - 5} more")
        else:
            print(f"    {profile}: No documents affected or CSV not found")

    # Check remaining duplicates after all changes
    print("\n[6] Checking for remaining duplicates after merge...")
    remaining_duplicates = find_duplicates(updated_specs)
    if remaining_duplicates:
        print(f"    WARNING: {len(remaining_duplicates)} duplicate LOINC codes remain:")
        for loinc, labs in sorted(remaining_duplicates.items()):
            print(f"      {loinc}: {labs}")
    else:
        print("    SUCCESS: No duplicate LOINC codes remaining!")

    # Save files (only when not in dry-run)
    if not args.dry_run:
        print("\n[7] Saving updated files...")

        # Sort before saving
        sorted_specs = dict(sorted(updated_specs.items()))
        save_json(config_path, sorted_specs)
        print(f"    Saved: {config_path}")

        sorted_cache = dict(sorted(updated_cache.items()))
        save_json(cache_path, sorted_cache)
        print(f"    Saved: {cache_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Original labs: {len(lab_specs) - 1}")
    print(f"  Labs after merge: {len(updated_specs) - 1}")
    print(f"  Labs removed: {len(removed_labs)}")
    print(f"  LOINC fixes applied: {len([c for c in spec_changes if 'LOINC FIX' in c])}")
    print(f"  Cache entries updated: {len(cache_changes)}")
    print(f"  Original duplicates: {len(duplicates)}")
    print(f"  Remaining duplicates: {len(remaining_duplicates)}")

    if all_affected:
        total_affected = sum(len(docs) for docs in all_affected.values())
        print(f"\n  Documents needing re-extraction: {total_affected}")
        for profile, docs in all_affected.items():
            if docs:
                print(f"    {profile}: {len(docs)} documents")

    if args.dry_run:
        print("\n*** DRY RUN COMPLETE - Run without --dry-run to apply changes ***")
    else:
        print("\n  Migration complete!")
        print("\n  Next steps:")
        print("    1. Run: python utils/validate_lab_specs_schema.py")
        print("    2. Re-extract affected profiles:")
        print("       python extract.py --profile tiago")
        print("       python extract.py --profile cristina")

    return 0 if not remaining_duplicates else 1


if __name__ == '__main__':
    sys.exit(main())
