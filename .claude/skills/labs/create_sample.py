#!/usr/bin/env python3
"""
Create an intelligent sample of extraction results for accuracy verification.

Usage:
    python create_sample.py <profile_name> [--sample-size N]

Outputs:
    - sample_for_verification.csv: Sample rows with metadata
    - sample_data.json: Machine-readable sample data
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


def load_profile(profile_name: str) -> dict:
    """Load profile configuration."""
    # Try yaml first, then json
    profile_path = Path(f"profiles/{profile_name}.yaml")
    if not profile_path.exists():
        profile_path = Path(f"profiles/{profile_name}.json")

    if not profile_path.exists():
        print(f"Error: Profile '{profile_name}' not found")
        sys.exit(1)

    with open(profile_path) as f:
        if profile_path.suffix == '.yaml':
            return yaml.safe_load(f)
        else:
            return json.load(f)


def create_intelligent_sample(df: pd.DataFrame, target_size: int = 50) -> pd.DataFrame:
    """
    Create an intelligent sample using multiple criteria.

    Selection criteria (in priority order):
    1. All rows with review_needed == True
    2. High-value labs (2 samples each)
    3. Stratified by source file (2-3 samples per file, up to 20 files)
    4. Low confidence rows (confidence < 0.8)
    5. Edge cases (below_limit/above_limit flags)
    """
    HIGH_VALUE_LABS = [
        "Blood - Glucose",
        "Blood - Hemoglobin (Hgb)",
        "Blood - Total Cholesterol",
        "Blood - LDL Cholesterol",
        "Blood - HDL Cholesterol",
        "Blood - Creatinine",
        "Blood - TSH"
    ]

    sample_indices = set()

    # 1. All flagged rows (highest priority)
    flagged = df[df['review_needed'].fillna(False) == True]
    sample_indices.update(flagged.index.tolist())
    print(f"✓ Flagged for review: {len(flagged)} rows")

    # 2. High-value labs (2 samples per lab)
    for lab in HIGH_VALUE_LABS:
        lab_data = df[df['lab_name'] == lab]
        if len(lab_data) > 0:
            n_samples = min(2, len(lab_data))
            sampled = lab_data.sample(n=n_samples, random_state=42)
            sample_indices.update(sampled.index.tolist())
    print(f"✓ High-value labs: {len([lab for lab in HIGH_VALUE_LABS if lab in df['lab_name'].values])} labs sampled")

    # 3. Stratified by source (2-3 samples per file, up to 20 files)
    source_files = sorted(df['source_file'].unique())[:20]
    for source_file in source_files:
        source_data = df[df['source_file'] == source_file]
        n_samples = min(2, len(source_data))
        sampled = source_data.sample(n=n_samples, random_state=42)
        sample_indices.update(sampled.index.tolist())
    print(f"✓ Source file stratification: {len(source_files)} files sampled")

    # 4. Low confidence
    low_conf = df[df['confidence'].fillna(1.0) < 0.8]
    if len(low_conf) > 0:
        sample_indices.update(low_conf.index.tolist())
        print(f"✓ Low confidence: {len(low_conf)} rows")
    else:
        print(f"✓ Low confidence: 0 rows")

    # 5. Edge cases
    edge_count = 0
    if 'is_below_limit' in df.columns:
        below_limit = df[df['is_below_limit'].fillna(False) == True]
        if len(below_limit) > 0:
            sample_indices.update(below_limit.sample(min(3, len(below_limit)), random_state=42).index.tolist())
            edge_count += len(below_limit)
    if 'is_above_limit' in df.columns:
        above_limit = df[df['is_above_limit'].fillna(False) == True]
        if len(above_limit) > 0:
            sample_indices.update(above_limit.sample(min(3, len(above_limit)), random_state=42).index.tolist())
            edge_count += len(above_limit)
    if edge_count > 0:
        print(f"✓ Edge cases: {edge_count} rows")

    # Limit to target size and create sample DataFrame
    sample_indices = list(sample_indices)[:target_size]
    sample_df = df.loc[sample_indices].copy()

    # Sort for easier verification
    sample_df = sample_df.sort_values(['source_file', 'page_number'])

    return sample_df


def main():
    parser = argparse.ArgumentParser(description='Create intelligent sample for verification')
    parser.add_argument('profile', help='Profile name (e.g., "tsilva")')
    parser.add_argument('--sample-size', type=int, default=50, help='Target sample size (default: 50)')
    args = parser.parse_args()

    # Load profile
    profile = load_profile(args.profile)
    output_path = Path(profile['output_path'])

    # Load data
    csv_path = output_path / 'all.csv'
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    print(f"\n=== DATASET OVERVIEW ===")
    print(f"Total rows: {len(df)}")
    print(f"Unique lab names: {df['lab_name'].nunique()}")
    print(f"Source files: {df['source_file'].nunique()}")
    print()

    # Create sample
    print(f"=== CREATING SAMPLE (target: {args.sample_size}) ===")
    sample_df = create_intelligent_sample(df, args.sample_size)

    print(f"\n=== SAMPLE SUMMARY ===")
    print(f"Total rows to verify: {len(sample_df)}")
    print(f"\nTop 10 labs in sample:")
    print(sample_df['lab_name'].value_counts().head(10))

    # Save CSV
    csv_output = output_path / 'sample_for_verification.csv'
    sample_df.to_csv(csv_output, index=True)
    print(f"\n✓ Saved: {csv_output}")

    # Save JSON for programmatic access
    sample_data = []
    for idx, row in sample_df.iterrows():
        sample_data.append({
            'index': int(idx),
            'source_file': row['source_file'],
            'page_number': int(row['page_number']),
            'lab_name': row['lab_name'],
            'lab_name_raw': row['lab_name_raw'],
            'value': float(row['value']) if pd.notna(row['value']) else None,
            'unit': row['unit'],
            'value_raw': row['value_raw'] if pd.notna(row['value_raw']) else None,
            'unit_raw': row['unit_raw'] if pd.notna(row['unit_raw']) else None,
            'confidence': float(row['confidence']) if pd.notna(row['confidence']) else 1.0,
            'review_needed': bool(row['review_needed']) if pd.notna(row['review_needed']) else False,
            'review_reason': row['review_reason'] if pd.notna(row['review_reason']) else ''
        })

    json_output = output_path / 'sample_data.json'
    with open(json_output, 'w') as f:
        json.dump({
            'profile': args.profile,
            'sample_size': len(sample_df),
            'total_dataset_size': len(df),
            'samples': sample_data
        }, f, indent=2)
    print(f"✓ Saved: {json_output}")


if __name__ == '__main__':
    main()
