#!/usr/bin/env python3
"""
Perform statistical analysis on extraction results.

Usage:
    python analyze_stats.py <profile_name> [--output report.md]

Outputs:
    Statistical report with:
    - Distribution analysis per lab
    - Precision patterns
    - Confidence calibration
    - Temporal trends (if applicable)
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import yaml


def load_profile(profile_name: str) -> dict:
    """Load profile configuration."""
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


def analyze_distribution(df: pd.DataFrame) -> dict:
    """Analyze value distributions per lab."""
    results = {}

    # Filter out NaN lab names
    valid_lab_names = df['lab_name'].dropna().unique()
    for lab_name in sorted(valid_lab_names):
        lab_data = df[df['lab_name'] == lab_name]['value'].dropna()

        if len(lab_data) < 5:
            continue  # Skip labs with too few values

        stats = {
            'count': int(len(lab_data)),
            'mean': float(lab_data.mean()),
            'median': float(lab_data.median()),
            'std': float(lab_data.std()),
            'min': float(lab_data.min()),
            'max': float(lab_data.max()),
            'q25': float(lab_data.quantile(0.25)),
            'q75': float(lab_data.quantile(0.75)),
        }

        # Detect outliers (>3 std from mean)
        outliers = lab_data[np.abs(lab_data - stats['mean']) > 3 * stats['std']]
        if len(outliers) > 0:
            stats['outliers'] = int(len(outliers))
            stats['outlier_values'] = outliers.tolist()

        results[lab_name] = stats

    return results


def analyze_precision(df: pd.DataFrame) -> dict:
    """Detect suspicious precision patterns."""
    results = {}

    # Filter out NaN lab names
    valid_lab_names = df['lab_name'].dropna().unique()
    for lab_name in sorted(valid_lab_names):
        lab_data = df[df['lab_name'] == lab_name]['value'].dropna()

        if len(lab_data) < 10:
            continue

        # Analyze decimal patterns
        decimal_parts = []
        for val in lab_data:
            val_str = str(val)
            if '.' in val_str:
                decimal_parts.append(val_str.split('.')[-1])
            else:
                decimal_parts.append('0')

        decimal_counts = pd.Series(decimal_parts).value_counts()

        # Flag if >80% end in same decimal pattern
        if len(decimal_counts) > 0 and decimal_counts.iloc[0] / len(decimal_parts) > 0.8:
            most_common = decimal_counts.index[0]
            percentage = 100 * decimal_counts.iloc[0] / len(decimal_parts)
            results[lab_name] = {
                'pattern': most_common,
                'percentage': float(percentage),
                'count': int(decimal_counts.iloc[0]),
                'total': len(decimal_parts)
            }

    return results


def analyze_confidence(df: pd.DataFrame) -> dict:
    """Analyze confidence score distribution."""
    confidence = df['confidence'].fillna(1.0)

    buckets = {
        'high': (confidence >= 0.9).sum(),
        'medium': ((confidence >= 0.7) & (confidence < 0.9)).sum(),
        'low': (confidence < 0.7).sum()
    }

    return {
        'distribution': buckets,
        'mean': float(confidence.mean()),
        'median': float(confidence.median()),
        'min': float(confidence.min()),
        'max': float(confidence.max())
    }


def analyze_per_lab_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate quality metrics per lab."""
    results = []

    # Filter out NaN lab names
    valid_lab_names = df['lab_name'].dropna().unique()
    for lab_name in sorted(valid_lab_names):
        lab_data = df[df['lab_name'] == lab_name]

        results.append({
            'Lab Name': lab_name,
            'Total Rows': len(lab_data),
            'Avg Confidence': lab_data['confidence'].fillna(1.0).mean(),
            'Flagged for Review': lab_data['review_needed'].fillna(False).sum(),
            'Low Confidence (<0.8)': (lab_data['confidence'].fillna(1.0) < 0.8).sum()
        })

    return pd.DataFrame(results)


def analyze_per_document_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate quality metrics per source document."""
    results = []

    for source_file in sorted(df['source_file'].unique()):
        doc_data = df[df['source_file'] == source_file]

        results.append({
            'Source File': source_file,
            'Total Rows': len(doc_data),
            'Avg Confidence': doc_data['confidence'].fillna(1.0).mean(),
            'Flagged for Review': doc_data['review_needed'].fillna(False).sum(),
            'Low Confidence (<0.8)': (doc_data['confidence'].fillna(1.0) < 0.8).sum()
        })

    return pd.DataFrame(results)


def generate_markdown_report(profile_name: str, df: pd.DataFrame) -> str:
    """Generate markdown statistical report."""
    report = [
        f"# Statistical Analysis Report",
        f"",
        f"**Profile:** {profile_name}",
        f"**Total Rows:** {len(df)}",
        f"**Unique Labs:** {df['lab_name'].nunique()}",
        f"**Source Files:** {df['source_file'].nunique()}",
        f"",
        f"## Distribution Analysis",
        f"",
    ]

    # Distribution analysis
    distributions = analyze_distribution(df)
    for lab_name, stats in list(distributions.items())[:20]:  # Top 20
        report.append(f"### {lab_name}")
        report.append(f"- Count: {stats['count']}")
        report.append(f"- Mean: {stats['mean']:.2f} (σ={stats['std']:.2f})")
        report.append(f"- Median: {stats['median']:.2f}")
        report.append(f"- Range: [{stats['min']:.2f}, {stats['max']:.2f}]")

        if 'outliers' in stats:
            report.append(f"- **⚠ Outliers:** {stats['outliers']} values >3σ from mean")

        report.append("")

    # Precision patterns
    report.append("## Precision Patterns")
    report.append("")
    precision_issues = analyze_precision(df)
    if precision_issues:
        for lab_name, pattern in precision_issues.items():
            report.append(f"**⚠ {lab_name}:** {pattern['percentage']:.1f}% of values end in '.{pattern['pattern']}' ({pattern['count']}/{pattern['total']})")
        report.append("")
    else:
        report.append("No suspicious precision patterns detected.")
        report.append("")

    # Confidence analysis
    report.append("## Confidence Score Analysis")
    report.append("")
    conf_stats = analyze_confidence(df)
    report.append(f"- Mean: {conf_stats['mean']:.3f}")
    report.append(f"- Median: {conf_stats['median']:.3f}")
    report.append(f"- Range: [{conf_stats['min']:.3f}, {conf_stats['max']:.3f}]")
    report.append("")
    report.append("**Distribution:**")
    report.append(f"- High (≥0.9): {conf_stats['distribution']['high']}")
    report.append(f"- Medium (0.7-0.9): {conf_stats['distribution']['medium']}")
    report.append(f"- Low (<0.7): {conf_stats['distribution']['low']}")
    report.append("")

    # Per-lab quality
    report.append("## Per-Lab Quality Metrics")
    report.append("")
    lab_quality = analyze_per_lab_quality(df)
    report.append(lab_quality.to_markdown(index=False))
    report.append("")

    # Per-document quality
    report.append("## Per-Document Quality Metrics")
    report.append("")
    doc_quality = analyze_per_document_quality(df)
    # Show top 20 documents
    report.append(doc_quality.head(20).to_markdown(index=False))
    report.append("")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Statistical analysis of extraction results')
    parser.add_argument('profile', help='Profile name (e.g., "tsilva")')
    parser.add_argument('--output', help='Output markdown file (default: stats_report.md)')
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

    # Generate report
    report = generate_markdown_report(args.profile, df)

    # Output
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = output_path / 'stats_report.md'

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"✓ Statistical report saved: {output_file}")

    # Also print to console
    print("\n" + "="*60)
    print(report)


if __name__ == '__main__':
    main()
