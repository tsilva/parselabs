#!/usr/bin/env python3
"""
Helper script for verification workflow.
Prepares verification data and tracks results.

Usage:
    # Initialize verification
    python verify_sample.py <profile_name> --init

    # Record verification result
    python verify_sample.py <profile_name> --record <index> --status <status> --notes "..."

    # Generate report
    python verify_sample.py <profile_name> --report

Verification statuses:
    - match: Extracted value matches source
    - wrong_digit: Single digit error
    - decimal_error: Decimal place misplacement
    - unit_mismatch: Correct number, wrong unit
    - missing: Value exists but not extracted
    - hallucination: Extracted value doesn't exist
    - ambiguous: Source unclear/illegible
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd


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


def init_verification(profile: dict, profile_name: str):
    """Initialize verification tracking."""
    output_path = Path(profile['output_path'])
    sample_json = output_path / 'sample_data.json'

    if not sample_json.exists():
        print(f"Error: Sample not found. Run create_sample.py first.")
        sys.exit(1)

    # Load sample
    with open(sample_json) as f:
        sample_data = json.load(f)

    # Initialize verification results
    verification_file = output_path / 'verification_results.json'
    if verification_file.exists():
        print(f"⚠ Verification file already exists: {verification_file}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    verification_data = {
        'profile': profile_name,
        'started': datetime.now().isoformat(),
        'sample_size': sample_data['sample_size'],
        'verified_count': 0,
        'results': {}
    }

    with open(verification_file, 'w') as f:
        json.dump(verification_data, f, indent=2)

    print(f"✓ Verification initialized: {verification_file}")
    print(f"  Sample size: {sample_data['sample_size']}")
    print(f"\nNext step: Use Claude to verify each sample row against source images")


def record_result(profile: dict, index: int, status: str, notes: str, actual_value: str = None, actual_unit: str = None):
    """Record a verification result."""
    output_path = Path(profile['output_path'])
    verification_file = output_path / 'verification_results.json'

    if not verification_file.exists():
        print("Error: Verification not initialized. Run with --init first.")
        sys.exit(1)

    # Load verification data
    with open(verification_file) as f:
        verification_data = json.load(f)

    # Record result
    result = {
        'status': status,
        'notes': notes,
        'timestamp': datetime.now().isoformat()
    }

    if actual_value:
        result['actual_value'] = actual_value
    if actual_unit:
        result['actual_unit'] = actual_unit

    verification_data['results'][str(index)] = result
    verification_data['verified_count'] = len(verification_data['results'])
    verification_data['last_updated'] = datetime.now().isoformat()

    # Save
    with open(verification_file, 'w') as f:
        json.dump(verification_data, f, indent=2)

    print(f"✓ Recorded result for index {index}: {status}")
    print(f"  Progress: {verification_data['verified_count']}/{verification_data['sample_size']}")


def generate_report(profile: dict, profile_name: str):
    """Generate verification report."""
    output_path = Path(profile['output_path'])
    verification_file = output_path / 'verification_results.json'
    sample_json = output_path / 'sample_data.json'

    if not verification_file.exists():
        print("Error: No verification results found.")
        sys.exit(1)

    # Load data
    with open(verification_file) as f:
        verification_data = json.load(f)

    with open(sample_json) as f:
        sample_data = json.load(f)

    # Load full dataset
    df = pd.read_csv(output_path / 'all.csv')

    # Analyze results
    statuses = {}
    for idx_str, result in verification_data['results'].items():
        status = result['status']
        statuses[status] = statuses.get(status, 0) + 1

    total_verified = verification_data['verified_count']
    sample_size = verification_data['sample_size']
    matches = statuses.get('match', 0)
    errors = total_verified - matches

    # Generate report
    report = [
        "# Extraction Accuracy Verification Report",
        "",
        f"**Profile:** {profile_name}",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Total Dataset:** {len(df)} rows",
        f"**Sample Size:** {sample_size} rows",
        f"**Verified:** {total_verified} rows ({100*total_verified/sample_size:.1f}%)",
        "",
        "## Executive Summary",
        "",
        f"- **Overall Accuracy:** {100*matches/total_verified:.1f}% ({matches}/{total_verified} verified rows)" if total_verified > 0 else "- No verifications completed yet",
        f"- **Errors Found:** {errors} ({100*errors/total_verified:.1f}%)" if total_verified > 0 else "",
        "",
        "## Verification Results",
        "",
        "### Status Breakdown",
        "",
        "| Status | Count | Percentage |",
        "|--------|-------|------------|",
    ]

    for status in sorted(statuses.keys()):
        count = statuses[status]
        pct = 100 * count / total_verified if total_verified > 0 else 0
        report.append(f"| {status} | {count} | {pct:.1f}% |")

    report.append("")
    report.append("## Detailed Results")
    report.append("")

    # Group by status
    for status in sorted(statuses.keys()):
        report.append(f"### {status.replace('_', ' ').title()} ({statuses[status]})")
        report.append("")

        for idx_str, result in verification_data['results'].items():
            if result['status'] == status:
                idx = int(idx_str)
                # Find sample row
                sample_row = next((s for s in sample_data['samples'] if s['index'] == idx), None)
                if sample_row:
                    report.append(f"**Row {idx}** - {sample_row['lab_name']}")
                    report.append(f"- Source: {sample_row['source_file']}, page {sample_row['page_number']}")
                    report.append(f"- Extracted: {sample_row['value']} {sample_row['unit']}")
                    if 'actual_value' in result:
                        report.append(f"- Actual: {result['actual_value']} {result.get('actual_unit', sample_row['unit'])}")
                    report.append(f"- Notes: {result['notes']}")
                    report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")

    if errors > 0:
        report.append("### Immediate Actions")
        report.append("")

        # Find documents with errors
        error_docs = set()
        for idx_str, result in verification_data['results'].items():
            if result['status'] != 'match':
                idx = int(idx_str)
                sample_row = next((s for s in sample_data['samples'] if s['index'] == idx), None)
                if sample_row:
                    error_docs.add(sample_row['source_file'])

        if error_docs:
            report.append("**Re-extract these documents:**")
            for doc in sorted(error_docs):
                report.append(f"- `{doc}`")
            report.append("")

    # Save report
    report_file = output_path / 'accuracy_report.md'
    with open(report_file, 'w') as f:
        f.write("\n".join(report))

    print(f"✓ Report generated: {report_file}")
    print("\n" + "="*60)
    print("\n".join(report[:30]))  # Print first part
    print("\n[... see full report in file ...]")


def main():
    parser = argparse.ArgumentParser(description='Verification workflow helper')
    parser.add_argument('profile', help='Profile name')
    parser.add_argument('--init', action='store_true', help='Initialize verification')
    parser.add_argument('--record', type=int, help='Record result for index')
    parser.add_argument('--status', help='Verification status')
    parser.add_argument('--notes', default='', help='Notes about verification')
    parser.add_argument('--actual-value', help='Actual value from source')
    parser.add_argument('--actual-unit', help='Actual unit from source')
    parser.add_argument('--report', action='store_true', help='Generate report')

    args = parser.parse_args()

    profile = load_profile(args.profile)

    if args.init:
        init_verification(profile, args.profile)
    elif args.record is not None:
        if not args.status:
            print("Error: --status required when recording result")
            sys.exit(1)
        record_result(profile, args.record, args.status, args.notes, args.actual_value, args.actual_unit)
    elif args.report:
        generate_report(profile, args.profile)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
