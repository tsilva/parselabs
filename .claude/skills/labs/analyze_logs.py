#!/usr/bin/env python3
"""
Analyze extraction logs for patterns, failures, and optimization opportunities.

Usage:
    python analyze_logs.py <profile_name> [--output report.md]

Outputs:
    Log analysis report with:
    - Processing summary (PDFs processed, pages, failures)
    - Strategy usage (TEXT vs VISION breakdown)
    - Extraction failures with document/page details
    - Unit inference statistics
    - Standardization cache performance
    - Timeline of extraction runs
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

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


@dataclass
class LogStats:
    """Container for extracted log statistics."""
    # Pipeline runs
    runs: list = field(default_factory=list)

    # PDF processing
    pdfs_started: set = field(default_factory=set)
    pdfs_completed: set = field(default_factory=set)
    pdfs_failed: set = field(default_factory=set)
    pdfs_no_results: set = field(default_factory=set)

    # Pipeline summary stats (from log summary)
    pdfs_found: int = 0
    pdfs_skipped: int = 0
    pdfs_to_process: int = 0
    pdfs_processed_total: int = 0
    extraction_failures_total: int = 0
    merged_rows: int = 0
    deduplicated_rows: int = 0

    # Strategy breakdown
    strategy_text: set = field(default_factory=set)
    strategy_text_cached: set = field(default_factory=set)
    strategy_vision: set = field(default_factory=set)
    strategy_text_to_vision: set = field(default_factory=set)

    # Page-level stats
    pages_processed: int = 0
    pages_cached: int = 0
    pages_failed: list = field(default_factory=list)

    # Standardization
    name_standardization_count: int = 0
    unit_standardization_count: int = 0

    # Normalization stats
    unknown_labs_filtered: int = 0
    comparison_operators: int = 0
    missing_units_inferred: int = 0
    suspicious_ranges_nullified: int = 0

    # Validation
    validation_flagged: int = 0
    validation_reasons: Counter = field(default_factory=Counter)

    # Errors
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def parse_timestamp(line: str) -> Optional[datetime]:
    """Extract timestamp from log line."""
    # Format: 2024-01-15 10:30:45,123
    match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return None
    return None


def parse_log_file(log_path: Path, stats: LogStats, is_error_log: bool = False) -> None:
    """Parse a log file and extract statistics."""
    if not log_path.exists():
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_run_start = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        timestamp = parse_timestamp(line)

        # Track pipeline runs
        if 'Using profile:' in line:
            current_run_start = timestamp
            profile_match = re.search(r'Using profile: (.+)$', line)
            if profile_match:
                stats.runs.append({
                    'start': timestamp,
                    'profile': profile_match.group(1)
                })

        # PDF processing started
        if match := re.search(r'\[([^\]]+)\] Processing\.\.\.', line):
            pdf_stem = match.group(1)
            stats.pdfs_started.add(pdf_stem)

        # PDF completed successfully
        if match := re.search(r'\[([^\]]+)\] Completed successfully', line):
            pdf_stem = match.group(1)
            stats.pdfs_completed.add(pdf_stem)

        # PDF no results
        if match := re.search(r'\[([^\]]+)\] No results extracted', line):
            pdf_stem = match.group(1)
            stats.pdfs_no_results.add(pdf_stem)

        # Strategy: TEXT (cached)
        if match := re.search(r'\[([^\]]+)\] Strategy: TEXT \(cached\)', line):
            pdf_stem = match.group(1)
            stats.strategy_text_cached.add(pdf_stem)
            stats.strategy_text.add(pdf_stem)

        # Strategy: TEXT (LLM classified)
        if match := re.search(r'\[([^\]]+)\] Strategy: TEXT \(LLM classified', line):
            pdf_stem = match.group(1)
            stats.strategy_text.add(pdf_stem)

        # Strategy: VISION
        if match := re.search(r'\[([^\]]+)\] Strategy: VISION', line):
            pdf_stem = match.group(1)
            stats.strategy_vision.add(pdf_stem)

        # Strategy: TEXT -> VISION (fallback)
        if match := re.search(r'\[([^\]]+)\] Strategy: TEXT -> VISION', line):
            pdf_stem = match.group(1)
            stats.strategy_text_to_vision.add(pdf_stem)

        # Page processing
        if 'Processing page' in line:
            stats.pages_processed += 1

        # Page cached
        if 'Loading cached extraction data' in line:
            stats.pages_cached += 1

        # Page extraction failed
        if match := re.search(r'\[([^\]]+)\] EXTRACTION FAILED: (.+)$', line):
            page_name = match.group(1)
            reason = match.group(2)
            stats.pages_failed.append({'page': page_name, 'reason': reason, 'timestamp': timestamp})
            # Also track PDF as failed
            pdf_stem = page_name.rsplit('.', 1)[0] if '.' in page_name else page_name
            stats.pdfs_failed.add(pdf_stem)

        # Pipeline summary stats
        if match := re.search(r'Found (\d+) PDF\(s\) matching', line):
            stats.pdfs_found = int(match.group(1))

        if match := re.search(r'Skipping (\d+) already-processed PDF\(s\)', line):
            stats.pdfs_skipped = int(match.group(1))

        if match := re.search(r'Processing (\d+) PDF\(s\)$', line):
            stats.pdfs_to_process = int(match.group(1))

        if match := re.search(r'Successfully processed (\d+) PDFs', line):
            stats.pdfs_processed_total = int(match.group(1))

        if match := re.search(r'PDFs processed: (\d+)', line):
            stats.pdfs_processed_total = int(match.group(1))

        if match := re.search(r'Extraction failures: (\d+)', line):
            stats.extraction_failures_total = int(match.group(1))

        if match := re.search(r'Merged data: (\d+) rows', line):
            stats.merged_rows = int(match.group(1))

        if match := re.search(r'After deduplication: (\d+) rows', line):
            stats.deduplicated_rows = int(match.group(1))

        # Standardization counts
        if match := re.search(r'Standardized (\d+) unique test names', line):
            stats.name_standardization_count += int(match.group(1))

        if match := re.search(r'Standardized (\d+) unique units', line):
            stats.unit_standardization_count += int(match.group(1))

        # Normalization stats
        if match := re.search(r'Filtering (\d+) rows with unknown lab names', line):
            stats.unknown_labs_filtered = int(match.group(1))

        if match := re.search(r'Extracted (\d+) comparison operators', line):
            stats.comparison_operators = int(match.group(1))

        if match := re.search(r'Attempting to infer (\d+) missing units', line):
            stats.missing_units_inferred = int(match.group(1))

        if match := re.search(r'Nullified (\d+) suspicious reference ranges', line):
            stats.suspicious_ranges_nullified = int(match.group(1))

        # Validation
        if match := re.search(r'Validation flagged (\d+) rows for review', line):
            stats.validation_flagged = int(match.group(1))

        if match := re.search(r'Validation complete: (\d+)/(\d+) rows flagged', line):
            stats.validation_flagged = int(match.group(1))

        if match := re.search(r'^\s+-\s+(\w+):\s+(\d+)', line):
            reason = match.group(1)
            count = int(match.group(2))
            stats.validation_reasons[reason] = count

        # Collect errors and warnings
        if is_error_log or ' - ERROR - ' in line:
            # Extract just the message part
            msg_match = re.search(r' - ERROR - (.+)$', line)
            if msg_match:
                stats.errors.append({
                    'message': msg_match.group(1),
                    'timestamp': timestamp,
                    'full_line': line
                })

        if ' - WARNING - ' in line:
            msg_match = re.search(r' - WARNING - (.+)$', line)
            if msg_match:
                stats.warnings.append({
                    'message': msg_match.group(1),
                    'timestamp': timestamp
                })


def generate_markdown_report(profile_name: str, stats: LogStats, output_path: Path) -> str:
    """Generate markdown report from log statistics."""
    report = [
        "# Extraction Log Analysis Report",
        "",
        f"**Profile:** {profile_name}",
        f"**Output Path:** {output_path}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Pipeline runs summary
    if stats.runs:
        report.append("## Pipeline Runs")
        report.append("")
        report.append(f"Total runs recorded: {len(stats.runs)}")
        if stats.runs:
            last_run = stats.runs[-1]
            report.append(f"Last run: {last_run['start'].strftime('%Y-%m-%d %H:%M:%S') if last_run['start'] else 'Unknown'}")
        report.append("")

    # Processing summary
    report.append("## Processing Summary")
    report.append("")

    # Use pipeline summary stats if available, otherwise use individual tracking
    total_started = len(stats.pdfs_started) or stats.pdfs_to_process
    total_completed = len(stats.pdfs_completed) or (stats.pdfs_processed_total - stats.extraction_failures_total)
    total_failed = len(stats.pdfs_failed) or stats.extraction_failures_total
    total_no_results = len(stats.pdfs_no_results)

    report.append(f"| Metric | Count |")
    report.append(f"|--------|-------|")
    if stats.pdfs_found > 0:
        report.append(f"| PDFs Found | {stats.pdfs_found} |")
    if stats.pdfs_skipped > 0:
        report.append(f"| PDFs Skipped (cached) | {stats.pdfs_skipped} |")
    if stats.pdfs_to_process > 0 or total_started > 0:
        report.append(f"| PDFs To Process | {stats.pdfs_to_process or total_started} |")
    if stats.pdfs_processed_total > 0:
        report.append(f"| PDFs Processed Total | {stats.pdfs_processed_total} |")
    if total_failed > 0:
        report.append(f"| Extraction Failures | {total_failed} |")
    if total_no_results > 0:
        report.append(f"| PDFs No Results | {total_no_results} |")
    report.append(f"| Pages Processed | {stats.pages_processed} |")
    if stats.pages_cached > 0:
        report.append(f"| Pages From Cache | {stats.pages_cached} |")
    report.append("")

    if stats.pdfs_processed_total > 0 and stats.extraction_failures_total >= 0:
        success_rate = ((stats.pdfs_processed_total - stats.extraction_failures_total) / stats.pdfs_processed_total) * 100 if stats.pdfs_processed_total > 0 else 0
        report.append(f"**Success Rate:** {success_rate:.1f}%")
        report.append("")

    # Data pipeline stats
    if stats.merged_rows > 0 or stats.deduplicated_rows > 0:
        report.append("### Data Pipeline")
        report.append("")
        if stats.merged_rows > 0:
            report.append(f"- Rows after merge: {stats.merged_rows}")
        if stats.deduplicated_rows > 0:
            report.append(f"- Rows after deduplication: {stats.deduplicated_rows}")
            if stats.merged_rows > 0:
                removed = stats.merged_rows - stats.deduplicated_rows
                report.append(f"- Duplicates removed: {removed}")
        report.append("")

    # Strategy breakdown
    report.append("## Strategy Breakdown")
    report.append("")

    text_count = len(stats.strategy_text)
    text_cached = len(stats.strategy_text_cached)
    vision_count = len(stats.strategy_vision)
    fallback_count = len(stats.strategy_text_to_vision)

    report.append(f"| Strategy | Count | Percentage |")
    report.append(f"|----------|-------|------------|")

    total_strategies = text_count + vision_count
    if total_strategies > 0:
        report.append(f"| TEXT (total) | {text_count} | {100*text_count/total_strategies:.1f}% |")
        report.append(f"| TEXT (cached) | {text_cached} | {100*text_cached/total_strategies:.1f}% |")
        report.append(f"| VISION | {vision_count} | {100*vision_count/total_strategies:.1f}% |")
        report.append(f"| TEXT→VISION fallback | {fallback_count} | {100*fallback_count/total_strategies:.1f}% |")
    else:
        report.append(f"| TEXT | {text_count} | - |")
        report.append(f"| VISION | {vision_count} | - |")
    report.append("")

    # Extraction failures
    if stats.pages_failed:
        report.append("## Extraction Failures")
        report.append("")
        report.append(f"Total page failures: {len(stats.pages_failed)}")
        report.append("")

        # Group by reason
        reasons = Counter(f['reason'] for f in stats.pages_failed)
        report.append("### Failures by Reason")
        report.append("")
        report.append("| Reason | Count |")
        report.append("|--------|-------|")
        for reason, count in reasons.most_common():
            # Truncate long reasons
            display_reason = reason[:60] + "..." if len(reason) > 60 else reason
            report.append(f"| {display_reason} | {count} |")
        report.append("")

        # List failed pages (limited)
        report.append("### Failed Pages (recent)")
        report.append("")
        for failure in stats.pages_failed[-10:]:  # Last 10
            ts = failure['timestamp'].strftime('%Y-%m-%d %H:%M') if failure['timestamp'] else 'Unknown'
            report.append(f"- `{failure['page']}` ({ts}): {failure['reason'][:80]}")
        report.append("")

    # Standardization stats
    if stats.name_standardization_count > 0 or stats.unit_standardization_count > 0:
        report.append("## Standardization")
        report.append("")
        report.append(f"- Unique test names standardized: {stats.name_standardization_count}")
        report.append(f"- Unique units standardized: {stats.unit_standardization_count}")
        report.append("")

    # Normalization stats
    norm_stats = [
        (stats.comparison_operators, "Comparison operators extracted"),
        (stats.missing_units_inferred, "Missing units inferred"),
        (stats.suspicious_ranges_nullified, "Suspicious reference ranges nullified"),
        (stats.unknown_labs_filtered, "Unknown lab names filtered"),
    ]
    norm_stats = [(count, desc) for count, desc in norm_stats if count > 0]

    if norm_stats:
        report.append("## Normalization")
        report.append("")
        for count, desc in norm_stats:
            report.append(f"- {desc}: {count}")
        report.append("")

    # Validation results
    if stats.validation_flagged > 0:
        report.append("## Validation Results")
        report.append("")
        report.append(f"**Rows flagged for review:** {stats.validation_flagged}")
        report.append("")

        if stats.validation_reasons:
            report.append("### Flags by Reason")
            report.append("")
            report.append("| Reason | Count |")
            report.append("|--------|-------|")
            for reason, count in stats.validation_reasons.most_common():
                report.append(f"| {reason} | {count} |")
            report.append("")

    # Errors
    if stats.errors:
        report.append("## Errors")
        report.append("")
        report.append(f"Total errors: {len(stats.errors)}")
        report.append("")

        # Group errors by type
        error_types = Counter()
        for err in stats.errors:
            # Extract error type from message
            msg = err['message']
            if match := re.search(r'\[([^\]]+)\]', msg):
                error_types[match.group(1)] += 1
            else:
                # Use first 40 chars as type
                error_types[msg[:40]] += 1

        report.append("### Errors by Type")
        report.append("")
        report.append("| Error Type | Count |")
        report.append("|------------|-------|")
        for err_type, count in error_types.most_common(10):
            report.append(f"| {err_type} | {count} |")
        report.append("")

        # Recent errors
        report.append("### Recent Errors")
        report.append("")
        for err in stats.errors[-10:]:  # Last 10
            ts = err['timestamp'].strftime('%Y-%m-%d %H:%M') if err['timestamp'] else 'Unknown'
            msg = err['message'][:100]
            report.append(f"- [{ts}] {msg}")
        report.append("")

    # Warnings summary
    if stats.warnings:
        report.append("## Warnings Summary")
        report.append("")
        report.append(f"Total warnings: {len(stats.warnings)}")
        report.append("")

        # Group by type
        warning_types = Counter()
        for w in stats.warnings:
            msg = w['message']
            # Extract document name if present
            if match := re.search(r'\[([^\]]+)\]', msg):
                # Categorize by the action after the document name
                rest = msg[match.end():].strip()
                category = rest[:50] if rest else match.group(1)
            else:
                category = msg[:50]
            warning_types[category] += 1

        report.append("### Top Warning Categories")
        report.append("")
        report.append("| Category | Count |")
        report.append("|----------|-------|")
        for category, count in warning_types.most_common(10):
            report.append(f"| {category} | {count} |")
        report.append("")

    # PDFs with issues
    problem_pdfs = stats.pdfs_failed | stats.pdfs_no_results
    if problem_pdfs:
        report.append("## PDFs Requiring Attention")
        report.append("")

        if stats.pdfs_failed:
            report.append("### Failed PDFs")
            report.append("")
            for pdf in sorted(stats.pdfs_failed):
                report.append(f"- `{pdf}`")
            report.append("")

        if stats.pdfs_no_results:
            report.append("### PDFs with No Results")
            report.append("")
            for pdf in sorted(stats.pdfs_no_results):
                report.append(f"- `{pdf}`")
            report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")

    recommendations = []

    if total_failed > 0:
        recommendations.append(f"- **Re-extract {total_failed} failed PDF(s)**: Review failures above and re-run extraction")

    if total_no_results > 0:
        recommendations.append(f"- **Investigate {total_no_results} PDF(s) with no results**: May be non-lab documents or require different processing")

    if fallback_count > text_count * 0.2:  # More than 20% fallbacks
        recommendations.append("- **High TEXT→VISION fallback rate**: Consider adjusting text extraction threshold or PDF preprocessing")

    if stats.validation_flagged > 0:
        recommendations.append(f"- **Review {stats.validation_flagged} flagged rows**: Use `python viewer.py --profile {profile_name}` to review")

    if not recommendations:
        recommendations.append("- No significant issues detected")

    for rec in recommendations:
        report.append(rec)
    report.append("")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Analyze extraction logs')
    parser.add_argument('profile', help='Profile name (e.g., "tsilva")')
    parser.add_argument('--output', help='Output markdown file (default: logs_report.md)')
    args = parser.parse_args()

    # Load profile
    profile = load_profile(args.profile)
    output_path = Path(profile['output_path'])

    # Log file paths
    log_dir = output_path / 'logs'
    info_log = log_dir / 'info.log'
    error_log = log_dir / 'error.log'

    # Check if logs exist
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        print("Run extraction first to generate logs.")
        sys.exit(1)

    if not info_log.exists() and not error_log.exists():
        print(f"Error: No log files found in {log_dir}")
        print("Run extraction first to generate logs.")
        sys.exit(1)

    # Parse logs
    stats = LogStats()

    if info_log.exists():
        parse_log_file(info_log, stats, is_error_log=False)
        print(f"Parsed info.log: {info_log.stat().st_size / 1024:.1f} KB")

    if error_log.exists():
        parse_log_file(error_log, stats, is_error_log=True)
        print(f"Parsed error.log: {error_log.stat().st_size / 1024:.1f} KB")

    # Generate report
    report = generate_markdown_report(args.profile, stats, output_path)

    # Output
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = output_path / 'logs_report.md'

    try:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\n✓ Log analysis report saved: {output_file}")
    except PermissionError:
        # Fall back to local file if output path is not writable
        fallback_file = Path(f"logs_report_{args.profile}.md")
        with open(fallback_file, 'w') as f:
            f.write(report)
        print(f"\n✓ Log analysis report saved: {fallback_file}")
        print(f"  (Could not write to {output_file} due to permissions)")

    # Also print to console
    print("\n" + "="*60)
    print(report)


if __name__ == '__main__':
    main()
