"""Sync approved regression fixtures from reviewed processed documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parselabs.config import LabSpecsConfig, ProfileConfig  # noqa: E402
from parselabs.regression import (  # noqa: E402
    APPROVED_FIXTURES_DIR,
    discover_approved_cases,
    write_canonical_csv,
)
from parselabs.review_sync import (  # noqa: E402
    build_document_expected_dataframe,
    build_document_review_dataframe,
    build_review_corpus_report,
    get_document_review_summary,
    iter_processed_documents,
    rebuild_document_csv,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Sync approved regression fixtures from reviewed processed documents")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Keep the original approve command as a backward-compatible alias.
    approve_parser = subparsers.add_parser("approve", help="Backward-compatible alias for sync-reviewed")
    approve_parser.add_argument("--profile", "-p", required=True, help="Profile name used to locate the processed output directory")

    sync_parser = subparsers.add_parser("sync-reviewed", help="Copy fixture-ready processed documents into regression fixtures")
    sync_parser.add_argument("--profile", "-p", required=True, help="Profile name used to locate the processed output directory")

    report_parser = subparsers.add_parser("report", help="Summarize reviewed-corpus accuracy signals")
    report_parser.add_argument("--profile", "-p", required=True, help="Profile name used to locate the processed output directory")
    return parser.parse_args()


def load_profile(profile_name: str) -> ProfileConfig:
    """Load a configured profile by name."""

    profile_path = ProfileConfig.find_path(profile_name)

    # Guard: The requested profile must exist.
    if not profile_path:
        raise SystemExit(f"Profile '{profile_name}' was not found.")

    profile = ProfileConfig.from_file(profile_path)

    # Guard: The fixture sync operates on processed outputs, so output_path is required.
    if not profile.output_path:
        raise SystemExit(f"Profile '{profile_name}' has no output_path defined.")

    # Guard: The processed output directory must already exist.
    if not profile.output_path.exists():
        raise SystemExit(f"Output path does not exist for profile '{profile_name}': {profile.output_path}")

    return profile


def compute_file_hash(file_path: Path, hash_length: int = 8) -> str:
    """Compute a short SHA-256 hash for a fixture case id."""

    hasher = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:hash_length]


def approve_cases(args: argparse.Namespace) -> None:
    """Sync reviewed processed documents into approved regression fixtures."""

    profile = load_profile(args.profile)
    lab_specs = LabSpecsConfig()
    APPROVED_FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Refresh every processed document CSV before deciding whether it is a valid fixture.
    valid_documents_by_stem: dict[str, tuple[Path, Path, pd.DataFrame]] = {}
    for document in iter_processed_documents(profile.output_path):
        rebuild_document_csv(document.doc_dir, lab_specs)
        review_df = build_document_review_dataframe(document.doc_dir, lab_specs)
        summary = get_document_review_summary(document.doc_dir, review_df)

        # Skip documents that still have pending rows or unresolved missing-row markers.
        if not summary.fixture_ready:
            continue

        expected_df = build_document_expected_dataframe(document.doc_dir, lab_specs)
        valid_documents_by_stem[document.stem] = (document.doc_dir, document.pdf_path, expected_df)

    existing_cases = {
        case.stem: case
        for case in discover_approved_cases()
        if case.profile == args.profile
    }

    removed_count = _remove_stale_cases(existing_cases, valid_documents_by_stem)
    synced_count = _write_valid_cases(args.profile, valid_documents_by_stem, existing_cases)

    logger.info(
        f"Synced {synced_count} fully valid processed document(s) for profile '{args.profile}'"
        f"; removed {removed_count} stale fixture(s)."
    )


def _remove_stale_cases(existing_cases: dict, valid_documents_by_stem: dict[str, tuple[Path, Path, pd.DataFrame]]) -> int:
    """Remove fixture cases that are no longer fully valid in processed output."""

    removed_count = 0

    # Remove old fixtures for this profile when the processed document is missing or no longer fully accepted.
    for stem, existing_case in existing_cases.items():
        if stem in valid_documents_by_stem:
            continue

        shutil.rmtree(existing_case.case_dir)
        removed_count += 1

    return removed_count


def _write_valid_cases(
    profile_name: str,
    valid_documents_by_stem: dict[str, tuple[Path, Path, pd.DataFrame]],
    existing_cases: dict,
) -> int:
    """Write approved fixtures for every fully accepted processed document."""

    approved_at = datetime.now(timezone.utc).isoformat()
    synced_count = 0

    # Copy each valid processed document into the private approved-fixture corpus.
    for stem, (_, pdf_path, expected_df) in sorted(valid_documents_by_stem.items()):
        file_hash = compute_file_hash(pdf_path)
        case_id = f"{stem}_{file_hash}"
        existing_case = existing_cases.get(stem)

        # Remove a stale hash-specific case directory before writing the new one.
        if existing_case and existing_case.case_id != case_id and existing_case.case_dir.exists():
            shutil.rmtree(existing_case.case_dir)

        case_dir = APPROVED_FIXTURES_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, case_dir / "document.pdf")
        write_canonical_csv(expected_df, case_dir / "expected.csv")

        metadata = {
            "case_id": case_id,
            "original_filename": pdf_path.name,
            "stem": stem,
            "file_hash": file_hash,
            "profile": profile_name,
            "approved_at": approved_at,
            "reviewed_at": approved_at,
        }
        (case_dir / "case.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        synced_count += 1

    return synced_count


def report_review_corpus(args: argparse.Namespace) -> None:
    """Print a compact review-corpus report for benchmark-driven improvements."""

    profile = load_profile(args.profile)
    lab_specs = LabSpecsConfig()
    report = build_review_corpus_report(profile.output_path, lab_specs)

    logger.info(f"Documents: {report.document_count}")
    logger.info(f"Fixture-ready documents: {report.fixture_ready_document_count}")
    logger.info(f"Rejected extracted rows: {report.rejected_rows}")
    logger.info(f"Unresolved missing-row markers: {report.unresolved_missing_rows}")
    logger.info(f"Unknown standardized names: {report.unknown_standardized_names}")
    logger.info(f"Unknown standardized units: {report.unknown_standardized_units}")

    _log_ranked_counts("Validation reasons", report.validation_reason_counts)
    _log_ranked_counts("Rejected raw names", report.rejected_raw_name_counts)
    _log_ranked_counts("Rejected raw units", report.rejected_raw_unit_counts)


def _log_ranked_counts(title: str, counts: dict[str, int], limit: int = 10) -> None:
    """Log the top ranked counts from a corpus-level report section."""

    logger.info(f"{title}:")

    # Guard: Empty sections should still print a stable placeholder.
    if not counts:
        logger.info("  (none)")
        return

    for key, count in list(counts.items())[:limit]:
        logger.info(f"  {key}: {count}")


def main() -> None:
    """CLI entry point."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if args.command in {"approve", "sync-reviewed"}:
        approve_cases(args)
        return

    if args.command == "report":
        report_review_corpus(args)


if __name__ == "__main__":
    main()
