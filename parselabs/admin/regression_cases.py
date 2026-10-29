"""Sync approved regression fixtures from reviewed processed documents."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from parselabs.config import LabSpecsConfig, ProfileConfig
from parselabs.exceptions import ConfigurationError
from parselabs.regression import (
    REVIEW_STATE_JSON_NAME,
    SOURCE_REVIEW_JSON_NAME,
    discover_approved_cases,
    export_fingerprint,
    get_approved_fixtures_dir,
    require_source_review,
    source_fingerprints,
    write_canonical_csv,
)
from parselabs.rows import (
    _rebuild_document_csv_with_review_dataframe,
    build_document_expected_dataframe,
    build_review_corpus_report,
    get_document_review_summary,
    iter_processed_documents,
    load_document_review_rows,
)
from parselabs.runtime import RuntimeContext
from parselabs.store import compute_file_hash, get_document_stem
from parselabs.utils import make_path_writable

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Sync approved regression fixtures from reviewed processed documents")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync-reviewed", help="Copy fixture-ready processed documents into regression fixtures")
    sync_parser.add_argument("--profile", "-p", required=True, help="Profile name used to locate the processed output directory")
    sync_selection = sync_parser.add_mutually_exclusive_group()
    sync_selection.add_argument("--source-file", action="append", help="Promote only this unambiguous PDF filename (repeatable)")
    sync_selection.add_argument("--document-key", action="append", help="Exact hashed processed-document directory name (repeatable)")
    sync_parser.add_argument("--replace", action="store_true", help="Explicitly replace existing golden cases after a fresh source and export audit")

    attest_parser = subparsers.add_parser("attest-reviewed", help="Record a completed independent source audit; does not perform the audit")
    attest_parser.add_argument("--profile", "-p", required=True)
    attest_selection = attest_parser.add_mutually_exclusive_group(required=True)
    attest_selection.add_argument("--source-file", help="Unambiguous PDF filename audited on every page")
    attest_selection.add_argument("--document-key", help="Exact hashed processed-document directory name audited on every page")
    attest_parser.add_argument("--review-note", required=True, help="Evidence and outcome of the source and normalized-export audit")

    report_parser = subparsers.add_parser("report", help="Summarize reviewed-corpus accuracy signals")
    report_parser.add_argument("--profile", "-p", required=True, help="Profile name used to locate the processed output directory")
    return parser.parse_args(argv)


def load_profile(profile_name: str) -> ProfileConfig:
    """Load a configured profile by name."""

    try:
        return RuntimeContext.from_profile(
            profile_name,
            need_input=False,
            need_output=True,
            need_api=False,
            setup_logs=False,
        ).profile
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc


def approve_cases(args: argparse.Namespace) -> None:
    """Sync reviewed processed documents into approved regression fixtures."""

    profile = load_profile(args.profile)
    lab_specs = LabSpecsConfig()
    fixtures_dir = get_approved_fixtures_dir()
    fixtures_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    fixtures_dir.chmod(0o700)
    selected = set(getattr(args, "source_file", None) or [])
    document_keys = set(getattr(args, "document_key", None) or [])
    documents = _select_documents(profile, selected, document_keys)

    # Refresh every processed document CSV before deciding whether it is a valid fixture.
    valid_documents_by_stem: dict[str, tuple[Path, Path, pd.DataFrame]] = {}
    for document in documents:
        _, review_df = _rebuild_document_csv_with_review_dataframe(document.doc_dir, lab_specs)
        summary = get_document_review_summary(document.doc_dir, review_df)

        # Skip documents that still have pending rows or unresolved missing-row markers.
        if not summary.fixture_ready:
            continue

        expected_df = build_document_expected_dataframe(document.doc_dir, lab_specs)

        # Existing accept/reject decisions require a separate, current source and export audit.
        try:
            require_source_review(document.doc_dir, document.source_pdf, expected_df)
        except ValueError as exc:
            logger.info("Skipped document without current source validation: %s", exc)
            continue

        case_key = f"{document.stem}_{compute_file_hash(document.source_pdf)}"
        valid_documents_by_stem[case_key] = (document.doc_dir, document.source_pdf, expected_df)

    # A requested promotion must not report success when its review is incomplete.
    if (selected or document_keys) and len(valid_documents_by_stem) != len(documents):
        raise ValueError("Selected documents are not all source-validated and fixture-ready; nothing was promoted.")

    existing_cases = {
        f"{case.stem}_{case.file_hash}": case
        for case in discover_approved_cases()
        if case.profile == args.profile
    }

    _check_replacements(existing_cases, valid_documents_by_stem, bool(getattr(args, "replace", False)))
    synced_count = _write_valid_cases(args.profile, valid_documents_by_stem, existing_cases, fixtures_dir)

    logger.info(
        f"Synced {synced_count} fully valid processed document(s) for profile '{args.profile}'"
        "; preserved existing cases outside this promotion."
    )


def _select_documents(profile: ProfileConfig, source_files: set[str], document_keys: set[str]) -> list:
    """Resolve exact source identities before touching derived files or golden baselines."""

    documents = iter_processed_documents(profile.output_path)

    # Hashed keys distinguish different PDFs that share a filename.
    if document_keys:
        chosen = [doc for doc in documents if doc.doc_dir.name in document_keys]
        if {doc.doc_dir.name for doc in chosen} != document_keys:
            raise ValueError("Some selected document keys were not found in processed output.")
        return chosen

    # An unfiltered promotion can include all audited content variants.
    if not source_files:
        return documents

    chosen = [doc for doc in documents if doc.source_pdf.name in source_files]

    # Missing filenames must fail rather than silently omit requested reports.
    if {doc.source_pdf.name for doc in chosen} != source_files:
        raise ValueError("Some selected source files were not found in processed output.")

    # A filename alone cannot identify two different content hashes safely.
    if len(chosen) != len(source_files):
        raise ValueError("Selected filename is ambiguous; use --document-key to select each hashed document explicitly.")

    return chosen


def _check_replacements(existing_cases: dict, documents: dict, replace: bool) -> None:
    """Keep frozen golden truth stable unless baseline replacement is explicit."""

    # Explicit replacement is checked against current source attestations by the caller.
    if replace:
        return

    for stem, (doc_dir, _, expected_df) in documents.items():
        previous = existing_cases.get(stem)

        # New cases do not replace any existing baseline.
        if previous is None:
            continue

        previous_review = previous.case_dir / SOURCE_REVIEW_JSON_NAME
        current_review = json.loads((doc_dir / SOURCE_REVIEW_JSON_NAME).read_text(encoding="utf-8"))
        previous_payload = json.loads(previous_review.read_text(encoding="utf-8")) if previous_review.exists() else {}

        # Identical source and export content makes repeat promotion idempotent.
        if previous_payload.get("fingerprints") == current_review["fingerprints"] and previous_payload.get("expected_export_sha256") == export_fingerprint(expected_df):
            continue

        raise ValueError("An existing golden case changed. Audit it and use --replace to intentionally update the baseline.")


def _write_valid_cases(
    profile_name: str,
    valid_documents_by_stem: dict[str, tuple[Path, Path, pd.DataFrame]],
    existing_cases: dict,
    fixtures_dir: Path,
) -> int:
    """Write approved fixtures for every fully accepted processed document."""

    approved_at = datetime.now(timezone.utc).isoformat()
    synced_count = 0

    # Copy each valid processed document into the private approved-fixture corpus.
    for document_key, (doc_dir, pdf_path, expected_df) in sorted(valid_documents_by_stem.items()):
        stem = get_document_stem(doc_dir)
        file_hash = compute_file_hash(pdf_path)
        case_id = f"{stem}_{file_hash}"
        existing_case = existing_cases.get(document_key)

        # Keep legacy cases in place and preserve other PDFs with the same filename.
        case_dir = existing_case.case_dir if existing_case else fixtures_dir / profile_name / case_id
        case_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        fixture_pdf_path = case_dir / "document.pdf"
        if fixture_pdf_path.exists():
            _prepare_fixture_pdf_for_replacement(fixture_pdf_path)
            fixture_pdf_path.unlink()
        fixture_pdf_path.write_bytes(pdf_path.read_bytes())
        fixture_pdf_path.chmod(0o600)
        write_canonical_csv(expected_df, case_dir / "expected.csv")
        _write_review_state_snapshot(doc_dir=doc_dir, case_dir=case_dir)

        # Preserve frozen page JSON for fully local regression without extraction API calls.
        snapshot_dir = case_dir / "reviewed"
        snapshot_dir.mkdir(exist_ok=True, mode=0o700)
        for stale_page in snapshot_dir.glob("*.json"):
            stale_page.unlink()
        for page_path in doc_dir.glob("*.json"):
            shutil.copyfile(page_path, snapshot_dir / page_path.name)
        shutil.copyfile(doc_dir / SOURCE_REVIEW_JSON_NAME, case_dir / SOURCE_REVIEW_JSON_NAME)

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

        # All medical artifacts are owner-only, even on machines with permissive umasks.
        for artifact in case_dir.rglob("*"):
            artifact.chmod(0o700 if artifact.is_dir() else 0o600)
        synced_count += 1

    return synced_count


def _write_review_state_snapshot(doc_dir: Path, case_dir: Path) -> None:
    """Persist reviewed row decisions so approved regressions can replay them."""

    review_rows = load_document_review_rows(doc_dir, include_statuses={"accepted", "rejected"})
    snapshot_rows: list[dict[str, int | str]] = []

    # Keep only the stable row identity plus the reviewed decision for regression replay.
    for row in review_rows.to_dict("records"):
        snapshot_rows.append(
            {
                "page_number": int(row["page_number"]),
                "result_index": int(row["result_index"]),
                "review_status": str(row["review_status"]),
            }
        )

    snapshot_payload = {"rows": snapshot_rows}
    snapshot_path = case_dir / REVIEW_STATE_JSON_NAME
    snapshot_path.write_text(json.dumps(snapshot_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _prepare_fixture_pdf_for_replacement(fixture_pdf_path: Path) -> None:
    """Clear restrictive flags and modes so an existing fixture PDF can be replaced."""

    make_path_writable(fixture_pdf_path, 0o644)


def attest_reviewed(args: argparse.Namespace) -> None:
    """Record the caller's completed source audit, bound to immutable input hashes."""

    profile = load_profile(args.profile)
    source_file = getattr(args, "source_file", None)
    document_key = getattr(args, "document_key", None)
    matches = _select_documents(profile, {source_file} if source_file else set(), {document_key} if document_key else set())

    # An attestation must identify exactly one fully reviewed document.
    if len(matches) != 1 or not args.review_note.strip():
        raise ValueError("Select exactly one processed PDF and provide a nonempty review note.")

    document = matches[0]
    expected_df = build_document_expected_dataframe(document.doc_dir, LabSpecsConfig())
    evidence = {
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "note": args.review_note.strip(),
        "checks": dict.fromkeys(("source_fidelity", "page_completeness", "patient_identity", "dates", "normalized_export"), True),
        "fingerprints": source_fingerprints(document.doc_dir, document.source_pdf),
        "expected_export_sha256": export_fingerprint(expected_df),
    }
    path = document.doc_dir / SOURCE_REVIEW_JSON_NAME
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    path.chmod(0o600)
    logger.info("Recorded source-review attestation. This command records the audit; it does not validate the source itself.")


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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    if args.command == "sync-reviewed":
        approve_cases(args)
        return 0

    if args.command == "report":
        report_review_corpus(args)
        return 0

    # Record explicit source-review evidence separately from fixture promotion.
    if args.command == "attest-reviewed":
        attest_reviewed(args)
        return 0

    raise RuntimeError(f"Unsupported regression command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
