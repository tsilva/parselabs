from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

import pytest

from parselabs.config import ExtractionConfig, LabSpecsConfig
from parselabs.regression import (
    build_case_diff,
    build_fresh_case_diff,
    discover_approved_cases,
    empty_export_dataframe,
    get_regression_runs_dir,
    get_required_regression_profile,
    private_regression_enabled,
    replay_review_decisions,
    require_source_review,
    select_regression_cases,
)
from parselabs.rows import build_document_expected_dataframe_from_reviewed_json
from parselabs.utils import setup_logging


def _close_root_logging_handlers() -> None:
    """Flush and close handlers before protecting retained private diagnostics."""

    root_logger = logging.getLogger()

    # Detach and close every root handler because approved-doc runs log into temp directories.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    # Flush any remaining logging state held by the stdlib logging module.
    logging.shutdown()


def _collect_profile_results(cases, output_dir, lab_specs):
    """Report all document-level mismatches without positional review replay."""

    actual = {}
    failures = []
    for case in cases:
        doc_dir = output_dir / f"{case.stem}_{case.file_hash}"

        # A missing document must not prevent exercising the remaining selected cases.
        if not doc_dir.exists():
            failures.append(f"{case.profile}/{case.case_id}: missing processed document. Artifacts: {output_dir}")
            continue

        # Match every raw observation before evaluating accepted-only exports.
        try:
            replay_review_decisions(case.case_dir / "reviewed", doc_dir)
            actual[(case.profile, case.case_id)] = build_document_expected_dataframe_from_reviewed_json(doc_dir, lab_specs)
        except ValueError as exc:
            failures.append(f"{case.profile}/{case.case_id}: {exc}. Artifacts: {doc_dir}")
    return actual, failures


@pytest.fixture(scope="module")
def approved_regression_run(pytestconfig):
    if not private_regression_enabled("RUN_APPROVED_DOCS"):
        pytest.skip("Set RUN_APPROVED_DOCS=1 to run approved document regressions.")

    cases = select_regression_cases(discover_approved_cases(), pytestconfig.getoption("--golden-suite"))
    if not cases:
        pytest.fail("RUN_APPROVED_DOCS=1 is set, but the private local approved corpus is empty.")

    # Verify frozen evidence before loading profiles or sending any source to the configured model.
    import pandas as pd

    for case in cases:
        expected = pd.read_csv(case.expected_csv_path, keep_default_na=False, dtype=str)
        with tempfile.TemporaryDirectory(prefix="parselabs-preflight-") as directory:
            doc_dir = Path(directory) / case.case_id
            shutil.copytree(case.case_dir / "reviewed", doc_dir)
            pdf_path = doc_dir / case.original_filename
            shutil.copyfile(case.document_path, pdf_path)
            require_source_review(doc_dir, pdf_path, expected)

    try:
        from parselabs.pipeline import run_pipeline_for_pdf_files
    except ModuleNotFoundError as exc:
        pytest.fail(f"Approved document regressions require the extraction runtime dependencies: {exc}")

    cases_by_profile = {}
    for case in cases:
        profile_name = case.profile
        if not profile_name:
            pytest.fail(f"Approved case '{case.case_id}' is missing its profile metadata.")
        cases_by_profile.setdefault(profile_name, []).append(case)

    actual_by_stem = {}
    failures = []
    for profile_name, profile_cases in cases_by_profile.items():
        try:
            profile = get_required_regression_profile(profile_name)
        except RuntimeError as exc:
            pytest.fail(str(exc))

        temp_root = Path(tempfile.mkdtemp(prefix="extraction-", dir=get_regression_runs_dir()))
        input_dir = temp_root / "input"
        output_dir = temp_root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = []
        for case in profile_cases:
            case_input_dir = input_dir / case.case_id
            case_input_dir.mkdir()
            target_pdf = case_input_dir / f"{case.stem}.pdf"
            target_pdf.write_bytes(case.document_path.read_bytes())
            pdf_files.append(target_pdf)

        setup_logging(output_dir / "logs", clear_logs=True)
        lab_specs = LabSpecsConfig()
        config = ExtractionConfig(
            input_path=input_dir,
            output_path=output_dir,
            openrouter_api_key=profile.openrouter_api_key,
            openrouter_base_url=profile.openrouter_base_url or "http://127.0.0.1:8082/api/v1",
            extract_model_id=profile.extract_model_id,
            input_file_regex="*.pdf",
            max_workers=2 if pytestconfig.getoption("--golden-suite") == "quick" else 1,
            expected_patient_name=profile.name,
        )
        try:
            result = run_pipeline_for_pdf_files(pdf_files, config, lab_specs)

            # A partial extraction cannot establish a clean regression result.
            if result.failed_pages or result.pdfs_failed:
                pytest.fail("Private regression extraction was incomplete; inspect local diagnostics.")

            # Independent cases continue after mismatches so all selected profiles are exercised.
            profile_actual, profile_failures = _collect_profile_results(profile_cases, output_dir, lab_specs)
            actual_by_stem.update(profile_actual)
            failures.extend(profile_failures)
        finally:
            # Retain private diagnostics for fresh extraction; approved fixtures are never modified.
            _close_root_logging_handlers()
            for path in temp_root.rglob("*"):
                path.chmod(0o700 if path.is_dir() else 0o600)

    return cases, actual_by_stem, failures


@pytest.mark.approved_docs
def test_approved_documents_match_expected_csv(approved_regression_run, pytestconfig):
    cases, actual_by_stem, failures = approved_regression_run

    diffs = list(failures)
    for case in cases:
        # Failed matching already has an actionable local diagnostic; do not add an empty-CSV diff.
        if (case.profile, case.case_id) not in actual_by_stem:
            continue
        expected_df = empty_export_dataframe() if case.expected_csv_path.stat().st_size == 0 else None
        if expected_df is None:
            import pandas as pd

            expected_df = pd.read_csv(case.expected_csv_path, keep_default_na=False, dtype=str)
        actual_df = actual_by_stem.get((case.profile, case.case_id), empty_export_dataframe())
        diff = build_fresh_case_diff(expected_df, actual_df, case.case_id)
        if diff:
            diffs.append(diff)

    # Keep actionable diagnostics beside the local corpus, never in Git or CI artifacts.
    report = get_regression_runs_dir() / "latest-result.json"
    report.write_text(json.dumps({"suite": pytestconfig.getoption("--golden-suite"), "documents": len(cases), "passed": not diffs, "differences": diffs}, indent=2) + "\n", encoding="utf-8")
    report.chmod(0o600)
    assert not diffs, "\n\n".join(diffs)


@pytest.mark.golden_local
def test_local_golden_reviewed_json_matches_expected_csv(monkeypatch, pytestconfig):
    """Exercise normalization and export against frozen private parses without any API."""

    # Private corpora are never required or accessed by ordinary CI runs.
    if not private_regression_enabled("RUN_LOCAL_GOLDENS"):
        pytest.skip("Set RUN_LOCAL_GOLDENS=1 to replay the private local golden corpus.")

    cases = select_regression_cases(discover_approved_cases(), pytestconfig.getoption("--golden-suite"))

    # Explicit local regression must fail rather than silently pass an empty corpus.
    if not cases:
        pytest.fail("RUN_LOCAL_GOLDENS=1 is set, but the private local golden corpus is empty.")

    import socket

    import pandas as pd

    # Enforce the local-only contract even if a future normalization change tries an API.
    def reject_network(*args, **kwargs):
        raise AssertionError("Offline golden regression must not access the network.")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    lab_specs = LabSpecsConfig()
    diffs = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="parselabs-golden-") as temp_dir:
            doc_dir = Path(temp_dir) / f"{case.stem}_{case.file_hash}"
            shutil.copytree(case.case_dir / "reviewed", doc_dir)
            pdf_path = doc_dir / case.original_filename
            shutil.copyfile(case.document_path, pdf_path)
            expected = pd.read_csv(case.expected_csv_path, keep_default_na=False, dtype=str)
            require_source_review(doc_dir, pdf_path, expected)
            actual = build_document_expected_dataframe_from_reviewed_json(doc_dir, lab_specs)
            diff = build_case_diff(expected, actual, f"{case.profile}/{case.case_id}")

            # Accumulate all drift so a local run reports the complete affected corpus.
            if diff:
                diffs.append(diff)

    assert not diffs, "\n\n".join(diffs)
