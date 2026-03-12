from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from parselabs.config import ExtractionConfig, LabSpecsConfig
from parselabs.regression import (
    build_case_diff,
    discover_approved_cases,
    empty_export_dataframe,
    get_required_regression_profile,
)
from parselabs.review_sync import build_document_expected_dataframe_from_reviewed_json, save_review_status
from parselabs.utils import setup_logging


@pytest.fixture(scope="module")
def approved_regression_run(tmp_path_factory):
    if os.getenv("RUN_APPROVED_DOCS") != "1":
        pytest.skip("Set RUN_APPROVED_DOCS=1 to run approved document regressions.")

    cases = discover_approved_cases()
    if not cases:
        pytest.fail("RUN_APPROVED_DOCS=1 is set, but no approved cases were found under tests/fixtures/approved/.")

    try:
        from main import run_pipeline_for_pdf_files
    except ModuleNotFoundError as exc:
        pytest.fail(f"Approved document regressions require the extraction runtime dependencies: {exc}")

    cases_by_profile = {}
    for case in cases:
        profile_name = case.profile
        if not profile_name:
            pytest.fail(f"Approved case '{case.case_id}' is missing its profile metadata.")
        cases_by_profile.setdefault(profile_name, []).append(case)

    actual_by_stem = {}
    for profile_name, profile_cases in cases_by_profile.items():
        try:
            profile = get_required_regression_profile(profile_name)
        except RuntimeError as exc:
            pytest.fail(str(exc))

        temp_root = tmp_path_factory.mktemp(f"approved-docs-{profile_name}")
        input_dir = temp_root / "input"
        output_dir = temp_root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = []
        for case in profile_cases:
            target_pdf = input_dir / f"{case.stem}.pdf"
            target_pdf.write_bytes(case.document_path.read_bytes())
            pdf_files.append(target_pdf)

        setup_logging(output_dir / "logs", clear_logs=True)
        lab_specs = LabSpecsConfig()
        config = ExtractionConfig(
            input_path=input_dir,
            output_path=output_dir,
            openrouter_api_key=profile.openrouter_api_key,
            openrouter_base_url=profile.openrouter_base_url or "https://openrouter.ai/api/v1",
            extract_model_id=profile.extract_model_id,
            input_file_regex="*.pdf",
            max_workers=1,
        )
        run_pipeline_for_pdf_files(pdf_files, config, lab_specs)

        # Replay the approved review decisions onto the fresh extraction before rebuilding reviewed truth.
        for case in profile_cases:
            doc_dir = output_dir / case.case_id

            # Guard: Each rerun fixture should recreate its own processed document directory.
            if not doc_dir.exists():
                pytest.fail(f"Approved case '{case.case_id}' did not recreate processed output at {doc_dir}.")

            _apply_review_state(case.review_state_path, doc_dir)
            actual_by_stem[case.stem] = build_document_expected_dataframe_from_reviewed_json(doc_dir, lab_specs)

    return cases, actual_by_stem


def _apply_review_state(review_state_path: Path | None, doc_dir: Path) -> None:
    """Replay approved review decisions onto a freshly extracted processed document."""

    # Guard: Approved fixtures must carry review decisions now that exports are review-driven.
    if review_state_path is None or not review_state_path.exists():
        pytest.fail(
            "Approved fixture is missing review_state.json. "
            "Resync fixtures with `uv run python utils/regression_cases.py sync-reviewed --profile ...`."
        )

    snapshot_payload = json.loads(review_state_path.read_text(encoding="utf-8"))
    snapshot_rows = snapshot_payload.get("rows", [])

    # Guard: Malformed snapshots should fail loudly so the fixture can be repaired.
    if not isinstance(snapshot_rows, list):
        pytest.fail(f"Approved fixture review state is malformed: {review_state_path}")

    # Replay each reviewed decision using the same persistence helper as the reviewer UI.
    for snapshot_row in snapshot_rows:
        page_number = int(snapshot_row["page_number"])
        result_index = int(snapshot_row["result_index"])
        review_status = str(snapshot_row["review_status"])
        ok, message = save_review_status(doc_dir, page_number, result_index, review_status)

        # Guard: A replay failure means extraction drifted or the snapshot no longer matches the document.
        if not ok:
            pytest.fail(
                f"Failed to replay review state for {doc_dir.name} page {page_number} row {result_index}: {message}"
            )


@pytest.mark.approved_docs
def test_approved_documents_match_expected_csv(approved_regression_run):
    cases, actual_by_stem = approved_regression_run

    diffs = []
    for case in cases:
        expected_df = empty_export_dataframe() if case.expected_csv_path.stat().st_size == 0 else None
        if expected_df is None:
            import pandas as pd

            expected_df = pd.read_csv(case.expected_csv_path, keep_default_na=False)
        actual_df = actual_by_stem.get(case.stem, empty_export_dataframe())
        diff = build_case_diff(expected_df, actual_df, case.case_id)
        if diff:
            diffs.append(diff)

    assert not diffs, "\n\n".join(diffs)
