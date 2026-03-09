from __future__ import annotations

import os
import shutil

import pytest

from parselabs.config import ExtractionConfig, LabSpecsConfig
from parselabs.regression import (
    build_case_diff,
    discover_approved_cases,
    empty_export_dataframe,
    get_required_regression_env,
    split_final_output_by_stem,
)
from parselabs.utils import setup_logging


@pytest.fixture(scope="module")
def approved_regression_run(tmp_path_factory):
    if os.getenv("RUN_APPROVED_DOCS") != "1":
        pytest.skip("Set RUN_APPROVED_DOCS=1 to run approved document regressions.")

    cases = discover_approved_cases()
    if not cases:
        pytest.fail("RUN_APPROVED_DOCS=1 is set, but no approved cases were found under tests/fixtures/approved/.")

    try:
        env = get_required_regression_env()
    except RuntimeError as exc:
        pytest.fail(str(exc))

    temp_root = tmp_path_factory.mktemp("approved-docs")
    input_dir = temp_root / "input"
    output_dir = temp_root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = []
    for case in cases:
        target_pdf = input_dir / f"{case.stem}.pdf"
        shutil.copy2(case.document_path, target_pdf)
        pdf_files.append(target_pdf)

    setup_logging(output_dir / "logs", clear_logs=True)
    lab_specs = LabSpecsConfig()
    config = ExtractionConfig(
        input_path=input_dir,
        output_path=output_dir,
        openrouter_api_key=env["OPENROUTER_API_KEY"],
        extract_model_id=env["EXTRACT_MODEL_ID"],
        input_file_regex="*.pdf",
        max_workers=1,
    )

    try:
        from main import build_final_output_dataframe
    except ModuleNotFoundError as exc:
        pytest.fail(f"Approved document regressions require the extraction runtime dependencies: {exc}")

    final_df = build_final_output_dataframe(pdf_files, config, lab_specs)
    return cases, split_final_output_by_stem(final_df)


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
