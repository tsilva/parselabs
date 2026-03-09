"""Manage private approved-document regression fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parselabs.config import ExtractionConfig, LabSpecsConfig, ProfileConfig  # noqa: E402
from parselabs.regression import (  # noqa: E402
    APPROVED_FIXTURES_DIR,
    ApprovedCase,
    discover_approved_cases,
    empty_export_dataframe,
    get_required_regression_profile,
    split_final_output_by_stem,
    write_canonical_csv,
)
from parselabs.utils import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Manage approved document regression cases")

    subparsers = parser.add_subparsers(dest="command", required=True)
    approve_parser = subparsers.add_parser("approve", help="Add/update approved document regression cases")
    approve_parser.add_argument("--profile", "-p", required=True, help="Profile name used to locate source PDFs")

    selection_group = approve_parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument(
        "--pattern",
        help="Glob pattern relative to the profile input path, e.g. '2024-*.pdf'",
    )
    selection_group.add_argument(
        "--files",
        nargs="+",
        help="Explicit PDF filenames relative to the profile input path, or absolute paths",
    )
    return parser.parse_args()


def load_profile(profile_name: str) -> ProfileConfig:
    """Load a configured profile by name."""

    try:
        profile = get_required_regression_profile(profile_name)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    if not profile.input_path:
        raise SystemExit(f"Profile '{profile_name}' has no input_path defined.")
    if not profile.input_path.exists():
        raise SystemExit(f"Input path does not exist for profile '{profile_name}': {profile.input_path}")
    return profile


def resolve_selected_pdfs(profile: ProfileConfig, args: argparse.Namespace) -> list[Path]:
    """Resolve selected PDF files for approval."""

    if args.pattern:
        pdfs = sorted(path for path in profile.input_path.glob(args.pattern) if path.is_file())
    else:
        pdfs = []
        for raw_path in args.files:
            candidate = Path(raw_path)
            pdf_path = candidate if candidate.is_absolute() else profile.input_path / candidate
            if not pdf_path.exists():
                raise SystemExit(f"Selected PDF not found: {pdf_path}")
            if not pdf_path.is_file():
                raise SystemExit(f"Selected path is not a file: {pdf_path}")
            pdfs.append(pdf_path)
        pdfs = sorted({path.resolve(): path.resolve() for path in pdfs}.values())

    if not pdfs:
        selector = args.pattern or ", ".join(args.files)
        raise SystemExit(f"No PDF files matched the selection: {selector}")

    stems = [path.stem for path in pdfs]
    duplicate_stems = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicate_stems:
        stems_str = ", ".join(duplicate_stems)
        raise SystemExit(f"Selected PDFs must have unique stems. Duplicates: {stems_str}")

    return pdfs


def compute_file_hash(file_path: Path, hash_length: int = 8) -> str:
    """Compute a short SHA-256 hash for the fixture case id."""

    hasher = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:hash_length]


def approve_cases(args: argparse.Namespace) -> None:
    """Create/update approved cases and rebuild all expected CSV baselines."""

    profile = load_profile(args.profile)
    selected_pdfs = resolve_selected_pdfs(profile, args)
    APPROVED_FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    existing_cases = {case.stem: case for case in discover_approved_cases()}
    selected_by_stem = {pdf_path.stem: pdf_path for pdf_path in selected_pdfs}

    for stem, pdf_path in selected_by_stem.items():
        file_hash = compute_file_hash(pdf_path)
        case_id = f"{stem}_{file_hash}"
        old_case = existing_cases.get(stem)
        if old_case and old_case.case_id != case_id and old_case.case_dir.exists():
            shutil.rmtree(old_case.case_dir)

        case_dir = APPROVED_FIXTURES_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, case_dir / "document.pdf")
        (case_dir / "expected.csv").write_text("", encoding="utf-8")
        (case_dir / "case.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "original_filename": pdf_path.name,
                    "stem": stem,
                    "file_hash": file_hash,
                    "profile": args.profile,
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )

    cases = discover_approved_cases()
    actual_by_stem = rebuild_expected_outputs(cases)
    approved_at = datetime.now(timezone.utc).isoformat()

    for case in cases:
        case_df = actual_by_stem.get(case.stem, empty_export_dataframe())
        write_canonical_csv(case_df, case.expected_csv_path)

        selected_pdf = selected_by_stem.get(case.stem)
        original_filename = selected_pdf.name if selected_pdf else case.original_filename
        profile_name = args.profile if selected_pdf else case.profile
        file_hash = compute_file_hash(case.document_path)
        metadata = {
            "case_id": case.case_id,
            "original_filename": original_filename,
            "stem": case.stem,
            "file_hash": file_hash,
            "profile": profile_name,
            "approved_at": approved_at,
            "model_id": get_required_regression_profile(profile_name).extract_model_id,
        }
        case.metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    logger.info(f"Approved {len(selected_pdfs)} PDF(s); rebuilt baselines for {len(cases)} approved case(s).")


def rebuild_expected_outputs(cases: list[ApprovedCase]) -> dict[str, object]:
    """Run approved cases grouped by profile and return final output split by stem."""

    try:
        from main import build_final_output_dataframe
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Approval workflow requires the extraction runtime dependencies: {exc}") from exc

    cases_by_profile: dict[str, list[ApprovedCase]] = {}
    for case in cases:
        profile_name = case.profile
        if not profile_name:
            raise SystemExit(f"Approved case '{case.case_id}' is missing its profile metadata.")
        cases_by_profile.setdefault(profile_name, []).append(case)

    actual_by_stem: dict[str, object] = {}
    for profile_name, profile_cases in cases_by_profile.items():
        profile = get_required_regression_profile(profile_name)
        with tempfile.TemporaryDirectory(prefix=f"approved-docs-{profile_name}-") as temp_dir_name:
            temp_root = Path(temp_dir_name)
            input_dir = temp_root / "input"
            output_dir = temp_root / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            pdf_files: list[Path] = []
            for case in profile_cases:
                target_pdf = input_dir / f"{case.stem}.pdf"
                shutil.copy2(case.document_path, target_pdf)
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
            final_df = build_final_output_dataframe(pdf_files, config, lab_specs)
            actual_by_stem.update(split_final_output_by_stem(final_df))

    return actual_by_stem


def main() -> None:
    """CLI entry point."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if args.command == "approve":
        approve_cases(args)


if __name__ == "__main__":
    main()
