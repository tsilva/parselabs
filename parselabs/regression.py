"""Helpers for approved-document regression fixtures and CSV comparison."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any

import pandas as pd

from parselabs.export_schema import COLUMN_ORDER, COLUMN_SCHEMA
from parselabs.utils import ensure_columns

APPROVED_FIXTURES_DIR = Path("tests/fixtures/approved")
CASE_JSON_NAME = "case.json"
EXPECTED_CSV_NAME = "expected.csv"
DOCUMENT_PDF_NAME = "document.pdf"
CANONICAL_SORT_COLUMNS = [
    "date",
    "lab_name",
    "page_number",
    "result_index",
    "raw_lab_name",
    "raw_value",
    "raw_unit",
]


@dataclass(frozen=True)
class ApprovedCase:
    """Single approved regression case."""

    case_id: str
    case_dir: Path
    document_path: Path
    expected_csv_path: Path
    metadata_path: Path
    stem: str
    file_hash: str
    original_filename: str
    profile: str | None
    metadata: dict[str, Any]


def get_required_regression_env() -> dict[str, str]:
    """Return required API env vars or raise a descriptive error."""

    required = {
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", "").strip(),
        "EXTRACT_MODEL_ID": os.getenv("EXTRACT_MODEL_ID", "").strip(),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(f"Missing required env vars for approved document regression: {missing_str}")
    return required


def empty_export_dataframe() -> pd.DataFrame:
    """Create an empty DataFrame in final export schema order."""

    return pd.DataFrame(columns=COLUMN_ORDER)


def discover_approved_cases(fixtures_dir: Path = APPROVED_FIXTURES_DIR) -> list[ApprovedCase]:
    """Discover approved regression fixtures and validate their shape."""

    if not fixtures_dir.exists():
        return []

    cases: list[ApprovedCase] = []
    seen_stems: dict[str, Path] = {}

    for case_dir in sorted(path for path in fixtures_dir.iterdir() if path.is_dir()):
        metadata_path = case_dir / CASE_JSON_NAME
        document_path = case_dir / DOCUMENT_PDF_NAME
        expected_csv_path = case_dir / EXPECTED_CSV_NAME

        missing = [path.name for path in (metadata_path, document_path, expected_csv_path) if not path.exists()]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Approved case '{case_dir.name}' is missing required files: {missing_str}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stem = str(metadata.get("stem", "")).strip()
        file_hash = str(metadata.get("file_hash", "")).strip()
        original_filename = str(metadata.get("original_filename", "")).strip()

        if not stem or not file_hash or not original_filename:
            raise ValueError(f"Approved case '{case_dir.name}' has incomplete metadata in {metadata_path}")

        previous_case_dir = seen_stems.get(stem)
        if previous_case_dir:
            raise ValueError(
                f"Approved cases must have unique stems. Stem '{stem}' appears in both "
                f"'{previous_case_dir.name}' and '{case_dir.name}'."
            )
        seen_stems[stem] = case_dir

        cases.append(
            ApprovedCase(
                case_id=case_dir.name,
                case_dir=case_dir,
                document_path=document_path,
                expected_csv_path=expected_csv_path,
                metadata_path=metadata_path,
                stem=stem,
                file_hash=file_hash,
                original_filename=original_filename,
                profile=metadata.get("profile"),
                metadata=metadata,
            )
        )

    return cases


def split_final_output_by_stem(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split merged final output into one DataFrame per document stem."""

    if df.empty or "source_file" not in df.columns:
        return {}

    docs: dict[str, pd.DataFrame] = {}
    for source_file, group in df.groupby("source_file", dropna=False, sort=False):
        source_name = str(source_file).strip()
        stem = Path(source_name).stem if source_name else ""
        if not stem:
            continue
        docs[stem] = group.copy()
    return docs


def canonicalize_export_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize final export output for exact CSV regression comparisons."""

    canonical = df.copy()
    ensure_columns(canonical, COLUMN_ORDER, default=None)
    canonical = canonical[COLUMN_ORDER].copy()

    if len(canonical) > 1:
        canonical = canonical.sort_values(CANONICAL_SORT_COLUMNS, kind="mergesort", na_position="first").reset_index(drop=True)
    else:
        canonical = canonical.reset_index(drop=True)

    for column in COLUMN_ORDER:
        dtype_name = COLUMN_SCHEMA[column]["dtype"]
        canonical[column] = canonical[column].map(lambda value, dtype_name=dtype_name: _normalize_cell(value, dtype_name))

    return canonical


def canonical_csv_text(df: pd.DataFrame) -> str:
    """Serialize canonicalized export rows to a stable CSV string."""

    canonical = canonicalize_export_df(df)
    return canonical.to_csv(index=False, lineterminator="\n")


def write_canonical_csv(df: pd.DataFrame, destination: Path) -> None:
    """Write canonical CSV output to disk."""

    destination.write_text(canonical_csv_text(df), encoding="utf-8")


def build_case_diff(expected_df: pd.DataFrame, actual_df: pd.DataFrame, case_id: str) -> str:
    """Build a unified diff message for a case comparison."""

    expected_text = canonical_csv_text(expected_df)
    actual_text = canonical_csv_text(actual_df)

    if expected_text == actual_text:
        return ""

    diff = "\n".join(
        unified_diff(
            expected_text.splitlines(),
            actual_text.splitlines(),
            fromfile=f"{case_id}/expected.csv",
            tofile=f"{case_id}/actual.csv",
            lineterm="",
        )
    )
    return f"Mismatch for approved case '{case_id}':\n{diff}"


def _normalize_cell(value: Any, dtype_name: str) -> str:
    """Normalize a single cell according to the final export schema."""

    if pd.isna(value):
        return ""

    if dtype_name == "datetime64[ns]":
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return str(value).strip()
        return parsed.strftime("%Y-%m-%d")

    if dtype_name == "float64":
        try:
            return format(float(value), ".15g")
        except (TypeError, ValueError):
            return str(value).strip()

    if dtype_name == "Int64":
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value).strip()

    if dtype_name == "boolean":
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1"}:
                return "true"
            if normalized in {"false", "0"}:
                return "false"
            return normalized
        return "true" if bool(value) else "false"

    return str(value).strip()
