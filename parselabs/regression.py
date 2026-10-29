"""Helpers for approved-document regression fixtures and CSV comparison."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path

import pandas as pd

from parselabs.config import ProfileConfig
from parselabs.exceptions import ConfigurationError
from parselabs.export_schema import COLUMN_ORDER, COLUMN_SCHEMA
from parselabs.paths import get_project_root
from parselabs.runtime import load_profile_config
from parselabs.types import ApprovedCaseMetadata
from parselabs.utils import ensure_columns

APPROVED_FIXTURES_DIR = Path.home() / ".local" / "share" / "parselabs" / "approved"
CASE_JSON_NAME = "case.json"
EXPECTED_CSV_NAME = "expected.csv"
DOCUMENT_PDF_NAME = "document.pdf"
REVIEW_STATE_JSON_NAME = "review_state.json"
SOURCE_REVIEW_JSON_NAME = "source_review.json"
QUICK_SUITE_JSON_NAME = "quick-suite.json"
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
    review_state_path: Path | None
    stem: str
    file_hash: str
    original_filename: str
    profile: str | None
    metadata: ApprovedCaseMetadata


def get_required_regression_profile(profile_name: str | None) -> ProfileConfig:
    """Load a profile with the runtime settings required for regression runs."""

    if not profile_name:
        raise RuntimeError("Approved document regression cases must record the source profile.")

    try:
        profile = load_profile_config(profile_name)
    except ConfigurationError as exc:
        raise RuntimeError(f"Approved document regression profile '{profile_name}' was not found.") from exc

    missing = []
    if not profile.extract_model_id:
        missing.append("extract_model_id")
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(f"Profile '{profile_name}' is missing required regression settings: {missing_str}")
    return profile


def empty_export_dataframe() -> pd.DataFrame:
    """Create an empty DataFrame in final export schema order."""

    return pd.DataFrame(columns=COLUMN_ORDER)


def get_approved_fixtures_dir() -> Path:
    """Resolve the private local corpus, rejecting locations inside Git checkouts."""

    return _resolve_private_regression_path(Path(os.environ.get("PARSELABS_APPROVED_FIXTURES_DIR", str(APPROVED_FIXTURES_DIR))))


def _resolve_private_regression_path(path: Path) -> Path:
    """Reject private artifacts inside Git, including symlinked manifests."""

    path = path.expanduser().resolve()

    # Private medical fixtures must never be written beneath a Git working tree.
    if path.is_relative_to(get_project_root()) or any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("The private golden corpus must be outside every Git checkout.")

    return path


def source_fingerprints(doc_dir: Path, pdf_path: Path) -> dict[str, str]:
    """Bind a source review to the complete PDF and every stored page payload."""

    from parselabs.store import parse_page_number

    paths = [pdf_path, *sorted(path for path in doc_dir.glob("*.json") if parse_page_number(path) is not None)]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def private_regression_enabled(flag: str) -> bool:
    """Permit explicit private runs only outside continuous integration."""

    # Ordinary tests must not discover private files or require a corpus.
    if os.environ.get(flag) != "1":
        return False

    # Refuse before accessing patient data, even when CI accidentally sets the opt-in flag.
    ci_flags = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "TF_BUILD", "BUILDKITE", "CIRCLECI", "JENKINS_URL", "TEAMCITY_VERSION")
    if any(os.environ.get(key, "").strip().lower() not in {"", "0", "false", "no"} for key in ci_flags):
        raise ValueError("Private medical regression tests cannot run in CI. Run them locally.")

    return True


def select_regression_cases(cases: list[ApprovedCase], suite: str) -> list[ApprovedCase]:
    """Select a frozen, explicitly curated local subset without copying patient data."""

    # Full replay keeps its existing coverage and does not require a quick-suite manifest.
    if suite == "full":
        return cases

    # Reject misspelled suites instead of silently falling back to an unrelated selection.
    if suite != "quick":
        raise ValueError("Unknown golden suite; choose quick or full.")

    path = _resolve_private_regression_path(get_approved_fixtures_dir().parent / QUICK_SUITE_JSON_NAME)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("cases")

    # A curated selection must be deliberate, versioned and nonempty.
    if manifest.get("version") != 1 or not isinstance(entries, list) or not entries:
        raise ValueError("The private quick-suite manifest must contain version 1 and a nonempty cases list.")

    selected = _resolve_quick_suite_entries(cases, entries)
    _validate_quick_suite_coverage(manifest, selected)
    return selected


def _resolve_quick_suite_entries(cases: list[ApprovedCase], entries: list[dict]) -> list[ApprovedCase]:
    """Resolve pinned corpus identities, rejecting missing or changed source PDFs."""

    available = {(case.profile, case.case_id): case for case in cases}
    selected: list[ApprovedCase] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        identity = (entry.get("profile"), entry.get("case_id"))

        # Missing identities and duplicates must not silently shrink or inflate coverage.
        if identity not in available or identity in seen:
            raise ValueError("The private quick suite references a missing or duplicate approved case.")

        # Each selected layout needs an explicit curation rationale and coverage labels.
        if not entry.get("rationale") or not isinstance(entry.get("tags"), list) or not entry["tags"]:
            raise ValueError("Every quick-suite case needs a rationale and coverage tags.")

        case = available[identity]

        # Changes to a selected source require deliberate re-curation, never automatic replacement.
        if hashlib.sha256(case.document_path.read_bytes()).hexdigest() != entry.get("pdf_sha256"):
            raise ValueError("A private quick-suite PDF changed; review the curated selection again.")

        seen.add(identity)
        selected.append(case)
    return selected


def _validate_quick_suite_coverage(manifest: dict, cases: list[ApprovedCase]) -> None:
    """Enforce the curator's profile coverage and document/page latency budgets."""

    required_profiles = manifest.get("required_profiles")

    # A suite without explicit profile coverage could quietly test only one person.
    if not isinstance(required_profiles, list) or not required_profiles or set(required_profiles) != {case.profile for case in cases}:
        raise ValueError("The private quick suite does not cover its required profiles exactly.")

    max_documents = manifest.get("max_documents")
    max_pages = manifest.get("max_pages")

    # The page budget keeps a curated sample from silently turning into a full extraction run.
    if type(max_documents) is not int or type(max_pages) is not int or max_documents < 1 or max_pages < 1:
        raise ValueError("The private quick suite needs positive document and page budgets.")

    from parselabs.store import parse_page_number

    page_counts = [sum(parse_page_number(path) is not None for path in (case.case_dir / "reviewed").glob("*.json")) for case in cases]

    # Missing frozen pages and exceeded budgets indicate stale or malformed curation.
    if not all(page_counts) or len(cases) > max_documents or sum(page_counts) > max_pages:
        raise ValueError("The private quick suite is missing pages or exceeds its curated document/page budget.")


def get_regression_runs_dir() -> Path:
    """Keep extraction diagnostics private and outside source checkouts."""

    directory = _resolve_private_regression_path(get_approved_fixtures_dir().parent / "regression-runs")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    return directory


def _review_row_key(row: dict) -> tuple[str, str, str]:
    """Identify the same raw observation despite cosmetic spelling or numeric formatting."""

    tokens = []
    for field in ("raw_lab_name", "raw_value", "raw_lab_unit"):
        text = unicodedata.normalize("NFKD", str(row.get(field) or "").casefold())
        text = "".join(char for char in text if not unicodedata.combining(char))
        text = re.sub(r"\s+", "", text)

        # Label punctuation and accent formatting do not change assay identity.
        if field == "raw_lab_name":
            text = re.sub(r"[^\w]", "", text)

        # Equal decimal spellings are the same observation; comparison signs remain significant.
        if field == "raw_value" and re.fullmatch(r"[<>]?[+-]?\d+(?:[.,]\d+)?", text):
            sign = text[0] if text[0] in "<>" else ""
            text = sign + format(float(text.lstrip("<>").replace(",", ".")), ".12g")
        tokens.append(text)
    return tuple(tokens)


def replay_review_decisions(reviewed_dir: Path, fresh_dir: Path) -> None:
    """Match frozen decisions by page and normalized observation, never by position."""

    from parselabs.store import parse_page_number

    expected_pages = {parse_page_number(path): path for path in reviewed_dir.glob("*.json") if parse_page_number(path) is not None}
    actual_pages = {parse_page_number(path): path for path in fresh_dir.glob("*.json") if parse_page_number(path) is not None}

    # Different page sets must be diagnosed before any fresh decisions are persisted.
    if not expected_pages or expected_pages.keys() != actual_pages.keys():
        raise ValueError("Fresh extraction page coverage differs from the reviewed document.")

    staged = []
    expected_keys = _normalized_review_keys(reviewed_dir)
    actual_keys = _normalized_review_keys(fresh_dir)
    for number, expected_path in expected_pages.items():
        expected_rows = json.loads(expected_path.read_text(encoding="utf-8"))["lab_results"]
        payload = json.loads(actual_pages[number].read_text(encoding="utf-8"))
        actual_rows = payload["lab_results"]
        decisions = _match_review_decisions(expected_rows, actual_rows, number, expected_keys[number], actual_keys[number])
        for row, status in zip(actual_rows, decisions, strict=True):
            row["review_status"] = status
        staged.append((actual_pages[number], payload))

    # Every page must match before applying any decision, preventing partial positional replay.
    for path, payload in staged:
        temporary = path.with_suffix(".review-tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)


def _normalized_review_keys(directory: Path) -> dict[int, list[tuple]]:
    """Use real mapping/conversion results; unknown assays retain literal source identity."""

    from parselabs.config import LabSpecsConfig
    from parselabs.rows import build_document_review_dataframe

    frame = build_document_review_dataframe(directory, LabSpecsConfig()).sort_values(["page_number", "result_index"])
    keys: dict[int, list[tuple]] = {}
    for row in frame.to_dict("records"):
        signature = tuple(_normalize_fresh_cell(name, row.get(name)) for name in ["lab_name", "value", "value_text", "lab_unit", "is_below_limit", "is_above_limit"])

        # Unmapped rows must not inherit a decision merely because their numeric values agree.
        if row.get("lab_name") == "$UNKNOWN$" or row.get("lab_unit") == "$UNKNOWN$":
            signature = ("unmapped", *_review_row_key(row))
        keys.setdefault(int(row["page_number"]), []).append(signature)
    return keys


def _match_review_decisions(expected: list[dict], actual: list[dict], page: int, expected_keys: list[tuple], actual_keys: list[tuple]) -> list[str]:
    """Reject changed, omitted, added or ambiguously reviewed observations."""

    expected_counts = Counter(expected_keys)

    # Unmatched extra rows must not be silently approved or discarded.
    if expected_counts != Counter(actual_keys):
        raise ValueError(f"Fresh source rows differ on page {page}: {len(expected)} reviewed, {len(actual)} extracted. Inspect private artifacts.")

    by_key: dict[tuple, set[str]] = {}
    for row, key in zip(expected, expected_keys, strict=True):
        by_key.setdefault(key, set()).add(row.get("review_status"))

    # Identical observations with conflicting decisions cannot be safely assigned by order.
    if any(len(statuses) != 1 or not statuses <= {"accepted", "rejected"} for statuses in by_key.values()):
        raise ValueError("Frozen review decisions are incomplete or ambiguous for identical source rows.")

    return [next(iter(by_key[key])) for key in actual_keys]


def require_source_review(doc_dir: Path, pdf_path: Path, expected_df: pd.DataFrame | None = None) -> dict:
    """Reject missing or stale source-review evidence before fixture promotion."""

    review_path = doc_dir / SOURCE_REVIEW_JSON_NAME

    # A reviewed status alone does not demonstrate an independent source audit.
    if not review_path.is_file():
        raise ValueError("Missing source review; audit every page and use attest-reviewed first.")

    review = json.loads(review_path.read_text(encoding="utf-8"))

    # Every audit must explicitly cover fidelity, completeness, identity, dates, and normalization.
    checks = ("source_fidelity", "page_completeness", "patient_identity", "dates", "normalized_export")
    if not review.get("note") or not review.get("reviewed_at") or any(review.get("checks", {}).get(key) is not True for key in checks):
        raise ValueError("Incomplete source-review evidence.")

    # Any source or decision edit invalidates the previous source review.
    if review.get("fingerprints") != source_fingerprints(doc_dir, pdf_path):
        raise ValueError("Stale source review: PDF or page JSON changed; review again before promotion.")

    # Normalization changes also require review before replacing expected output.
    if expected_df is not None and review.get("expected_export_sha256") != export_fingerprint(expected_df):
        raise ValueError("Stale normalized-export review: canonical CSV changed; review again before promotion.")

    return review


def export_fingerprint(df: pd.DataFrame) -> str:
    """Hash reviewed export content independently of incidental CSV formatting."""

    return hashlib.sha256(canonical_csv_text(df).encode("utf-8")).hexdigest()


def discover_approved_cases(fixtures_dir: Path | None = None) -> list[ApprovedCase]:
    """Discover approved regression fixtures and validate their shape."""

    fixtures_dir = fixtures_dir if fixtures_dir is not None else get_approved_fixtures_dir()

    # An absent local corpus is normal on machines without private fixtures.
    if not fixtures_dir.exists():
        return []

    cases: list[ApprovedCase] = []
    seen_cases: dict[tuple[str | None, str, str], Path] = {}

    for case_dir in _list_case_directories(fixtures_dir):
        metadata_path = case_dir / CASE_JSON_NAME
        document_path = case_dir / DOCUMENT_PDF_NAME
        expected_csv_path = case_dir / EXPECTED_CSV_NAME
        review_state_path = case_dir / REVIEW_STATE_JSON_NAME

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

        identity = (metadata.get("profile"), stem, file_hash)
        previous_case_dir = seen_cases.get(identity)
        if previous_case_dir:
            raise ValueError(
                f"Approved cases must have unique profile/stem/hash identities. Case '{stem}_{file_hash}' appears in both "
                f"'{previous_case_dir.name}' and '{case_dir.name}'."
            )
        seen_cases[identity] = case_dir

        cases.append(
            ApprovedCase(
                case_id=case_dir.name,
                case_dir=case_dir,
                document_path=document_path,
                expected_csv_path=expected_csv_path,
                metadata_path=metadata_path,
                review_state_path=review_state_path if review_state_path.exists() else None,
                stem=stem,
                file_hash=file_hash,
                original_filename=original_filename,
                profile=metadata.get("profile"),
                metadata=metadata,
            )
        )

    return cases


def _list_case_directories(fixtures_dir: Path) -> list[Path]:
    """Find profile-nested and legacy cases without hiding missing metadata files."""

    cases: list[Path] = []
    for directory in sorted(path for path in fixtures_dir.iterdir() if path.is_dir()):
        # Recognize legacy cases directly under the root by any surviving artifact.
        if any((directory / name).exists() for name in (CASE_JSON_NAME, DOCUMENT_PDF_NAME, EXPECTED_CSV_NAME, "reviewed")):
            cases.append(directory)
            continue

        # Profile containers hold cases one level below the corpus root.
        cases.extend(sorted(path for path in directory.iterdir() if path.is_dir()))

    return cases


def canonicalize_export_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize final export output for exact CSV regression comparisons."""

    canonical = df.copy()
    ensure_columns(canonical, COLUMN_ORDER, default=None)
    canonical = canonical[COLUMN_ORDER].copy()

    # Normalize types before sorting so CSV-loaded strings and in-memory numbers agree.
    for column in COLUMN_ORDER:
        dtype_name = COLUMN_SCHEMA[column]["dtype"]
        canonical[column] = canonical[column].map(lambda value, dtype_name=dtype_name: _normalize_cell(value, dtype_name))

    # Sort page/result identities numerically while preserving exact raw text for audit.
    canonical = canonical.sort_values(
        CANONICAL_SORT_COLUMNS,
        key=lambda column: pd.to_numeric(column, errors="coerce") if column.name in {"page_number", "result_index"} else column,
        kind="mergesort",
        na_position="first",
    ).reset_index(drop=True)

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


def build_fresh_case_diff(expected_df: pd.DataFrame, actual_df: pd.DataFrame, case_id: str) -> str:
    """Compare measurements independently of fresh row order and cosmetic raw formatting."""

    columns = ["date", "lab_name", "value", "value_text", "lab_unit", "page_number", "reference_min", "reference_max", "is_below_limit", "is_above_limit", "lab_type"]
    expected = canonicalize_export_df(expected_df)[columns]
    actual = canonicalize_export_df(actual_df)[columns]
    for frame in (expected, actual):
        frame["value_text"] = frame["value_text"].map(lambda value: _normalize_fresh_cell("value_text", value))
    expected = expected.sort_values(columns, kind="stable").reset_index(drop=True)
    actual = actual.sort_values(columns, kind="stable").reset_index(drop=True)
    expected_text = expected.to_csv(index=False, lineterminator="\n")
    actual_text = actual.to_csv(index=False, lineterminator="\n")

    # Source observations are separately matched before this semantic export comparison.
    if expected_text == actual_text:
        return ""

    diff = "\n".join(unified_diff(expected_text.splitlines(), actual_text.splitlines(), fromfile=f"{case_id}/expected.csv", tofile=f"{case_id}/actual.csv", lineterm=""))
    return f"Mismatch for freshly extracted case '{case_id}':\n{diff}"


def _normalize_fresh_cell(name: str, value: object) -> str:
    """Ignore terminal sentence punctuation only for qualitative fresh-extraction text."""

    text = _normalize_cell(value, COLUMN_SCHEMA[name]["dtype"])

    # A sentence-ending period does not change a qualitative result such as NORMAL.
    if name == "value_text":
        return re.sub(r"\s+", " ", text).rstrip(".").strip()

    return text


def _normalize_cell(value: object, dtype_name: str) -> str:
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
