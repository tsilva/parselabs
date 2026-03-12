"""Shared helpers for reviewed processed-document outputs."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from parselabs.config import UNKNOWN_VALUE, LabSpecsConfig
from parselabs.export_schema import COLUMN_ORDER, COLUMN_SCHEMA, get_column_lists
from parselabs.normalization import (
    apply_dtype_conversions,
    apply_normalizations,
    deduplicate_results,
    flag_duplicate_entries,
)
from parselabs.standardization import standardize_lab_names, standardize_lab_units
from parselabs.utils import ensure_columns
from parselabs.validation import ValueValidator

logger = logging.getLogger(__name__)

REVIEW_MISSING_ROWS_KEY = "review_missing_rows"
EXTRACTION_FAILED_REASON = "EXTRACTION_FAILED"
UNKNOWN_LAB_MAPPING_REASON = "UNKNOWN_LAB_MAPPING"
UNKNOWN_UNIT_MAPPING_REASON = "UNKNOWN_UNIT_MAPPING"
AMBIGUOUS_PERCENTAGE_VARIANT_REASON = "AMBIGUOUS_PERCENTAGE_VARIANT"
SUSPICIOUS_REFERENCE_RANGE_REASON = "SUSPICIOUS_REFERENCE_RANGE"

DOCUMENT_REVIEW_COLUMNS = [
    "date",
    "source_file",
    "page_number",
    "result_index",
    "raw_lab_name",
    "raw_value",
    "raw_lab_unit",
    "raw_reference_range",
    "raw_reference_min",
    "raw_reference_max",
    "raw_comments",
    "lab_name_standardized",
    "lab_unit_standardized",
    "lab_name",
    "value",
    "lab_unit",
    "reference_min",
    "reference_max",
    "review_needed",
    "review_reason",
    "is_below_limit",
    "is_above_limit",
    "lab_type",
    "review_status",
    "review_completed_at",
]


@dataclass(frozen=True)
class ProcessedDocument:
    """Single processed document directory under an output path."""

    doc_dir: Path
    stem: str
    pdf_path: Path
    csv_path: Path


@dataclass(frozen=True)
class ReviewSummary:
    """Aggregate review state for a processed document."""

    total: int
    accepted: int
    rejected: int
    pending: int
    missing_row_markers: int

    @property
    def reviewed(self) -> int:
        """Return the number of rows with an explicit accept/reject decision."""

        return self.accepted + self.rejected

    @property
    def fixture_ready(self) -> bool:
        """Return whether the document can be used as reviewed truth."""

        return self.pending == 0 and self.missing_row_markers == 0


@dataclass(frozen=True)
class ReviewCorpusReport:
    """Aggregate review-corpus signals for benchmark-driven improvements."""

    document_count: int
    fixture_ready_document_count: int
    rejected_rows: int
    unresolved_missing_rows: int
    unknown_standardized_names: int
    unknown_standardized_units: int
    validation_reason_counts: dict[str, int]
    rejected_raw_name_counts: dict[str, int]
    rejected_raw_unit_counts: dict[str, int]


class ReviewStateError(RuntimeError):
    """Raised when reviewed JSON cannot be promoted into fixture truth."""


def iter_processed_documents(output_path: Path) -> list[ProcessedDocument]:
    """Discover processed document directories under an output path."""

    documents: list[ProcessedDocument] = []

    # Guard: Missing output directories simply mean there is nothing to review.
    if not output_path.exists():
        return documents

    # Inspect only child directories because processed documents are stored per folder.
    for doc_dir in sorted(path for path in output_path.iterdir() if path.is_dir()):
        # Skip operational folders that do not contain document artifacts.
        if doc_dir.name == "logs":
            continue

        # Skip directories that do not use the canonical hash-suffixed layout.
        if not _is_hashed_document_dir(doc_dir):
            continue

        # Skip directories that do not contain a copied source PDF.
        pdf_path = _find_document_pdf(doc_dir)
        if pdf_path is None:
            continue

        # Build the canonical per-document CSV path from the copied PDF stem.
        stem = pdf_path.stem
        csv_path = doc_dir / f"{stem}.csv"
        documents.append(ProcessedDocument(doc_dir=doc_dir, stem=stem, pdf_path=pdf_path, csv_path=csv_path))

    return documents


def get_review_summary(review_df: pd.DataFrame, missing_row_markers: int = 0) -> ReviewSummary:
    """Summarize accept/reject/pending counts for a document review frame."""

    # Guard: Empty frames have no reviewable rows.
    if review_df.empty:
        return ReviewSummary(
            total=0,
            accepted=0,
            rejected=0,
            pending=0,
            missing_row_markers=missing_row_markers,
        )

    # Normalize review_status so missing values are treated consistently.
    statuses = review_df["review_status"].fillna("").astype(str).str.strip().str.lower()
    total = len(statuses)
    accepted = int((statuses == "accepted").sum())
    rejected = int((statuses == "rejected").sum())
    pending = total - accepted - rejected
    return ReviewSummary(
        total=total,
        accepted=accepted,
        rejected=rejected,
        pending=pending,
        missing_row_markers=missing_row_markers,
    )


def get_document_review_summary(doc_dir: Path, review_df: pd.DataFrame | None = None) -> ReviewSummary:
    """Summarize document review state, including unresolved missing-row markers."""

    # Reuse a prebuilt dataframe when the caller already loaded one.
    if review_df is None:
        review_df = load_document_review_rows(doc_dir)

    missing_row_markers = count_review_missing_rows(doc_dir)
    return get_review_summary(review_df, missing_row_markers=missing_row_markers)


def build_document_review_dataframe(doc_dir: Path, lab_specs: LabSpecsConfig) -> pd.DataFrame:
    """Build a 1:1 review dataframe from per-page JSON extraction files."""

    # Load one row per extracted JSON result so review actions map back cleanly.
    review_df = load_document_review_rows(doc_dir)

    # Guard: Documents with no extracted rows still get a stable empty schema.
    if review_df.empty:
        empty_df = pd.DataFrame(columns=DOCUMENT_REVIEW_COLUMNS)
        return empty_df

    # Normalize extracted rows into the review dataframe shape without dropping reviewable rows.
    review_df = _prepare_rows_for_review(review_df, lab_specs)

    # Ensure the review CSV keeps a stable column order even when some fields are absent.
    ensure_columns(review_df, DOCUMENT_REVIEW_COLUMNS, default=None)
    review_df = review_df[DOCUMENT_REVIEW_COLUMNS].copy()

    # Keep rows in source order so the reviewer advances through the document naturally.
    review_df = review_df.sort_values(["page_number", "result_index"], kind="mergesort").reset_index(drop=True)
    return review_df


def rebuild_document_csv(doc_dir: Path, lab_specs: LabSpecsConfig) -> Path:
    """Rebuild a per-document review CSV from page JSON files."""

    # Recompute the review frame from JSON so CSV contents always match persisted review state.
    review_df = build_document_review_dataframe(doc_dir, lab_specs)
    csv_path = _get_document_csv_path(doc_dir)
    review_df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def build_document_expected_dataframe(doc_dir: Path, lab_specs: LabSpecsConfig) -> pd.DataFrame:
    """Build the final export-shape dataframe for one processed document."""

    return build_document_expected_dataframe_from_reviewed_json(doc_dir, lab_specs)


def build_document_expected_dataframe_from_reviewed_json(
    doc_dir: Path,
    lab_specs: LabSpecsConfig,
    allow_pending: bool = False,
) -> pd.DataFrame:
    """Build the final export-shape dataframe for one processed document."""

    # Guard: Strict rebuilds require a fixture-ready document review state.
    if not allow_pending:
        ensure_document_fixture_ready(doc_dir, lab_specs)

    # Start from only the explicitly accepted rows in the reviewed JSON payloads.
    final_df = load_document_review_rows(doc_dir, include_statuses={"accepted"})

    # Guard: Empty documents produce an empty export frame.
    if final_df.empty:
        return pd.DataFrame(columns=COLUMN_ORDER)

    # Transform accepted review rows into the canonical export schema used by all.csv.
    final_df, _ = transform_rows_to_final_export(
        final_df,
        lab_specs,
        apply_standardization=True,
    )
    return final_df


def transform_rows_to_final_export(
    rows_df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
    apply_standardization: bool,
) -> tuple[pd.DataFrame, dict[str, int | dict[str, int]]]:
    """Transform extracted rows into the canonical final export dataframe."""

    export_cols, _, _, dtypes = get_column_lists(COLUMN_SCHEMA)

    # Guard: Empty inputs still return a stable export schema and empty validation stats.
    if rows_df.empty:
        return (
            pd.DataFrame(columns=export_cols),
            {"total_rows": 0, "rows_flagged": 0, "flags_by_reason": {}},
        )

    prepared_df, validation_stats = _prepare_rows_for_export(
        rows_df,
        lab_specs,
        apply_standardization=apply_standardization,
    )

    # Guard: Rows that all map to unknown labs still export a stable empty schema.
    if prepared_df.empty:
        return pd.DataFrame(columns=export_cols), validation_stats

    # Select and type-convert only the canonical export columns expected by final outputs.
    ensure_columns(prepared_df, export_cols, default=None)
    final_df = prepared_df[export_cols].copy()
    final_df = apply_dtype_conversions(final_df, dtypes)
    return final_df, validation_stats


def save_review_status(doc_dir: Path, page_number: int, result_index: int, status: str | None) -> tuple[bool, str]:
    """Persist a review decision to the page JSON backing a CSV row."""

    normalized_status = _normalize_review_status(status)

    # Guard: Persist only supported review decisions or an explicit reset to pending.
    if normalized_status not in {"accepted", "rejected", None}:
        return False, f"Unsupported review status: {status}"

    # Resolve the JSON file for the selected page before attempting any mutation.
    json_path = _get_page_json_path(doc_dir, page_number)

    # Guard: The page JSON must exist for review persistence to work.
    if not json_path.exists():
        return False, f"JSON file not found: {json_path}"

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        # Invalid JSON should surface as an actionable error rather than being ignored.
        return False, f"Failed to read JSON file: {exc}"

    # Guard: Review writes only make sense for extraction payloads with lab_results.
    results = data.get("lab_results")
    if not isinstance(results, list):
        return False, "No lab_results in JSON file."

    # Guard: The selected result index must point at a real row.
    if result_index < 0 or result_index >= len(results):
        return False, f"result_index {result_index} out of range."

    # Persist explicit review decisions plus a timestamp for auditability.
    if normalized_status in {"accepted", "rejected"}:
        results[result_index]["review_status"] = normalized_status
        results[result_index]["review_completed_at"] = pd.Timestamp.now(tz="UTC").isoformat()

    # Clearing a decision should restore the row to the canonical pending state.
    if normalized_status is None:
        results[result_index].pop("review_status", None)
        results[result_index].pop("review_completed_at", None)

    try:
        # Rewrite the JSON atomically enough for local desktop usage.
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as exc:
        # Surface filesystem errors directly so the caller can warn the reviewer.
        return False, f"Failed to write JSON file: {exc}"

    return True, ""


def save_missing_row_marker(doc_dir: Path, page_number: int, anchor_result_index: int) -> tuple[bool, str]:
    """Persist a missing-row marker to the page JSON backing the current review row."""

    # Resolve the JSON file for the selected page before attempting any mutation.
    json_path = _get_page_json_path(doc_dir, page_number)

    # Guard: The page JSON must exist for missing-row markers to persist.
    if not json_path.exists():
        return False, f"JSON file not found: {json_path}"

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        # Invalid JSON should surface as an actionable error rather than being ignored.
        return False, f"Failed to read JSON file: {exc}"

    results = data.get("lab_results")

    # Guard: Missing-row markers must anchor to an existing extracted result.
    if not isinstance(results, list):
        return False, "No lab_results in JSON file."

    # Guard: Reject anchors that do not point at an existing extracted row.
    if anchor_result_index < 0 or anchor_result_index >= len(results):
        return False, f"anchor_result_index {anchor_result_index} out of range."

    markers = data.get(REVIEW_MISSING_ROWS_KEY)

    # Guard: Normalize absent marker lists into a fresh mutable list.
    if markers is None:
        markers = []

    # Guard: Reject malformed marker containers so the reviewer can repair the JSON.
    if not isinstance(markers, list):
        return False, f"{REVIEW_MISSING_ROWS_KEY} must be a list in {json_path.name}."

    markers.append(
        {
            "anchor_result_index": int(anchor_result_index),
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }
    )
    data[REVIEW_MISSING_ROWS_KEY] = markers

    try:
        # Rewrite the JSON atomically enough for local desktop usage.
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as exc:
        # Surface filesystem errors directly so the caller can warn the reviewer.
        return False, f"Failed to write JSON file: {exc}"

    return True, ""


def get_review_missing_rows(doc_dir: Path, page_number: int | None = None) -> list[dict]:
    """Return unresolved missing-row markers for a document or page."""

    markers: list[dict] = []

    # Scan every processed page JSON so unresolved omissions are counted consistently.
    for page_json_path in sorted(doc_dir.glob("*.json")):
        current_page_number = _parse_page_number(page_json_path)

        # Skip files that do not follow the processed page naming convention.
        if current_page_number is None:
            continue

        # Skip non-target pages when the caller requested a single page summary.
        if page_number is not None and current_page_number != page_number:
            continue

        try:
            page_payload = json.loads(page_json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            # Corrupt JSON files should be logged and skipped instead of breaking review summaries.
            logger.warning(f"Failed to read processed page JSON {page_json_path}: {exc}")
            continue

        page_markers = page_payload.get(REVIEW_MISSING_ROWS_KEY, [])

        # Skip malformed marker containers because they cannot be summarized safely.
        if not isinstance(page_markers, list):
            logger.warning(f"Invalid {REVIEW_MISSING_ROWS_KEY} payload in {page_json_path}")
            continue

        for marker in page_markers:
            # Skip malformed marker entries so one bad item does not poison the summary.
            if not isinstance(marker, dict):
                continue

            markers.append(
                {
                    "page_number": current_page_number,
                    "anchor_result_index": marker.get("anchor_result_index"),
                    "created_at": marker.get("created_at"),
                }
            )

    return markers


def count_review_missing_rows(doc_dir: Path, page_number: int | None = None) -> int:
    """Return the number of unresolved missing-row markers for a document or page."""

    return len(get_review_missing_rows(doc_dir, page_number=page_number))


def ensure_document_fixture_ready(doc_dir: Path, lab_specs: LabSpecsConfig) -> ReviewSummary:
    """Raise when a processed document is not ready to become reviewed truth."""

    review_df = build_document_review_dataframe(doc_dir, lab_specs)
    summary = get_document_review_summary(doc_dir, review_df)

    # Guard: Fixture sync requires every extracted row to be reviewed and every omission marker resolved.
    if summary.fixture_ready:
        return summary

    issue_parts: list[str] = []

    # Include pending rows so the reviewer knows what still needs an explicit decision.
    if summary.pending > 0:
        issue_parts.append(f"{summary.pending} pending row(s)")

    # Include unresolved missing markers because they block fixture truth even when rows are reviewed.
    if summary.missing_row_markers > 0:
        issue_parts.append(f"{summary.missing_row_markers} unresolved missing-row marker(s)")

    issue_text = ", ".join(issue_parts) if issue_parts else "document review incomplete"
    raise ReviewStateError(f"{_get_document_stem(doc_dir)} is not fixture-ready: {issue_text}.")


def build_review_corpus_report(output_path: Path, lab_specs: LabSpecsConfig) -> ReviewCorpusReport:
    """Aggregate benchmark-driving review signals across a processed output corpus."""

    documents = iter_processed_documents(output_path)
    validation_reason_counts: Counter[str] = Counter()
    rejected_raw_name_counts: Counter[str] = Counter()
    rejected_raw_unit_counts: Counter[str] = Counter()
    rejected_rows = 0
    unresolved_missing_rows = 0
    unknown_standardized_names = 0
    unknown_standardized_units = 0
    fixture_ready_document_count = 0

    # Rebuild each review dataframe from JSON so the report matches the persisted source of truth.
    for document in documents:
        review_df = build_document_review_dataframe(document.doc_dir, lab_specs)
        summary = get_document_review_summary(document.doc_dir, review_df)
        unresolved_missing_rows += summary.missing_row_markers

        # Count fixture-ready documents separately so review progress is visible.
        if summary.fixture_ready:
            fixture_ready_document_count += 1

        if not review_df.empty:
            unknown_standardized_names += int((review_df["lab_name_standardized"] == UNKNOWN_VALUE).sum())
            unknown_standardized_units += int((review_df["lab_unit_standardized"] == UNKNOWN_VALUE).sum())

        rejected_mask = review_df["review_status"].fillna("").astype(str).str.strip().str.lower() == "rejected"
        rejected_rows += int(rejected_mask.sum())

        # Count rejected raw names so standardization or prompt fixes can target recurring failures.
        for raw_name, count in review_df.loc[rejected_mask, "raw_lab_name"].fillna("").astype(str).value_counts().items():
            if raw_name:
                rejected_raw_name_counts[raw_name] += int(count)

        # Count rejected raw units separately because unit confusion often points to OCR or mapping issues.
        for raw_unit, count in review_df.loc[rejected_mask, "raw_lab_unit"].fillna("").astype(str).value_counts().items():
            if raw_unit:
                rejected_raw_unit_counts[raw_unit] += int(count)

        # Split semicolon-delimited validation reason codes into aggregate counts.
        for reason_text in review_df["review_reason"].fillna("").astype(str):
            for reason_code in [code.strip() for code in reason_text.split(";") if code.strip()]:
                validation_reason_counts[reason_code] += 1

    return ReviewCorpusReport(
        document_count=len(documents),
        fixture_ready_document_count=fixture_ready_document_count,
        rejected_rows=rejected_rows,
        unresolved_missing_rows=unresolved_missing_rows,
        unknown_standardized_names=unknown_standardized_names,
        unknown_standardized_units=unknown_standardized_units,
        validation_reason_counts=dict(validation_reason_counts.most_common()),
        rejected_raw_name_counts=dict(rejected_raw_name_counts.most_common()),
        rejected_raw_unit_counts=dict(rejected_raw_unit_counts.most_common()),
    )


def get_page_image_path(doc_dir: Path, page_number: int) -> Path | None:
    """Return the primary page image for a review row when it exists."""

    # Resolve the page image path using the processed document stem and page number.
    stem = _get_document_stem(doc_dir)
    image_path = doc_dir / f"{stem}.{page_number:03d}.jpg"

    # Guard: Return None when the image is unavailable.
    if not image_path.exists():
        return None

    return image_path


def _find_document_pdf(doc_dir: Path) -> Path | None:
    """Return the copied source PDF for a processed document directory."""

    pdf_paths = sorted(doc_dir.glob("*.pdf"))

    # Guard: Processed document directories without a PDF are incomplete.
    if not pdf_paths:
        return None

    return pdf_paths[0]


def _get_document_stem(doc_dir: Path) -> str:
    """Resolve the logical document stem for a processed output directory."""

    # Prefer the copied PDF because it preserves the original filename exactly.
    pdf_path = _find_document_pdf(doc_dir)
    if pdf_path is not None:
        return pdf_path.stem

    # Fall back to the directory name for partially-built outputs.
    hashed_match = re.match(r"^(?P<stem>.+)_[0-9a-fA-F]{8}$", doc_dir.name)
    if hashed_match:
        return str(hashed_match.group("stem"))

    return doc_dir.name


def _get_document_csv_path(doc_dir: Path) -> Path:
    """Return the canonical CSV path for a processed document directory."""

    stem = _get_document_stem(doc_dir)
    return doc_dir / f"{stem}.csv"


def _get_page_json_path(doc_dir: Path, page_number: int) -> Path:
    """Resolve the page JSON path for a document row."""

    stem = _get_document_stem(doc_dir)
    return doc_dir / f"{stem}.{page_number:03d}.json"


def _normalize_review_status(status: object) -> str | None:
    """Normalize persisted review statuses to the supported values."""

    # Guard: Missing statuses stay unset.
    if status is None:
        return None

    normalized = str(status).strip().lower()

    # Guard: Empty strings are treated as unset review decisions.
    if not normalized:
        return None

    # Guard: Pass through only the supported persisted status values.
    if normalized in {"accepted", "rejected"}:
        return normalized

    return normalized


def _parse_page_number(page_json_path: Path) -> int | None:
    """Extract the 1-based page number from a processed page JSON filename."""

    match = re.search(r"\.(\d{3})\.json$", page_json_path.name)

    # Guard: Unexpected filenames cannot contribute review rows reliably.
    if not match:
        return None

    return int(match.group(1))


def _extract_document_date(page_payload: dict, doc_dir: Path) -> str | None:
    """Extract the document date from page metadata or the filename."""

    # Prefer the explicit collection date captured by extraction.
    doc_date = page_payload.get("collection_date") or page_payload.get("report_date")

    # Ignore the legacy placeholder date used by bad model responses.
    if doc_date == "0000-00-00":
        doc_date = None

    # Fall back to a YYYY-MM-DD token in the document stem when metadata is absent.
    if not doc_date:
        stem = _get_document_stem(doc_dir)
        match = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
        if match:
            doc_date = match.group(1)

    return doc_date


def load_document_review_rows(
    doc_dir: Path,
    include_statuses: set[str] | None = None,
) -> pd.DataFrame:
    """Load one review row per extracted result from page JSON files."""

    rows: list[dict] = []
    page_json_paths = sorted(doc_dir.glob("*.json"))
    source_file = f"{_get_document_stem(doc_dir)}.csv"
    doc_date: str | None = None

    # Read each page JSON once and flatten its lab_results into row records.
    for page_json_path in page_json_paths:
        page_number = _parse_page_number(page_json_path)

        # Skip malformed filenames because they cannot round-trip to review actions.
        if page_number is None:
            continue

        try:
            page_payload = json.loads(page_json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            # Corrupt JSON files should be logged and skipped instead of breaking the whole review UI.
            logger.warning(f"Failed to read processed page JSON {page_json_path}: {exc}")
            continue

        # Capture the first usable document date for all rows in this document.
        if doc_date is None:
            doc_date = _extract_document_date(page_payload, doc_dir)

        # Skip pages without extracted lab results.
        results = page_payload.get("lab_results", [])
        if not isinstance(results, list):
            continue

        page_failed = bool(page_payload.get("_extraction_failed"))

        # Flatten each extracted row while preserving its original page-local index.
        for result_index, result in enumerate(results):
            # Skip malformed result payloads so one bad item does not poison the file.
            if not isinstance(result, dict):
                continue

            status = _normalize_review_status(result.get("review_status"))
            review_needed = bool(result.get("review_needed")) or page_failed
            review_reason = str(result.get("review_reason") or "").strip()

            # Failed extractions should stay visible to the reviewer even when salvage produced rows.
            if page_failed and EXTRACTION_FAILED_REASON not in review_reason:
                if review_reason and not review_reason.endswith(";"):
                    review_reason = f"{review_reason}; "
                review_reason = f"{review_reason}{EXTRACTION_FAILED_REASON}; " if review_reason else f"{EXTRACTION_FAILED_REASON}; "

            # Skip rows outside the requested review-status subset.
            if include_statuses is not None and status not in include_statuses:
                continue

            rows.append(
                {
                    "date": doc_date,
                    "source_file": source_file,
                    "page_number": page_number,
                    "result_index": result_index,
                    "raw_lab_name": result.get("raw_lab_name"),
                    "raw_value": result.get("raw_value"),
                    "raw_lab_unit": result.get("raw_lab_unit"),
                    "raw_reference_range": result.get("raw_reference_range"),
                    "raw_reference_min": result.get("raw_reference_min"),
                    "raw_reference_max": result.get("raw_reference_max"),
                    "raw_comments": result.get("raw_comments"),
                    "review_needed": review_needed,
                    "review_reason": review_reason,
                    "review_status": status,
                    "review_completed_at": result.get("review_completed_at"),
                }
            )

    # Guard: Empty results still return a dataframe with the expected columns.
    if not rows:
        return pd.DataFrame(columns=DOCUMENT_REVIEW_COLUMNS)

    review_df = pd.DataFrame(rows)

    # Apply the resolved document date to every row so later pages can still fill earlier blanks.
    if doc_date is not None:
        review_df["date"] = doc_date

    ensure_columns(review_df, DOCUMENT_REVIEW_COLUMNS, default=None)
    return review_df


def apply_cached_standardization(review_df: pd.DataFrame, lab_specs: LabSpecsConfig) -> pd.DataFrame:
    """Apply cache-backed name and unit mappings to extracted review rows."""

    # Guard: Skip standardization entirely when lab specs are unavailable.
    if not lab_specs.exists:
        ensure_columns(review_df, ["lab_name_standardized", "lab_unit_standardized"], default=None)
        return review_df

    # Standardize lab names in one pass so repeated raw labels share the same cached mapping.
    raw_names = review_df["raw_lab_name"].fillna("").astype(str).tolist()
    name_map = standardize_lab_names(raw_names)
    review_df["lab_name_standardized"] = [name_map.get(name, UNKNOWN_VALUE) for name in raw_names]

    # Standardize units only for rows whose lab names mapped successfully.
    unit_contexts: list[tuple[str, str]] = []
    for raw_unit, standardized_name in zip(review_df["raw_lab_unit"].fillna("").astype(str), review_df["lab_name_standardized"].fillna("").astype(str), strict=False):
        # Skip rows without a usable standardized lab name.
        if not standardized_name or standardized_name == UNKNOWN_VALUE:
            unit_contexts.append(("", ""))
            continue

        unit_contexts.append((raw_unit, standardized_name))

    mapped_units = standardize_lab_units([context for context in unit_contexts if context != ("", "")])

    standardized_units: list[str | None] = []
    safe_missing_unit_primary_units = {"boolean", "pH", "unitless"}

    for raw_unit, standardized_name in unit_contexts:
        # Leave units empty when the lab name did not map.
        if (raw_unit, standardized_name) == ("", ""):
            standardized_units.append(None)
            continue

        standardized_unit = mapped_units.get((raw_unit, standardized_name))

        # Some labs are intrinsically unitless or use a conventional implied unit.
        # When extraction omits the printed unit entirely, prefer the primary unit
        # from lab_specs instead of forcing a cache entry for blank input.
        if standardized_unit == UNKNOWN_VALUE and raw_unit.strip() == "":
            primary_unit = lab_specs.get_primary_unit(standardized_name)
            if primary_unit in safe_missing_unit_primary_units:
                standardized_unit = primary_unit

        standardized_units.append(standardized_unit)

    review_df["lab_unit_standardized"] = standardized_units
    return review_df


def _add_export_column_aliases(review_df: pd.DataFrame) -> pd.DataFrame:
    """Add public export-name aliases without losing internal normalized columns."""

    review_df = review_df.copy()

    # Preserve the standardized mapping columns and add export-style aliases alongside them.
    if "lab_name_standardized" in review_df.columns:
        review_df["lab_name"] = review_df["lab_name_standardized"]

    # Expose normalized numeric values under the final export column names.
    if "value_primary" in review_df.columns:
        review_df["value"] = review_df["value_primary"]
    if "lab_unit_primary" in review_df.columns:
        review_df["lab_unit"] = review_df["lab_unit_primary"]
    if "reference_min_primary" in review_df.columns:
        review_df["reference_min"] = review_df["reference_min_primary"]
    if "reference_max_primary" in review_df.columns:
        review_df["reference_max"] = review_df["reference_max_primary"]

    # Add the public raw-unit alias required by the final export schema.
    if "raw_lab_unit" in review_df.columns:
        review_df["raw_unit"] = review_df["raw_lab_unit"]

    return review_df


def _append_review_reason_code(
    review_df: pd.DataFrame,
    mask: pd.Series,
    reason_code: str,
) -> pd.DataFrame:
    """Append a review reason code without duplicating existing reason text."""

    # Guard: No matching rows means there is nothing to update.
    if review_df.empty or not mask.any():
        return review_df

    # Ensure review columns exist before appending reason codes.
    if "review_needed" not in review_df.columns:
        review_df["review_needed"] = False
    if "review_reason" not in review_df.columns:
        review_df["review_reason"] = ""

    review_df.loc[mask, "review_needed"] = True
    current_reasons = review_df.loc[mask, "review_reason"].fillna("").astype(str)
    review_df.loc[mask, "review_reason"] = current_reasons.apply(lambda value: f"{value}{reason_code}; " if reason_code not in value else value)
    return review_df


def _flag_unknown_mappings(review_df: pd.DataFrame) -> pd.DataFrame:
    """Mark unresolved name and unit mappings for explicit reviewer attention."""

    # Unknown standardized lab names cannot be trusted for export.
    unknown_lab_mask = review_df["lab_name_standardized"].fillna("") == UNKNOWN_VALUE
    review_df = _append_review_reason_code(review_df, unknown_lab_mask, UNKNOWN_LAB_MAPPING_REASON)

    # Known labs with unknown or missing units stay reviewable but cannot publish yet.
    unit_values = review_df["lab_unit_standardized"].fillna("")
    known_lab_mask = review_df["lab_name_standardized"].fillna("") != UNKNOWN_VALUE
    unknown_unit_mask = known_lab_mask & unit_values.isin(["", UNKNOWN_VALUE])
    review_df = _append_review_reason_code(review_df, unknown_unit_mask, UNKNOWN_UNIT_MAPPING_REASON)
    return review_df


def _flag_percentage_variant_ambiguity(
    review_df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """Flag rows where explicit units disagree with percentage-vs-absolute lab variants."""

    # Guard: Variant checks require lab specs and standardized lab names.
    if not lab_specs.exists or review_df.empty:
        return review_df

    ambiguous_indices: list[int] = []

    for idx in review_df.index:
        std_name = review_df.at[idx, "lab_name_standardized"]
        std_unit = review_df.at[idx, "lab_unit_standardized"]

        # Skip rows without a usable standardized lab name.
        if pd.isna(std_name) or std_name == UNKNOWN_VALUE or not str(std_name).strip():
            continue

        # Skip rows without an explicit usable standardized unit.
        if pd.isna(std_unit) or std_unit in {"", UNKNOWN_VALUE}:
            continue

        # Percentage units paired with non-percentage names are ambiguous when a percentage variant exists.
        if std_unit == "%" and not std_name.endswith("(%)"):
            potential_pct_name = f"{std_name} (%)"
            if potential_pct_name in lab_specs._specs:
                ambiguous_indices.append(idx)
            continue

        # Non-percentage units paired with percentage names are also ambiguous.
        if std_unit != "%" and std_name.endswith("(%)"):
            ambiguous_indices.append(idx)

    return _append_review_reason_code(
        review_df,
        review_df.index.isin(ambiguous_indices),
        AMBIGUOUS_PERCENTAGE_VARIANT_REASON,
    )


def _flag_suspicious_reference_ranges(
    review_df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """Flag reference ranges that look incompatible with the standardized lab definition."""

    # Guard: Range checks require lab specs and normalized reference columns.
    if not lab_specs.exists or review_df.empty:
        return review_df

    suspicious_indices: list[int] = []

    for idx in review_df.index:
        std_name = review_df.at[idx, "lab_name_standardized"]
        ref_min = review_df.at[idx, "reference_min_primary"]
        ref_max = review_df.at[idx, "reference_max_primary"]
        unit = review_df.at[idx, "lab_unit_primary"]

        # Skip rows without a known lab mapping or a full reference range.
        if pd.isna(std_name) or std_name == UNKNOWN_VALUE:
            continue
        if pd.isna(ref_min) or pd.isna(ref_max):
            continue

        # Inverted ranges are always suspicious.
        if ref_min > ref_max:
            suspicious_indices.append(idx)
            continue

        # Percentage ranges above 100 are likely mixed-unit artifacts.
        if unit == "%" and ref_max > 100:
            suspicious_indices.append(idx)
            continue

        expected_range = lab_specs._specs.get(std_name, {}).get("ranges", {}).get("default", [])

        # Skip labs without expected ranges to compare against.
        if len(expected_range) < 2:
            continue

        expected_min, expected_max = expected_range[0], expected_range[1]

        # Skip comparisons that would divide by zero or use missing expectations.
        if not expected_min or not expected_max:
            continue

        ratio_min = abs(ref_min / expected_min)
        ratio_max = abs(ref_max / expected_max)
        if (ratio_min > 10 or ratio_min < 0.1) and (ratio_max > 10 or ratio_max < 0.1):
            suspicious_indices.append(idx)

    return _append_review_reason_code(
        review_df,
        review_df.index.isin(suspicious_indices),
        SUSPICIOUS_REFERENCE_RANGE_REASON,
    )


def _flag_review_ambiguities(
    review_df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """Attach reviewer-facing ambiguity reasons without mutating the extracted values."""

    review_df = _flag_unknown_mappings(review_df)
    review_df = _flag_percentage_variant_ambiguity(review_df, lab_specs)
    review_df = _flag_suspicious_reference_ranges(review_df, lab_specs)
    return review_df


def _filter_exportable_rows(review_df: pd.DataFrame) -> pd.DataFrame:
    """Remove unresolved rows that cannot safely participate in final publish outputs."""

    # Guard: Frames without normalized name/unit columns cannot be filtered safely.
    if "lab_name_standardized" not in review_df.columns or "lab_unit_primary" not in review_df.columns:
        return review_df

    unresolved_lab_mask = review_df["lab_name_standardized"].fillna("") == UNKNOWN_VALUE
    unresolved_unit_mask = review_df["lab_unit_primary"].fillna("").isin(["", UNKNOWN_VALUE])
    unresolved_mask = unresolved_lab_mask | unresolved_unit_mask

    # Guard: Fast-path when every row has a resolved lab and publishable unit.
    if not unresolved_mask.any():
        return review_df

    return review_df[~unresolved_mask].reset_index(drop=True)


def _prepare_rows_for_review(
    review_df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """Normalize extracted rows into the review dataframe shape."""

    # Apply cached standardization so the reviewer sees the mapped interpretation.
    review_df = apply_cached_standardization(review_df, lab_specs)

    # Apply normalization logic without deduplication so every extracted row remains reviewable.
    review_df = apply_normalizations(review_df, lab_specs)

    # Surface unresolved mappings and ambiguous interpretations before any validation flags are added.
    review_df = _flag_review_ambiguities(review_df, lab_specs)

    # Add duplicate-review flags before export-name aliases are added.
    review_df = flag_duplicate_entries(review_df)

    # Add the public export-name aliases used throughout the viewer and rebuild path.
    review_df = _add_export_column_aliases(review_df)

    # Run value validation so the review tool surfaces the same warnings as final export.
    validator = ValueValidator(lab_specs)
    return validator.validate(review_df)


def _prepare_rows_for_export(
    rows_df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
    apply_standardization: bool,
) -> tuple[pd.DataFrame, dict[str, int | dict[str, int]]]:
    """Normalize, deduplicate, and validate extracted rows for final export."""

    if apply_standardization:
        rows_df = apply_cached_standardization(rows_df, lab_specs)

    rows_df = apply_normalizations(rows_df, lab_specs)
    rows_df = _flag_review_ambiguities(rows_df, lab_specs)
    rows_df = _filter_exportable_rows(rows_df)

    # Guard: Rows that still lack publishable mappings cannot contribute to the final export.
    if rows_df.empty:
        return rows_df, {"total_rows": 0, "rows_flagged": 0, "flags_by_reason": {}}

    rows_df = flag_duplicate_entries(rows_df)

    if lab_specs.exists:
        rows_df = deduplicate_results(rows_df, lab_specs)

    rows_df = _add_export_column_aliases(rows_df)
    validator = ValueValidator(lab_specs)
    rows_df = validator.validate(rows_df)
    return rows_df, validator.validation_stats


def _is_hashed_document_dir(doc_dir: Path) -> bool:
    """Return whether a processed document directory uses the canonical hash suffix."""

    return re.match(r"^.+_[0-9a-fA-F]{8}$", doc_dir.name) is not None
