"""Shared helpers for reviewed processed-document outputs."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
from parselabs.store import (
    DocumentRef,
    get_document_csv_path,
    get_document_stem,
    parse_page_number,
    read_page_payload,
)
from parselabs.store import (
    count_review_missing_rows as count_review_missing_rows_in_store,
)
from parselabs.store import (
    get_page_image_path as get_page_image_path_from_store,
)
from parselabs.store import (
    get_review_missing_rows as get_review_missing_rows_from_store,
)
from parselabs.store import (
    iter_processed_documents as iter_processed_documents_from_store,
)
from parselabs.store import (
    save_missing_row_marker as save_missing_row_marker_in_store,
)
from parselabs.store import (
    save_review_status as save_review_status_in_store,
)
from parselabs.utils import ensure_columns
from parselabs.validation import ValueValidator

logger = logging.getLogger(__name__)

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
    "bbox_left",
    "bbox_top",
    "bbox_right",
    "bbox_bottom",
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

ProcessedDocument = DocumentRef


@dataclass(frozen=True)
class RowBuildResult:
    """Rows plus validation stats from one row-build request."""

    frame: pd.DataFrame
    validation_stats: dict[str, int | dict[str, int]]


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

    return iter_processed_documents_from_store(output_path)


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
    review_df, _ = prepare_rows(review_df, lab_specs, mode="review")

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
    csv_path = get_document_csv_path(doc_dir)
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

    prepared_df, validation_stats = prepare_rows(
        rows_df,
        lab_specs,
        mode="export",
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


def build_review_rows(
    source: Path | Iterable[dict],
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """Return review rows for a processed document or page-payload iterable."""

    # Document directories already have a stable JSON-to-review builder.
    if isinstance(source, Path):
        return build_document_review_dataframe(source, lab_specs)

    rows_df = _flatten_page_payloads(source)

    # Guard: Empty payload collections still return a stable review schema.
    if rows_df.empty:
        return pd.DataFrame(columns=DOCUMENT_REVIEW_COLUMNS)

    prepared_df, _ = prepare_rows(rows_df, lab_specs, mode="review")
    ensure_columns(prepared_df, DOCUMENT_REVIEW_COLUMNS, default=None)
    return prepared_df[DOCUMENT_REVIEW_COLUMNS].copy()


def build_export_rows(
    source: Path | Iterable[dict],
    lab_specs: LabSpecsConfig,
    *,
    accepted_only: bool,
    apply_standardization: bool = True,
) -> RowBuildResult:
    """Return canonical export rows for a document or page-payload iterable."""

    if isinstance(source, Path):
        statuses = {"accepted"} if accepted_only else None
        rows_df = load_document_review_rows(source, include_statuses=statuses)
    else:
        rows_df = _flatten_page_payloads(source, accepted_only=accepted_only)

    final_df, validation_stats = transform_rows_to_final_export(
        rows_df,
        lab_specs,
        apply_standardization=apply_standardization,
    )
    return RowBuildResult(frame=final_df, validation_stats=validation_stats)


def build_corpus_review_rows(
    output_path: Path,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """Return the combined review dataset for every processed document."""

    review_frames = [
        build_review_rows(document.doc_dir, lab_specs)
        for document in iter_processed_documents(output_path)
    ]

    # Guard: No processed documents means there is nothing to concatenate.
    if not review_frames:
        return pd.DataFrame(columns=DOCUMENT_REVIEW_COLUMNS)

    return pd.concat(review_frames, ignore_index=True, sort=False)


def _flatten_page_payloads(
    page_payloads: Iterable[dict],
    *,
    accepted_only: bool = False,
) -> pd.DataFrame:
    """Flatten canonical page payloads into the shared review-row shape."""

    rows: list[dict] = []
    document_date: str | None = None

    for page_idx, payload in enumerate(page_payloads, start=1):
        # Skip malformed payloads so callers can pass best-effort collections.
        if not isinstance(payload, dict):
            continue

        page_number = int(payload.get("page_number") or page_idx)
        page_failed = bool(payload.get("_extraction_failed"))
        page_results = payload.get("lab_results", [])

        # Preserve the first usable document date across every row.
        if document_date is None:
            document_date = payload.get("collection_date") or payload.get("report_date")
            if document_date == "0000-00-00":
                document_date = None

        # Skip non-list payloads so one malformed page does not poison the batch.
        if not isinstance(page_results, list):
            continue

        for result_index, result in enumerate(page_results):
            # Skip malformed result payloads.
            if not isinstance(result, dict):
                continue

            status_text = str(result.get("review_status") or "").strip().lower()
            status = status_text if status_text in {"accepted", "rejected"} else None

            # Export-only callers may request just accepted reviewed rows.
            if accepted_only and status != "accepted":
                continue

            rows.append(
                {
                    "date": document_date,
                    "source_file": payload.get("source_file"),
                    "page_number": page_number,
                    "result_index": result_index,
                    "raw_lab_name": result.get("raw_lab_name"),
                    "raw_value": result.get("raw_value"),
                    "raw_lab_unit": result.get("raw_lab_unit"),
                    "raw_reference_range": result.get("raw_reference_range"),
                    "raw_reference_min": result.get("raw_reference_min"),
                    "raw_reference_max": result.get("raw_reference_max"),
                    "raw_comments": result.get("raw_comments"),
                    "bbox_left": result.get("bbox_left"),
                    "bbox_top": result.get("bbox_top"),
                    "bbox_right": result.get("bbox_right"),
                    "bbox_bottom": result.get("bbox_bottom"),
                    "review_needed": bool(result.get("review_needed")) or page_failed,
                    "review_reason": str(result.get("review_reason") or "").strip(),
                    "review_status": status,
                    "review_completed_at": result.get("review_completed_at"),
                }
            )

    # Guard: Empty payload collections still return the shared schema.
    if not rows:
        return pd.DataFrame(columns=DOCUMENT_REVIEW_COLUMNS)

    flattened_df = pd.DataFrame(rows)
    ensure_columns(flattened_df, DOCUMENT_REVIEW_COLUMNS, default=None)
    return flattened_df


def save_review_status(doc_dir: Path, page_number: int, result_index: int, status: str | None) -> tuple[bool, str]:
    """Persist a review decision to the page JSON backing a CSV row."""

    return save_review_status_in_store(doc_dir, page_number, result_index, status)


def save_missing_row_marker(doc_dir: Path, page_number: int, anchor_result_index: int) -> tuple[bool, str]:
    """Persist a missing-row marker to the page JSON backing the current review row."""

    return save_missing_row_marker_in_store(doc_dir, page_number, anchor_result_index)


def get_review_missing_rows(doc_dir: Path, page_number: int | None = None) -> list[dict]:
    """Return unresolved missing-row markers for a document or page."""

    return get_review_missing_rows_from_store(doc_dir, page_number=page_number)


def count_review_missing_rows(doc_dir: Path, page_number: int | None = None) -> int:
    """Return the number of unresolved missing-row markers for a document or page."""

    return count_review_missing_rows_in_store(doc_dir, page_number=page_number)


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
    raise ReviewStateError(f"{get_document_stem(doc_dir)} is not fixture-ready: {issue_text}.")


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

    return get_page_image_path_from_store(doc_dir, page_number)


def _extract_document_date(page_payload: dict, doc_dir: Path) -> str | None:
    """Extract the document date from page metadata or the filename."""

    # Prefer the explicit collection date captured by extraction.
    doc_date = page_payload.get("collection_date") or page_payload.get("report_date")

    # Ignore the legacy placeholder date used by bad model responses.
    if doc_date == "0000-00-00":
        doc_date = None

    # Fall back to a YYYY-MM-DD token in the document stem when metadata is absent.
    if not doc_date:
        stem = get_document_stem(doc_dir)
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
    source_file = f"{get_document_stem(doc_dir)}.csv"
    doc_date: str | None = None

    # Read each page JSON once and flatten its lab_results into row records.
    for page_json_path in page_json_paths:
        page_number = parse_page_number(page_json_path)

        # Skip malformed filenames because they cannot round-trip to review actions.
        if page_number is None:
            continue

        page_payload = read_page_payload(page_json_path)

        # Guard: Invalid JSON files should not break the whole review UI.
        if page_payload is None:
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

            status_text = str(result.get("review_status") or "").strip().lower()
            status = status_text if status_text in {"accepted", "rejected"} else None
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
                    "bbox_left": result.get("bbox_left"),
                    "bbox_top": result.get("bbox_top"),
                    "bbox_right": result.get("bbox_right"),
                    "bbox_bottom": result.get("bbox_bottom"),
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
    review_df = _remap_percentage_variant_lab_names(review_df, lab_specs)
    return review_df


def _remap_percentage_variant_lab_names(
    review_df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """Canonicalize percentage-vs-absolute sibling lab names from standardized units."""

    # Guard: Remapping requires both standardized columns and loaded lab specs.
    if (
        review_df.empty
        or not lab_specs.exists
        or "lab_name_standardized" not in review_df.columns
        or "lab_unit_standardized" not in review_df.columns
    ):
        return review_df

    remapped_names: list[object] = []
    remap_count = 0

    for idx in review_df.index:
        standardized_name = review_df.at[idx, "lab_name_standardized"]
        standardized_unit = review_df.at[idx, "lab_unit_standardized"]

        # Keep rows without a usable standardized lab name unchanged.
        if pd.isna(standardized_name) or standardized_name == UNKNOWN_VALUE or not str(standardized_name).strip():
            remapped_names.append(standardized_name)
            continue

        standardized_name = str(standardized_name)

        # Keep rows without an explicit standardized unit unchanged.
        if pd.isna(standardized_unit) or standardized_unit in {"", UNKNOWN_VALUE}:
            remapped_names.append(standardized_name)
            continue

        # Percentage units belong on the configured (%) sibling when one exists.
        if standardized_unit == "%":
            percentage_variant = lab_specs.get_percentage_variant(standardized_name)
            if percentage_variant is None:
                remapped_names.append(standardized_name)
                continue

            remapped_names.append(percentage_variant)
            remap_count += int(percentage_variant != standardized_name)
            continue

        # Non-percentage units belong on the configured absolute sibling when one exists.
        non_percentage_variant = lab_specs.get_non_percentage_variant(standardized_name)
        if non_percentage_variant is None:
            remapped_names.append(standardized_name)
            continue

        remapped_names.append(non_percentage_variant)
        remap_count += int(non_percentage_variant != standardized_name)

    review_df["lab_name_standardized"] = remapped_names

    # Log only real name changes so rebuild output stays readable.
    if remap_count > 0:
        logger.info(
            "[rows] Remapped %s percentage-variant lab name(s) from standardized units",
            remap_count,
        )

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
    """Flag rows whose percentage-vs-absolute interpretation still cannot be resolved."""

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

        # Missing units stay ambiguous when the lab has a sibling percentage variant.
        if pd.isna(std_unit) or std_unit in {"", UNKNOWN_VALUE}:
            if (
                lab_specs.get_percentage_variant(str(std_name)) is not None
                or lab_specs.get_non_percentage_variant(str(std_name)) is not None
            ):
                ambiguous_indices.append(idx)
            continue

        # Percentage units paired with non-percentage names remain ambiguous only when no sibling can fix them.
        if std_unit == "%" and not std_name.endswith("(%)"):
            ambiguous_indices.append(idx)
            continue

        # Non-percentage units paired with percentage names are also ambiguous when remapping could not resolve them.
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


def prepare_rows(
    rows_df: pd.DataFrame,
    context: LabSpecsConfig,
    *,
    mode: str,
    include_statuses: set[str] | None = None,
    apply_standardization: bool = True,
) -> tuple[pd.DataFrame, dict[str, int | dict[str, int]]]:
    """Prepare extracted rows for either review rendering or final export."""

    lab_specs = context

    # Guard: Empty inputs still return stable validation stats.
    if rows_df.empty:
        return rows_df, {"total_rows": 0, "rows_flagged": 0, "flags_by_reason": {}}

    prepared_df = rows_df.copy()

    # Filter by persisted review decision when the caller wants only a subset.
    if include_statuses is not None and "review_status" in prepared_df.columns:
        allowed_statuses = {str(status).strip().lower() for status in include_statuses}
        status_series = prepared_df["review_status"].fillna("").astype(str).str.strip().str.lower()
        prepared_df = prepared_df[status_series.isin(allowed_statuses)].reset_index(drop=True)

    # Guard: Filtering can legitimately remove every row.
    if prepared_df.empty:
        return prepared_df, {"total_rows": 0, "rows_flagged": 0, "flags_by_reason": {}}

    # Apply cache-backed standardization once so both modes share one mapping path.
    if apply_standardization:
        prepared_df = apply_cached_standardization(prepared_df, lab_specs)

    # Normalize raw values and reference bounds before any mode-specific handling.
    prepared_df = apply_normalizations(prepared_df, lab_specs)
    prepared_df = _flag_review_ambiguities(prepared_df, lab_specs)

    # Review mode keeps every row visible but still surfaces duplicates and validation flags.
    if mode == "review":
        prepared_df = flag_duplicate_entries(prepared_df)
        prepared_df = _add_export_column_aliases(prepared_df)
        validator = ValueValidator(lab_specs)
        prepared_df = validator.validate(prepared_df)
        return prepared_df, validator.validation_stats

    # Guard: Export mode only publishes rows with resolved mappings and publishable units.
    if mode == "export":
        prepared_df = _filter_exportable_rows(prepared_df)
        if prepared_df.empty:
            return prepared_df, {"total_rows": 0, "rows_flagged": 0, "flags_by_reason": {}}

        prepared_df = flag_duplicate_entries(prepared_df)
        if lab_specs.exists:
            prepared_df = deduplicate_results(prepared_df, lab_specs)
        prepared_df = _add_export_column_aliases(prepared_df)
        validator = ValueValidator(lab_specs)
        prepared_df = validator.validate(prepared_df)
        return prepared_df, validator.validation_stats

    raise ValueError(f"Unsupported row-preparation mode: {mode}")


def _prepare_rows_for_review(
    review_df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """Normalize extracted rows into the review dataframe shape."""

    prepared_df, _ = prepare_rows(review_df, lab_specs, mode="review")
    return prepared_df


def _prepare_rows_for_export(
    rows_df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
    apply_standardization: bool,
) -> tuple[pd.DataFrame, dict[str, int | dict[str, int]]]:
    """Normalize, deduplicate, and validate extracted rows for final export."""

    return prepare_rows(
        rows_df,
        lab_specs,
        mode="export",
        apply_standardization=apply_standardization,
    )


def _is_hashed_document_dir(doc_dir: Path) -> bool:
    """Return whether a processed document directory uses the canonical hash suffix."""

    return re.match(r"^.+_[0-9a-fA-F]{8}$", doc_dir.name) is not None
