"""Shared helpers for reviewed processed-document outputs."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from parselabs.config import CANONICAL_QUALITATIVE_SUFFIX, UNKNOWN_VALUE, LabSpecsConfig
from parselabs.export_schema import COLUMN_ORDER, COLUMN_SCHEMA, get_column_lists
from parselabs.normalization import (
    apply_dtype_conversions,
    apply_normalizations,
    classify_qualitative_value,
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
from parselabs.types import (
    PageLabResultPayload,
    PagePayload,
    PersistedReviewStatus,
    ReviewMissingRowRecord,
    ReviewRow,
    coerce_persisted_review_status,
)
from parselabs.utils import ensure_columns
from parselabs.validation import ValueValidator

logger = logging.getLogger(__name__)

EXTRACTION_FAILED_REASON = "EXTRACTION_FAILED"
UNKNOWN_LAB_MAPPING_REASON = "UNKNOWN_LAB_MAPPING"
UNKNOWN_UNIT_MAPPING_REASON = "UNKNOWN_UNIT_MAPPING"
AMBIGUOUS_PERCENTAGE_VARIANT_REASON = "AMBIGUOUS_PERCENTAGE_VARIANT"
SUSPICIOUS_REFERENCE_RANGE_REASON = "SUSPICIOUS_REFERENCE_RANGE"

_INFERRED_URINE_SECTION_BY_NAME = {
    "cor": "< TIPO II >",
    "ph": "< TIPO II >",
    "densidade a 15o c": "< TIPO II >",
    "proteinas": "Elementos anormais",
    "glicose": "Elementos anormais",
    "corpos cetonicos": "Elementos anormais",
    "bilirrubina": "Elementos anormais",
    "nitritos": "Elementos anormais",
    "sangue": "Elementos anormais",
    "urobilinogenio": "Elementos anormais",
    "celulas epiteliais": "EXAME MICROSCOPICO DO SEDIMENTO",
    "leucocitos": "EXAME MICROSCOPICO DO SEDIMENTO",
    "eritrocitos": "EXAME MICROSCOPICO DO SEDIMENTO",
}

_MIN_URINE_CONTEXT_ROWS = 3

DOCUMENT_REVIEW_COLUMNS = [
    "date",
    "source_file",
    "page_number",
    "result_index",
    "raw_lab_name",
    "raw_section_name",
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


def _normalize_section_inference_key(value: object) -> str:
    """Normalize raw labels into stable keys for section backfilling."""

    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("º", "o")
    return re.sub(r"\s+", " ", text)


def _backfill_missing_raw_sections(review_df: pd.DataFrame) -> pd.DataFrame:
    """Infer urine section labels for sectionless rows when the page context is clear."""

    required_columns = {"page_number", "raw_lab_name", "raw_section_name"}

    # Guard: Only dataframes with row identity and raw labels can be enriched.
    if review_df.empty or not required_columns.issubset(review_df.columns):
        return review_df

    enriched_df = review_df.copy()

    for _, page_df in enriched_df.groupby("page_number", sort=False):
        normalized_names = page_df["raw_lab_name"].fillna("").astype(str).map(_normalize_section_inference_key)
        urine_context_count = int(normalized_names.isin(_INFERRED_URINE_SECTION_BY_NAME).sum())

        # Guard: Sparse matches are too weak to infer a urine section safely.
        if urine_context_count < _MIN_URINE_CONTEXT_ROWS:
            continue

        for idx, normalized_name in zip(page_df.index, normalized_names, strict=False):
            current_section = enriched_df.at[idx, "raw_section_name"]

            # Preserve explicit section metadata from extraction.
            if pd.notna(current_section) and str(current_section).strip():
                continue

            inferred_section = _INFERRED_URINE_SECTION_BY_NAME.get(normalized_name)
            if inferred_section is None:
                continue

            enriched_df.at[idx, "raw_section_name"] = inferred_section

    return enriched_df


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


def _extract_page_payload_date(page_payload: PagePayload) -> str | None:
    """Return the canonical date stored on one page payload."""

    document_date = page_payload.get("collection_date") or page_payload.get("report_date")
    return None if document_date == "0000-00-00" else document_date


def _format_review_reason(
    review_reason: object,
    *,
    page_failed: bool,
    include_extraction_failed_reason: bool,
) -> str:
    """Return the persisted review reason text for one flattened row."""

    normalized_reason = str(review_reason or "").strip()
    if not page_failed or not include_extraction_failed_reason:
        return normalized_reason
    if EXTRACTION_FAILED_REASON in normalized_reason:
        return normalized_reason
    if normalized_reason and not normalized_reason.endswith(";"):
        normalized_reason = f"{normalized_reason}; "
    if normalized_reason:
        return f"{normalized_reason}{EXTRACTION_FAILED_REASON}; "
    return f"{EXTRACTION_FAILED_REASON}; "


def _build_flattened_review_row(
    result: PageLabResultPayload,
    *,
    document_date: str | None,
    source_file: str | None,
    page_number: int,
    result_index: int,
    page_failed: bool,
    include_extraction_failed_reason: bool,
) -> ReviewRow:
    """Return one flattened review-row payload from a persisted page result."""

    status = coerce_persisted_review_status(result.get("review_status"))
    return {
        "date": document_date,
        "source_file": source_file,
        "page_number": page_number,
        "result_index": result_index,
        "raw_lab_name": result.get("raw_lab_name"),
        "raw_section_name": result.get("raw_section_name"),
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
        "review_reason": _format_review_reason(
            result.get("review_reason"),
            page_failed=page_failed,
            include_extraction_failed_reason=include_extraction_failed_reason,
        ),
        "review_status": status,
        "review_completed_at": result.get("review_completed_at"),
    }


def _iter_flattened_review_rows(
    payload: PagePayload,
    *,
    source_file: str | None,
    page_number: int,
    document_date: str | None,
    include_extraction_failed_reason: bool,
    include_statuses: set[PersistedReviewStatus] | None = None,
) -> Iterable[ReviewRow]:
    """Yield flattened review rows for one persisted page payload."""

    page_results = payload.get("lab_results", [])
    if not isinstance(page_results, list):
        return

    page_failed = bool(payload.get("_extraction_failed"))
    for result_index, result in enumerate(page_results):
        if not isinstance(result, dict):
            continue

        typed_result: PageLabResultPayload = result
        status = coerce_persisted_review_status(typed_result.get("review_status"))
        if include_statuses is not None and status not in include_statuses:
            continue

        yield _build_flattened_review_row(
            typed_result,
            document_date=document_date,
            source_file=source_file,
            page_number=page_number,
            result_index=result_index,
            page_failed=page_failed,
            include_extraction_failed_reason=include_extraction_failed_reason,
        )


def _flatten_page_payloads(
    page_payloads: Iterable[PagePayload],
    *,
    accepted_only: bool = False,
) -> pd.DataFrame:
    """Flatten canonical page payloads into the shared review-row shape."""

    rows: list[ReviewRow] = []
    document_date: str | None = None

    for page_idx, payload in enumerate(page_payloads, start=1):
        # Skip malformed payloads so callers can pass best-effort collections.
        if not isinstance(payload, dict):
            continue

        page_number = int(payload.get("page_number") or page_idx)

        # Preserve the first usable document date across every row.
        if document_date is None:
            document_date = _extract_page_payload_date(payload)

        rows.extend(
            _iter_flattened_review_rows(
                payload,
                source_file=payload.get("source_file"),
                page_number=page_number,
                document_date=document_date,
                include_extraction_failed_reason=False,
                include_statuses={"accepted"} if accepted_only else None,
            )
        )

    # Guard: Empty payload collections still return the shared schema.
    if not rows:
        return pd.DataFrame(columns=DOCUMENT_REVIEW_COLUMNS)

    flattened_df = pd.DataFrame(rows)
    flattened_df = _backfill_missing_raw_sections(flattened_df)
    ensure_columns(flattened_df, DOCUMENT_REVIEW_COLUMNS, default=None)
    return flattened_df


def save_review_status(doc_dir: Path, page_number: int, result_index: int, status: str | None) -> tuple[bool, str]:
    """Persist a review decision to the page JSON backing a CSV row."""

    return save_review_status_in_store(doc_dir, page_number, result_index, status)


def save_missing_row_marker(doc_dir: Path, page_number: int, anchor_result_index: int) -> tuple[bool, str]:
    """Persist a missing-row marker to the page JSON backing the current review row."""

    return save_missing_row_marker_in_store(doc_dir, page_number, anchor_result_index)


def get_review_missing_rows(doc_dir: Path, page_number: int | None = None) -> list[ReviewMissingRowRecord]:
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


def _extract_document_date(page_payload: PagePayload, doc_dir: Path) -> str | None:
    """Extract the document date from page metadata or the filename."""

    # Prefer the explicit collection date captured by extraction.
    doc_date = _extract_page_payload_date(page_payload)

    # Ignore the legacy placeholder date used by bad model responses.
    if not doc_date:
        stem = get_document_stem(doc_dir)
        match = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
        if match:
            doc_date = match.group(1)

    return doc_date


def load_document_review_rows(
    doc_dir: Path,
    include_statuses: set[PersistedReviewStatus] | None = None,
) -> pd.DataFrame:
    """Load one review row per extracted result from page JSON files."""

    rows: list[ReviewRow] = []
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

        rows.extend(
            _iter_flattened_review_rows(
                page_payload,
                source_file=source_file,
                page_number=page_number,
                document_date=doc_date,
                include_extraction_failed_reason=True,
                include_statuses=include_statuses,
            )
        )

    # Guard: Empty results still return a dataframe with the expected columns.
    if not rows:
        return pd.DataFrame(columns=DOCUMENT_REVIEW_COLUMNS)

    review_df = pd.DataFrame(rows)

    # Apply the resolved document date to every row so later pages can still fill earlier blanks.
    if doc_date is not None:
        review_df["date"] = doc_date

    review_df = _backfill_missing_raw_sections(review_df)
    ensure_columns(review_df, DOCUMENT_REVIEW_COLUMNS, default=None)
    return review_df


def apply_cached_standardization(review_df: pd.DataFrame, lab_specs: LabSpecsConfig) -> pd.DataFrame:
    """Apply cache-backed name and unit mappings to extracted review rows."""

    # Guard: Skip standardization entirely when lab specs are unavailable.
    if not lab_specs.exists:
        ensure_columns(review_df, ["raw_section_name", "lab_name_standardized", "lab_unit_standardized"], default=None)
        return review_df

    # Ensure the optional section-context column exists before building contextual cache keys.
    ensure_columns(review_df, ["raw_section_name"], default=None)

    # Prefer explicit qualitative/quantitative companion mappings when the same assay is
    # printed twice on one page as both a numeric result and an interpreted label.
    raw_names = review_df["raw_lab_name"].fillna("").astype(str).tolist()
    raw_section_names = [str(value).strip() if pd.notna(value) and str(value).strip() else None for value in review_df["raw_section_name"].tolist()]
    base_name_contexts = list(zip(raw_names, raw_section_names, strict=False))
    variant_name_contexts = _build_variant_aware_name_contexts(review_df, base_name_contexts)
    lookup_contexts = list(dict.fromkeys(base_name_contexts + variant_name_contexts))
    name_map = standardize_lab_names(lookup_contexts)

    standardized_names: list[str] = []
    for base_context, variant_context in zip(base_name_contexts, variant_name_contexts, strict=False):
        standardized_name = name_map.get(variant_context, UNKNOWN_VALUE)

        # Fall back to the undecorated raw label when no variant-specific cache entry exists.
        if standardized_name == UNKNOWN_VALUE and variant_context != base_context:
            standardized_name = name_map.get(base_context, UNKNOWN_VALUE)

        standardized_names.append(standardized_name)

    review_df["lab_name_standardized"] = standardized_names
    review_df = _remap_boolean_companion_lab_names(review_df, lab_specs)
    review_df = _remap_unit_bearing_rows_off_qualitative_variants(review_df, lab_specs)

    # Standardize units only for rows whose lab names mapped successfully.
    unit_contexts: list[tuple[str, str]] = []
    for raw_unit, standardized_name in zip(review_df["raw_lab_unit"].fillna("").astype(str), review_df["lab_name_standardized"].fillna("").astype(str), strict=False):
        # Skip rows without a usable standardized lab name.
        if not standardized_name or standardized_name == UNKNOWN_VALUE:
            unit_contexts.append(("", ""))
            continue

        # Route explicit percent-vs-absolute units onto the correct sibling before cache lookup.
        effective_name = _infer_effective_lab_name_for_unit_lookup(
            raw_unit,
            standardized_name,
            lab_specs,
        )
        unit_contexts.append((raw_unit, effective_name))

    mapped_units = standardize_lab_units([context for context in unit_contexts if context != ("", "")])

    standardized_units: list[str | None] = []
    safe_missing_unit_primary_units = {"boolean", "mm/h", "pH", "unitless"}

    for idx, (raw_unit, standardized_name) in zip(review_df.index, unit_contexts, strict=False):
        # Leave units empty when the lab name did not map.
        if (raw_unit, standardized_name) == ("", ""):
            standardized_units.append(None)
            continue

        standardized_unit = mapped_units.get((raw_unit, standardized_name))

        # Preserve explicit conversion units when stale cache entries collapse them into
        # the primary unit before apply_unit_conversions() can apply the factor.
        if raw_unit.strip() != "":
            inferred_unit = _infer_explicit_conversion_unit_from_raw_unit(
                raw_unit,
                standardized_name,
                lab_specs,
            )
            if inferred_unit is not None:
                standardized_unit = inferred_unit

        # Explicit non-percent units should force percentage-vs-absolute siblings onto the
        # absolute path even when an old cache entry still points to the (%) variant.
        if raw_unit.strip() != "":
            inferred_unit = _infer_explicit_variant_unit_from_raw_unit(
                raw_unit,
                standardized_name,
                lab_specs,
            )
            if inferred_unit is not None:
                standardized_unit = inferred_unit

        # Blank raw units can still be recoverable when sibling variants have
        # distinct reference ranges in the source report.
        if raw_unit.strip() == "":
            inferred_unit = _infer_missing_variant_unit_from_ranges(
                review_df.loc[idx],
                standardized_name,
                lab_specs,
            )
            if inferred_unit is not None:
                standardized_unit = inferred_unit

        # Some labs are intrinsically unitless or use a conventional implied unit.
        # When extraction omits the printed unit entirely, prefer the primary unit
        # from lab_specs instead of forcing a cache entry for blank input.
        if standardized_unit == UNKNOWN_VALUE and raw_unit.strip() == "":
            primary_unit = lab_specs.get_primary_unit(standardized_name)
            if primary_unit in safe_missing_unit_primary_units or (
                primary_unit == "%" and standardized_name.endswith("(%)")
            ):
                standardized_unit = primary_unit

        standardized_units.append(standardized_unit)

    review_df["lab_unit_standardized"] = standardized_units
    review_df = _remap_percentage_variant_lab_names(review_df, lab_specs)
    review_df = _remap_complement_variant_lab_names(review_df, lab_specs)
    return review_df


def _build_variant_aware_name_contexts(
    review_df: pd.DataFrame,
    base_name_contexts: list[tuple[str, str | None]],
) -> list[tuple[str, str | None]]:
    """Decorate row names when one page prints both qualitative and numeric assay companions."""

    required_columns = {"page_number", "raw_lab_name", "raw_lab_unit", "raw_value"}

    # Guard: Missing row identity means callers can only use the undecorated cache path.
    if review_df.empty or not required_columns.issubset(review_df.columns):
        return base_name_contexts

    variant_contexts = list(base_name_contexts)
    helper_df = review_df.copy()
    helper_df["context_position"] = range(len(helper_df))
    helper_df["name_key"] = helper_df["raw_lab_name"].map(_normalize_section_inference_key)
    helper_df["section_key"] = helper_df["raw_section_name"].map(_normalize_section_inference_key)
    helper_df["is_qualitative_companion"] = helper_df.apply(_row_is_qualitative_companion, axis=1)
    helper_df["is_quantitative_companion"] = helper_df.apply(_row_is_quantitative_companion, axis=1)

    for _, group_df in helper_df.groupby(["page_number", "name_key", "section_key"], sort=False):
        # Guard: Companion decoration only applies when both forms are present in one source group.
        if not group_df["is_qualitative_companion"].any() or not group_df["is_quantitative_companion"].any():
            continue

        for _, row in group_df.iterrows():
            position = int(row["context_position"])
            raw_name, raw_section_name = base_name_contexts[position]

            if row["is_qualitative_companion"]:
                variant_contexts[position] = (f"{raw_name} (qualitative)", raw_section_name)
                continue

            if row["is_quantitative_companion"]:
                variant_contexts[position] = (f"{raw_name} (quantitative)", raw_section_name)

    return variant_contexts


def _remap_boolean_companion_lab_names(
    review_df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """Expose qualitative boolean rows as canonical `(Qualitative)` siblings."""

    required_columns = {"page_number", "raw_lab_name", "raw_section_name", "raw_lab_unit", "raw_value", "lab_name_standardized"}

    # Guard: Remapping requires the raw fields needed to identify qualitative rows safely.
    if review_df.empty or not required_columns.issubset(review_df.columns):
        return review_df

    remapped_names = review_df["lab_name_standardized"].tolist()
    helper_df = review_df.copy()
    helper_df["context_position"] = range(len(helper_df))
    helper_df["name_key"] = helper_df["raw_lab_name"].map(_normalize_section_inference_key)
    helper_df["section_key"] = helper_df["raw_section_name"].map(_normalize_section_inference_key)
    helper_df["is_qualitative_companion"] = helper_df.apply(_row_is_qualitative_companion, axis=1)
    remap_count = 0

    for _, row in helper_df[helper_df["is_qualitative_companion"]].iterrows():
        position = int(row["context_position"])
        standardized_name = remapped_names[position]

        # Skip rows without a usable base name.
        if pd.isna(standardized_name) or standardized_name == UNKNOWN_VALUE or not str(standardized_name).strip():
            continue

        qualitative_variant = lab_specs.get_qualitative_variant(str(standardized_name))

        # Guard: Leave rows unchanged when there is no boolean-backed sibling to expose.
        if qualitative_variant is None or qualitative_variant == standardized_name:
            continue

        remapped_names[position] = qualitative_variant
        remap_count += 1

    if remap_count:
        logger.info("[rows] Remapped %s qualitative boolean row(s) to qualitative siblings", remap_count)

    review_df["lab_name_standardized"] = remapped_names
    return review_df


def _row_is_qualitative_companion(row: pd.Series) -> bool:
    """Return True when one row is a text-only interpretation of an assay result."""

    raw_unit = _normalize_optional_text(row.get("raw_lab_unit"))
    raw_value = _normalize_optional_text(row.get("raw_value"))

    # Guard: Explicit units already indicate this row is the numeric companion.
    if raw_unit:
        return False

    return classify_qualitative_value(raw_value) is not None


def _row_is_quantitative_companion(row: pd.Series) -> bool:
    """Return True when one row carries the numeric/unit-bearing companion result."""

    raw_unit = _normalize_optional_text(row.get("raw_lab_unit"))

    # Guard: Unit-bearing rows already carry the numeric assay form.
    if raw_unit:
        return True

    raw_value = _normalize_optional_text(row.get("raw_value")).replace(",", ".")
    return pd.notna(pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0])


def _remap_unit_bearing_rows_off_qualitative_variants(
    review_df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """Move explicit unit-bearing rows back onto the quantitative base analyte."""

    required_columns = {"lab_name_standardized", "raw_lab_unit"}

    # Guard: Remapping requires both the standardized name and explicit raw units.
    if review_df.empty or not required_columns.issubset(review_df.columns):
        return review_df

    remapped_names = review_df["lab_name_standardized"].tolist()
    remap_count = 0

    for idx in review_df.index:
        standardized_name = remapped_names[idx]
        raw_unit = _normalize_optional_text(review_df.at[idx, "raw_lab_unit"])

        if not raw_unit:
            continue

        if pd.isna(standardized_name) or standardized_name == UNKNOWN_VALUE:
            continue

        canonical_name = lab_specs.get_canonical_lab_name(str(standardized_name))
        if not canonical_name.endswith(CANONICAL_QUALITATIVE_SUFFIX):
            continue

        base_name = canonical_name.removesuffix(CANONICAL_QUALITATIVE_SUFFIX).strip()
        if lab_specs.resolve_lab_name(base_name) is None:
            continue

        remapped_names[idx] = base_name
        remap_count += 1

    if remap_count:
        logger.info(
            "[rows] Remapped %s unit-bearing row(s) off qualitative siblings",
            remap_count,
        )

    review_df["lab_name_standardized"] = remapped_names
    return review_df


def _normalize_optional_text(value: object) -> str:
    """Return a stripped text token while collapsing NaN-like values to blank."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def _normalize_conversion_unit_token(value: object) -> str:
    """Normalize unit text for explicit conversion-unit matching."""

    normalized_value = unicodedata.normalize("NFKC", str(value).strip().lower())
    normalized_value = normalized_value.replace("μ", "µ")
    normalized_value = normalized_value.replace("ˆ", "^")
    normalized_value = normalized_value.replace("×", "x")
    normalized_value = re.sub(r"\s+", "", normalized_value)
    return normalized_value


def _iter_conversion_units_for_lab(
    standardized_name: str,
    lab_specs: LabSpecsConfig,
) -> list[str]:
    """Return the primary and alternative units that can be converted for one lab."""

    candidate_units: list[str] = []
    primary_unit = lab_specs.get_primary_unit(standardized_name)

    # Include the configured primary unit first so exact raw-primary matches stay stable.
    if isinstance(primary_unit, str) and primary_unit.strip():
        candidate_units.append(primary_unit)

    lab_config = lab_specs.specs.get(standardized_name, {})
    for alternative in lab_config.get("alternatives", []):
        unit = alternative.get("unit")

        # Skip malformed alternative entries instead of crashing the review pipeline.
        if not isinstance(unit, str) or not unit.strip():
            continue

        candidate_units.append(unit)

    return list(dict.fromkeys(candidate_units))


def _infer_explicit_conversion_unit_from_raw_unit(
    raw_unit: str,
    standardized_name: str,
    lab_specs: LabSpecsConfig,
) -> str | None:
    """Preserve explicit conversion units so downstream factor application can run."""

    raw_unit_token = _normalize_conversion_unit_token(raw_unit)

    # Guard: Blank raw units have no explicit conversion path to preserve.
    if not raw_unit_token:
        return None

    for candidate_unit in _iter_conversion_units_for_lab(standardized_name, lab_specs):
        # Match directly against the canonical conversion units configured for this lab.
        if _normalize_conversion_unit_token(candidate_unit) == raw_unit_token:
            return candidate_unit

    return None


def _infer_missing_variant_unit_from_ranges(
    row: pd.Series,
    standardized_name: str,
    lab_specs: LabSpecsConfig,
) -> str | None:
    """Infer a missing unit when percentage and absolute sibling variants have distinct ranges."""

    candidate_names = [standardized_name]

    percentage_variant = lab_specs.get_percentage_variant(standardized_name)
    if percentage_variant is not None:
        candidate_names.append(percentage_variant)

    non_percentage_variant = lab_specs.get_non_percentage_variant(standardized_name)
    if non_percentage_variant is not None:
        candidate_names.append(non_percentage_variant)

    # Guard: No sibling variants means there is no deterministic missing-unit inference path.
    candidate_names = list(dict.fromkeys(candidate_names))
    if len(candidate_names) < 2:
        return None

    raw_min = row.get("raw_reference_min")
    raw_max = row.get("raw_reference_max")
    observed_value = row.get("raw_value")

    scored_candidates: list[tuple[float, str]] = []
    for candidate_name in candidate_names:
        score = _score_variant_range_match(candidate_name, raw_min, raw_max, observed_value, lab_specs)
        if score is None:
            continue
        scored_candidates.append((score, candidate_name))

    # Guard: Missing or unusable range metadata leaves the unit unresolved for review.
    if len(scored_candidates) < 2:
        return None

    scored_candidates.sort(key=lambda item: item[0])
    best_score, best_name = scored_candidates[0]
    next_score = scored_candidates[1][0]

    # Require a clear winner so we do not silently force ambiguous sibling variants.
    if next_score - best_score < 0.25:
        return None

    return lab_specs.get_primary_unit(best_name)


def _infer_explicit_variant_unit_from_raw_unit(
    raw_unit: str,
    standardized_name: str,
    lab_specs: LabSpecsConfig,
) -> str | None:
    """Infer percentage-vs-absolute sibling units from an explicit raw unit token."""

    raw_unit_text = str(raw_unit).strip()

    # Guard: Blank raw units are handled by the range-based inference path instead.
    if not raw_unit_text:
        return None

    percentage_variant = lab_specs.get_percentage_variant(standardized_name)
    non_percentage_variant = lab_specs.get_non_percentage_variant(standardized_name)

    # Guard: Labs without sibling variants do not need explicit unit correction.
    if percentage_variant is None and non_percentage_variant is None:
        return None

    # Explicit percent markers belong to the configured percentage sibling.
    if _raw_unit_looks_percentage(raw_unit_text):
        return "%"

    absolute_variant = non_percentage_variant or standardized_name
    inferred_conversion_unit = _infer_explicit_conversion_unit_from_raw_unit(
        raw_unit_text,
        absolute_variant,
        lab_specs,
    )

    # Preserve explicit conversion units like /mm3 so factor application can still run.
    if inferred_conversion_unit is not None:
        return inferred_conversion_unit

    # Incomplete OCR like ``x 10³/`` or qualitative labels like ``V.abs.`` still
    # unambiguously point to the absolute differential-count sibling.
    if _raw_unit_looks_absolute_cell_count(raw_unit_text):
        return lab_specs.get_primary_unit(absolute_variant)

    return lab_specs.get_primary_unit(absolute_variant)


def _infer_effective_lab_name_for_unit_lookup(
    raw_unit: str,
    standardized_name: str,
    lab_specs: LabSpecsConfig,
) -> str:
    """Choose the sibling analyte that should own unit-cache lookup for one row."""

    raw_unit_text = str(raw_unit).strip()

    # Guard: Blank raw units do not provide a decisive sibling signal before cache lookup.
    if not raw_unit_text:
        return standardized_name

    percentage_variant = lab_specs.get_percentage_variant(standardized_name)
    non_percentage_variant = lab_specs.get_non_percentage_variant(standardized_name)

    # Guard: Labs without a sibling pair keep their original standardized name.
    if percentage_variant is None and non_percentage_variant is None:
        return standardized_name

    # Explicit percent markers belong on the configured percentage sibling.
    if _raw_unit_looks_percentage(raw_unit_text):
        return percentage_variant or standardized_name

    # Collapse absolute-count and conversion-unit rows onto the non-percentage sibling.
    absolute_variant = non_percentage_variant or standardized_name
    inferred_conversion_unit = _infer_explicit_conversion_unit_from_raw_unit(
        raw_unit_text,
        absolute_variant,
        lab_specs,
    )

    # Explicit conversion units like /mm3 should query the absolute sibling cache entry.
    if inferred_conversion_unit is not None:
        return absolute_variant

    # Incomplete OCR like ``x 10³/`` or qualitative labels like ``V.abs.`` still
    # unambiguously point to the absolute differential-count sibling.
    if _raw_unit_looks_absolute_cell_count(raw_unit_text):
        return absolute_variant

    # Non-decisive raw units keep the name-cache result unchanged.
    return standardized_name


def _raw_unit_looks_percentage(raw_unit: str) -> bool:
    """Return True when the raw unit text explicitly denotes a percentage."""

    normalized_unit = str(raw_unit).strip().lower()

    # Recognize literal percent units as well as common textual percentage labels.
    return "%" in normalized_unit or "percent" in normalized_unit


def _raw_unit_looks_absolute_cell_count(raw_unit: str) -> bool:
    """Return True when the raw unit text clearly denotes an absolute cell count."""

    normalized_unit = unicodedata.normalize("NFKC", str(raw_unit).strip().lower())
    normalized_unit = normalized_unit.replace("μ", "µ")
    compact_unit = re.sub(r"\s+", "", normalized_unit)

    # Recognize explicit scientific-notation counts after OCR or NFKC folding.
    if re.search(r"(?:^|x)10(?:\^)?(?:3|9)/", compact_unit):
        return True

    return any(
        marker in compact_unit
        for marker in (
            "v.abs",
            "vabs",
            "10^9/",
            "10^3/",
            "10e3/",
            "10³/",
            "x10^9/",
            "x10^3/",
            "x10e3/",
            "x10³/",
            "/mm3",
            "/mm³",
            "/ul",
            "/µl",
        )
    )


def _score_variant_range_match(
    candidate_name: str,
    raw_min: object,
    raw_max: object,
    observed_value: object,
    lab_specs: LabSpecsConfig,
) -> float | None:
    """Score how well one candidate lab variant matches the observed report range metadata."""

    expected_range = lab_specs.specs.get(candidate_name, {}).get("ranges", {}).get("default")
    if not isinstance(expected_range, list) or len(expected_range) < 2:
        return None

    expected_min, expected_max = expected_range[0], expected_range[1]
    score = 0.0
    comparisons = 0

    if pd.notna(raw_min) and isinstance(expected_min, (int, float)):
        score += _scaled_range_delta(float(raw_min), float(expected_min))
        comparisons += 1

    if pd.notna(raw_max) and isinstance(expected_max, (int, float)):
        score += _scaled_range_delta(float(raw_max), float(expected_max))
        comparisons += 1

    if comparisons == 0 and pd.notna(observed_value):
        parsed_value = pd.to_numeric(pd.Series([observed_value]).map(lambda value: str(value).replace(",", ".")), errors="coerce").iloc[0]
        if pd.notna(parsed_value) and isinstance(expected_min, (int, float)) and isinstance(expected_max, (int, float)):
            midpoint = (float(expected_min) + float(expected_max)) / 2
            score += _scaled_range_delta(float(parsed_value), midpoint)
            comparisons += 1

    if comparisons == 0:
        return None

    return score / comparisons


def _scaled_range_delta(observed: float, expected: float) -> float:
    """Return a unit-agnostic distance score between two range bounds."""

    if expected == 0:
        return abs(observed)

    return abs(observed - expected) / max(abs(expected), 1.0)


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


def _remap_complement_variant_lab_names(
    review_df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """Disambiguate CH50 placeholder mappings into C3/C4 siblings from source ranges."""

    required_columns = {
        "lab_name_standardized",
        "raw_lab_unit",
        "raw_reference_min",
        "raw_reference_max",
        "lab_unit_standardized",
    }

    # Guard: Complement disambiguation requires the full source-range context.
    if review_df.empty or not required_columns.issubset(review_df.columns):
        return review_df

    remapped_names = review_df["lab_name_standardized"].tolist()
    remapped_units = review_df["lab_unit_standardized"].tolist()
    remap_count = 0
    complement_candidates = ("Blood - Complement C3", "Blood - Complement C4")

    for idx in review_df.index:
        standardized_name = remapped_names[idx]
        if standardized_name != "Blood - Complement CH50":
            continue

        raw_unit = _normalize_optional_text(review_df.at[idx, "raw_lab_unit"])
        if _normalize_conversion_unit_token(raw_unit) != _normalize_conversion_unit_token("mg/dL"):
            continue

        scored_candidates: list[tuple[float, str]] = []
        for candidate_name in complement_candidates:
            score = _score_variant_range_match(
                candidate_name,
                review_df.at[idx, "raw_reference_min"],
                review_df.at[idx, "raw_reference_max"],
                review_df.at[idx, "raw_value"],
                lab_specs,
            )
            if score is not None:
                scored_candidates.append((score, candidate_name))

        if not scored_candidates:
            continue

        scored_candidates.sort(key=lambda item: item[0])
        best_score, best_name = scored_candidates[0]
        next_score = scored_candidates[1][0] if len(scored_candidates) > 1 else None

        # Guard: Keep ambiguous rows on CH50 so review can surface them later.
        if next_score is not None and next_score - best_score < 0.25:
            continue

        remapped_names[idx] = best_name
        remapped_units[idx] = lab_specs.get_primary_unit(best_name)
        remap_count += 1

    if remap_count:
        logger.info(
            "[rows] Remapped %s complement row(s) from CH50 placeholders to C3/C4 siblings",
            remap_count,
        )

    review_df["lab_name_standardized"] = remapped_names
    review_df["lab_unit_standardized"] = remapped_units
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


def _remove_review_reason_code(
    review_df: pd.DataFrame,
    reason_code: str,
) -> pd.DataFrame:
    """Remove one review reason code and clear review_needed when no reasons remain."""

    if review_df.empty or "review_reason" not in review_df.columns:
        return review_df

    updated_reasons = (
        review_df["review_reason"]
        .fillna("")
        .astype(str)
        .apply(
            lambda value: "; ".join(
                part for part in [piece.strip() for piece in value.split(";")] if part and part != reason_code
            )
        )
        .apply(lambda value: f"{value}; " if value else "")
    )
    review_df["review_reason"] = updated_reasons

    if "review_needed" in review_df.columns:
        review_df["review_needed"] = updated_reasons.str.strip().ne("")

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
        if std_unit == "%" and not std_name.endswith("(%)") and lab_specs.get_percentage_variant(str(std_name)) is not None:
            ambiguous_indices.append(idx)
            continue

        # Non-percentage units paired with percentage names are also ambiguous when remapping could not resolve them.
        if std_unit != "%" and std_name.endswith("(%)") and lab_specs.get_non_percentage_variant(str(std_name)) is not None:
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

        # Absolute-unit protein fraction rows often print percentage ranges alongside the
        # mass concentration. When the configured (%) sibling is a clearly better match,
        # keep the row reviewable through its mapped unit rather than flagging the range.
        if unit != "%" and _reference_range_matches_percentage_sibling_better(str(std_name), ref_min, ref_max, review_df.at[idx, "raw_value"], lab_specs):
            continue

        expected_range = lab_specs._specs.get(std_name, {}).get("ranges", {}).get("default", [])

        # Skip labs without expected ranges to compare against.
        if len(expected_range) < 2:
            continue

        expected_min, expected_max = expected_range[0], expected_range[1]

        ratios: list[float] = []
        if isinstance(expected_min, (int, float)) and expected_min != 0:
            ratios.append(abs(ref_min / expected_min))
        if isinstance(expected_max, (int, float)) and expected_max != 0:
            ratios.append(abs(ref_max / expected_max))

        if ratios and all(ratio > 10 or ratio < 0.1 for ratio in ratios):
            suspicious_indices.append(idx)

    return _append_review_reason_code(
        review_df,
        review_df.index.isin(suspicious_indices),
        SUSPICIOUS_REFERENCE_RANGE_REASON,
    )


def _reference_range_matches_percentage_sibling_better(
    standardized_name: str,
    raw_min: object,
    raw_max: object,
    observed_value: object,
    lab_specs: LabSpecsConfig,
) -> bool:
    """Return True when the report range aligns with the (%) sibling more than the absolute lab."""

    percentage_variant = lab_specs.get_percentage_variant(standardized_name)

    # Guard: No (%) sibling means there is no alternate scale to compare against.
    if percentage_variant is None:
        return False

    # Percentage-style report ranges should stay within ordinary percentage bounds.
    if pd.notna(raw_min) and float(raw_min) < 0:
        return False
    if pd.notna(raw_max) and float(raw_max) > 100:
        return False

    current_score = _score_variant_range_match(standardized_name, raw_min, raw_max, observed_value, lab_specs)
    percentage_score = _score_variant_range_match(percentage_variant, raw_min, raw_max, observed_value, lab_specs)

    # Guard: Missing comparable range metadata leaves the regular suspicious-range logic in place.
    if percentage_score is None:
        return False
    if current_score is None:
        return True

    # Require a meaningful improvement so close calls still surface to review.
    return (current_score - percentage_score) >= 0.25


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
            prepared_df = _remove_review_reason_code(prepared_df, "DUPLICATE_ENTRY")
        prepared_df = _add_export_column_aliases(prepared_df)
        validator = ValueValidator(lab_specs)
        prepared_df = validator.validate(prepared_df)
        return prepared_df, validator.validation_stats

    raise ValueError(f"Unsupported row-preparation mode: {mode}")
