"""Package-native row builders for review and final export flows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from parselabs.config import LabSpecsConfig
from parselabs.review_sync import (
    DOCUMENT_REVIEW_COLUMNS,
    build_document_review_dataframe,
    iter_processed_documents,
    load_document_review_rows,
    prepare_rows,
    transform_rows_to_final_export,
)
from parselabs.utils import ensure_columns


@dataclass(frozen=True)
class RowPipelineResult:
    """Rows plus validation stats from one pipeline transform."""

    frame: pd.DataFrame
    validation_stats: dict[str, int | dict[str, int]]


class RowPipeline:
    """Build review and export dataframes from canonical page JSON."""

    @staticmethod
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

    @staticmethod
    def build_export_rows(
        source: Path | Iterable[dict],
        lab_specs: LabSpecsConfig,
        *,
        accepted_only: bool,
        apply_standardization: bool = True,
    ) -> RowPipelineResult:
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
        return RowPipelineResult(frame=final_df, validation_stats=validation_stats)

    @staticmethod
    def build_corpus_review_rows(
        output_path: Path,
        lab_specs: LabSpecsConfig,
    ) -> pd.DataFrame:
        """Return the combined review dataset for every processed document."""

        review_frames = [
            RowPipeline.build_review_rows(document.doc_dir, lab_specs)
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
