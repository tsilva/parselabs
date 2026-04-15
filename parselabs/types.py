"""Shared typed payloads and identifiers used across runtime boundaries."""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict

ReviewAction = Literal["accept", "reject", "clear", "missing_row"]
ReviewStatus = Literal["accepted", "rejected", "pending"]
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
BBoxTuple: TypeAlias = tuple[float, float, float, float]
PixelBBoxTuple: TypeAlias = tuple[int, int, int, int]


class RowIdentity(TypedDict):
    """Stable identity for one extracted row across review flows."""

    source_file: str
    page_number: int
    result_index: int


class BBox(TypedDict):
    """JSON-friendly bounding box coordinates."""

    left: float | int
    top: float | int
    right: float | int
    bottom: float | int


class PageLabResultPayload(TypedDict, total=False):
    """Persisted extracted row payload stored in per-page JSON."""

    raw_lab_name: str | None
    raw_section_name: str | None
    raw_value: str | None
    raw_lab_unit: str | None
    raw_reference_range: str | None
    raw_reference_min: float | None
    raw_reference_max: float | None
    raw_comments: str | None
    bbox_left: float | None
    bbox_top: float | None
    bbox_right: float | None
    bbox_bottom: float | None
    review_needed: bool
    review_reason: str
    review_status: Literal["accepted", "rejected"] | None
    review_completed_at: str | None


class PagePayload(TypedDict, total=False):
    """Persisted page-level extraction payload."""

    collection_date: str | None
    report_date: str | None
    lab_facility: str | None
    page_has_lab_data: bool | None
    source_file: str | None
    page_number: int
    lab_results: list[PageLabResultPayload]
    _extraction_failed: bool
    _failure_reason: str
    _retry_count: int
    review_missing_rows: list[dict[str, JsonValue]]


class ReviewRow(RowIdentity, total=False):
    """Flattened review dataframe row contract."""

    date: str | None
    raw_lab_name: str | None
    raw_section_name: str | None
    raw_value: str | None
    raw_lab_unit: str | None
    raw_reference_range: str | None
    raw_reference_min: float | None
    raw_reference_max: float | None
    raw_comments: str | None
    bbox_left: float | None
    bbox_top: float | None
    bbox_right: float | None
    bbox_bottom: float | None
    lab_name_standardized: str | None
    lab_unit_standardized: str | None
    lab_name: str | None
    value: float | int | str | None
    lab_unit: str | None
    reference_min: float | None
    reference_max: float | None
    review_needed: bool
    review_reason: str
    is_below_limit: bool | None
    is_above_limit: bool | None
    lab_type: str | None
    review_status: Literal["accepted", "rejected"] | None
    review_completed_at: str | None


class ReviewArtifactPayload(TypedDict, total=False):
    """Structured payload returned by review artifact flows."""

    done: bool
    profile: str
    row_id: str
    doc_dir: str
    doc_dir_name: str
    doc_stem: str
    source_pdf_path: str
    page_json_path: str
    page_image_path: str | None
    bbox_clip_path: str | None
    page_number: int
    result_index: int
    bbox: BBox | None
    bbox_pixels: BBox | None
    crop_box_pixels: BBox | None
    artifact_error: str | None
    stored_result: PageLabResultPayload
    artifacts_dir: str


class ReviewDecisionResult(TypedDict, total=False):
    """Structured result for one persisted review decision."""

    ok: bool
    profile: str
    row_id: str
    decision: str
    error: str
    doc_dir: str
    page_number: int
    result_index: int


class StandardizationNameMatch(TypedDict, total=False):
    """LLM response item for one raw lab-name mapping."""

    raw_lab_name: str
    raw_section_name: str | None
    standardized_name: str


class StandardizationUnitMatch(TypedDict, total=False):
    """LLM response item for one raw unit mapping."""

    raw_unit: str
    lab_name: str
    standardized_unit: str


class ApprovedCaseMetadata(TypedDict, total=False):
    """Metadata stored next to an approved regression fixture."""

    case_id: str
    original_filename: str
    stem: str
    file_hash: str
    profile: str | None
    approved_at: str
    reviewed_at: str
