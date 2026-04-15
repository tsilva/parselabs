"""Shared typed payloads and identifiers used across runtime boundaries."""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict

ReviewAction = Literal["accept", "reject", "clear", "missing_row"]
PersistedReviewStatus = Literal["accepted", "rejected"]
ReviewStatus = Literal["accepted", "rejected", "pending"]
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
BBoxTuple: TypeAlias = tuple[float, float, float, float]
PixelBBoxTuple: TypeAlias = tuple[int, int, int, int]


def coerce_review_action(value: object) -> ReviewAction | None:
    """Return one supported review action or None for invalid inputs."""

    normalized = str(value or "").strip().lower()
    if normalized == "accept":
        return "accept"
    if normalized == "reject":
        return "reject"
    if normalized == "clear":
        return "clear"
    if normalized == "missing_row":
        return "missing_row"
    return None


def coerce_persisted_review_status(value: object) -> PersistedReviewStatus | None:
    """Return one persisted review status or None for blank/invalid values."""

    normalized = str(value or "").strip().lower()
    if normalized == "accepted":
        return "accepted"
    if normalized == "rejected":
        return "rejected"
    return None


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
    review_status: PersistedReviewStatus | None
    review_completed_at: str | None


class RawExtractionLabResultPayload(TypedDict, total=False):
    """Best-effort raw lab-result item parsed from a tool-call payload."""

    raw_lab_name: JsonValue
    raw_section_name: JsonValue
    raw_value: JsonValue
    raw_lab_unit: JsonValue
    raw_unit: JsonValue
    lab_unit_raw: JsonValue
    raw_reference_range: JsonValue
    reference_range: JsonValue
    raw_reference_min: JsonValue
    reference_min: JsonValue
    raw_reference_max: JsonValue
    reference_max: JsonValue
    raw_comments: JsonValue
    bbox_left: JsonValue
    bbox_top: JsonValue
    bbox_right: JsonValue
    bbox_bottom: JsonValue
    left: JsonValue
    top: JsonValue
    right: JsonValue
    bottom: JsonValue
    section_name: JsonValue


class ReviewMissingRowMarker(TypedDict):
    """Persisted missing-row marker stored on one page payload."""

    anchor_result_index: int
    created_at: str


class ReviewMissingRowRecord(ReviewMissingRowMarker):
    """Missing-row marker resolved with its page number for review summaries."""

    page_number: int


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
    review_missing_rows: list[ReviewMissingRowMarker]


class RawExtractionPayload(TypedDict, total=False):
    """Best-effort raw tool-call payload before validation normalizes it."""

    collection_date: JsonValue
    report_date: JsonValue
    lab_facility: JsonValue
    page_has_lab_data: JsonValue
    lab_results: JsonValue


class ExtractionValidationResult(TypedDict):
    """Validation outcome for one raw extraction payload."""

    success: bool
    data: PagePayload | None
    should_retry: bool
    error_msg: str | None


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
    review_status: PersistedReviewStatus | None
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
