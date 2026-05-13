"""Filesystem helpers for hashed processed documents and page JSON state."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path

from parselabs.types import (
    PageLabResultPayload,
    PagePayload,
    ReviewAction,
    ReviewMissingRowMarker,
    ReviewMissingRowRecord,
    coerce_persisted_review_status,
)

logger = logging.getLogger(__name__)

HASHED_DOCUMENT_DIR_RE = re.compile(r"^(?P<stem>.+)_(?P<file_hash>[0-9a-fA-F]{8})$")
REVIEW_MISSING_ROWS_KEY = "review_missing_rows"


@dataclass(frozen=True, init=False)
class DocumentRef:
    """Reference to a processed document directory and its source PDF."""

    doc_dir: Path
    stem: str
    source_pdf: Path
    file_hash: str
    page_count: int

    def __init__(
        self,
        *,
        doc_dir: Path,
        stem: str,
        source_pdf: Path,
        file_hash: str = "",
        page_count: int = 0,
    ) -> None:
        """Build a processed-document reference from canonical runtime fields."""

        resolved_doc_dir = Path(doc_dir)
        resolved_stem = str(stem)
        resolved_hash = file_hash or _extract_hash_from_dir_name(resolved_doc_dir)
        resolved_page_count = int(page_count)

        object.__setattr__(self, "doc_dir", resolved_doc_dir)
        object.__setattr__(self, "stem", resolved_stem)
        object.__setattr__(self, "source_pdf", Path(source_pdf))
        object.__setattr__(self, "file_hash", resolved_hash)
        object.__setattr__(self, "page_count", resolved_page_count)


@dataclass(frozen=True)
class PdfRunPlan:
    """Exact-content unique PDFs to process for one pipeline invocation."""

    documents_to_process: list[DocumentRef]
    duplicates: list[tuple[Path, Path]]


def compute_file_hash(file_path: Path, hash_length: int = 8) -> str:
    """Compute a short SHA-256 hash for the given file."""

    hasher = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:hash_length]


def build_document_ref(pdf_path: Path, output_path: Path, file_hash: str, page_count: int = 0) -> DocumentRef:
    """Build a processed-document reference from a source PDF and content hash."""

    pdf_path = pdf_path.expanduser().resolve()
    doc_dir = output_path / f"{pdf_path.stem}_{file_hash}"
    return DocumentRef(
        doc_dir=doc_dir,
        stem=pdf_path.stem,
        source_pdf=pdf_path,
        file_hash=file_hash,
        page_count=page_count,
    )


def build_hashed_csv_path(pdf_path: Path, output_path: Path, file_hash: str) -> Path:
    """Return the per-document CSV path for a hashed output directory."""

    return get_document_csv_path(build_document_ref(pdf_path, output_path, file_hash).doc_dir)


def discover_pdf_files(input_path: Path, input_file_regex: str | None) -> list[Path]:
    """Return top-level PDF files matching the configured glob pattern."""

    try:
        entries = list(input_path.iterdir())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input directory does not exist: {input_path}") from exc
    except PermissionError as exc:
        detail = exc.strerror or str(exc)
        raise PermissionError(f"Cannot access input directory {input_path}: {detail}") from exc
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise OSError(f"Cannot enumerate input directory {input_path}: {detail}") from exc

    pattern = (input_file_regex or "*.pdf").lower()
    return sorted(path for path in entries if path.is_file() and fnmatch.fnmatch(path.name.lower(), pattern))


def plan_pdf_run(pdf_files: list[Path], output_path: Path) -> PdfRunPlan:
    """Hash every PDF once, derive hashed directories, and dedupe exact duplicates."""

    documents_to_process: list[DocumentRef] = []
    duplicates: list[tuple[Path, Path]] = []
    seen_by_hash: dict[str, Path] = {}

    for pdf_path in pdf_files:
        file_hash = compute_file_hash(pdf_path)

        # Reuse only the first exact-content PDF encountered in this run.
        if file_hash in seen_by_hash:
            duplicates.append((pdf_path, seen_by_hash[file_hash]))
            continue

        seen_by_hash[file_hash] = pdf_path
        documents_to_process.append(build_document_ref(pdf_path, output_path, file_hash))

    return PdfRunPlan(
        documents_to_process=documents_to_process,
        duplicates=duplicates,
    )


def iter_processed_documents(output_path: Path) -> list[DocumentRef]:
    """Discover hashed processed document directories under an output path."""

    documents: list[DocumentRef] = []

    # Guard: Missing output roots simply mean there is nothing to inspect.
    if not output_path.exists():
        return documents

    for doc_dir in sorted(path for path in output_path.iterdir() if path.is_dir()):
        if doc_dir.name == "logs":
            continue

        match = HASHED_DOCUMENT_DIR_RE.match(doc_dir.name)
        if match is None:
            continue

        pdf_paths = sorted(doc_dir.glob("*.pdf"))
        if not pdf_paths:
            continue

        stem = str(match.group("stem"))
        file_hash = str(match.group("file_hash"))
        page_count = len(list(doc_dir.glob("*.json")))
        documents.append(
            DocumentRef(
                doc_dir=doc_dir,
                stem=stem,
                source_pdf=pdf_paths[0],
                file_hash=file_hash,
                page_count=page_count,
            )
        )

    return documents


def resolve_document_dir(stem: str, output_path: Path) -> Path | None:
    """Resolve a processed document directory by logical stem."""

    matches = sorted(output_path.glob(f"{stem}_????????"))
    if matches:
        return matches[0]
    return None


def get_document_stem(doc_dir: Path) -> str:
    """Return the logical document stem for a processed directory."""

    pdf_paths = sorted(doc_dir.glob("*.pdf"))
    if pdf_paths:
        return pdf_paths[0].stem

    match = HASHED_DOCUMENT_DIR_RE.match(doc_dir.name)
    if match is not None:
        return str(match.group("stem"))

    return doc_dir.name


def get_document_csv_path(doc_dir: Path) -> Path:
    """Return the canonical per-document review CSV path."""

    return doc_dir / f"{get_document_stem(doc_dir)}.csv"


def get_page_json_path(doc_dir: Path, page_number: int) -> Path:
    """Return the canonical JSON path for one processed page."""

    return doc_dir / f"{get_document_stem(doc_dir)}.{page_number:03d}.json"


def get_page_image_path(doc_dir: Path, page_number: int) -> Path | None:
    """Return the primary page image for a processed page when available."""

    image_path = doc_dir / f"{get_document_stem(doc_dir)}.{page_number:03d}.jpg"
    if not image_path.exists():
        return None
    return image_path


def resolve_page_path(output_path: Path, source_file: str, page_number: int | None, suffix: str) -> Path:
    """Resolve a page-level artifact path from merged review row metadata."""

    if not source_file:
        return Path()

    stem = source_file.rsplit(".", 1)[0] if "." in source_file else source_file
    page_str = f"{int(page_number):03d}" if page_number is not None else "001"
    doc_dir = resolve_document_dir(stem, output_path)

    if doc_dir is None:
        return Path()

    return doc_dir / f"{stem}.{page_str}{suffix}"


def read_page_payload(json_path: Path) -> PagePayload | None:
    """Read a page JSON payload, returning None for missing or invalid files."""

    if not json_path.exists():
        return None

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, JSONDecodeError) as exc:
        logger.warning(f"Failed to read processed page JSON {json_path}: {exc}")
        return None

    if not isinstance(payload, dict):
        logger.warning(f"Failed to read processed page JSON {json_path}: payload is not an object")
        return None

    return payload


def write_page_payload(json_path: Path, payload: PagePayload) -> None:
    """Persist a page JSON payload with stable formatting."""

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


_BBOX_FIELDS = [
    "bbox_left",
    "bbox_top",
    "bbox_right",
    "bbox_bottom",
]
_COLUMN_BBOX_MAX_WIDTH = 40.0
_COLUMN_BBOX_MIN_HEIGHT = 250.0
_COLUMN_BBOX_ASPECT_RATIO = 4.0
_COLUMN_BBOX_CLUSTER_GAP = 30.0
_COLUMN_BBOX_CLUSTER_MIN_COUNT = 3


def _coerce_cached_bbox(result: dict) -> tuple[float, float, float, float] | None:
    """Return cached bbox coordinates when all four values are usable numbers."""

    # Missing bbox values make the cached extraction stale under the current contract.
    if any(result.get(field_name) is None for field_name in _BBOX_FIELDS):
        return None

    # Cached JSON may contain strings from older outputs, but they still need to be numeric.
    try:
        left, top, right, bottom = [float(result[field_name]) for field_name in _BBOX_FIELDS]
    except (TypeError, ValueError):
        return None

    return left, top, right, bottom


def _is_column_like_cached_bbox(bbox: tuple[float, float, float, float]) -> bool:
    """Return whether a cached bbox looks like a table column instead of a row."""

    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    return (
        width <= _COLUMN_BBOX_MAX_WIDTH
        and height >= _COLUMN_BBOX_MIN_HEIGHT
        and height >= (width * _COLUMN_BBOX_ASPECT_RATIO)
    )


def _cached_results_have_reusable_bboxes(results: object) -> bool:
    """Return whether every cached result satisfies the current bbox contract."""

    # No-result pages are reusable because there are no row bboxes to validate.
    if results is None:
        return True

    # Malformed result containers should be regenerated instead of reused.
    if not isinstance(results, list):
        return False

    column_like_bboxes = []

    for result in results:
        # Malformed result entries should be regenerated instead of reused.
        if not isinstance(result, dict):
            return False

        bbox = _coerce_cached_bbox(result)

        # Rows with incomplete or non-numeric bboxes need a fresh extraction.
        if bbox is None:
            return False

        # Track column slices so both isolated and clustered malformed bboxes are invalidated.
        if _is_column_like_cached_bbox(bbox):
            column_like_bboxes.append(bbox)

    # A single column slice is enough to mark the page cache unsafe.
    if len(column_like_bboxes) == 1:
        return False

    left_positions = sorted(bbox[0] for bbox in column_like_bboxes)
    adjacent_column_slices = sum(
        1
        for idx in range(1, len(left_positions))
        if (left_positions[idx] - left_positions[idx - 1]) <= _COLUMN_BBOX_CLUSTER_GAP
    )

    # Clustered column slices are the common malformed extraction shape.
    return len(column_like_bboxes) < _COLUMN_BBOX_CLUSTER_MIN_COUNT and adjacent_column_slices == 0


def is_page_payload_reusable(payload: dict | None) -> bool:
    """Return whether a cached page JSON payload can be reused safely."""

    if not isinstance(payload, dict):
        return False

    if payload.get("_extraction_failed"):
        return False

    return _cached_results_have_reusable_bboxes(payload.get("lab_results"))


def save_review_status(doc_dir: Path, page_number: int, result_index: int, status: str | None) -> tuple[bool, str]:
    """Persist a review decision to the page JSON backing a review row."""

    normalized_status = coerce_persisted_review_status(status)
    if status is not None and normalized_status is None:
        return False, f"Unsupported review status: {status}"

    json_path = get_page_json_path(doc_dir, page_number)
    payload = read_page_payload(json_path)

    if payload is None:
        return False, f"JSON file not found: {json_path}"

    results = payload.get("lab_results")
    if not isinstance(results, list):
        return False, "No lab_results in JSON file."
    if result_index < 0 or result_index >= len(results):
        return False, f"result_index {result_index} out of range."
    if not isinstance(results[result_index], dict):
        return False, f"lab_results[{result_index}] is not an object."

    result = results[result_index]
    typed_result: PageLabResultPayload = result

    if normalized_status in {"accepted", "rejected"}:
        typed_result["review_status"] = normalized_status
        typed_result["review_completed_at"] = datetime.now(timezone.utc).isoformat()
    else:
        typed_result.pop("review_status", None)
        typed_result.pop("review_completed_at", None)

    try:
        write_page_payload(json_path, payload)
    except (OSError, PermissionError) as exc:
        return False, f"Failed to write JSON file: {exc}"

    return True, ""


def apply_review_action(
    doc_dir: Path,
    page_number: int,
    result_index: int,
    action: ReviewAction,
) -> tuple[bool, str]:
    """Persist one supported review action for a processed row."""

    # Missing-row markers are stored on the page payload instead of the row.
    if action == "missing_row":
        return save_missing_row_marker(doc_dir, page_number, result_index)

    status = {
        "accept": "accepted",
        "reject": "rejected",
        "clear": None,
    }[action]
    return save_review_status(doc_dir, page_number, result_index, status)


def save_missing_row_marker(doc_dir: Path, page_number: int, anchor_result_index: int) -> tuple[bool, str]:
    """Persist a missing-row marker for a processed page."""

    json_path = get_page_json_path(doc_dir, page_number)
    payload = read_page_payload(json_path)

    if payload is None:
        return False, f"JSON file not found: {json_path}"

    results = payload.get("lab_results")
    if not isinstance(results, list):
        return False, "No lab_results in JSON file."
    if anchor_result_index < 0 or anchor_result_index >= len(results):
        return False, f"anchor_result_index {anchor_result_index} out of range."

    markers = payload.get(REVIEW_MISSING_ROWS_KEY)
    if markers is None:
        markers = []
    if not isinstance(markers, list):
        return False, f"{REVIEW_MISSING_ROWS_KEY} must be a list in {json_path.name}."

    marker: ReviewMissingRowMarker = {
        "anchor_result_index": int(anchor_result_index),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    markers.append(marker)
    payload[REVIEW_MISSING_ROWS_KEY] = markers

    try:
        write_page_payload(json_path, payload)
    except (OSError, PermissionError) as exc:
        return False, f"Failed to write JSON file: {exc}"

    return True, ""


def get_review_missing_rows(doc_dir: Path, page_number: int | None = None) -> list[ReviewMissingRowRecord]:
    """Return unresolved missing-row markers for one processed document."""

    markers: list[ReviewMissingRowRecord] = []

    for page_json_path in sorted(doc_dir.glob("*.json")):
        current_page_number = parse_page_number(page_json_path)

        if current_page_number is None:
            continue
        if page_number is not None and current_page_number != page_number:
            continue

        payload = read_page_payload(page_json_path)
        if payload is None:
            continue

        page_markers = payload.get(REVIEW_MISSING_ROWS_KEY, [])
        if not isinstance(page_markers, list):
            logger.warning(f"Invalid {REVIEW_MISSING_ROWS_KEY} payload in {page_json_path}")
            continue

        for marker in page_markers:
            if not isinstance(marker, dict):
                continue

            anchor_result_index = marker.get("anchor_result_index")
            created_at = marker.get("created_at")
            if not isinstance(anchor_result_index, int) or not isinstance(created_at, str):
                continue

            markers.append(
                {
                    "page_number": current_page_number,
                    "anchor_result_index": anchor_result_index,
                    "created_at": created_at,
                }
            )

    return markers


def count_review_missing_rows(doc_dir: Path, page_number: int | None = None) -> int:
    """Return the count of unresolved missing-row markers for a document."""

    return len(get_review_missing_rows(doc_dir, page_number=page_number))


def parse_page_number(page_json_path: Path) -> int | None:
    """Extract the 1-based page number from a processed page JSON filename."""

    match = re.search(r"\.(\d{3})\.json$", page_json_path.name)
    if match is None:
        return None
    return int(match.group(1))


def _extract_hash_from_dir_name(doc_dir: Path) -> str:
    """Return the hash suffix embedded in a processed document directory name."""

    match = HASHED_DOCUMENT_DIR_RE.match(doc_dir.name)
    if match is None:
        return ""
    return str(match.group("file_hash"))


__all__ = [
    "DocumentRef",
    "HASHED_DOCUMENT_DIR_RE",
    "PdfRunPlan",
    "REVIEW_MISSING_ROWS_KEY",
    "ReviewAction",
    "apply_review_action",
    "build_document_ref",
    "build_hashed_csv_path",
    "compute_file_hash",
    "count_review_missing_rows",
    "discover_pdf_files",
    "get_document_csv_path",
    "get_document_stem",
    "get_page_image_path",
    "get_page_json_path",
    "get_review_missing_rows",
    "is_page_payload_reusable",
    "iter_processed_documents",
    "parse_page_number",
    "plan_pdf_run",
    "read_page_payload",
    "resolve_document_dir",
    "resolve_page_path",
    "save_missing_row_marker",
    "save_review_status",
    "write_page_payload",
]
