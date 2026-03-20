"""Shared deterministic review-artifact helpers for CLI and MCP flows."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from parselabs.review import get_bbox_coordinates, scale_bbox_to_pixels
from parselabs.review_state import ReviewTarget, apply_review_action_for_target
from parselabs.runtime import RuntimeContext
from parselabs.store import iter_processed_documents, parse_page_number, read_page_payload

DEFAULT_ARTIFACTS_DIRNAME = ".review-artifacts"
DEFAULT_CROP_PADDING_PX = 24


@dataclass(frozen=True)
class PendingReviewRow:
    """Resolved pending review row plus the backing document metadata."""

    doc_dir: Path
    doc_dir_name: str
    doc_stem: str
    source_pdf_path: Path
    page_json_path: Path
    page_image_path: Path | None
    page_number: int
    result_index: int
    stored_result: dict


def load_review_context(profile_name: str) -> RuntimeContext:
    """Load the output-backed runtime context for one review profile."""

    return RuntimeContext.from_profile(
        profile_name,
        need_input=False,
        need_output=True,
        need_api=False,
        setup_logs=False,
    )


def resolve_artifacts_dir(output_path: Path, requested_dir: Path | None) -> Path:
    """Resolve and create the artifact directory for bbox crops."""

    artifacts_dir = requested_dir if requested_dir is not None else output_path / DEFAULT_ARTIFACTS_DIRNAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir.resolve()


def find_next_pending_row(output_path: Path, *, review_needed_only: bool) -> PendingReviewRow | None:
    """Return the first pending row in deterministic document/page/result order."""

    for document in iter_processed_documents(output_path):
        page_json_paths = sorted(document.doc_dir.glob("*.json"))

        for page_json_path in page_json_paths:
            page_number = parse_page_number(page_json_path)

            # Skip malformed page filenames that cannot round-trip to review writes.
            if page_number is None:
                continue

            payload = read_page_payload(page_json_path)

            # Skip unreadable page payloads so one bad JSON file does not block the queue.
            if payload is None:
                continue

            results = payload.get("lab_results")

            # Skip malformed page payloads that do not expose result rows.
            if not isinstance(results, list):
                continue

            for result_index, stored_result in enumerate(results):
                # Skip malformed rows so the queue remains traversable.
                if not isinstance(stored_result, dict):
                    continue

                status = str(stored_result.get("review_status") or "").strip().lower()

                # Skip rows that already have an explicit persisted review decision.
                if status in {"accepted", "rejected"}:
                    continue

                # Skip non-flagged rows when the caller requested only validation hits.
                if review_needed_only and not bool(stored_result.get("review_needed")):
                    continue

                return PendingReviewRow(
                    doc_dir=document.doc_dir,
                    doc_dir_name=document.doc_dir.name,
                    doc_stem=document.stem,
                    source_pdf_path=document.source_pdf,
                    page_json_path=page_json_path,
                    page_image_path=document.doc_dir / f"{document.stem}.{page_number:03d}.jpg",
                    page_number=int(page_number),
                    result_index=int(result_index),
                    stored_result=stored_result,
                )

    # Guard: Exhausted iteration means there are no rows left for this filter.
    return None


def build_next_payload(profile_name: str, row: PendingReviewRow, artifacts_dir: Path) -> dict:
    """Build the JSON payload for one pending review row."""

    page_image_path = row.page_image_path if row.page_image_path and row.page_image_path.exists() else None
    crop_path, bbox_pixels, crop_pixels, artifact_error = write_bbox_clip(row, artifacts_dir)

    return {
        "done": False,
        "profile": profile_name,
        "row_id": encode_row_id(row),
        "doc_dir": str(row.doc_dir),
        "doc_dir_name": row.doc_dir_name,
        "doc_stem": row.doc_stem,
        "source_pdf_path": str(row.source_pdf_path),
        "page_json_path": str(row.page_json_path),
        "page_image_path": str(page_image_path) if page_image_path is not None else None,
        "bbox_clip_path": str(crop_path) if crop_path is not None else None,
        "page_number": row.page_number,
        "result_index": row.result_index,
        "bbox": format_bbox_payload(get_bbox_coordinates(row.stored_result)),
        "bbox_pixels": format_bbox_payload(bbox_pixels),
        "crop_box_pixels": format_bbox_payload(crop_pixels),
        "artifact_error": artifact_error,
        "stored_result": row.stored_result,
    }


def get_next_review_payload(
    profile_name: str,
    *,
    artifacts_dir: Path | None = None,
    review_needed_only: bool = False,
) -> dict:
    """Return the next pending row payload for one review profile."""

    context = load_review_context(profile_name)
    output_path = context.output_path

    # Guard: Review-artifact flows require a resolved output directory.
    if output_path is None:
        raise RuntimeError(f"Profile '{profile_name}' does not define an output path.")

    resolved_artifacts_dir = resolve_artifacts_dir(output_path, artifacts_dir)
    pending_row = find_next_pending_row(output_path, review_needed_only=review_needed_only)

    # Guard: Exhausted corpora return a stable done payload instead of failing.
    if pending_row is None:
        return {
            "done": True,
            "profile": profile_name,
            "artifacts_dir": str(resolved_artifacts_dir),
        }

    return build_next_payload(profile_name, pending_row, resolved_artifacts_dir)


def apply_review_decision(profile_name: str, row_id: str, decision: str) -> tuple[bool, dict]:
    """Persist one review decision for the requested row identifier."""

    context = load_review_context(profile_name)
    output_path = context.output_path

    # Guard: Review-artifact flows require a resolved output directory.
    if output_path is None:
        raise RuntimeError(f"Profile '{profile_name}' does not define an output path.")

    try:
        target = decode_row_id(row_id, output_path)
    except ValueError as exc:
        return False, {
            "ok": False,
            "profile": profile_name,
            "row_id": row_id,
            "decision": decision,
            "error": str(exc),
        }

    success, error = apply_review_action_for_target(target, decision)

    # Guard: Persisted review failures should surface as structured errors.
    if not success:
        return False, {
            "ok": False,
            "profile": profile_name,
            "row_id": row_id,
            "decision": decision,
            "error": error,
        }

    return True, {
        "ok": True,
        "profile": profile_name,
        "row_id": row_id,
        "decision": decision,
        "doc_dir": str(target.doc_dir),
        "page_number": target.page_number,
        "result_index": target.result_index,
    }


def write_bbox_clip(
    row: PendingReviewRow,
    artifacts_dir: Path,
) -> tuple[Path | None, tuple[int, int, int, int] | None, tuple[int, int, int, int] | None, str | None]:
    """Write one deterministic bbox crop for the requested review row."""

    bbox = get_bbox_coordinates(row.stored_result)

    # Guard: Rows without a page image cannot produce a visual clip.
    if row.page_image_path is None or not row.page_image_path.exists():
        return None, None, None, "Page image not found for pending row."

    # Guard: Rows without a valid bbox still return the row metadata for manual handling.
    if bbox is None:
        return None, None, None, "Row does not contain a valid bounding box."

    with Image.open(row.page_image_path) as image:
        pixel_bbox = scale_bbox_to_pixels(bbox, image.size)

        # Guard: Invalid scaled bounds should not crash the caller.
        if pixel_bbox is None:
            return None, None, None, "Bounding box could not be scaled into image pixels."

        crop_box = expand_crop_bounds(pixel_bbox, image.size)
        crop_image = image.crop(crop_box)
        clip_path = artifacts_dir / f"{row.doc_dir_name}.p{row.page_number:03d}.r{row.result_index:04d}.png"
        crop_image.save(clip_path)

    return clip_path.resolve(), pixel_bbox, crop_box, None


def expand_crop_bounds(
    pixel_bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Expand one pixel bbox with deterministic fixed padding."""

    left, top, right, bottom = pixel_bbox
    image_width, image_height = image_size
    crop_left = max(0, left - DEFAULT_CROP_PADDING_PX)
    crop_top = max(0, top - DEFAULT_CROP_PADDING_PX)
    crop_right = min(image_width, right + DEFAULT_CROP_PADDING_PX)
    crop_bottom = min(image_height, bottom + DEFAULT_CROP_PADDING_PX)
    return crop_left, crop_top, crop_right, crop_bottom


def encode_row_id(row: PendingReviewRow) -> str:
    """Encode one pending row identity into an opaque stable token."""

    token_payload = {
        "doc_dir_name": row.doc_dir_name,
        "page_number": row.page_number,
        "result_index": row.result_index,
    }
    raw_token = json.dumps(token_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw_token).decode("ascii")
    return encoded.rstrip("=")


def decode_row_id(row_id: str, output_path: Path) -> ReviewTarget:
    """Decode one opaque row token back into a persisted review target."""

    padded_row_id = row_id + ("=" * (-len(row_id) % 4))

    try:
        raw_payload = base64.urlsafe_b64decode(padded_row_id.encode("ascii"))
        token_payload = json.loads(raw_payload.decode("utf-8"))
    except Exception as exc:  # pragma: no cover - defensive parse guard
        raise ValueError(f"Invalid row_id: {exc}") from exc

    doc_dir_name = str(token_payload.get("doc_dir_name") or "").strip()
    page_number = token_payload.get("page_number")
    result_index = token_payload.get("result_index")

    # Guard: Row identifiers must always point to one hashed processed-document directory.
    if not doc_dir_name:
        raise ValueError("Invalid row_id: missing doc_dir_name.")

    # Guard: Row identifiers must include the page number needed for JSON resolution.
    if not isinstance(page_number, int):
        raise ValueError("Invalid row_id: missing integer page_number.")

    # Guard: Row identifiers must include the page-local result index.
    if not isinstance(result_index, int):
        raise ValueError("Invalid row_id: missing integer result_index.")

    doc_dir = output_path / doc_dir_name

    # Guard: Row identifiers must resolve to an existing processed document directory.
    if not doc_dir.exists():
        raise ValueError(f"Invalid row_id: document directory not found: {doc_dir_name}.")

    return ReviewTarget(
        doc_dir=doc_dir,
        page_number=page_number,
        result_index=result_index,
    )


def format_bbox_payload(bbox: tuple[int, int, int, int] | tuple[float, float, float, float] | None) -> dict | None:
    """Return bbox coordinates as a JSON-friendly object."""

    # Guard: Missing bbox values stay null in the response payload.
    if bbox is None:
        return None

    left, top, right, bottom = bbox
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def format_json_text(payload: dict) -> str:
    """Return one stable human-readable JSON string."""

    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
