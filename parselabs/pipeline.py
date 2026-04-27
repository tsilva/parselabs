"""Main entry point for lab results extraction and processing."""

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from multiprocessing import Manager, Pool  # noqa: E402
from pathlib import Path  # noqa: E402
from queue import Empty  # noqa: E402
from time import sleep  # noqa: E402
from typing import Callable  # noqa: E402

import pandas as pd  # noqa: E402
import pdf2image  # noqa: E402
from openai import OpenAI  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Local imports
from parselabs.config import (  # noqa: E402
    ExtractionConfig,
    LabSpecsConfig,
    ProfileConfig,
)
from parselabs.exceptions import ConfigurationError, ExtractionAPIError, PipelineError  # noqa: E402
from parselabs.export_schema import COLUMN_SCHEMA, get_column_lists  # noqa: E402
from parselabs.extraction import (  # noqa: E402
    extract_labs_from_page_image,
)
from parselabs.paths import (
    get_env_file,  # noqa: E402
    get_profiles_dir,  # noqa: E402
)
from parselabs.rows import (  # noqa: E402
    DOCUMENT_REVIEW_COLUMNS,
    _rebuild_document_csv_with_review_dataframe,
    get_document_review_summary,
    iter_processed_documents,
    load_document_review_rows,
    rebuild_document_csv,
    transform_rows_to_final_export,
)
from parselabs.runtime import (  # noqa: E402
    RuntimeContext,
    add_profile_arguments,
    get_openai_client,
)
from parselabs.standardization_refresh import (  # noqa: E402
    StandardizationRefreshResult,
    refresh_standardization_caches_from_dataframe,
)
from parselabs.store import (  # noqa: E402
    DocumentRef,
    discover_pdf_files,
    get_document_csv_path,
    is_page_payload_reusable,
    plan_pdf_run,
    read_page_payload,
)
from parselabs.types import ExtractionFailureRecord, PagePayload  # noqa: E402
from parselabs.utils import (  # noqa: E402
    ConsoleLogMode,
    create_page_image_variants,
    log_user_info,
    log_user_warning,
    setup_logging,
)

# Module-level logger (file handlers added after config is loaded)
logger = logging.getLogger(__name__)
PROFILES_DIR = get_profiles_dir()
EXTRACTION_FAILURE_RAW_NAME = "[EXTRACTION FAILED]"
ACTIVE_CONSOLE_MODE: ConsoleLogMode = "normal"
WORKER_PROGRESS_QUEUE = None


# ========================================
# PDF Processing - Helper Functions
# ========================================


def _setup_pdf_processing(pdf_path: Path, output_dir: Path, file_hash: str) -> tuple[Path, Path, list[ExtractionFailureRecord]]:
    """Initialize processing: create directories, return paths."""
    pdf_stem = pdf_path.stem  # Extract filename without extension for directory naming
    dir_name = f"{pdf_stem}_{file_hash}"  # Include file hash for uniqueness
    doc_out_dir = output_dir / dir_name  # Create document-specific output directory

    doc_out_dir.mkdir(exist_ok=True, parents=True)  # Ensure directory exists, create parents if needed
    csv_path = doc_out_dir / f"{pdf_stem}.csv"  # Define CSV output path within document directory
    return (
        doc_out_dir,
        csv_path,
        [],
    )  # Return paths and empty failed_pages list for population


def _copy_pdf_to_output(pdf_path: Path, doc_out_dir: Path) -> Path:
    """Copy PDF to output directory if not already present."""
    copied_pdf = doc_out_dir / pdf_path.name  # Define destination path for PDF copy

    # Copy only if missing or file size differs (ensures we have latest version)
    if not copied_pdf.exists() or copied_pdf.stat().st_size != pdf_path.stat().st_size:
        shutil.copy2(pdf_path, copied_pdf)  # copy2 preserves metadata like timestamps
    return copied_pdf  # Return path to the copied PDF (whether newly copied or existing)


def _convert_pdf_to_images(copied_pdf: Path, pdf_stem: str) -> list:
    """Convert PDF to PIL images."""

    # Use pdf2image to convert PDF pages to PIL Image objects
    return pdf2image.convert_from_path(str(copied_pdf))


def _get_page_image_paths(doc_out_dir: Path, page_name: str) -> dict[str, Path]:
    """Build filesystem paths for the extraction image variants."""

    return {
        "primary": doc_out_dir / f"{page_name}.jpg",
        "fallback": doc_out_dir / f"{page_name}.fallback.jpg",
    }


def _prepare_page_images(page_image, page_name: str, doc_out_dir: Path) -> dict[str, Path]:
    """Create and cache the image variants used for page extraction."""

    image_paths = _get_page_image_paths(doc_out_dir, page_name)
    if all(path.exists() for path in image_paths.values()):
        return image_paths

    variants = create_page_image_variants(page_image)
    for variant_name, processed_image in variants.items():
        image_path = image_paths[variant_name]
        if image_path.exists():
            continue
        processed_image.save(image_path, "JPEG", quality=95)
        logger.info(f"[{page_name}] Saved {variant_name} extraction image")

    return image_paths


def _page_requires_image_fallback(page_data: PagePayload) -> bool:
    """Return whether a page result is too weak to keep without another pass."""

    # Missing or malformed payloads cannot be trusted.
    if not isinstance(page_data, dict):
        return True

    # Explicit extraction failures should trigger the next fallback.
    if page_data.get("_extraction_failed"):
        return True

    results = [result for result in page_data.get("lab_results", []) if isinstance(result, dict)]

    # Pages explicitly marked as non-lab content do not need another read.
    if page_data.get("page_has_lab_data") is False:
        return False

    # Empty likely-lab pages need another pass.
    if not results:
        return True

    return False


def _ensure_extraction_failure_placeholder(page_data: PagePayload) -> PagePayload:
    """Insert a synthetic review row for failed pages that returned no results."""

    # Successful pages or pages with extracted rows already have reviewable content.
    if not page_data.get("_extraction_failed") or page_data.get("lab_results"):
        return page_data

    failure_reason = page_data.get("_failure_reason", "Unknown extraction failure")
    page_data["lab_results"] = [
        {
            "raw_lab_name": EXTRACTION_FAILURE_RAW_NAME,
            "raw_value": None,
            "raw_lab_unit": None,
            "raw_reference_range": None,
            "raw_reference_min": None,
            "raw_reference_max": None,
            "raw_comments": failure_reason,
            "bbox_left": None,
            "bbox_top": None,
            "bbox_right": None,
            "bbox_bottom": None,
        }
    ]
    return page_data


def _extract_page_data_from_image(
    image_path: Path,
    config: ExtractionConfig,
    page_name: str,
    variant_name: str,
) -> PagePayload:
    """Extract a single page from an image variant."""

    logger.info(f"[{page_name}] Attempting VISION extraction ({variant_name})")
    return extract_labs_from_page_image(
        image_path,
        config.extract_model_id,
        get_openai_client(config),
    )


def _extract_or_load_page_data(
    image_paths: dict[str, Path],
    json_path: Path,
    page_name: str,
    config: ExtractionConfig,
    pdf_stem: str,
    page_idx: int,
    failed_pages: list[ExtractionFailureRecord],
) -> PagePayload:
    """Extract data from image or load from cache."""

    # Reuse only valid cached JSON payloads so failed pages are retried automatically.
    if json_path.exists():
        cached_payload = read_page_payload(json_path)

        # Cached failures and unreadable JSON should be retried on the next run.
        if is_page_payload_reusable(cached_payload):
            logger.info(f"[{page_name}] Loading cached extraction data")
            page_data = cached_payload
            page_data["source_file"] = page_name
            page_data = _ensure_extraction_failure_placeholder(page_data)
            _check_and_record_failure(page_data, pdf_stem, page_idx, failed_pages)
            return page_data

        logger.info(f"[{page_name}] Re-running extraction because cached JSON is missing, invalid, or failed")

    attempts: list[tuple[str, Callable[[], PagePayload]]] = []

    # Fall back through the primary image and then the alternate image.
    attempts.extend(
        [
            (
                "Primary vision extraction",
                lambda: _extract_page_data_from_image(
                    image_paths["primary"],
                    config,
                    page_name,
                    "primary",
                ),
            ),
            (
                "Fallback vision extraction",
                lambda: _extract_page_data_from_image(
                    image_paths["fallback"],
                    config,
                    page_name,
                    "fallback",
                ),
            ),
        ]
    )

    page_data: PagePayload | None = None

    # Stop at the first extraction attempt that produces a reviewable payload.
    for attempt_idx, (attempt_label, attempt_fn) in enumerate(attempts):
        candidate = attempt_fn()

        if not _page_requires_image_fallback(candidate):
            page_data = candidate
            break

        has_more_attempts = attempt_idx < len(attempts) - 1
        if has_more_attempts:
            logger.info(f"[{page_name}] {attempt_label} needs fallback; trying next route")
            continue

        page_data = candidate
        break

    if page_data is None:
        page_data = {"lab_results": []}

    # Set source_file programmatically (LLM can't know the filename)
    page_data["source_file"] = page_name
    page_data = _ensure_extraction_failure_placeholder(page_data)

    # Record any extraction failures for reporting
    _check_and_record_failure(page_data, pdf_stem, page_idx, failed_pages, page_name)

    # Cache all reviewable outcomes, including explicit failures and confirmed blank pages.
    has_results = bool(page_data.get("lab_results"))
    confirmed_no_lab_data = page_data.get("page_has_lab_data") is False
    extraction_failed = bool(page_data.get("_extraction_failed"))
    if has_results or confirmed_no_lab_data or extraction_failed:
        json_path.write_text(
            json.dumps(page_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        logger.info(f"[{page_name}] Empty extraction, not caching (will retry next run)")
    return page_data


def _check_and_record_failure(
    page_data: PagePayload,
    pdf_stem: str,
    page_idx: int,
    failed_pages: list[ExtractionFailureRecord],
    page_name: str | None = None,
) -> None:
    """Check if extraction failed and record failure reason."""

    # Skip if extraction succeeded
    if not page_data.get("_extraction_failed"):
        return

    # Extract failure reason from extraction data
    failure_reason = page_data.get("_failure_reason", "Unknown error")

    # Record failure for summary reporting
    failed_pages.append({"page": f"{pdf_stem} page {page_idx + 1}", "reason": str(failure_reason)})

    # Log failure if page name available
    if page_name:
        logger.error(f"[{page_name}] EXTRACTION FAILED: {failure_reason}")


def _add_page_metadata(results: list, page_idx: int, page_name: str) -> list:
    """Add page metadata to extraction results."""

    # Initialize collection for enriched results
    enriched = []

    # Add metadata to each result for tracking
    for result_idx, result in enumerate(results):
        result["result_index"] = result_idx  # Track position within page
        result["page_number"] = page_idx + 1  # 1-indexed page number
        result["source_file"] = page_name  # Track source for debugging
        enriched.append(result)

    return enriched


def _process_single_page(
    page_image,
    page_idx: int,
    total_pages: int,
    pdf_stem: str,
    doc_out_dir: Path,
    config: ExtractionConfig,
    failed_pages: list[ExtractionFailureRecord],
) -> tuple[list, PagePayload | None]:
    """Process a single PDF page: preprocess, extract, and return results with metadata.

    Returns tuple of (page_results, page_data) where page_data is the full extraction data.
    """

    # Generate unique page identifier with zero-padding
    page_name = f"{pdf_stem}.{page_idx + 1:03d}"
    json_path = doc_out_dir / f"{page_name}.json"

    logger.info(f"[{page_name}] Processing page {page_idx + 1}/{total_pages}...")

    # Preprocess and cache page image variants
    image_paths = _prepare_page_images(page_image, page_name, doc_out_dir)

    # Extract data using page-level vision routing or load from cache.
    page_data = _extract_or_load_page_data(
        image_paths,
        json_path,
        page_name,
        config,
        pdf_stem,
        page_idx,
        failed_pages,
    )

    # Add page metadata to results
    page_results = _add_page_metadata(page_data.get("lab_results", []), page_idx, page_name)

    return page_results, page_data


def _extract_via_pages(
    copied_pdf: Path,
    config: ExtractionConfig,
    doc_out_dir: Path,
    pdf_stem: str,
    failed_pages: list[ExtractionFailureRecord],
) -> tuple[list, str | None]:
    """Extract lab results using per-page vision extraction."""

    # Convert PDF pages to PIL images
    pil_pages = _convert_pdf_to_images(copied_pdf, pdf_stem)

    # Guard: Handle PDF conversion failure
    if pil_pages is None:
        return [], None

    logger.info(f"[{pdf_stem}] Processing {len(pil_pages)} page(s) with vision extraction...")

    # Initialize collection for all extracted results
    all_results = []
    doc_date = None

    # Process each page independently
    for page_idx, page_image in enumerate(pil_pages):
        # Process single page and get results
        page_results, page_data = _process_single_page(
            page_image,
            page_idx,
            len(pil_pages),
            pdf_stem,
            doc_out_dir,
            config,
            failed_pages,
        )

        # Extract document date from first page only
        if page_idx == 0 and page_data is not None:
            doc_date = _extract_document_date(page_data, pdf_stem)

        # Append page results to collection
        all_results.extend(page_results)

    return all_results, doc_date


def _extract_data_from_pdf(
    copied_pdf: Path,
    config: ExtractionConfig,
    doc_out_dir: Path,
    pdf_stem: str,
    failed_pages: list[ExtractionFailureRecord],
) -> tuple[list, str | None]:
    """Extract lab data from PDF using per-page vision extraction.

    Returns tuple of (all_results, doc_date) or raises exception on failure.
    """

    return _extract_via_pages(
        copied_pdf,
        config,
        doc_out_dir,
        pdf_stem,
        failed_pages,
    )


def process_single_pdf(
    pdf_path: Path,
    file_hash: str,
    output_dir: Path,
    config: ExtractionConfig,
    lab_specs: LabSpecsConfig,
) -> tuple[Path | None, list[ExtractionFailureRecord]]:
    """Process a single PDF file: extract page JSON and rebuild the review CSV.

    Returns:
        Tuple of (csv_path, failed_pages) where:
        - csv_path: Path to output CSV, or None if processing failed entirely
        - failed_pages: List of dicts with 'page' and 'reason' for any extraction failures
    """

    # Initialize output directory structure and paths
    pdf_stem = pdf_path.stem
    doc_out_dir, csv_path, failed_pages = _setup_pdf_processing(pdf_path, output_dir, file_hash)

    try:
        logger.info(f"[{pdf_stem}] Processing...")

        # Copy source PDF to output directory for archival
        copied_pdf = _copy_pdf_to_output(pdf_path, doc_out_dir)

        # Extract page JSON using the primary image with one fallback variant.
        all_results, _ = _extract_data_from_pdf(
            copied_pdf,
            config,
            doc_out_dir,
            pdf_stem,
            failed_pages,
        )

        # Log documents that produced no review rows beyond explicit blank-page confirmations.
        if not all_results:
            logger.warning(f"[{pdf_stem}] No lab rows extracted; review CSV will contain only derived page state")

        # Rebuild the review CSV from page JSON so the persisted CSV is always derived state.
        csv_path = rebuild_document_csv(doc_out_dir, lab_specs)

        logger.info(f"[{pdf_stem}] Completed successfully")
        return csv_path, failed_pages

    except (OSError, PermissionError, PipelineError, ExtractionAPIError) as e:
        logger.error(f"[{pdf_stem}] Processing failed: {e}", exc_info=True)
        return None, failed_pages


def export_excel(
    df: pd.DataFrame,
    excel_path: Path,
    hidden_cols: list,
    widths: dict,
) -> None:
    """Export DataFrame to Excel with formatting."""

    # Create Excel writer with xlsxwriter engine and date formatting
    with pd.ExcelWriter(
        excel_path,
        engine="xlsxwriter",
        datetime_format="yyyy-mm-dd",
        date_format="yyyy-mm-dd",
    ) as writer:
        # Export data to worksheet
        df.to_excel(writer, sheet_name="Data", index=False)

        # Get worksheet for formatting
        ws = writer.sheets["Data"]

        # Freeze first row (header) for scrolling
        ws.freeze_panes(1, 0)

        # Apply column widths and visibility settings
        for idx, col_name in enumerate(df.columns):
            width = widths.get(col_name, 12)  # Default width if not specified
            options = {"hidden": True} if col_name in hidden_cols else {}
            ws.set_column(idx, idx, width, None, options)

    logger.info(f"Saved Excel: {excel_path}")


# ========================================
# Helpers
# ========================================


def _extract_document_date(data_dict: dict, pdf_stem: str) -> str | None:
    """Extract document date from extraction data or filename."""

    # Try to get date from extraction data fields
    doc_date = data_dict.get("collection_date") or data_dict.get("report_date")

    # Handle invalid placeholder date
    if doc_date == "0000-00-00":
        doc_date = None

    # Fallback: Extract date from filename pattern (YYYY-MM-DD)
    if not doc_date:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", pdf_stem)

        # Use date from filename if pattern matches
        if match:
            doc_date = match.group(1)

    return doc_date


REQUIRED_CSV_COLS = ["result_index", "page_number", "source_file"]


@dataclass
class PdfPreflightResult:
    """Unique PDFs plus duplicate metadata for one pipeline invocation."""

    pdfs_to_process: list[DocumentRef]
    duplicates: list[tuple[Path, Path]]
    cached_csv_paths: list[Path] = field(default_factory=list)


@dataclass
class PipelineRunResult:
    """Final dataframe plus pipeline metadata for one corpus run."""

    final_df: pd.DataFrame
    merged_review_df: pd.DataFrame
    csv_paths: list[Path]
    failed_pages: list[ExtractionFailureRecord]
    pdfs_failed: int


@dataclass(frozen=True)
class ReviewedCorpusResult:
    """Rebuilt per-document CSVs plus merged review and accepted-export data."""

    csv_paths: list[Path]
    merged_review_df: pd.DataFrame
    final_df: pd.DataFrame


def _console_mode_shows_detail(console_mode: str) -> bool:
    """Return whether console output should include diagnostic progress details."""

    # Verbose/debug modes are explicitly for detail; normal/quiet should stay concise.
    return console_mode in {"verbose", "debug"}


def _prepare_pdf_run(pdf_files: list[Path], output_path: Path, *, show_progress: bool = False) -> PdfPreflightResult:
    """Hash every PDF once, dedupe exact duplicates, and derive hashed output paths."""

    pdf_iterator = pdf_files

    # Hashing is usually instant, so only show its progress bar in detailed console modes.
    if show_progress and len(pdf_files) > 1:
        logger.info("Hashing PDFs for deduplication...")
        pdf_iterator = tqdm(pdf_files, desc="Hashing PDFs", unit="pdf")
    run_plan = plan_pdf_run(list(pdf_iterator), output_path)

    return PdfPreflightResult(
        pdfs_to_process=run_plan.documents_to_process,
        duplicates=run_plan.duplicates,
    )


def _log_validation_stats(validation_stats: dict) -> None:
    """Log validation statistics if any rows were flagged."""

    # Check if any rows were flagged for review
    flagged_count = validation_stats.get("rows_flagged", 0)
    if flagged_count > 0:
        log_user_info(logger, "Validation flagged %s rows for review", flagged_count)

        # Log breakdown of flag reasons
        for reason, count in validation_stats.get("flags_by_reason", {}).items():
            log_user_info(logger, "  - %s: %s", reason, count)


def _progress_doc_label(pdf_path: Path) -> str:
    """Return a compact document label for progress-bar display."""

    # Keep long lab filenames from consuming the full terminal width.
    label = pdf_path.stem
    if len(label) <= 36:
        return label

    return f"{label[:33]}..."


def _format_active_docs(active_docs: set[str], limit: int = 3) -> str:
    """Return a concise progress-bar postfix for currently active documents."""

    # Guard: no active workers means there is no useful postfix to show.
    if not active_docs:
        return ""

    # Show a stable subset so the tqdm line stays readable with many workers.
    active_names = sorted(active_docs)
    visible_names = active_names[:limit]
    suffix = f" +{len(active_names) - limit}" if len(active_names) > limit else ""
    return f"active: {', '.join(visible_names)}{suffix}"


def _set_active_docs_postfix(pbar, active_docs: set[str]) -> None:
    """Update the tqdm postfix with active document names."""

    # Replace the postfix in-place rather than emitting separate log lines.
    pbar.set_postfix_str(_format_active_docs(active_docs), refresh=True)


def _drain_worker_progress(progress_queue, active_docs: set[str], pbar) -> None:
    """Apply all queued worker start/done events to the progress bar."""

    updated = False

    # Drain every pending event so the displayed active set catches up quickly.
    while True:
        try:
            event, doc_label = progress_queue.get_nowait()
        except Empty:
            break

        # Worker started a document.
        if event == "start":
            active_docs.add(doc_label)
            updated = True
            continue

        # Worker finished or aborted a document.
        if event == "done":
            active_docs.discard(doc_label)
            updated = True

    # Refresh the tqdm postfix only when the active set changed.
    if updated:
        _set_active_docs_postfix(pbar, active_docs)


def _process_pdfs_in_parallel(
    pdfs_to_process: list[DocumentRef],
    config: ExtractionConfig,
    lab_specs: LabSpecsConfig,
    log_dir: Path,
    console_mode: ConsoleLogMode = "normal",
) -> tuple[list[Path], list[ExtractionFailureRecord], int]:
    """
    Process PDFs in parallel using multiprocessing pool.

    Returns:
        Tuple of (csv_paths, all_failed_pages, pdfs_failed_count)
    """

    # Calculate optimal worker count (don't exceed CPU count or task count)
    n_workers = min(config.max_workers, len(pdfs_to_process))
    logger.info("Using %s worker(s) for PDF processing", n_workers)

    # Build task tuples for each PDF to process
    tasks = [(task.source_pdf, task.file_hash, config.output_path, config, lab_specs) for task in pdfs_to_process]

    if n_workers == 1:
        _init_worker_logging(log_dir, console_mode)
        results = []
        with tqdm(total=len(tasks), desc="Processing PDFs", unit="pdf") as pbar:
            for task in tasks:
                active_docs = {_progress_doc_label(task[0])}
                _set_active_docs_postfix(pbar, active_docs)
                results.append(_process_pdf_wrapper(task))
                _set_active_docs_postfix(pbar, set())
                pbar.update(1)
    else:
        # Execute parallel processing with live worker progress events.
        with Manager() as manager:
            progress_queue = manager.Queue()
            with Pool(n_workers, initializer=_init_worker_logging, initargs=(log_dir, console_mode, progress_queue)) as pool:
                results = _collect_parallel_pdf_results(pool, tasks, progress_queue)

    # Unpack results and collect statistics.
    pdfs_failed = sum(1 for csv_path, _ in results if csv_path is None)
    all_failed_pages: list[ExtractionFailureRecord] = []
    for _, failed_pages in results:
        all_failed_pages.extend(failed_pages)

    # Keep only successful CSV outputs from the worker run.
    csv_paths = [csv_path for csv_path, _ in results if csv_path and csv_path.exists()]

    return csv_paths, all_failed_pages, pdfs_failed


def _collect_parallel_pdf_results(pool: Pool, tasks: list[tuple], progress_queue) -> list[tuple[Path | None, list[ExtractionFailureRecord]]]:
    """Collect multiprocessing results while updating active document progress."""

    # Submit all tasks asynchronously so the parent can keep polling progress events.
    pending_results = [pool.apply_async(_process_pdf_wrapper, (task,)) for task in tasks]
    results: list[tuple[Path | None, list[ExtractionFailureRecord]] | None] = [None] * len(pending_results)
    remaining_indices = set(range(len(pending_results)))
    active_docs: set[str] = set()

    # Track both completion count and currently active worker document labels.
    with tqdm(total=len(tasks), desc="Processing PDFs", unit="pdf") as pbar:
        while remaining_indices:
            _drain_worker_progress(progress_queue, active_docs, pbar)

            # Collect any completed tasks without blocking progress updates.
            completed_indices = [idx for idx in remaining_indices if pending_results[idx].ready()]
            for idx in completed_indices:
                results[idx] = pending_results[idx].get()
                remaining_indices.remove(idx)
                pbar.update(1)

            # Avoid a busy loop while workers are processing long PDFs.
            if remaining_indices:
                sleep(0.1)

        _drain_worker_progress(progress_queue, active_docs, pbar)
        _set_active_docs_postfix(pbar, set())

    return [result for result in results if result is not None]


def _init_worker_logging(log_dir: Path, console_mode: ConsoleLogMode = "normal", progress_queue=None):
    """Initialize logging in worker processes."""
    global WORKER_PROGRESS_QUEUE

    # Workers report document start/done events so the parent can update tqdm.
    WORKER_PROGRESS_QUEUE = progress_queue
    setup_logging(log_dir, clear_logs=False, console_mode=console_mode)


def _process_pdf_wrapper(args):
    """Wrapper function for multiprocessing.

    Returns tuple of (csv_path, failed_pages) from process_single_pdf.
    """

    pdf_path = args[0]
    doc_label = _progress_doc_label(pdf_path)

    # Tell the parent process which document this worker is handling.
    _send_worker_progress("start", doc_label)
    try:
        return process_single_pdf(*args)
    finally:
        _send_worker_progress("done", doc_label)


def _send_worker_progress(event: str, doc_label: str) -> None:
    """Send a worker progress event when a progress queue is configured."""

    # Guard: single-process and tests may not configure a progress queue.
    if WORKER_PROGRESS_QUEUE is None:
        return

    WORKER_PROGRESS_QUEUE.put((event, doc_label))


# ========================================
# Argument Parsing
# ========================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Lab Results Parser - Extract lab results from PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Run all profiles:
  parselabs

  # Run specific profile:
  parselabs extract --profile tsilva

  # List available profiles:
  parselabs --list-profiles

  # Override settings:
  parselabs extract --profile tsilva --model google/gemini-2.5-pro
        """,
    )

    # Shared profile-selection arguments
    add_profile_arguments(
        parser,
        profile_help="Profile name (without extension). If not specified, runs all profiles.",
    )

    # Overrides
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model ID for extraction (overrides the profile value)",
    )
    parser.add_argument("--workers", "-w", type=int, help="Number of parallel workers")
    parser.add_argument(
        "--pattern",
        type=str,
        help="Glob pattern for input files (overrides profile, default: *.pdf)",
    )
    parser.add_argument(
        "--rebuild-from-json",
        action="store_true",
        help="Rebuild per-document CSVs and merged outputs from reviewed page JSON files",
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Compatibility flag for reviewed-truth helpers; merged all.csv exports are unaffected",
    )
    parser.add_argument(
        "--no-auto-standardize",
        dest="auto_standardize",
        action="store_false",
        help="Skip the end-of-run cache refresh for uncached standardization mappings",
    )

    # Console verbosity controls only terminal output; log files keep detailed diagnostics.
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument("--quiet", dest="console_mode", action="store_const", const="quiet", help="Show only errors on the console")
    verbosity_group.add_argument("--verbose", dest="console_mode", action="store_const", const="verbose", help="Show detailed INFO logs on the console")
    verbosity_group.add_argument("--debug", dest="console_mode", action="store_const", const="debug", help="Show DEBUG and higher logs on the console")

    parser.set_defaults(auto_standardize=True, console_mode="normal")

    return parser.parse_args()


def _setup_rebuild_environment(profile_name: str) -> tuple[ProfileConfig, LabSpecsConfig]:
    """Setup logging and lab specs for a reviewed-JSON rebuild."""

    context = RuntimeContext.from_profile(
        profile_name,
        need_input=False,
        need_output=True,
        need_api=False,
        setup_logs=True,
        clear_logs=False,
        console_mode=ACTIVE_CONSOLE_MODE,
    )

    global logger
    logger = context.logger
    log_user_info(logger, "Processing profile: %s", profile_name)
    log_user_info(logger, "Using profile: %s", context.profile.name)
    log_user_info(logger, "Output: %s", context.output_path)

    # Copy lab specs to the output folder so rebuild artifacts stay reproducible.
    copied_path = context.copy_lab_specs_to_output()
    if copied_path is not None:
        logger.info("Copied lab specs to output: %s", copied_path)

    return context.profile, context.lab_specs


# ========================================
# Main Pipeline
# ========================================


def _classify_api_check_error(error_msg: str, timeout: int) -> tuple[bool, str]:
    """Classify an API validation error into a diagnostic result.

    Returns tuple of (is_available, diagnostic_message).
    """

    # Authentication errors (invalid or missing API key)
    if "401" in error_msg or "Unauthorized" in error_msg:
        return (
            False,
            f"Authentication failed - check OPENROUTER_API_KEY in {get_env_file()} "
            f"or the shell environment: {error_msg}",
        )

    # Authorization / permission failures
    if "403" in error_msg or "Forbidden" in error_msg or "permission" in error_msg.lower():
        return False, f"Authorization failed - key or account cannot use this model: {error_msg}"

    # Timeout errors (server slow or unreachable)
    if "timeout" in error_msg.lower():
        return False, f"Server timeout after {timeout}s - server may be unreachable"

    # Model / endpoint not found
    if "404" in error_msg:
        return False, f"Model or endpoint not found - check EXTRACT_MODEL_ID and OPENROUTER_BASE_URL: {error_msg}"

    # Invalid request / unsupported model configuration
    if "400" in error_msg or "BadRequest" in error_msg or "invalid" in error_msg.lower():
        return False, f"API validation request was rejected - check EXTRACT_MODEL_ID, OPENROUTER_BASE_URL, or --model: {error_msg}"

    # Connection failures (network issues, DNS problems, server down)
    if "Connection" in error_msg or "refused" in error_msg.lower() or "reset" in error_msg.lower() or "Name or service not known" in error_msg or "getaddrinfo" in error_msg:
        return False, f"Cannot connect to server: {error_msg}"

    # Unknown errors - assume unavailable to be safe
    return False, f"API validation failed: {error_msg}"


def validate_api_access(client: OpenAI, model_id: str, timeout: int = 10) -> tuple[bool, str]:
    """Validate API access by running a minimal completion with the configured model.

    Args:
        client: OpenAI client configured for OpenRouter
        model_id: Model used for extraction
        timeout: Timeout in seconds for the check

    Returns:
        Tuple of (is_available, message)
    """
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Reply with OK."}],
            temperature=0,
            max_tokens=5,
            timeout=timeout,
        )
        if not completion or not getattr(completion, "choices", None):
            return False, "API validation failed: empty completion response"
        return True, "API key and model validation passed"
    except Exception as e:
        return _classify_api_check_error(str(e), timeout)


def _merge_unique_csv_paths(csv_paths: list[Path]) -> list[Path]:
    """Return readable CSV paths in stable order without duplicates."""

    unique_paths: list[Path] = []
    seen_paths: set[Path] = set()

    # Preserve first-seen order so merges remain stable across reruns.
    for csv_path in csv_paths:
        # Skip missing paths because downstream merge expects readable CSV files.
        if not csv_path.exists():
            continue
        if csv_path in seen_paths:
            continue

        seen_paths.add(csv_path)
        unique_paths.append(csv_path)

    return unique_paths


def _get_console_mode(args) -> ConsoleLogMode:
    """Resolve the requested console logging mode from parsed arguments."""

    # Validate defensively for callers that construct argparse namespaces directly in tests.
    console_mode = getattr(args, "console_mode", "normal")
    if console_mode in {"normal", "verbose", "debug", "quiet"}:
        return console_mode

    # Fallback to the normal default when a custom namespace omits or corrupts the mode.
    return "normal"


def _discover_pdf_files(input_path: Path, input_file_regex: str | None) -> list[Path]:
    """Discover PDFs and translate filesystem errors into pipeline errors."""

    try:
        return discover_pdf_files(input_path, input_file_regex)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        raise PipelineError(str(exc)) from exc


def _process_pdfs_or_use_cache(
    preflight: PdfPreflightResult,
    config: ExtractionConfig,
    lab_specs: LabSpecsConfig,
    log_dir: Path,
    console_mode: ConsoleLogMode = "normal",
) -> tuple[list[Path], list[ExtractionFailureRecord], int]:
    """Process the exact-content unique PDFs for this run.

    Returns tuple of (csv_paths, all_failed_pages, pdfs_failed).
    """

    if not preflight.pdfs_to_process:
        return _merge_unique_csv_paths(preflight.cached_csv_paths), [], 0

    csv_paths, all_failed_pages, pdfs_failed = _process_pdfs_in_parallel(
        preflight.pdfs_to_process,
        config,
        lab_specs,
        log_dir,
        console_mode,
    )
    return _merge_unique_csv_paths(csv_paths), all_failed_pages, pdfs_failed


def _setup_profile_environment(args, profile_name: str) -> tuple[ExtractionConfig, LabSpecsConfig]:
    """Setup environment for a profile: config, logging, lab specs.

    Returns tuple of (config, lab_specs).
    Raises ConfigurationError if setup fails.
    """

    context = RuntimeContext.from_profile(
        profile_name,
        need_input=True,
        need_output=True,
        need_api=True,
        overrides={
            "model": getattr(args, "model", None),
            "workers": getattr(args, "workers", None),
            "pattern": getattr(args, "pattern", None),
        },
        create_output_dir=True,
        setup_logs=True,
        clear_logs=True,
        console_mode=_get_console_mode(args),
    )

    # Guard: Extraction flows always require a validated extraction config.
    if context.extraction_config is None:
        raise ConfigurationError(f"Profile '{profile_name}' is missing extraction runtime settings.")

    global logger
    logger = context.logger
    log_user_info(logger, "Processing profile: %s", profile_name)
    log_user_info(logger, "Input: %s", context.extraction_config.input_path)
    log_user_info(logger, "Output: %s", context.extraction_config.output_path)
    log_user_info(logger, "Model: %s", context.extraction_config.extract_model_id)

    copied_path = context.copy_lab_specs_to_output()
    if copied_path is not None:
        logger.info("Copied lab specs to output: %s", copied_path)

    return context.extraction_config, context.lab_specs


def _export_final_results(
    final_df: pd.DataFrame,
    hidden_cols: list,
    widths: dict,
    output_path: Path,
) -> None:
    """Export final results to CSV and Excel formats."""

    # Export merged results to CSV format
    log_user_info(logger, "Saving merged CSV...")
    csv_path = output_path / "all.csv"
    final_df.to_csv(csv_path, index=False, encoding="utf-8")
    log_user_info(logger, "Saved merged CSV: %s", csv_path)

    # Export merged results to Excel format with formatting
    log_user_info(logger, "Exporting to Excel...")
    excel_path = output_path / "all.xlsx"
    export_excel(final_df, excel_path, hidden_cols, widths)


def _load_merged_review_dataframe(output_path: Path) -> pd.DataFrame:
    """Load the merged review dataframe that was just exported for a profile."""

    csv_path = output_path / "all.csv"

    # Guard: Auto-refresh scans need an exported merged review CSV.
    if not csv_path.exists():
        raise PipelineError(f"Merged review CSV not found at {csv_path}")

    return pd.read_csv(csv_path, encoding="utf-8")


def _rebuild_review_outputs_from_processed_documents(
    output_path: Path,
    lab_specs: LabSpecsConfig,
    *,
    allow_pending: bool,
) -> ReviewedCorpusResult:
    """Rebuild per-document CSVs and merged outputs from persisted page JSON."""

    documents = iter_processed_documents(output_path)

    # Guard: Auto-refresh rebuilds still require processed document directories.
    if not documents:
        raise PipelineError(f"No processed documents found in {output_path}")

    return _collect_reviewed_corpus_from_document_dirs(
        [document.doc_dir for document in documents],
        lab_specs,
        allow_pending=allow_pending,
    )


def _list_processed_document_csv_paths(output_path: Path) -> list[Path]:
    """Return the canonical per-document CSV paths under one output root."""

    return [get_document_csv_path(document.doc_dir) for document in iter_processed_documents(output_path)]


def _log_standardization_refresh_summary(
    result: StandardizationRefreshResult,
    *,
    auto_standardize: bool,
    profile_name: str,
) -> None:
    """Log the end-of-run standardization refresh summary."""

    manual_command = f"parselabs admin update-standardization-caches --profile {profile_name}"

    # Disabled auto-refresh still gets a final scan summary for manual follow-up.
    if not auto_standardize:
        if not result.attempted:
            log_user_info(logger, "[standardization] Auto-refresh disabled; no uncached mappings were found.")
            return

        log_user_warning(
            logger,
            "[standardization] Auto-refresh disabled; %s name(s) and %s unit pair(s) remain uncached. Manual fallback: %s",
            len(result.uncached_names),
            len(result.uncached_unit_pairs),
            manual_command,
        )
        return

    # Guard: No unresolved work means the run was already fully cached.
    if not result.attempted and not result.changed:
        log_user_info(logger, "[standardization] Auto-refresh not needed; all mappings were already cached.")
        return

    # Summarize any cache mutations before reporting unresolved leftovers.
    log_user_info(
        logger,
        "[standardization] Auto-refresh summary: +%s name mapping(s), +%s unit mapping(s)",
        result.name_updates,
        result.unit_updates,
    )

    if result.pruned_name_entries or result.pruned_unit_entries:
        log_user_info(
            logger,
            f"[standardization] Removed {result.pruned_name_entries} stale name entr{'y' if result.pruned_name_entries == 1 else 'ies'} "
            f"and {result.pruned_unit_entries} stale unit entr{'y' if result.pruned_unit_entries == 1 else 'ies'}"
        )

    if result.name_error:
        log_user_warning(logger, "[standardization] Name auto-refresh failed: %s", result.name_error)

    if result.unit_error:
        log_user_warning(logger, "[standardization] Unit auto-refresh failed: %s", result.unit_error)

    # Guard: Fully resolved refreshes can stop after the positive summary.
    if not result.unresolved_names and not result.unresolved_unit_pairs:
        log_user_info(logger, "[standardization] Auto-refresh complete; no uncached mappings remain.")
        return

    log_user_warning(
        logger,
        "[standardization] Remaining uncached mappings after auto-refresh: %s name(s), %s unit pair(s)",
        len(result.unresolved_names),
        len(result.unresolved_unit_pairs),
    )


def _maybe_auto_standardize_outputs(
    *,
    output_path: Path,
    lab_specs: LabSpecsConfig,
    hidden_cols: list,
    widths: dict,
    model_id: str | None,
    base_url: str | None,
    api_key: str | None,
    auto_standardize: bool,
    profile_name: str,
    allow_pending: bool,
) -> list[Path]:
    """Refresh standardization caches after export and rebuild outputs when needed."""

    merged_review_df = _load_merged_review_dataframe(output_path)

    # Scan only when auto-refresh is disabled so the final summary can stay explicit.
    if not auto_standardize:
        result = refresh_standardization_caches_from_dataframe(
            merged_review_df,
            lab_specs,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            dry_run=True,
        )
        _log_standardization_refresh_summary(result, auto_standardize=False, profile_name=profile_name)
        return _list_processed_document_csv_paths(output_path)

    result = refresh_standardization_caches_from_dataframe(
        merged_review_df,
        lab_specs,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
    )

    updated_csv_paths = _list_processed_document_csv_paths(output_path)

    # Apply successful cache updates to all persisted outputs without re-extracting PDFs.
    if result.rebuild_required:
        log_user_info(logger, "[standardization] Rebuilding outputs from page JSON to apply refreshed caches...")

        try:
            reviewed_corpus = _rebuild_review_outputs_from_processed_documents(
                output_path,
                lab_specs,
                allow_pending=allow_pending,
            )
        except PipelineError as exc:
            log_user_warning(logger, "[standardization] Output rebuild after auto-refresh failed: %s", exc)
        else:
            rebuilt_publish_df = reviewed_corpus.merged_review_df if allow_pending else reviewed_corpus.final_df
            _export_final_results(
                rebuilt_publish_df,
                hidden_cols,
                widths,
                output_path,
            )
            updated_csv_paths = reviewed_corpus.csv_paths

    _log_standardization_refresh_summary(result, auto_standardize=True, profile_name=profile_name)
    return updated_csv_paths


def _build_merged_review_dataframe_from_csv_paths(csv_paths: list[Path]) -> pd.DataFrame:
    """Return the merged review-dataframe snapshot for the processed documents."""

    review_frames: list[pd.DataFrame] = []

    # Load each per-document CSV exactly as persisted so all.csv mirrors document CSV state.
    for csv_path in csv_paths:
        review_frames.append(pd.read_csv(csv_path))

    # Guard: No document CSVs means the merged review dataset must keep a stable schema.
    if not review_frames:
        return pd.DataFrame(columns=DOCUMENT_REVIEW_COLUMNS)

    return pd.concat(review_frames, ignore_index=True, sort=False)


def _collect_reviewed_corpus_from_document_dirs(
    doc_dirs: list[Path],
    lab_specs: LabSpecsConfig,
    *,
    allow_pending: bool,
) -> ReviewedCorpusResult:
    """Rebuild document CSVs and collect merged review plus accepted export data."""

    csv_paths: list[Path] = []
    review_frames: list[pd.DataFrame] = []
    accepted_rows: list[pd.DataFrame] = []
    review_issues: list[str] = []

    # Rebuild every document CSV from canonical page JSON before collecting rows.
    for doc_dir in doc_dirs:
        csv_path, review_df = _rebuild_document_csv_with_review_dataframe(doc_dir, lab_specs)
        csv_paths.append(csv_path)
        review_frames.append(review_df)
        summary = get_document_review_summary(doc_dir, review_df)

        # Strict publish mode blocks on pending review state unless explicitly allowed.
        if not allow_pending and not summary.fixture_ready:
            issue_parts: list[str] = []

            # Include pending rows so the reviewer knows what still lacks a decision.
            if summary.pending > 0:
                issue_parts.append(f"{summary.pending} pending row(s)")

            # Include unresolved omission markers because they also block reviewed truth.
            if summary.missing_row_markers > 0:
                issue_parts.append(f"{summary.missing_row_markers} unresolved missing-row marker(s)")

            issue_text = ", ".join(issue_parts) if issue_parts else "document review incomplete"
            review_issues.append(f"{doc_dir.name}: {issue_text}")

        accepted_df = load_document_review_rows(doc_dir, include_statuses={"accepted"})

        # Skip documents that have not produced any accepted reviewed rows yet.
        if accepted_df.empty:
            continue

        accepted_rows.append(accepted_df)

    # Guard: Strict publish mode must stop before writing outputs when review is incomplete.
    if review_issues:
        issue_text = "\n".join(f"- {issue}" for issue in review_issues)
        raise PipelineError(
            "Reviewed JSON rebuild blocked because some documents are not fixture-ready:\n"
            f"{issue_text}"
        )

    merged_review_df = (
        pd.concat(review_frames, ignore_index=True, sort=False)
        if review_frames
        else pd.DataFrame(columns=DOCUMENT_REVIEW_COLUMNS)
    )

    # Guard: Empty accepted corpora still produce a stable empty export schema.
    if not accepted_rows:
        export_cols, _, _, _ = get_column_lists(COLUMN_SCHEMA)
        return ReviewedCorpusResult(
            csv_paths=csv_paths,
            merged_review_df=merged_review_df,
            final_df=pd.DataFrame(columns=export_cols),
        )

    merged_accepted_df = pd.concat(accepted_rows, ignore_index=True)
    log_user_info(logger, "Applying shared export pipeline to accepted reviewed rows...")
    final_df, validation_stats = transform_rows_to_final_export(
        merged_accepted_df,
        lab_specs,
        apply_standardization=True,
    )
    _log_validation_stats(validation_stats)
    return ReviewedCorpusResult(
        csv_paths=csv_paths,
        merged_review_df=merged_review_df,
        final_df=final_df,
    )


def _build_final_export_from_document_dirs(
    doc_dirs: list[Path],
    lab_specs: LabSpecsConfig,
    allow_pending: bool,
) -> tuple[pd.DataFrame, list[Path]]:
    """Rebuild review CSVs and publish only accepted reviewed rows for the target documents."""

    reviewed_corpus = _collect_reviewed_corpus_from_document_dirs(
        doc_dirs,
        lab_specs,
        allow_pending=allow_pending,
    )
    return reviewed_corpus.final_df, reviewed_corpus.csv_paths


def run_pipeline_for_pdf_files(
    pdf_files: list[Path],
    config: ExtractionConfig,
    lab_specs: LabSpecsConfig,
) -> PipelineRunResult:
    """Process explicit PDF files and return the final export DataFrame plus metadata."""

    pdf_files = sorted(pdf_files)
    logger.info("Found %s PDF(s) for explicit pipeline run", len(pdf_files))

    if not pdf_files:
        raise PipelineError("No PDF files provided for processing.")

    console_mode = getattr(config, "console_mode", "normal")
    preflight = _prepare_pdf_run(
        pdf_files,
        config.output_path,
        show_progress=_console_mode_shows_detail(console_mode),
    )
    unique_pdf_count = len(preflight.pdfs_to_process)

    # Report duplicate-content PDFs before moving on to processing stats.
    for dup_path, orig_path in preflight.duplicates:
        logger.warning(f"Skipping duplicate PDF: {dup_path.name} (same content as {orig_path.name})")
    if preflight.duplicates:
        log_user_info(
            logger,
            "Skipped %s duplicate PDF(s), %s unique PDF(s) remaining",
            len(preflight.duplicates),
            unique_pdf_count,
        )

    logger.info("Processing %s PDF(s)", len(preflight.pdfs_to_process))

    log_dir = config.output_path / "logs"
    csv_paths, all_failed_pages, pdfs_failed = _process_pdfs_or_use_cache(
        preflight,
        config,
        lab_specs,
        log_dir,
        console_mode,
    )

    if not csv_paths:
        raise PipelineError("No PDFs successfully processed.")

    log_user_info(logger, "Successfully processed %s document review snapshots", len(csv_paths))
    reviewed_corpus = _collect_reviewed_corpus_from_document_dirs(
        [csv_path.parent for csv_path in csv_paths],
        lab_specs,
        allow_pending=True,
    )

    return PipelineRunResult(
        final_df=reviewed_corpus.final_df,
        merged_review_df=reviewed_corpus.merged_review_df,
        csv_paths=reviewed_corpus.csv_paths,
        failed_pages=all_failed_pages,
        pdfs_failed=pdfs_failed,
    )
def build_final_output_dataframe_from_reviewed_json(
    output_path: Path,
    lab_specs: LabSpecsConfig,
    allow_pending: bool = False,
) -> tuple[pd.DataFrame, list[Path]]:
    """Return the final post-validation export dataframe from reviewed page JSON files."""
    documents = iter_processed_documents(output_path)

    # Guard: Reviewed rebuilds require at least one processed document directory.
    if not documents:
        raise PipelineError(f"No processed documents found in {output_path}")

    return _build_final_export_from_document_dirs(
        [document.doc_dir for document in documents],
        lab_specs,
        allow_pending=allow_pending,
    )


def _run_reviewed_json_rebuild(args, profile_name: str, allow_pending: bool) -> None:
    """Rebuild per-document CSVs and merged outputs from reviewed page JSON files."""

    global ACTIVE_CONSOLE_MODE

    # Make the rebuild environment use the same console mode as the parsed CLI args.
    ACTIVE_CONSOLE_MODE = _get_console_mode(args)
    profile, lab_specs = _setup_rebuild_environment(profile_name)
    _, hidden_cols, widths, _ = get_column_lists(COLUMN_SCHEMA)
    log_user_info(logger, "Rebuilding document CSVs and merged outputs from reviewed page JSON files...")
    reviewed_corpus = _rebuild_review_outputs_from_processed_documents(
        profile.output_path,
        lab_specs,
        allow_pending=allow_pending,
    )

    publish_df = reviewed_corpus.merged_review_df if allow_pending else reviewed_corpus.final_df
    _export_final_results(
        publish_df,
        hidden_cols,
        widths,
        profile.output_path,
    )
    final_csv_paths = _maybe_auto_standardize_outputs(
        output_path=profile.output_path,
        lab_specs=lab_specs,
        hidden_cols=hidden_cols,
        widths=widths,
        model_id=getattr(args, "model", None) or profile.extract_model_id,
        base_url=profile.openrouter_base_url,
        api_key=profile.openrouter_api_key,
        auto_standardize=bool(getattr(args, "auto_standardize", True)),
        profile_name=profile_name,
        allow_pending=allow_pending,
    )
    _report_extraction_failures([], profile.output_path / "all.csv", final_csv_paths)


def run_for_profile(args, profile_name: str) -> None:
    """Run extraction pipeline for a single profile.

    Raises ConfigurationError for config/setup failures.
    Raises PipelineError for runtime pipeline failures.
    """

    # Setup environment: config, logging, lab specs (raises ConfigurationError on failure)
    config, lab_specs = _setup_profile_environment(args, profile_name)

    log_user_info(logger, "Validating API access with a simple prompt...")
    is_available, message = validate_api_access(get_openai_client(config), config.extract_model_id)
    if not is_available:
        raise ConfigurationError(f"Cannot start extraction for profile '{profile_name}' - {message}")
    log_user_info(logger, "API validation passed: %s", message)

    # Get column configuration for export formatting
    _, hidden_cols, widths, _ = get_column_lists(COLUMN_SCHEMA)

    # Discover PDF files matching the input pattern
    try:
        pdf_files = discover_pdf_files(config.input_path, config.input_file_regex)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        raise PipelineError(str(exc)) from exc
    log_user_info(logger, "Found %s PDF(s) matching '%s'", len(pdf_files), config.input_file_regex)

    if not pdf_files:
        raise PipelineError(f"No PDF files found matching '{config.input_file_regex}' in {config.input_path}")

    pipeline_result = run_pipeline_for_pdf_files(pdf_files, config, lab_specs)

    # Export the merged review corpus so all.csv mirrors the per-document review CSVs.
    _export_final_results(
        pipeline_result.merged_review_df,
        hidden_cols,
        widths,
        config.output_path,
    )
    final_csv_paths = _maybe_auto_standardize_outputs(
        output_path=config.output_path,
        lab_specs=lab_specs,
        hidden_cols=hidden_cols,
        widths=widths,
        model_id=config.extract_model_id,
        base_url=config.openrouter_base_url,
        api_key=config.openrouter_api_key,
        auto_standardize=bool(getattr(args, "auto_standardize", True)),
        profile_name=profile_name,
        allow_pending=True,
    )
    _report_extraction_failures(pipeline_result.failed_pages, config.output_path / "all.csv", final_csv_paths)


def _report_extraction_failures(all_failed_pages: list[ExtractionFailureRecord], csv_path: Path, csv_paths: list[Path]) -> None:
    """Log and report any extraction failures to user."""

    # Log final pipeline summary
    log_user_info(logger, "Pipeline completed: %s PDF(s) processed; output: %s", len(csv_paths), csv_path)

    # Report any extraction failures to user and log
    if all_failed_pages:
        log_user_warning(logger, "Pages with extraction failures: %s", len(all_failed_pages))
        for failure in all_failed_pages:
            log_user_warning(logger, "  - %s: %s", failure["page"], failure["reason"])
    else:
        log_user_info(logger, "Extraction failures: 0")


def main():
    """Main pipeline orchestration."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Parse command-line arguments
    args = parse_args()

    # Handle --list-profiles flag (just list and exit)
    if args.list_profiles:
        profiles = ProfileConfig.list_profiles()

        # Display available profiles
        if profiles:
            logger.info("Available profiles:")
            for name in profiles:
                logger.info(f"  - {name}")

        # No profiles configured yet
        else:
            logger.info(f"No profiles found. Create profile files in {PROFILES_DIR}.")
        return

    # Determine which profiles to run (single specified or all available)
    if args.profile:
        profiles_to_run = [args.profile]

    # No profile specified - run all available
    else:
        profiles_to_run = ProfileConfig.list_profiles()

        # Guard: No profiles configured
        if not profiles_to_run:
            logger.error(f"No profiles found. Create profile files in {PROFILES_DIR}.")
            logger.error("Or use --profile to specify one.")
            sys.exit(1)
        logger.info("Running profiles: %s", ", ".join(profiles_to_run))

    # Initialize results tracking for each profile
    results = {}
    rebuild_from_json = bool(getattr(args, "rebuild_from_json", False))
    allow_pending = bool(getattr(args, "allow_pending", False))

    # Run extraction pipeline for each profile
    for profile_name in profiles_to_run:
        try:
            # Route reviewed-JSON rebuilds through the JSON-only path instead of the extraction pipeline.
            if rebuild_from_json:
                _run_reviewed_json_rebuild(args, profile_name, allow_pending=allow_pending)
            else:
                run_for_profile(args, profile_name)
            results[profile_name] = "success"
        except (ConfigurationError, PipelineError) as e:
            logger.error(f"\nError in profile '{profile_name}':\n{e}")
            results[profile_name] = "failed"
            sys.exit(1)

    # Print summary if multiple profiles were processed
    if len(profiles_to_run) > 1:
        logger.info("Profile summary:")
        for profile_name, status in results.items():
            logger.info("  %s: %s", profile_name, status)


if __name__ == "__main__":
    main()
