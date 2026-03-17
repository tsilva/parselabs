"""Main entry point for lab results extraction and processing."""

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402
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
from parselabs.exceptions import ConfigurationError, PipelineError  # noqa: E402
from parselabs.export_schema import COLUMN_SCHEMA, get_column_lists  # noqa: E402
from parselabs.extraction import (  # noqa: E402
    extract_labs_from_page_image,
    extract_labs_from_text,
)
from parselabs.paths import get_profiles_dir  # noqa: E402
from parselabs.rows import (  # noqa: E402
    DOCUMENT_REVIEW_COLUMNS,
    build_document_review_dataframe,
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
from parselabs.utils import (  # noqa: E402
    create_page_image_variants,
    setup_logging,
)

# Module-level logger (file handlers added after config is loaded)
logger = logging.getLogger(__name__)
PROFILES_DIR = get_profiles_dir()
EXTRACTION_FAILURE_RAW_NAME = "[EXTRACTION FAILED]"


# ========================================
# PDF Text Extraction (Cost Optimization)
# ========================================


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, bool]:
    """
    Extract text from PDF using pdftotext (from poppler).

    Returns:
        Tuple of (extracted_text, success). Returns ("", False) on non-zero exit code.
        May raise subprocess.TimeoutExpired on timeout.
    """

    # Execute pdftotext command with layout preservation and 30s timeout
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Check if command succeeded
    if result.returncode != 0:
        logger.debug(f"pdftotext returned non-zero exit code: {result.returncode}")
        return "", False

    return result.stdout, True


_MIN_TEXT_CHARS = 200  # Minimum non-whitespace characters to attempt text extraction
_MIN_PAGE_TEXT_CHARS = 80  # Lower threshold for page-level text routing


def _text_has_enough_content(text: str, min_chars: int = _MIN_TEXT_CHARS) -> bool:
    """Check if extracted text has enough content to attempt LLM extraction."""
    clean_text = text.replace(" ", "").replace("\n", "").replace("\t", "")
    return len(clean_text) >= min_chars


def _split_pdf_text_into_pages(text: str) -> list[str]:
    """Split pdftotext output into per-page chunks."""

    return [page.strip() for page in text.split("\f")]


def _extract_page_texts_from_pdf(pdf_path: Path, expected_pages: int) -> list[str]:
    """Extract PDF text once and return one text chunk per page."""

    pdf_text, pdftotext_success = extract_text_from_pdf(pdf_path)
    if not pdftotext_success:
        return [""] * expected_pages

    page_texts = _split_pdf_text_into_pages(pdf_text)

    # pdftotext often emits a trailing empty page after the final form-feed.
    while page_texts and page_texts[-1] == "":
        page_texts.pop()

    if len(page_texts) < expected_pages:
        page_texts.extend([""] * (expected_pages - len(page_texts)))
    elif len(page_texts) > expected_pages:
        page_texts = page_texts[:expected_pages]

    return page_texts


# ========================================
# PDF Processing - Helper Functions
# ========================================


def _setup_pdf_processing(pdf_path: Path, output_dir: Path, file_hash: str) -> tuple[Path, Path, list]:
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


def _page_requires_image_fallback(page_data: dict) -> bool:
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


def _ensure_extraction_failure_placeholder(page_data: dict) -> dict:
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


def _extract_page_data_from_text(page_text: str, config: ExtractionConfig, page_name: str) -> dict:
    """Extract a single page from pdftotext output."""

    logger.info(f"[{page_name}] Attempting TEXT extraction")
    return extract_labs_from_text(page_text, config.extract_model_id, get_openai_client(config))


def _extract_page_data_from_image(
    image_path: Path,
    config: ExtractionConfig,
    page_name: str,
    variant_name: str,
) -> dict:
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
    failed_pages: list,
    page_text: str,
) -> dict:
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

    attempts: list[tuple[str, Callable[[], dict]]] = []

    # Prefer text extraction only when the embedded page text is substantial enough.
    if _text_has_enough_content(page_text, min_chars=_MIN_PAGE_TEXT_CHARS):
        attempts.append(("TEXT extraction", lambda: _extract_page_data_from_text(page_text, config, page_name)))

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

    page_data: dict | None = None

    # Stop at the first extraction attempt that produces a reviewable payload.
    for attempt_idx, (attempt_label, attempt_fn) in enumerate(attempts):
        candidate = attempt_fn()

        if not _page_requires_image_fallback(candidate):
            page_data = candidate
            break

        has_more_attempts = attempt_idx < len(attempts) - 1
        if has_more_attempts:
            logger.warning(f"[{page_name}] {attempt_label} needs fallback; trying next route")
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
    page_data: dict,
    pdf_stem: str,
    page_idx: int,
    failed_pages: list,
    page_name: str | None = None,
) -> None:
    """Check if extraction failed and record failure reason."""

    # Skip if extraction succeeded
    if not page_data.get("_extraction_failed"):
        return

    # Extract failure reason from extraction data
    failure_reason = page_data.get("_failure_reason", "Unknown error")

    # Record failure for summary reporting
    failed_pages.append({"page": f"{pdf_stem} page {page_idx + 1}", "reason": failure_reason})

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
    failed_pages: list,
    page_text: str,
) -> tuple[list, dict | None]:
    """Process a single PDF page: preprocess, extract, and return results with metadata.

    Returns tuple of (page_results, page_data) where page_data is the full extraction data.
    """

    # Generate unique page identifier with zero-padding
    page_name = f"{pdf_stem}.{page_idx + 1:03d}"
    json_path = doc_out_dir / f"{page_name}.json"

    logger.info(f"[{page_name}] Processing page {page_idx + 1}/{total_pages}...")

    # Preprocess and cache page image variants
    image_paths = _prepare_page_images(page_image, page_name, doc_out_dir)

    # Extract data using page-level text/vision routing or load from cache
    page_data = _extract_or_load_page_data(
        image_paths,
        json_path,
        page_name,
        config,
        pdf_stem,
        page_idx,
        failed_pages,
        page_text,
    )

    # Add page metadata to results
    page_results = _add_page_metadata(page_data.get("lab_results", []), page_idx, page_name)

    return page_results, page_data


def _extract_via_pages(
    copied_pdf: Path,
    config: ExtractionConfig,
    doc_out_dir: Path,
    pdf_stem: str,
    failed_pages: list,
) -> tuple[list, str | None]:
    """Extract lab results using per-page hybrid text/vision routing."""

    # Convert PDF pages to PIL images
    pil_pages = _convert_pdf_to_images(copied_pdf, pdf_stem)

    # Guard: Handle PDF conversion failure
    if pil_pages is None:
        return [], None

    logger.info(f"[{pdf_stem}] Processing {len(pil_pages)} page(s) with hybrid extraction...")

    # Extract page text once so each page can be routed independently.
    page_texts = _extract_page_texts_from_pdf(copied_pdf, len(pil_pages))

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
            page_texts[page_idx],
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
    failed_pages: list,
) -> tuple[list, str | None]:
    """Extract lab data from PDF using per-page hybrid extraction.

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
) -> tuple[Path | None, list[dict]]:
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

        # Extract page JSON using the simplified text-first strategy with one image fallback.
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

    except Exception as e:
        # Catch-all for errors during extraction or review CSV rebuild.
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
    failed_pages: list[dict]
    pdfs_failed: int


@dataclass(frozen=True)
class ReviewedCorpusResult:
    """Rebuilt per-document CSVs plus merged review and accepted-export data."""

    csv_paths: list[Path]
    merged_review_df: pd.DataFrame
    final_df: pd.DataFrame


def _prepare_pdf_run(pdf_files: list[Path], output_path: Path) -> PdfPreflightResult:
    """Hash every PDF once, dedupe exact duplicates, and derive hashed output paths."""

    pdf_iterator = pdf_files

    # Show hashing progress for larger explicit runs.
    if len(pdf_files) > 1:
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
        logger.info(f"Validation flagged {flagged_count} rows for review")

        # Log breakdown of flag reasons
        for reason, count in validation_stats.get("flags_by_reason", {}).items():
            logger.info(f"  - {reason}: {count}")


def _process_pdfs_in_parallel(
    pdfs_to_process: list[DocumentRef],
    config: ExtractionConfig,
    lab_specs: LabSpecsConfig,
    log_dir: Path,
) -> tuple[list[Path], list[dict], int]:
    """
    Process PDFs in parallel using multiprocessing pool.

    Returns:
        Tuple of (csv_paths, all_failed_pages, pdfs_failed_count)
    """

    # Calculate optimal worker count (don't exceed CPU count or task count)
    n_workers = min(config.max_workers, len(pdfs_to_process))
    logger.info(f"Using {n_workers} worker(s) for PDF processing")

    # Build task tuples for each PDF to process
    tasks = [(task.source_pdf, task.file_hash, config.output_path, config, lab_specs) for task in pdfs_to_process]

    if n_workers == 1:
        _init_worker_logging(log_dir)
        results = []
        with tqdm(total=len(tasks), desc="Processing PDFs", unit="pdf") as pbar:
            for task in tasks:
                results.append(_process_pdf_wrapper(task))
                pbar.update(1)
    else:
        # Execute parallel processing with progress bar
        with Pool(n_workers, initializer=_init_worker_logging, initargs=(log_dir,)) as pool:
            results = []

            # Track progress with tqdm as tasks complete
            with tqdm(total=len(tasks), desc="Processing PDFs", unit="pdf") as pbar:
                for result in pool.imap(_process_pdf_wrapper, tasks):
                    results.append(result)
                    pbar.update(1)

    # Unpack results and collect statistics.
    pdfs_failed = sum(1 for csv_path, _ in results if csv_path is None)
    all_failed_pages = []
    for _, failed_pages in results:
        all_failed_pages.extend(failed_pages)

    # Keep only successful CSV outputs from the worker run.
    csv_paths = [csv_path for csv_path, _ in results if csv_path and csv_path.exists()]

    return csv_paths, all_failed_pages, pdfs_failed


def _init_worker_logging(log_dir: Path):
    """Initialize logging in worker processes."""
    setup_logging(log_dir, clear_logs=False)


def _process_pdf_wrapper(args):
    """Wrapper function for multiprocessing.

    Returns tuple of (csv_path, failed_pages) from process_single_pdf.
    """
    return process_single_pdf(*args)


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
    parser.set_defaults(auto_standardize=True)

    return parser.parse_args()


def _load_profile_for_rebuild(profile_name: str) -> ProfileConfig:
    """Load the minimal profile state required for a reviewed-JSON rebuild."""

    profile_path = ProfileConfig.find_path(profile_name)

    # Guard: The requested profile must exist before rebuild can start.
    if not profile_path:
        raise ConfigurationError(f"Profile '{profile_name}' not found. Use --list-profiles to see available profiles.")

    profile = ProfileConfig.from_file(profile_path)

    # Guard: Reviewed rebuilds operate only on processed outputs, so output_path is required.
    if not profile.output_path:
        raise ConfigurationError(f"Profile '{profile_name}' has no output_path defined.")

    # Guard: The processed output directory must already exist.
    if not profile.output_path.exists():
        raise ConfigurationError(f"Output path does not exist: {profile.output_path}")

    return profile


def _setup_rebuild_environment(profile_name: str) -> tuple[ProfileConfig, LabSpecsConfig]:
    """Setup logging and lab specs for a reviewed-JSON rebuild."""

    context = RuntimeContext.from_profile(
        profile_name,
        need_input=False,
        need_output=True,
        need_api=False,
        setup_logs=True,
        clear_logs=False,
    )

    global logger
    logger = context.logger
    logger.info(f"Using profile: {context.profile.name}")
    logger.info(f"Output: {context.output_path}")

    # Copy lab specs to the output folder so rebuild artifacts stay reproducible.
    copied_path = context.copy_lab_specs_to_output()
    if copied_path is not None:
        logger.info(f"Copied lab specs to output: {copied_path}")

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
        return False, f"Authentication failed - check the profile openrouter_api_key: {error_msg}"

    # Authorization / permission failures
    if "403" in error_msg or "Forbidden" in error_msg or "permission" in error_msg.lower():
        return False, f"Authorization failed - key or account cannot use this model: {error_msg}"

    # Timeout errors (server slow or unreachable)
    if "timeout" in error_msg.lower():
        return False, f"Server timeout after {timeout}s - server may be unreachable"

    # Model / endpoint not found
    if "404" in error_msg:
        return False, f"Model or endpoint not found - check extract_model_id and base_url: {error_msg}"

    # Invalid request / unsupported model configuration
    if "400" in error_msg or "BadRequest" in error_msg or "invalid" in error_msg.lower():
        return False, f"API validation request was rejected - check extract_model_id and profile settings: {error_msg}"

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
        # Classify the error to provide helpful diagnostic messages
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


def _process_pdfs_or_use_cache(
    preflight: PdfPreflightResult,
    config: ExtractionConfig,
    lab_specs: LabSpecsConfig,
    log_dir: Path,
) -> tuple[list[Path], list[dict], int]:
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
    )

    # Guard: Extraction flows always require a validated extraction config.
    if context.extraction_config is None:
        raise ConfigurationError(f"Profile '{profile_name}' is missing extraction runtime settings.")

    global logger
    logger = context.logger
    logger.info(f"Input: {context.extraction_config.input_path}")
    logger.info(f"Output: {context.extraction_config.output_path}")
    logger.info(f"Model: {context.extraction_config.extract_model_id}")

    copied_path = context.copy_lab_specs_to_output()
    if copied_path is not None:
        logger.info(f"Copied lab specs to output: {copied_path}")

    return context.extraction_config, context.lab_specs


def _export_final_results(
    final_df: pd.DataFrame,
    hidden_cols: list,
    widths: dict,
    output_path: Path,
) -> None:
    """Export final results to CSV and Excel formats."""

    # Export merged results to CSV format
    logger.info("Saving merged CSV...")
    csv_path = output_path / "all.csv"
    final_df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved merged CSV: {csv_path}")

    # Export merged results to Excel format with formatting
    logger.info("Exporting to Excel...")
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
            logger.info("[standardization] Auto-refresh disabled; no uncached mappings were found.")
            return

        logger.warning(
            f"[standardization] Auto-refresh disabled; {len(result.uncached_names)} name(s) and "
            f"{len(result.uncached_unit_pairs)} unit pair(s) remain uncached. "
            f"Manual fallback: {manual_command}"
        )
        return

    # Guard: No unresolved work means the run was already fully cached.
    if not result.attempted and not result.changed:
        logger.info("[standardization] Auto-refresh not needed; all mappings were already cached.")
        return

    # Summarize any cache mutations before reporting unresolved leftovers.
    logger.info(
        f"[standardization] Auto-refresh summary: +{result.name_updates} name mapping(s), "
        f"+{result.unit_updates} unit mapping(s)"
    )

    if result.pruned_name_entries or result.pruned_unit_entries:
        logger.info(
            f"[standardization] Removed {result.pruned_name_entries} stale name entr{'y' if result.pruned_name_entries == 1 else 'ies'} "
            f"and {result.pruned_unit_entries} stale unit entr{'y' if result.pruned_unit_entries == 1 else 'ies'}"
        )

    if result.name_error:
        logger.warning(f"[standardization] Name auto-refresh failed: {result.name_error}")

    if result.unit_error:
        logger.warning(f"[standardization] Unit auto-refresh failed: {result.unit_error}")

    # Guard: Fully resolved refreshes can stop after the positive summary.
    if not result.unresolved_names and not result.unresolved_unit_pairs:
        logger.info("[standardization] Auto-refresh complete; no uncached mappings remain.")
        return

    logger.warning(
        f"[standardization] Remaining uncached mappings after auto-refresh: "
        f"{len(result.unresolved_names)} name(s), {len(result.unresolved_unit_pairs)} unit pair(s)"
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

    try:
        result = refresh_standardization_caches_from_dataframe(
            merged_review_df,
            lab_specs,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
        )
    except Exception as exc:
        logger.warning(f"[standardization] Auto-refresh failed before completion: {exc}")
        return _list_processed_document_csv_paths(output_path)

    updated_csv_paths = _list_processed_document_csv_paths(output_path)

    # Apply successful cache updates to all persisted outputs without re-extracting PDFs.
    if result.rebuild_required:
        logger.info("[standardization] Rebuilding outputs from page JSON to apply refreshed caches...")

        try:
            reviewed_corpus = _rebuild_review_outputs_from_processed_documents(
                output_path,
                lab_specs,
                allow_pending=allow_pending,
            )
        except PipelineError as exc:
            logger.warning(f"[standardization] Output rebuild after auto-refresh failed: {exc}")
        else:
            _export_final_results(
                reviewed_corpus.merged_review_df,
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
        csv_paths.append(rebuild_document_csv(doc_dir, lab_specs))
        review_df = build_document_review_dataframe(doc_dir, lab_specs)
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
    logger.info("Applying shared export pipeline to accepted reviewed rows...")
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
    logger.info(f"Found {len(pdf_files)} PDF(s) for explicit pipeline run")

    if not pdf_files:
        raise PipelineError("No PDF files provided for processing.")

    preflight = _prepare_pdf_run(pdf_files, config.output_path)
    unique_pdf_count = len(preflight.pdfs_to_process)

    # Report duplicate-content PDFs before moving on to processing stats.
    for dup_path, orig_path in preflight.duplicates:
        logger.warning(f"Skipping duplicate PDF: {dup_path.name} (same content as {orig_path.name})")
    if preflight.duplicates:
        logger.info(
            f"Skipped {len(preflight.duplicates)} duplicate PDF(s), "
            f"{unique_pdf_count} unique PDF(s) remaining"
        )

    logger.info(f"Processing {len(preflight.pdfs_to_process)} PDF(s)")

    log_dir = config.output_path / "logs"
    csv_paths, all_failed_pages, pdfs_failed = _process_pdfs_or_use_cache(
        preflight,
        config,
        lab_specs,
        log_dir,
    )

    if not csv_paths:
        raise PipelineError("No PDFs successfully processed.")

    logger.info(f"Successfully processed {len(csv_paths)} document review snapshots")
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


def build_final_output_dataframe(
    pdf_files: list[Path],
    config: ExtractionConfig,
    lab_specs: LabSpecsConfig,
) -> pd.DataFrame:
    """Return the final post-validation export DataFrame for explicit PDFs."""

    return run_pipeline_for_pdf_files(pdf_files, config, lab_specs).final_df


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

    profile, lab_specs = _setup_rebuild_environment(profile_name)
    _, hidden_cols, widths, _ = get_column_lists(COLUMN_SCHEMA)
    logger.info("Rebuilding document CSVs and merged outputs from reviewed page JSON files...")
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

    logger.info("Validating API access with a simple prompt...")
    is_available, message = validate_api_access(get_openai_client(config), config.extract_model_id)
    if not is_available:
        raise ConfigurationError(f"Cannot start extraction for profile '{profile_name}' - {message}")
    logger.info(f"API validation passed: {message}")

    # Get column configuration for export formatting
    _, hidden_cols, widths, _ = get_column_lists(COLUMN_SCHEMA)

    # Discover PDF files matching the input pattern
    try:
        pdf_files = discover_pdf_files(config.input_path, config.input_file_regex)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        raise PipelineError(str(exc)) from exc
    logger.info(f"Found {len(pdf_files)} PDF(s) matching '{config.input_file_regex}'")

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


def _report_extraction_failures(all_failed_pages: list[dict], csv_path: Path, csv_paths: list[Path]) -> None:
    """Log and report any extraction failures to user."""

    # Log final pipeline summary
    logger.info("=" * 50)
    logger.info("Pipeline completed")
    logger.info(f"  PDFs processed: {len(csv_paths)}")
    logger.info(f"  Output: {csv_path}")

    # Report any extraction failures to user and log
    if all_failed_pages:
        logger.warning(f"  Pages with extraction failures: {len(all_failed_pages)}")
        for failure in all_failed_pages:
            logger.warning(f"    - {failure['page']}: {failure['reason']}")
    else:
        logger.info("  Extraction failures: 0")


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
        logger.info(f"Running all profiles: {', '.join(profiles_to_run)}")

    # Initialize results tracking for each profile
    results = {}
    rebuild_from_json = bool(getattr(args, "rebuild_from_json", False))
    allow_pending = bool(getattr(args, "allow_pending", False))

    # Run extraction pipeline for each profile
    for profile_name in profiles_to_run:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing profile: {profile_name}")
        logger.info(f"{'=' * 60}")
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
        # Catch-all for unexpected errors during profile processing
        except Exception as e:
            logger.error(f"\nUnexpected error in profile '{profile_name}': {e}")
            results[profile_name] = f"error: {e}"
            sys.exit(1)

    # Print summary if multiple profiles were processed
    if len(profiles_to_run) > 1:
        logger.info(f"\n{'=' * 60}")
        logger.info("Summary:")
        logger.info(f"{'=' * 60}")
        for profile_name, status in results.items():
            logger.info(f"  {profile_name}: {status}")


if __name__ == "__main__":
    main()
