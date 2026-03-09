"""Main entry point for lab results extraction and processing."""

from parselabs.utils import load_dotenv_with_env

load_dotenv_with_env()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import pdf2image  # noqa: E402
from openai import OpenAI  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Local imports
from parselabs.config import (  # noqa: E402
    UNKNOWN_VALUE,
    ExtractionConfig,
    LabSpecsConfig,
    ProfileConfig,
)
from parselabs.exceptions import ConfigurationError, PipelineError  # noqa: E402
from parselabs.export_schema import COLUMN_SCHEMA, get_column_lists  # noqa: E402
from parselabs.extraction import (  # noqa: E402
    LabResult,
    extract_labs_from_page_image,
    extract_labs_from_text,
)
from parselabs.normalization import (  # noqa: E402
    apply_dtype_conversions,
    apply_normalizations,
    deduplicate_results,
    flag_duplicate_entries,
)
from parselabs.standardization import (  # noqa: E402
    standardize_lab_names,
    standardize_lab_units,
)
from parselabs.paths import get_profiles_dir  # noqa: E402
from parselabs.utils import (  # noqa: E402
    create_page_image_variants,
    ensure_columns,
    setup_logging,
)
from parselabs.validation import ValueValidator  # noqa: E402

# Module-level logger (file handlers added after config is loaded)
logger = logging.getLogger(__name__)
PROFILES_DIR = get_profiles_dir()

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


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
# File Hashing
# ========================================

_file_hash_cache: dict[Path, str] = {}


def _compute_file_hash(file_path: Path, hash_length: int = 8) -> str:
    """Compute SHA-256 hash of a file, returning first `hash_length` hex chars.

    Results are cached to avoid re-hashing the same file.
    """

    # Return cached result if available
    resolved = file_path.resolve()
    if resolved in _file_hash_cache:
        return _file_hash_cache[resolved]

    h = hashlib.sha256()
    with open(resolved, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    result = h.hexdigest()[:hash_length]
    _file_hash_cache[resolved] = result
    return result


# ========================================
# PDF Processing - Helper Functions
# ========================================


def _setup_pdf_processing(pdf_path: Path, output_dir: Path, file_hash: str) -> tuple[Path, Path, list]:
    """Initialize processing: create directories, return paths."""
    pdf_stem = pdf_path.stem  # Extract filename without extension for directory naming
    dir_name = f"{pdf_stem}_{file_hash}"  # Include file hash for uniqueness
    doc_out_dir = output_dir / dir_name  # Create document-specific output directory

    # Backwards-compat migration: rename legacy directory if it exists
    legacy_dir = output_dir / pdf_stem
    if legacy_dir.exists() and not doc_out_dir.exists() and legacy_dir.is_dir():
        legacy_dir.rename(doc_out_dir)
        logger.info(f"Migrated legacy directory: {pdf_stem} -> {dir_name}")

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


def _page_extraction_quality(page_data: dict) -> int:
    """Heuristic score for choosing between page extraction attempts."""

    results = [result for result in page_data.get("lab_results", []) if isinstance(result, dict)]
    if not results:
        return 25 if page_data.get("page_has_lab_data") is False else -25

    score = len(results) * 12
    null_value_count = sum(1 for result in results if not result.get("raw_value"))
    empty_name_count = sum(1 for result in results if not str(result.get("raw_lab_name", "")).strip())
    unique_name_count = len({str(result.get("raw_lab_name", "")).strip() for result in results if str(result.get("raw_lab_name", "")).strip()})

    score -= null_value_count * 4
    score -= empty_name_count * 8
    score += unique_name_count

    if null_value_count and (null_value_count / len(results)) > 0.5:
        score -= 20

    return score


def _page_extraction_needs_reread(page_data: dict) -> bool:
    """Detect weak page extractions that should trigger another pass."""

    results = [result for result in page_data.get("lab_results", []) if isinstance(result, dict)]
    if page_data.get("page_has_lab_data") is False:
        return False
    if not results:
        return True

    null_value_count = sum(1 for result in results if not result.get("raw_value"))
    if (null_value_count / len(results)) > 0.5:
        return True

    empty_name_count = sum(1 for result in results if not str(result.get("raw_lab_name", "")).strip())
    return empty_name_count > 0


def _finalize_page_candidate(page_data: dict, method: str) -> dict:
    """Annotate a page candidate with internal selection metadata."""

    page_data["_extraction_method"] = method
    page_data["_extraction_quality"] = _page_extraction_quality(page_data)
    return page_data


def _extract_page_data_from_text(page_text: str, config: ExtractionConfig, page_name: str) -> dict:
    """Extract a single page from pdftotext output."""

    logger.info(f"[{page_name}] Attempting TEXT extraction")
    page_data = extract_labs_from_text(page_text, config.extract_model_id, client)
    return _finalize_page_candidate(page_data, "text")


def _extract_page_data_from_image(
    image_path: Path,
    config: ExtractionConfig,
    page_name: str,
    variant_name: str,
) -> dict:
    """Extract a single page from an image variant."""

    logger.info(f"[{page_name}] Attempting VISION extraction ({variant_name})")
    page_data = extract_labs_from_page_image(
        image_path,
        config.extract_model_id,
        client,
    )
    return _finalize_page_candidate(page_data, f"vision:{variant_name}")


def _select_best_page_candidate(page_name: str, candidates: list[dict]) -> dict:
    """Choose the strongest extraction candidate for a page."""

    best_candidate = max(
        candidates,
        key=lambda candidate: (
            candidate.get("_extraction_quality", float("-inf")),
            len(candidate.get("lab_results", [])),
            candidate.get("_extraction_method") == "text",
        ),
    )
    logger.info(
        f"[{page_name}] Selected {best_candidate.get('_extraction_method')} "
        f"(score={best_candidate.get('_extraction_quality')}, "
        f"results={len(best_candidate.get('lab_results', []))})"
    )
    return best_candidate


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

    # Check if extraction results already cached for this page
    if json_path.exists():
        logger.info(f"[{page_name}] Loading cached extraction data")
        page_data = json.loads(json_path.read_text(encoding="utf-8"))
        page_data["source_file"] = page_name
        _check_and_record_failure(page_data, pdf_stem, page_idx, failed_pages)
        return page_data

    candidates = []

    if _text_has_enough_content(page_text, min_chars=_MIN_PAGE_TEXT_CHARS):
        text_candidate = _extract_page_data_from_text(page_text, config, page_name)
        candidates.append(text_candidate)
        if not _page_extraction_needs_reread(text_candidate):
            page_data = _select_best_page_candidate(page_name, candidates)
        else:
            logger.warning(
                f"[{page_name}] TEXT extraction looked weak "
                f"(score={text_candidate.get('_extraction_quality')}); trying VISION"
            )
            page_data = None
    else:
        page_data = None

    if page_data is None:
        primary_candidate = _extract_page_data_from_image(
            image_paths["primary"],
            config,
            page_name,
            "primary",
        )
        candidates.append(primary_candidate)

        if _page_extraction_needs_reread(primary_candidate):
            fallback_candidate = _extract_page_data_from_image(
                image_paths["fallback"],
                config,
                page_name,
                "fallback",
            )
            candidates.append(fallback_candidate)

        page_data = _select_best_page_candidate(page_name, candidates)

    # Set source_file programmatically (LLM can't know the filename)
    page_data["source_file"] = page_name

    # Record any extraction failures for reporting
    _check_and_record_failure(page_data, pdf_stem, page_idx, failed_pages, page_name)

    # Only cache extractions with meaningful results
    has_results = bool(page_data.get("lab_results"))
    confirmed_no_lab_data = page_data.get("page_has_lab_data") is False
    if has_results or confirmed_no_lab_data:
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


def _apply_name_standardization(
    all_results: list,
    lab_specs: LabSpecsConfig,
    pdf_stem: str,
) -> int:
    """Apply name standardization fallback. Returns count of updated results."""

    # Collect raw names that need standardization (not already standardized or marked as unknown)
    names_to_standardize = [result.get("raw_lab_name", "") for result in all_results if not result.get("lab_name_standardized") or result.get("lab_name_standardized") == UNKNOWN_VALUE]

    # Guard: Skip if no names need standardization
    if not names_to_standardize:
        return 0

    # Look up standardization mappings from cache
    name_mappings = standardize_lab_names(
        names_to_standardize,
        lab_specs.standardized_names,
    )

    # Apply mappings to results and count updates
    fallback_count = 0
    for result in all_results:
        # Skip already standardized results
        std_name = result.get("lab_name_standardized")
        if std_name and std_name != UNKNOWN_VALUE:
            continue

        # Apply mapping if available
        raw_name = result.get("raw_lab_name", "")
        mapped = name_mappings.get(raw_name)
        if mapped:
            result["lab_name_standardized"] = mapped
            fallback_count += 1

    # Log summary if any mappings were applied
    if fallback_count > 0:
        logger.info(f"[{pdf_stem}] Name standardization applied to {fallback_count} results")
    return fallback_count


def _apply_unit_standardization(
    all_results: list,
    lab_specs: LabSpecsConfig,
    pdf_stem: str,
) -> int:
    """Apply unit standardization fallback. Returns count of updated results."""

    # Collect unit contexts that need standardization (with standardized names)
    unit_contexts = [
        (
            result.get("raw_lab_unit", ""),
            result.get("lab_name_standardized", ""),
        )
        for result in all_results
        if not result.get("lab_unit_standardized") and result.get("lab_name_standardized") and result.get("lab_name_standardized") != UNKNOWN_VALUE
    ]

    # Guard: Skip if no units need standardization
    if not unit_contexts:
        return 0

    # Look up unit standardization mappings from cache
    unit_mappings = standardize_lab_units(
        unit_contexts,
        lab_specs.standardized_units,
        lab_specs,
    )

    # Apply mappings to results and count updates
    unit_count = 0
    for result in all_results:
        # Skip already standardized units
        if result.get("lab_unit_standardized"):
            continue

        # Skip results without valid standardized names
        std_name = result.get("lab_name_standardized", "")
        if not std_name or std_name == UNKNOWN_VALUE:
            continue

        # Apply mapping if available
        pair = (result.get("raw_lab_unit", ""), std_name)
        mapped = unit_mappings.get(pair)
        if mapped:
            result["lab_unit_standardized"] = mapped
            unit_count += 1

    # Log summary if any mappings were applied
    if unit_count > 0:
        logger.info(f"[{pdf_stem}] Unit standardization applied to {unit_count} results")
    return unit_count


def _apply_standardization(
    all_results: list,
    lab_specs: LabSpecsConfig,
    pdf_stem: str,
) -> None:
    """Apply name and unit standardization via cache-based mapping."""

    # Guard: Skip standardization if lab specs not available
    if not lab_specs.exists:
        return

    # Apply name standardization to unmapped results
    _apply_name_standardization(all_results, lab_specs, pdf_stem)

    # Apply unit standardization to unmapped results
    _apply_unit_standardization(all_results, lab_specs, pdf_stem)


def _save_results_to_csv(
    all_results: list,
    doc_date: str | None,
    csv_path: Path,
    pdf_stem: str,
) -> None:
    """Create DataFrame and save to CSV."""

    # Convert results to DataFrame for structured export
    df = pd.DataFrame(all_results)

    # Add document date to all rows
    df["date"] = doc_date

    # Ensure all required LabResult fields are present (fill missing with None)
    core_cols = list(LabResult.model_fields.keys()) + ["date"]
    ensure_columns(df, core_cols, default=None)

    # Select only core columns that exist in the DataFrame
    df = df[[col for col in core_cols if col in df.columns]]

    # Export to CSV without index
    df.to_csv(csv_path, index=False, encoding="utf-8")


def _handle_empty_results(csv_path: Path, pdf_stem: str) -> tuple[Path, list]:
    """Handle case when no results were extracted."""

    # Log warning for debugging
    logger.warning(f"[{pdf_stem}] No results extracted")

    # Create empty CSV file as placeholder
    pd.DataFrame().to_csv(csv_path, index=False)

    # Return empty result indicators
    return csv_path, []


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
    output_dir: Path,
    config: ExtractionConfig,
    lab_specs: LabSpecsConfig,
) -> tuple[Path | None, list[dict]]:
    """Process a single PDF file: extract, standardize, and save results.

    Returns:
        Tuple of (csv_path, failed_pages) where:
        - csv_path: Path to output CSV, or None if processing failed entirely
        - failed_pages: List of dicts with 'page' and 'reason' for any extraction failures
    """

    # Initialize output directory structure and paths
    pdf_stem = pdf_path.stem
    file_hash = _compute_file_hash(pdf_path)
    doc_out_dir, csv_path, failed_pages = _setup_pdf_processing(pdf_path, output_dir, file_hash)

    try:
        logger.info(f"[{pdf_stem}] Processing...")

        # Copy source PDF to output directory for archival
        copied_pdf = _copy_pdf_to_output(pdf_path, doc_out_dir)

        # Extract lab data using text-first strategy with vision fallback
        all_results, doc_date = _extract_data_from_pdf(
            copied_pdf,
            config,
            doc_out_dir,
            pdf_stem,
            failed_pages,
        )

        # Guard: Check if any results were successfully extracted
        if not all_results:
            return _handle_empty_results(csv_path, pdf_stem)[0], failed_pages

        # Apply standardization via cache-based mapping
        _apply_standardization(all_results, lab_specs, pdf_stem)

        # Export final results to CSV format
        _save_results_to_csv(all_results, doc_date, csv_path, pdf_stem)

        logger.info(f"[{pdf_stem}] Completed successfully")
        return csv_path, failed_pages

    except Exception as e:
        # Catch-all for errors during processing (extraction, standardization, export)
        logger.error(f"[{pdf_stem}] Processing failed: {e}", exc_info=True)
        return None, failed_pages


# ========================================
# Data Merging & Export
# ========================================


def merge_csv_files(csv_paths: list[Path]) -> pd.DataFrame:
    """Merge multiple CSV files into a single DataFrame."""

    # Initialize collection for valid dataframes
    dataframes = []

    # Read each CSV and append to collection
    for csv_path in csv_paths:
        # Skip empty or non-existent files
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            continue

        # Read CSV and add source metadata
        df = pd.read_csv(csv_path, encoding="utf-8")
        df["source_file"] = csv_path.name
        dataframes.append(df)

    # Return empty DataFrame if no valid files found
    if not dataframes:
        return pd.DataFrame()

    # Concatenate all dataframes into single result
    return pd.concat(dataframes, ignore_index=True)


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


def _get_csv_path(pdf_path: Path, output_path: Path) -> Path:
    """Get the output CSV path for a given PDF file."""

    # Build path: output/{stem}_{hash}/{stem}.csv
    file_hash = _compute_file_hash(pdf_path)
    new_path = output_path / f"{pdf_path.stem}_{file_hash}" / f"{pdf_path.stem}.csv"

    # Check new path first, then fall back to legacy path for pre-migration output
    if new_path.exists():
        return new_path

    legacy_path = output_path / pdf_path.stem / f"{pdf_path.stem}.csv"
    if legacy_path.exists():
        return legacy_path

    # Neither exists — return new-style path for creation
    return new_path


REQUIRED_CSV_COLS = ["result_index", "page_number", "source_file"]


@dataclass
class PipelineRunResult:
    """Final dataframe plus pipeline metadata for one corpus run."""

    final_df: pd.DataFrame
    csv_paths: list[Path]
    failed_pages: list[dict]
    pdfs_failed: int


def _deduplicate_pdf_files(pdf_files: list[Path]) -> tuple[list[Path], list[tuple[Path, Path]]]:
    """Deduplicate PDF files by content hash.

    Returns:
        (unique_files, duplicates) where duplicates is a list of (duplicate_path, original_path) tuples
    """

    seen_hashes: dict[str, Path] = {}
    unique_files: list[Path] = []
    duplicates: list[tuple[Path, Path]] = []

    for pdf_path in pdf_files:
        file_hash = _compute_file_hash(pdf_path)
        if file_hash in seen_hashes:
            duplicates.append((pdf_path, seen_hashes[file_hash]))
        else:
            seen_hashes[file_hash] = pdf_path
            unique_files.append(pdf_path)

    return unique_files, duplicates


def _filter_pdfs_to_process(pdf_files: list[Path], output_path: Path) -> tuple[list[Path], int]:
    """Filter out PDFs that already have valid CSV outputs."""

    # Initialize collections for tracking
    pdfs_to_process = []
    skipped_count = 0

    # Check each PDF for existing valid CSV
    for pdf_path in pdf_files:
        csv_path = _get_csv_path(pdf_path, output_path)

        if _is_csv_valid(csv_path):
            # Skip PDFs with valid existing outputs (cache hit)
            skipped_count += 1
            continue

        # Log warning if CSV exists but is invalid (missing columns, etc.)
        if csv_path.exists():
            logger.warning(f"Re-processing {pdf_path.name}: CSV missing required columns")

        # Add to processing queue
        pdfs_to_process.append(pdf_path)

    return pdfs_to_process, skipped_count


def _is_csv_valid(csv_path: Path, required_cols: list[str] = REQUIRED_CSV_COLS) -> bool:
    """Check if CSV exists and has all required columns."""

    # Guard: File must exist
    if not csv_path.exists():
        return False

    try:
        # Read only header row to check column names
        df = pd.read_csv(csv_path, nrows=0)

        # Verify all required columns are present
        return all(col in df.columns for col in required_cols)
    except Exception:
        # Any error (corrupt file, permissions, etc.) means invalid
        return False


def _filter_unknown_labs(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows that couldn't be mapped to known lab tests."""

    # Identify rows with unknown lab names
    unknown_mask = merged_df["lab_name_standardized"] == UNKNOWN_VALUE

    # Log and remove unknown lab rows
    if unknown_mask.any():
        unknown_count = unknown_mask.sum()
        logger.error(f"Filtering {unknown_count} rows with unknown lab names")
        merged_df = merged_df[~unknown_mask].reset_index(drop=True)

    return merged_df


def _rename_columns_for_export(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns from internal names to simplified export schema."""

    # Define mapping from internal to export column names
    column_renames = {
        "lab_name_standardized": "lab_name",
        "value_primary": "value",
        "lab_unit_primary": "lab_unit",
        "raw_lab_unit": "raw_unit",
        "reference_min_primary": "reference_min",
        "reference_max_primary": "reference_max",
    }

    return merged_df.rename(columns=column_renames)


def _run_value_validation(merged_df: pd.DataFrame, lab_specs: LabSpecsConfig) -> tuple[pd.DataFrame, dict]:
    """Run value-based validation and return validated DataFrame with stats."""

    # Initialize validator and run validation checks
    validator = ValueValidator(lab_specs)
    merged_df = validator.validate(merged_df)

    return merged_df, validator.validation_stats


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
    pdfs_to_process: list[Path],
    pdf_files: list[Path],
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
    tasks = [(pdf, config.output_path, config, lab_specs) for pdf in pdfs_to_process]

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

    # Unpack results and collect statistics
    pdfs_failed = sum(1 for csv_path, _ in results if csv_path is None)
    all_failed_pages = []
    for _, failed_pages in results:
        all_failed_pages.extend(failed_pages)

    # Build list of valid CSV paths from all PDFs (processed + skipped)
    csv_paths = [p for pdf in pdf_files if _is_csv_valid(p := _get_csv_path(pdf, config.output_path))]

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
  parselabs --profile tsilva

  # List available profiles:
  parselabs --list-profiles

  # Override settings:
  parselabs --profile tsilva --model google/gemini-2.5-pro

  # Use alternate environment (loads ~/.config/parselabs/.env and .env, then ~/.config/parselabs/.env.local and .env.local):
  parselabs --profile tsilva --env local
        """,
    )

    # Profile-based
    parser.add_argument(
        "--profile",
        "-p",
        type=str,
        help="Profile name (without extension). If not specified, runs all profiles.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available profiles and exit",
    )

    # Overrides
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model ID for extraction (overrides EXTRACT_MODEL_ID from .env)",
    )
    parser.add_argument("--workers", "-w", type=int, help="Number of parallel workers")
    parser.add_argument(
        "--pattern",
        type=str,
        help="Glob pattern for input files (overrides profile, default: *.pdf)",
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Environment name to load (loads .env plus .env.{name}, preferring ~/.config/parselabs first)",
    )

    return parser.parse_args()


def _validate_profile_and_env(args) -> ProfileConfig:
    """Validate profile and environment configuration.

    Returns validated ProfileConfig.
    Raises ConfigurationError if validation fails.
    """
    errors = []

    # Find and validate profile exists
    profile_path = ProfileConfig.find_path(args.profile)
    if not profile_path:
        raise ConfigurationError(f"Profile '{args.profile}' not found. Use --list-profiles to see available profiles.")

    # Load profile configuration
    profile = ProfileConfig.from_file(profile_path)

    # Validate required paths are defined in profile
    if not profile.input_path:
        errors.append(f"Profile '{args.profile}' has no input_path defined.")
    if not profile.output_path:
        errors.append(f"Profile '{args.profile}' has no output_path defined.")

    # Validate required environment variables
    if not os.getenv("OPENROUTER_API_KEY"):
        errors.append("OPENROUTER_API_KEY environment variable not set.")
    if not os.getenv("EXTRACT_MODEL_ID"):
        errors.append("EXTRACT_MODEL_ID environment variable not set.")

    if errors:
        raise ConfigurationError("\n".join(errors))
    return profile


def _build_config_from_profile(profile: ProfileConfig, args) -> ExtractionConfig:
    """Build ExtractionConfig from validated profile and CLI args."""

    # Load required environment variables
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    extract_model_id = os.getenv("EXTRACT_MODEL_ID")

    # Build base configuration from profile and environment
    config = ExtractionConfig(
        input_path=profile.input_path,
        output_path=profile.output_path,
        openrouter_api_key=openrouter_api_key,
        extract_model_id=extract_model_id,
        input_file_regex=profile.input_file_regex or "*.pdf",
        max_workers=profile.workers or int(os.getenv("MAX_WORKERS", "0")) or (os.cpu_count() or 1),
    )

    # Apply CLI argument overrides (highest priority)
    if args.model:
        config.extract_model_id = args.model
    if args.workers:
        config.max_workers = args.workers
    if args.pattern:
        config.input_file_regex = args.pattern

    return config


def build_config(args) -> ExtractionConfig:
    """Build ExtractionConfig from args and env.

    Returns validated ExtractionConfig.
    Raises ConfigurationError if validation fails.
    """

    # Validate profile and environment (raises ConfigurationError on failure)
    profile = _validate_profile_and_env(args)

    # Build configuration from validated profile
    config = _build_config_from_profile(profile, args)

    # Validate input path exists (runtime check)
    if not config.input_path.exists():
        raise ConfigurationError(f"Input path does not exist: {config.input_path}")

    # Ensure output directory exists
    config.output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using profile: {profile.name}")
    return config


# ========================================
# Main Pipeline
# ========================================


def _classify_server_error(error_msg: str, timeout: int) -> tuple[bool, str]:
    """Classify a server connectivity error into a diagnostic result.

    Returns tuple of (is_available, diagnostic_message).
    """

    # Authentication errors (invalid or missing API key)
    if "401" in error_msg or "Unauthorized" in error_msg:
        return False, f"Authentication failed - check your OPENROUTER_API_KEY: {error_msg}"

    # Timeout errors (server slow or unreachable)
    if "timeout" in error_msg.lower():
        return False, f"Server timeout after {timeout}s - server may be unreachable"

    # 404 errors - endpoint not implemented (common with local servers), server is still available
    if "404" in error_msg:
        return True, "Server is available (models endpoint not implemented)"

    # Connection failures (network issues, DNS problems, server down)
    if "Connection" in error_msg or "refused" in error_msg.lower() or "reset" in error_msg.lower() or "Name or service not known" in error_msg or "getaddrinfo" in error_msg:
        return False, f"Cannot connect to server: {error_msg}"

    # Unknown errors - assume unavailable to be safe
    return False, f"Server check failed: {error_msg}"


def check_server_availability(client: OpenAI, model_id: str, timeout: int = 10) -> tuple[bool, str]:
    """Check if OpenRouter API server is available and responsive.

    Args:
        client: OpenAI client configured for OpenRouter
        model_id: Model ID to check availability for
        timeout: Timeout in seconds for the check

    Returns:
        Tuple of (is_available, message)
    """
    try:
        # Check that the server is reachable with a lightweight request
        client.models.list(timeout=timeout)
        return True, "Server is available"
    except Exception as e:
        # Classify the error to provide helpful diagnostic messages
        return _classify_server_error(str(e), timeout)


def _build_and_validate_config(args, profile_name: str) -> ExtractionConfig:
    """Build and validate configuration for a profile.

    Returns validated ExtractionConfig.
    Raises ConfigurationError if validation fails.
    """

    # Temporarily override args.profile for config building
    original_profile = args.profile
    args.profile = profile_name

    try:
        # Build and validate configuration (raises ConfigurationError on failure)
        return build_config(args)
    finally:
        # Restore original profile name regardless of outcome
        args.profile = original_profile


def _process_pdfs_or_use_cache(
    pdf_files: list[Path],
    pdfs_to_process: list[Path],
    config: ExtractionConfig,
    lab_specs: LabSpecsConfig,
    log_dir: Path,
) -> tuple[list[Path], list[dict], int]:
    """Process PDFs or use cached results if all already processed.

    Returns tuple of (csv_paths, all_failed_pages, pdfs_failed).
    """

    # All PDFs already have valid CSVs - just collect paths
    if not pdfs_to_process:
        logger.info("All PDFs already processed. Moving to merge step...")
        csv_paths = [p for pdf in pdf_files if _is_csv_valid(p := _get_csv_path(pdf, config.output_path))]
        return csv_paths, [], 0

    # Process remaining PDFs using parallel worker pool
    return _process_pdfs_in_parallel(
        pdfs_to_process,
        pdf_files,
        config,
        lab_specs,
        log_dir,
    )


def _setup_profile_environment(args, profile_name: str) -> tuple[ExtractionConfig, LabSpecsConfig]:
    """Setup environment for a profile: config, logging, lab specs.

    Returns tuple of (config, lab_specs).
    Raises ConfigurationError if setup fails.
    """

    # Build and validate configuration (raises ConfigurationError on failure)
    config = _build_and_validate_config(args, profile_name)

    # Setup logging to output folder for later review
    global logger
    log_dir = config.output_path / "logs"
    logger = setup_logging(log_dir, clear_logs=True)

    # Log configuration for debugging/auditing
    logger.info(f"Input: {config.input_path}")
    logger.info(f"Output: {config.output_path}")
    logger.info(f"Model: {config.extract_model_id}")

    # Load lab specifications for normalization and validation
    lab_specs = LabSpecsConfig()

    # Copy lab specs to output folder for reproducibility (if available)
    if lab_specs.exists:
        lab_specs_dest = config.output_path / "lab_specs.json"
        shutil.copy2(lab_specs.config_path, lab_specs_dest)
        logger.info(f"Copied lab specs to output: {lab_specs_dest}")

    return config, lab_specs


def _process_and_transform_data(
    merged_df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
    export_cols: list,
) -> pd.DataFrame | None:
    """Apply all transformations to merged data: normalize, filter, dedupe, validate.

    Returns transformed DataFrame or None if processing fails.
    """

    # Apply value normalizations and unit conversions
    logger.info("Applying normalizations...")
    merged_df = apply_normalizations(merged_df, lab_specs)

    # Filter out rows that couldn't be mapped to known lab tests
    merged_df = _filter_unknown_labs(merged_df)

    # Flag duplicate (date, lab_name) entries before deduplication so reviewers can verify
    merged_df = flag_duplicate_entries(merged_df)

    # Remove duplicate results from same date/lab combinations (requires lab specs for priority rules)
    if lab_specs.exists:
        logger.info("Deduplicating results...")
        merged_df = deduplicate_results(merged_df, lab_specs)
        logger.info(f"After deduplication: {len(merged_df)} rows")

    # Rename columns from internal names to simplified export schema
    merged_df = _rename_columns_for_export(merged_df)

    # Run value-based validation to flag suspicious values
    logger.info("Running value-based validation...")
    merged_df, validation_stats = _run_value_validation(merged_df, lab_specs)
    _log_validation_stats(validation_stats)

    return merged_df


def _export_final_results(
    final_df: pd.DataFrame,
    hidden_cols: list,
    widths: dict,
    output_path: Path,
    all_failed_pages: list[dict],
    csv_paths: list[Path],
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

    # Report extraction results and failures
    _report_extraction_failures(all_failed_pages, csv_path, csv_paths)


def _prepare_final_export_dataframe(
    merged_df: pd.DataFrame,
    export_cols: list[str],
    dtypes: dict[str, str],
) -> pd.DataFrame:
    """Finalize transformed rows into the export schema used by all.csv."""

    final_cols = [col for col in export_cols if col in merged_df.columns]
    final_df = merged_df[final_cols].copy()

    logger.info("Applying data type conversions...")
    return apply_dtype_conversions(final_df, dtypes)


def run_pipeline_for_pdf_files(
    pdf_files: list[Path],
    config: ExtractionConfig,
    lab_specs: LabSpecsConfig,
) -> PipelineRunResult:
    """Process explicit PDF files and return the final export DataFrame plus metadata."""

    export_cols, _, _, dtypes = get_column_lists(COLUMN_SCHEMA)

    pdf_files = sorted(pdf_files)
    logger.info(f"Found {len(pdf_files)} PDF(s) for explicit pipeline run")

    if not pdf_files:
        raise PipelineError("No PDF files provided for processing.")

    pdf_files, duplicate_pdfs = _deduplicate_pdf_files(pdf_files)
    for dup_path, orig_path in duplicate_pdfs:
        logger.warning(f"Skipping duplicate PDF: {dup_path.name} (same content as {orig_path.name})")
    if duplicate_pdfs:
        logger.info(f"Skipped {len(duplicate_pdfs)} duplicate PDF(s), {len(pdf_files)} unique PDF(s) remaining")

    pdfs_to_process, skipped_count = _filter_pdfs_to_process(pdf_files, config.output_path)
    logger.info(f"Skipping {skipped_count} already-processed PDF(s)")
    logger.info(f"Processing {len(pdfs_to_process)} PDF(s)")

    log_dir = config.output_path / "logs"
    csv_paths, all_failed_pages, pdfs_failed = _process_pdfs_or_use_cache(
        pdf_files,
        pdfs_to_process,
        config,
        lab_specs,
        log_dir,
    )

    if not csv_paths:
        raise PipelineError("No PDFs successfully processed.")

    logger.info(f"Successfully processed {len(csv_paths)} PDFs")
    logger.info("Merging CSV files...")
    merged_df = merge_csv_files(csv_paths)
    rows_after_merge = len(merged_df)
    logger.info(f"Merged data: {rows_after_merge} rows")

    if merged_df.empty:
        raise PipelineError("No data after merging CSV files.")

    merged_df = _process_and_transform_data(
        merged_df,
        lab_specs,
        export_cols,
    )

    if merged_df is None:
        raise PipelineError("Data processing failed.")

    final_df = _prepare_final_export_dataframe(merged_df, export_cols, dtypes)
    return PipelineRunResult(
        final_df=final_df,
        csv_paths=csv_paths,
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


def run_for_profile(args, profile_name: str) -> None:
    """Run extraction pipeline for a single profile.

    Raises ConfigurationError for config/setup failures.
    Raises PipelineError for runtime pipeline failures.
    """

    # Setup environment: config, logging, lab specs (raises ConfigurationError on failure)
    config, lab_specs = _setup_profile_environment(args, profile_name)

    # Get column configuration for export formatting
    _, hidden_cols, widths, _ = get_column_lists(COLUMN_SCHEMA)

    # Discover PDF files matching the input pattern
    pdf_files = sorted(config.input_path.glob(config.input_file_regex))
    logger.info(f"Found {len(pdf_files)} PDF(s) matching '{config.input_file_regex}'")

    if not pdf_files:
        raise PipelineError(f"No PDF files found matching '{config.input_file_regex}' in {config.input_path}")

    pipeline_result = run_pipeline_for_pdf_files(pdf_files, config, lab_specs)

    # Export final results to CSV and Excel
    _export_final_results(
        pipeline_result.final_df,
        hidden_cols,
        widths,
        config.output_path,
        pipeline_result.failed_pages,
        pipeline_result.csv_paths,
    )


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

    # Check server availability once before processing any profiles
    logger.info("Checking OpenRouter server availability...")
    is_available, message = check_server_availability(client, os.getenv("EXTRACT_MODEL_ID", ""))

    # Guard: Abort if API server is unreachable
    if not is_available:
        logger.error(f"Cannot start extraction - {message}")
        logger.error("Please check your internet connection and API key, then try again.")
        sys.exit(1)
    logger.info(f"Server check passed: {message}")

    # Initialize results tracking for each profile
    results = {}

    # Run extraction pipeline for each profile
    for profile_name in profiles_to_run:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing profile: {profile_name}")
        logger.info(f"{'=' * 60}")
        try:
            run_for_profile(args, profile_name)
            results[profile_name] = "success"
        except (ConfigurationError, PipelineError) as e:
            logger.error(f"\nError in profile '{profile_name}':\n{e}")
            results[profile_name] = "failed"
        # Catch-all for unexpected errors during profile processing
        except Exception as e:
            logger.error(f"\nUnexpected error in profile '{profile_name}': {e}")
            results[profile_name] = f"error: {e}"

    # Print summary if multiple profiles were processed
    if len(profiles_to_run) > 1:
        logger.info(f"\n{'=' * 60}")
        logger.info("Summary:")
        logger.info(f"{'=' * 60}")
        for profile_name, status in results.items():
            logger.info(f"  {profile_name}: {status}")


if __name__ == "__main__":
    main()
