"""Main entry point for lab results extraction and processing."""

from labs_parser.utils import load_dotenv_with_env

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
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import pdf2image  # noqa: E402
from openai import OpenAI  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Local imports
from labs_parser.config import (  # noqa: E402
    UNKNOWN_VALUE,
    ExtractionConfig,
    LabSpecsConfig,
    ProfileConfig,
)
from labs_parser.exceptions import ConfigurationError, PipelineError  # noqa: E402
from labs_parser.extraction import (  # noqa: E402
    LabResult,
    _build_standardized_names_section,
    extract_labs_from_page_image,
    extract_labs_from_text,
)
from labs_parser.normalization import (  # noqa: E402
    apply_dtype_conversions,
    apply_normalizations,
    deduplicate_results,
    flag_duplicate_entries,
)
from labs_parser.standardization import (  # noqa: E402
    standardize_lab_names,
    standardize_lab_units,
)
from labs_parser.utils import (  # noqa: E402
    ensure_columns,
    preprocess_page_image,
    setup_logging,
)
from labs_parser.validation import ValueValidator  # noqa: E402

# Module-level logger (file handlers added after config is loaded)
logger = logging.getLogger(__name__)

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


# ========================================
# Column Schema (simplified - 13 columns)
# ========================================

COLUMN_SCHEMA = {
    # Core identification
    "date": {"dtype": "datetime64[ns]", "excel_width": 13},
    "source_file": {"dtype": "str", "excel_width": 25},
    "page_number": {"dtype": "Int64", "excel_width": 8},
    # Extracted values (standardized)
    "lab_name": {"dtype": "str", "excel_width": 35},
    "value": {"dtype": "float64", "excel_width": 12},
    "unit": {"dtype": "str", "excel_width": 15},
    # Reference ranges from PDF
    "reference_min": {"dtype": "float64", "excel_width": 12},
    "reference_max": {"dtype": "float64", "excel_width": 12},
    # Raw values (for audit)
    "lab_name_raw": {"dtype": "str", "excel_width": 35},
    "value_raw": {"dtype": "str", "excel_width": 12},
    "unit_raw": {"dtype": "str", "excel_width": 15},
    # Review flags (from validation)
    "review_needed": {"dtype": "boolean", "excel_width": 12},
    "review_reason": {"dtype": "str", "excel_width": 30},
    # Limit indicators (for values like <0.05 or >738)
    "is_below_limit": {"dtype": "boolean", "excel_width": 12},
    "is_above_limit": {"dtype": "boolean", "excel_width": 12},
    # Internal (hidden in Excel)
    "lab_type": {"dtype": "str", "excel_width": 10, "excel_hidden": True},
    "result_index": {
        "dtype": "Int64",
        "excel_width": 10,
        "excel_hidden": True,
    },
}

# Canonical column order for CSV export
COLUMN_ORDER = [
    "date",
    "source_file",
    "page_number",
    "lab_name",
    "value",
    "unit",
    "reference_min",
    "reference_max",
    "lab_name_raw",
    "value_raw",
    "unit_raw",
    "review_needed",
    "review_reason",
    "is_below_limit",
    "is_above_limit",
    "lab_type",
    "result_index",
]


def get_column_lists(schema: dict):
    """Extract ordered lists from schema."""

    # Filter to only columns present in the schema
    export_cols = [k for k in COLUMN_ORDER if k in schema]

    # Identify columns that should be hidden in Excel output
    hidden_cols = [col for col, props in schema.items() if props.get("excel_hidden")]

    # Build column width mapping for Excel formatting
    widths = {col: props["excel_width"] for col, props in schema.items() if "excel_width" in props}

    # Build data type mapping for column conversion
    dtypes = {col: props["dtype"] for col, props in schema.items() if "dtype" in props}

    return export_cols, hidden_cols, widths, dtypes


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


def _text_has_enough_content(text: str, min_chars: int = _MIN_TEXT_CHARS) -> bool:
    """Check if extracted text has enough content to attempt LLM extraction."""
    clean_text = text.replace(" ", "").replace("\n", "").replace("\t", "")
    return len(clean_text) >= min_chars


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


def _try_load_cached_text_extraction(doc_out_dir: Path, pdf_stem: str) -> dict | None:
    """Try to load cached text extraction results."""

    # Build path to cached JSON results
    text_json_path = doc_out_dir / f"{pdf_stem}.json"

    # Return None if no cache exists (caller will perform fresh extraction)
    if not text_json_path.exists():
        return None

    # Parse and return cached data - let exceptions propagate
    return json.loads(text_json_path.read_text(encoding="utf-8"))


def _extract_labs_from_pdf_text(
    pdf_text: str,
    config: ExtractionConfig,
    standardization_section: str | None,
) -> dict:
    """Extract lab results from PDF text using LLM."""

    # Delegate to extraction module with configured model and client
    return extract_labs_from_text(
        pdf_text,
        config.extract_model_id,
        client,
        standardization_section=standardization_section,
    )


def _cache_text_extraction(
    text_extraction_data: dict,
    pdf_text: str,
    doc_out_dir: Path,
    pdf_stem: str,
) -> None:
    """Cache text extraction results and raw PDF text."""

    # Build paths for both structured results and raw text
    text_json_path = doc_out_dir / f"{pdf_stem}.json"  # Path for structured extraction results
    text_txt_path = doc_out_dir / f"{pdf_stem}.txt"  # Path for raw text content

    # Serialize and save structured extraction data
    text_json_path.write_text(
        json.dumps(text_extraction_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save raw PDF text for debugging and audit purposes
    text_txt_path.write_text(pdf_text, encoding="utf-8")


def _try_text_extraction(
    copied_pdf: Path,
    config: ExtractionConfig,
    standardization_section: str | None,
    doc_out_dir: Path,
    pdf_stem: str,
) -> tuple[bool, dict | None]:
    """Attempt text-first extraction with fallback to vision.

    Strategy: Text extraction is cheaper than vision, so we try it first.
    Falls back to vision if PDF has no text, insufficient content, or extraction fails.

    Returns (used_text_extraction, extraction_data).
    """

    # Check for cached text extraction results from previous runs
    cached_data = _try_load_cached_text_extraction(doc_out_dir, pdf_stem)
    if cached_data:
        logger.info(f"[{pdf_stem}] Strategy: TEXT (cached)")
        return True, cached_data

    # Extract raw text from PDF using pdftotext (fast, no AI cost)
    pdf_text, pdftotext_success = extract_text_from_pdf(copied_pdf)

    # Guard: Fall back to vision if PDF has no extractable text layer
    if not pdftotext_success:
        logger.info(f"[{pdf_stem}] Strategy: VISION (no embedded text in PDF)")
        return False, None

    # Guard: Fall back to vision if text content is too sparse for reliable extraction
    if not _text_has_enough_content(pdf_text):
        logger.info(f"[{pdf_stem}] Strategy: VISION (insufficient text content)")
        return False, None

    logger.info(f"[{pdf_stem}] Strategy: TEXT (sufficient content, {len(pdf_text)} chars)")

    # Attempt LLM-based extraction from extracted text
    try:
        text_extraction_data = _extract_labs_from_pdf_text(pdf_text, config, standardization_section)
    except Exception as e:
        # Text extraction failed - log and fall back to vision
        logger.warning(f"[{pdf_stem}] Text extraction failed: {e}")
        return False, None

    # Guard: Fall back to vision if extraction succeeded but returned no lab results
    if not text_extraction_data or not text_extraction_data.get("lab_results"):
        logger.warning(f"[{pdf_stem}] Strategy: TEXT -> VISION (no results from text)")
        return False, None

    # Cache successful text extraction for future runs
    _cache_text_extraction(text_extraction_data, pdf_text, doc_out_dir, pdf_stem)
    logger.info(f"[{pdf_stem}] Text extraction complete: {len(text_extraction_data['lab_results'])} results")
    return True, text_extraction_data


def _process_text_results(text_extraction_data: dict, pdf_stem: str) -> tuple[list, str | None]:
    """Process text extraction results into standardized format."""

    # Initialize collection for all extracted results
    all_results = []

    # Extract document date from extraction data or filename
    doc_date = _extract_document_date(text_extraction_data, pdf_stem)

    # Iterate through each lab result and add metadata
    for result_idx, result in enumerate(text_extraction_data.get("lab_results", [])):
        result["result_index"] = result_idx  # Track position within document
        result["page_number"] = 1  # Text extraction treats entire PDF as single page
        result["source_file"] = f"{pdf_stem}.text"  # Mark as text-based extraction
        all_results.append(result)

    return all_results, doc_date


def _convert_pdf_to_images(copied_pdf: Path, pdf_stem: str) -> list:
    """Convert PDF to PIL images."""

    # Use pdf2image to convert PDF pages to PIL Image objects
    return pdf2image.convert_from_path(str(copied_pdf))


def _preprocess_and_save_image(page_image, page_name: str, jpg_path: Path) -> None:
    """Preprocess page image and save if not cached."""

    # Skip processing if image already exists (cache hit)
    if jpg_path.exists():
        return

    # Apply preprocessing (grayscale, contrast enhancement, etc.)
    processed = preprocess_page_image(page_image)

    # Save preprocessed image with high quality JPEG
    processed.save(jpg_path, "JPEG", quality=95)
    logger.info(f"[{page_name}] Image preprocessed and saved")


def _extract_or_load_page_data(
    jpg_path: Path,
    json_path: Path,
    page_name: str,
    config: ExtractionConfig,
    standardization_section: str | None,
    pdf_stem: str,
    page_idx: int,
    failed_pages: list,
) -> dict:
    """Extract data from image or load from cache."""

    # Check if extraction results already cached for this page
    if json_path.exists():
        logger.info(f"[{page_name}] Loading cached extraction data")
        page_data = json.loads(json_path.read_text(encoding="utf-8"))
        _check_and_record_failure(page_data, pdf_stem, page_idx, failed_pages)
        return page_data

    # Perform fresh extraction using vision model
    logger.info(f"[{page_name}] Extracting data from image...")
    page_data = extract_labs_from_page_image(
        jpg_path,
        config.extract_model_id,
        client,
        standardization_section=standardization_section,
    )
    logger.info(f"[{page_name}] Extraction completed")

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
    standardization_section: str | None,
    failed_pages: list,
) -> tuple[list, dict | None]:
    """Process a single PDF page: preprocess, extract, and return results with metadata.

    Returns tuple of (page_results, page_data) where page_data is the full extraction data.
    """

    # Generate unique page identifier with zero-padding
    page_name = f"{pdf_stem}.{page_idx + 1:03d}"
    jpg_path = doc_out_dir / f"{page_name}.jpg"
    json_path = doc_out_dir / f"{page_name}.json"

    logger.info(f"[{page_name}] Processing page {page_idx + 1}/{total_pages}...")

    # Preprocess and cache page image
    _preprocess_and_save_image(page_image, page_name, jpg_path)

    # Extract data using vision model or load from cache
    page_data = _extract_or_load_page_data(
        jpg_path,
        json_path,
        page_name,
        config,
        standardization_section,
        pdf_stem,
        page_idx,
        failed_pages,
    )

    # Add page metadata to results
    page_results = _add_page_metadata(page_data.get("lab_results", []), page_idx, page_name)

    return page_results, page_data


def _extract_via_vision(
    copied_pdf: Path,
    config: ExtractionConfig,
    standardization_section: str | None,
    doc_out_dir: Path,
    pdf_stem: str,
    failed_pages: list,
) -> tuple[list, str | None]:
    """Extract lab results using vision-based processing."""

    # Convert PDF pages to PIL images
    pil_pages = _convert_pdf_to_images(copied_pdf, pdf_stem)

    # Guard: Handle PDF conversion failure
    if pil_pages is None:
        return [], None

    logger.info(f"[{pdf_stem}] Processing {len(pil_pages)} page(s) with vision...")

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
            standardization_section,
            failed_pages,
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
    names_to_standardize = [result.get("lab_name_raw", "") for result in all_results if not result.get("lab_name_standardized") or result.get("lab_name_standardized") == UNKNOWN_VALUE]

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
        raw_name = result.get("lab_name_raw", "")
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
            result.get("lab_unit_raw", ""),
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
        pair = (result.get("lab_unit_raw", ""), std_name)
        mapped = unit_mappings.get(pair)
        if mapped:
            result["lab_unit_standardized"] = mapped
            unit_count += 1

    # Log summary if any mappings were applied
    if unit_count > 0:
        logger.info(f"[{pdf_stem}] Unit standardization applied to {unit_count} results")
    return unit_count


def _apply_standardization_fallbacks(
    all_results: list,
    lab_specs: LabSpecsConfig,
    pdf_stem: str,
) -> None:
    """Apply name and unit standardization fallbacks."""

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
    standardization_section: str | None,
    doc_out_dir: Path,
    pdf_stem: str,
    failed_pages: list,
) -> tuple[list, str | None]:
    """Extract lab data from PDF using text-first strategy with vision fallback.

    Returns tuple of (all_results, doc_date) or raises exception on failure.
    """

    # Attempt text-first extraction (cheaper), fall back to vision if needed
    used_text, text_data = _try_text_extraction(copied_pdf, config, standardization_section, doc_out_dir, pdf_stem)

    if used_text:
        # Guard: Ensure text_data is valid before processing
        if not text_data:
            raise ValueError("Text extraction indicated success but returned no data")

        # Process text extraction results
        return _process_text_results(text_data, pdf_stem)

    # Fall back to vision-based extraction
    return _extract_via_vision(
        copied_pdf,
        config,
        standardization_section,
        doc_out_dir,
        pdf_stem,
        failed_pages,
    )


def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    config: ExtractionConfig,
    lab_specs: LabSpecsConfig,
    standardization_section: str | None = None,
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
            standardization_section,
            doc_out_dir,
            pdf_stem,
            failed_pages,
        )

        # Guard: Check if any results were successfully extracted
        if not all_results:
            return _handle_empty_results(csv_path, pdf_stem)[0], failed_pages

        # Apply standardization fallbacks for any missing standardized names/units
        _apply_standardization_fallbacks(all_results, lab_specs, pdf_stem)

        # Persist standardized values back to per-page JSON files for future runs
        _update_json_with_standardized_values(all_results, doc_out_dir)

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


def _find_text_extraction_json(doc_out_dir: Path, page_results: list[dict]) -> list[Path]:
    """Find JSON file for text extraction results."""

    # Check if any results are from text extraction (marked with .text suffix)
    is_text_extraction = any(str(r.get("source_file", "")).endswith(".text") for r in page_results)

    # Return empty list if not text extraction
    if not is_text_extraction:
        return []

    # Check if the document-level JSON exists for text extraction
    candidate = doc_out_dir / f"{doc_out_dir.name}.json"
    if candidate.exists():
        return [candidate]

    return []


def _apply_standardized_values_to_json(json_path: Path, page_results: list[dict]) -> None:
    """Update a single JSON file with standardized lab names and units from page results."""

    # Load JSON data - let exceptions propagate to orchestrator
    data = json.loads(json_path.read_text(encoding="utf-8"))
    lab_results = data.get("lab_results", [])

    # Update each result by result_index with standardized values
    for result in page_results:
        idx = result.get("result_index")

        # Update only if index is valid and within bounds
        if idx is not None and 0 <= idx < len(lab_results):
            lab_results[idx]["lab_name_standardized"] = result.get("lab_name_standardized")
            lab_results[idx]["lab_unit_standardized"] = result.get("lab_unit_standardized")

    # Write updated data back to JSON file
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _update_json_with_standardized_values(all_results: list[dict], doc_out_dir: Path) -> None:
    """Update JSON files with standardized lab names and units."""

    # Group results by page number to minimize file I/O
    results_by_page: dict[int, list[dict]] = {}
    for result in all_results:
        page_num = result.get("page_number")

        # Skip results without page number (can't locate JSON file)
        if page_num is not None:
            results_by_page.setdefault(page_num, []).append(result)

    # Update each JSON file with standardized values
    for page_num, page_results in results_by_page.items():
        # Find the JSON file for this page using glob pattern
        json_files = list(doc_out_dir.glob(f"*.{page_num:03d}.json"))

        # Handle text extraction results which use {pdf_stem}.json naming
        if not json_files:
            json_files = _find_text_extraction_json(doc_out_dir, page_results)

        # Skip if no JSON file found
        if not json_files:
            continue

        # Apply standardized values to this page's JSON file
        _apply_standardized_values_to_json(json_files[0], page_results)


REQUIRED_CSV_COLS = ["result_index", "page_number", "source_file"]


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
        "lab_unit_primary": "unit",
        "lab_unit_raw": "unit_raw",
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
    standardization_section: str | None,
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
    tasks = [(pdf, config.output_path, config, lab_specs, standardization_section) for pdf in pdfs_to_process]

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
  python extract.py

  # Run specific profile:
  python extract.py --profile tsilva

  # List available profiles:
  python extract.py --list-profiles

  # Override settings:
  python extract.py --profile tsilva --model google/gemini-2.5-pro

  # Use alternate environment (loads .env.local after .env):
  python extract.py --profile tsilva --env local
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
        help="Environment name to load (loads .env.{name} instead of .env)",
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
    standardization_section: str | None,
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
        standardization_section,
        log_dir,
    )


def _setup_profile_environment(args, profile_name: str) -> tuple[ExtractionConfig, LabSpecsConfig, str | None]:
    """Setup environment for a profile: config, logging, lab specs.

    Returns tuple of (config, lab_specs, standardization_section).
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

    # Build standardization section for inline standardization during extraction
    standardization_section: str | None = None
    if lab_specs.exists:
        standardization_section = _build_standardized_names_section(lab_specs.standardized_names, lab_specs._specs)
        logger.info(f"Built standardization section: {len(lab_specs.standardized_names)} lab names")

    return config, lab_specs, standardization_section


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
    merged_df: pd.DataFrame,
    export_cols: list,
    hidden_cols: list,
    widths: dict,
    dtypes: dict,
    output_path: Path,
    all_failed_pages: list[dict],
    csv_paths: list[Path],
) -> None:
    """Export final results to CSV and Excel formats."""

    # Select only the columns needed for final export
    final_cols = [col for col in export_cols if col in merged_df.columns]
    merged_df = merged_df[final_cols].copy()

    # Convert columns to their proper data types
    logger.info("Applying data type conversions...")
    merged_df = apply_dtype_conversions(merged_df, dtypes)

    # Export merged results to CSV format
    logger.info("Saving merged CSV...")
    csv_path = output_path / "all.csv"
    merged_df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved merged CSV: {csv_path}")

    # Export merged results to Excel format with formatting
    logger.info("Exporting to Excel...")
    excel_path = output_path / "all.xlsx"
    export_excel(merged_df, excel_path, hidden_cols, widths)

    # Report extraction results and failures
    _report_extraction_failures(all_failed_pages, csv_path, csv_paths)


def run_for_profile(args, profile_name: str) -> None:
    """Run extraction pipeline for a single profile.

    Raises ConfigurationError for config/setup failures.
    Raises PipelineError for runtime pipeline failures.
    """

    # Setup environment: config, logging, lab specs (raises ConfigurationError on failure)
    config, lab_specs, standardization_section = _setup_profile_environment(args, profile_name)

    # Get column configuration for export formatting
    export_cols, hidden_cols, widths, dtypes = get_column_lists(COLUMN_SCHEMA)

    # Discover PDF files matching the input pattern
    pdf_files = sorted(config.input_path.glob(config.input_file_regex))
    logger.info(f"Found {len(pdf_files)} PDF(s) matching '{config.input_file_regex}'")

    # Guard: No PDFs found
    if not pdf_files:
        raise PipelineError(f"No PDF files found matching '{config.input_file_regex}' in {config.input_path}")

    # Deduplicate PDFs by file hash (skip identical files with different names)
    pdf_files, duplicate_pdfs = _deduplicate_pdf_files(pdf_files)
    for dup_path, orig_path in duplicate_pdfs:
        logger.warning(f"Skipping duplicate PDF: {dup_path.name} (same content as {orig_path.name})")
    if duplicate_pdfs:
        logger.info(f"Skipped {len(duplicate_pdfs)} duplicate PDF(s), {len(pdf_files)} unique PDF(s) remaining")

    # Filter out PDFs that already have valid CSV outputs (cache check)
    pdfs_to_process, skipped_count = _filter_pdfs_to_process(pdf_files, config.output_path)

    logger.info(f"Skipping {skipped_count} already-processed PDF(s)")
    logger.info(f"Processing {len(pdfs_to_process)} PDF(s)")

    # Process PDFs or use cached results (handles both cache hit and miss cases)
    log_dir = config.output_path / "logs"
    csv_paths, all_failed_pages, pdfs_failed = _process_pdfs_or_use_cache(
        pdf_files,
        pdfs_to_process,
        config,
        lab_specs,
        standardization_section,
        log_dir,
    )

    # Guard: No PDFs were successfully processed
    if not csv_paths:
        raise PipelineError("No PDFs successfully processed.")

    logger.info(f"Successfully processed {len(csv_paths)} PDFs")

    # Merge all individual PDF CSVs into single dataset
    logger.info("Merging CSV files...")
    merged_df = merge_csv_files(csv_paths)
    rows_after_merge = len(merged_df)
    logger.info(f"Merged data: {rows_after_merge} rows")

    # Guard: Merged dataset is empty
    if merged_df.empty:
        raise PipelineError("No data after merging CSV files.")

    # Apply all data transformations: normalize, filter, dedupe, validate
    merged_df = _process_and_transform_data(
        merged_df,
        lab_specs,
        export_cols,
    )

    # Guard: Data processing failed
    if merged_df is None:
        raise PipelineError("Data processing failed.")

    # Export final results to CSV and Excel
    _export_final_results(
        merged_df,
        export_cols,
        hidden_cols,
        widths,
        dtypes,
        config.output_path,
        all_failed_pages,
        csv_paths,
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
            logger.info("No profiles found. Create profiles in the 'profiles/' directory.")
        return

    # Determine which profiles to run (single specified or all available)
    if args.profile:
        profiles_to_run = [args.profile]

    # No profile specified - run all available
    else:
        profiles_to_run = ProfileConfig.list_profiles()

        # Guard: No profiles configured
        if not profiles_to_run:
            logger.error("No profiles found. Create profiles in the 'profiles/' directory.")
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
