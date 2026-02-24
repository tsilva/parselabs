"""Main entry point for lab results extraction and processing."""

from labs_parser.utils import load_dotenv_with_env

load_dotenv_with_env()

import os
import sys
from pathlib import Path
import re
import json
import shutil
import logging
import argparse
import subprocess
import pandas as pd
import pdf2image
from multiprocessing import Pool
from openai import OpenAI
from tqdm import tqdm

# Local imports
from labs_parser.config import (
    ExtractionConfig,
    LabSpecsConfig,
    ProfileConfig,
    UNKNOWN_VALUE,
)
from labs_parser.utils import preprocess_page_image, setup_logging, ensure_columns
from labs_parser.extraction import (
    LabResult,
    HealthLabReport,
    extract_labs_from_page_image,
    extract_labs_from_text,
    self_consistency,
    _build_standardized_names_section,
)
from labs_parser.standardization import load_cache
from labs_parser.normalization import (
    apply_normalizations,
    deduplicate_results,
    apply_dtype_conversions,
)
from labs_parser.validation import ValueValidator

# Module-level logger (file handlers added after config is loaded)
logger = logging.getLogger(__name__)

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


# ========================================
# Column Schema (simplified - 15 columns)
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
    # Quality
    "confidence": {"dtype": "float64", "excel_width": 12},
    # Review flags (from validation)
    "review_needed": {"dtype": "boolean", "excel_width": 12},
    "review_reason": {"dtype": "str", "excel_width": 30},
    "review_confidence": {"dtype": "float64", "excel_width": 14},
    # Limit indicators (for values like <0.05 or >738)
    "is_below_limit": {"dtype": "boolean", "excel_width": 12},
    "is_above_limit": {"dtype": "boolean", "excel_width": 12},
    # Internal (hidden in Excel)
    "lab_type": {"dtype": "str", "excel_width": 10, "excel_hidden": True},
    "result_index": {"dtype": "Int64", "excel_width": 10, "excel_hidden": True},
}


def get_column_lists(schema: dict):
    """Extract ordered lists from schema."""
    ordered = [
        # Core columns in logical order
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
        "confidence",
        "review_needed",
        "review_reason",
        "review_confidence",
        "is_below_limit",
        "is_above_limit",
        "lab_type",
        "result_index",
    ]
    export_cols = [k for k in ordered if k in schema]
    hidden_cols = [col for col, props in schema.items() if props.get("excel_hidden")]
    widths = {col: props["excel_width"] for col, props in schema.items() if "excel_width" in props}
    dtypes = {col: props["dtype"] for col, props in schema.items() if "dtype" in props}

    return export_cols, hidden_cols, widths, dtypes


# ========================================
# PDF Text Extraction (Cost Optimization)
# ========================================


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, bool]:
    """
    Extract text from PDF using pdftotext (from poppler).

    Returns:
        Tuple of (extracted_text, success).
        If pdftotext fails or is not installed, returns ("", False).
    """
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.debug(f"pdftotext returned non-zero exit code: {result.returncode}")
            return "", False
        return result.stdout, True
    except subprocess.TimeoutExpired:
        logger.warning(f"pdftotext timed out for {pdf_path.name}")
        return "", False
    except FileNotFoundError:
        logger.debug("pdftotext not installed (install with: brew install poppler)")
        return "", False
    except Exception as e:
        logger.warning(f"pdftotext failed for {pdf_path.name}: {e}")
        return "", False


_MIN_TEXT_CHARS = 200  # Minimum non-whitespace characters to attempt text extraction


def _text_has_enough_content(text: str, min_chars: int = _MIN_TEXT_CHARS) -> bool:
    """Check if extracted text has enough content to attempt LLM extraction.

    Uses a simple character count threshold instead of an LLM classifier.
    If the PDF has at least min_chars non-whitespace characters, we attempt
    text extraction. If it returns 0 results, we fall through to vision.

    Args:
        text: Extracted PDF text.
        min_chars: Minimum non-whitespace characters required.

    Returns:
        True if text has enough content to attempt extraction.
    """
    clean_text = text.replace(" ", "").replace("\n", "").replace("\t", "")
    return len(clean_text) >= min_chars


# ========================================
# PDF Processing
# ========================================



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
    pdf_stem = pdf_path.stem
    doc_out_dir = output_dir / pdf_stem
    doc_out_dir.mkdir(exist_ok=True, parents=True)
    csv_path = doc_out_dir / f"{pdf_stem}.csv"
    failed_pages = []  # Track pages that failed extraction

    try:
        logger.info(f"[{pdf_stem}] Processing...")

        # Copy PDF to output directory
        copied_pdf = doc_out_dir / pdf_path.name
        if not copied_pdf.exists() or copied_pdf.stat().st_size != pdf_path.stat().st_size:
            shutil.copy2(pdf_path, copied_pdf)

        # ========================================
        # Text-First Extraction (Cost Optimization)
        # ========================================
        # Try to extract text from PDF first - if successful, use text-only LLM (much cheaper)
        # Falls back to vision-based extraction if text extraction fails or is not viable

        used_text_extraction = False
        text_extraction_data = None

        # Check for existing text extraction cache
        text_json_path = doc_out_dir / f"{pdf_stem}.text.json"
        if text_json_path.exists():
            try:
                text_extraction_data = json.loads(text_json_path.read_text(encoding="utf-8"))
                used_text_extraction = True
                logger.info(f"[{pdf_stem}] Strategy: TEXT (cached)")
            except Exception as e:
                logger.warning(f"[{pdf_stem}] Failed to load text extraction cache: {e}")
                text_extraction_data = None

        # Try text extraction if no cache
        if text_extraction_data is None:
            pdf_text, pdftotext_success = extract_text_from_pdf(copied_pdf)

            if pdftotext_success and _text_has_enough_content(pdf_text):
                logger.info(
                    f"[{pdf_stem}] Strategy: TEXT (sufficient content, {len(pdf_text)} chars)"
                )

                try:
                    text_extraction_data = extract_labs_from_text(
                        pdf_text,
                        config.extract_model_id,
                        client,
                        standardization_section=standardization_section,
                    )

                    # Validate text extraction results
                    if text_extraction_data and text_extraction_data.get("lab_results"):
                        used_text_extraction = True
                        # Cache the text extraction results
                        text_json_path.write_text(
                            json.dumps(text_extraction_data, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        logger.info(
                            f"[{pdf_stem}] Text extraction complete: "
                            f"{len(text_extraction_data['lab_results'])} results"
                        )
                    else:
                        logger.warning(
                            f"[{pdf_stem}] Strategy: TEXT -> VISION (no results from text, falling back)"
                        )
                        text_extraction_data = None

                except Exception as e:
                    logger.warning(f"[{pdf_stem}] Strategy: TEXT -> VISION (text failed: {e})")
                    text_extraction_data = None
            else:
                if pdftotext_success:
                    logger.info(f"[{pdf_stem}] Strategy: VISION (insufficient text content)")
                else:
                    logger.info(f"[{pdf_stem}] Strategy: VISION (no embedded text in PDF)")

        # ========================================
        # Vision-Based Extraction (Fallback)
        # ========================================
        # If text extraction was not successful, use the traditional vision-based approach

        all_results = []
        doc_date = None

        if used_text_extraction and text_extraction_data:
            # Use text extraction results
            doc_date = _extract_document_date(text_extraction_data, pdf_stem)

            # Add page metadata to results (all from "page 1" since text extraction is whole-document)
            for result_idx, result in enumerate(text_extraction_data.get("lab_results", [])):
                result["result_index"] = result_idx
                result["page_number"] = 1
                result["source_file"] = f"{pdf_stem}.text"  # Mark as text extraction
                all_results.append(result)

        else:
            # Fall back to vision-based extraction
            # Convert PDF to images
            try:
                pil_pages = pdf2image.convert_from_path(str(copied_pdf))
            except Exception as e:
                logger.error(f"[{pdf_stem}] Failed to convert PDF: {e}")
                return None, failed_pages

            logger.info(f"[{pdf_stem}] Processing {len(pil_pages)} page(s) with vision...")
            for page_idx, page_image in enumerate(pil_pages):
                page_name = f"{pdf_stem}.{page_idx + 1:03d}"
                jpg_path = doc_out_dir / f"{page_name}.jpg"
                json_path = doc_out_dir / f"{page_name}.json"

                logger.info(f"[{page_name}] Processing page {page_idx + 1}/{len(pil_pages)}...")

                # Preprocess and save image
                if not jpg_path.exists():
                    processed = preprocess_page_image(page_image)
                    processed.save(jpg_path, "JPEG", quality=95)
                    logger.info(f"[{page_name}] Image preprocessed and saved")

                # Extract or load results
                if not json_path.exists():
                    logger.info(f"[{page_name}] Extracting data from image...")
                    try:
                        page_data, _ = self_consistency(
                            extract_labs_from_page_image,
                            config.self_consistency_model_id,
                            config.n_extractions,
                            jpg_path,
                            config.extract_model_id,
                            client,
                            standardization_section=standardization_section,
                        )
                        logger.info(f"[{page_name}] Extraction completed")

                        # Check for extraction failure (temperature retry exhausted)
                        if page_data.get("_extraction_failed"):
                            failure_reason = page_data.get("_failure_reason", "Unknown error")
                            failed_pages.append(
                                {
                                    "page": f"{pdf_stem} page {page_idx + 1}",
                                    "reason": failure_reason,
                                }
                            )
                            logger.error(f"[{page_name}] EXTRACTION FAILED: {failure_reason}")

                        json_path.write_text(
                            json.dumps(page_data, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    except Exception as e:
                        logger.error(f"[{page_name}] Extraction failed: {e}")
                        page_data = HealthLabReport(lab_results=[]).model_dump(mode="json")
                else:
                    logger.info(f"[{page_name}] Loading cached extraction data")
                    page_data = json.loads(json_path.read_text(encoding="utf-8"))
                    # Check for cached extraction failure
                    if page_data.get("_extraction_failed"):
                        failure_reason = page_data.get("_failure_reason", "Unknown error")
                        failed_pages.append(
                            {
                                "page": f"{pdf_stem} page {page_idx + 1}",
                                "reason": failure_reason,
                            }
                        )

                # Extract date from first page
                if page_idx == 0:
                    doc_date = _extract_document_date(page_data, pdf_stem)

                # Add page metadata and result index to results
                for result_idx, result in enumerate(page_data.get("lab_results", [])):
                    result["result_index"] = result_idx
                    result["page_number"] = page_idx + 1
                    result["source_file"] = page_name
                    all_results.append(result)

        if not all_results:
            logger.warning(f"[{pdf_stem}] No results extracted")
            pd.DataFrame().to_csv(csv_path, index=False)
            return csv_path, failed_pages

        # Standardized names/units are now set inline during extraction (via lab_name/unit fields).
        # Apply cache-based fallback for any results still missing lab_name_standardized.
        if lab_specs.exists:
            name_cache = load_cache("name_standardization")
            fallback_count = 0
            for result in all_results:
                std_name = result.get("lab_name_standardized")
                if not std_name or std_name == UNKNOWN_VALUE:
                    raw_name = result.get("lab_name_raw", "")
                    cached = name_cache.get(raw_name.lower()) if raw_name else None
                    if cached and cached != UNKNOWN_VALUE:
                        result["lab_name_standardized"] = cached
                        fallback_count += 1
                    elif not std_name:
                        result["lab_name_standardized"] = UNKNOWN_VALUE
            if fallback_count > 0:
                logger.info(f"[{pdf_stem}] Cache fallback applied to {fallback_count} names")

        # Update JSON files with standardized values
        _update_json_with_standardized_values(all_results, doc_out_dir)

        # Create DataFrame and save
        df = pd.DataFrame(all_results)
        df["date"] = doc_date

        # Ensure core columns exist
        core_cols = list(LabResult.model_fields.keys()) + ["date"]
        ensure_columns(df, core_cols, default=None)

        df = df[[col for col in core_cols if col in df.columns]]
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info(f"[{pdf_stem}] Completed successfully")
        return csv_path, failed_pages

    except Exception as e:
        logger.error(f"[{pdf_stem}] Unhandled exception: {e}", exc_info=True)
        return None, failed_pages


# ========================================
# Data Merging & Export
# ========================================


def merge_csv_files(csv_paths: list[Path]) -> pd.DataFrame:
    """Merge multiple CSV files into a single DataFrame."""
    dataframes = []
    for csv_path in csv_paths:
        try:
            if csv_path.stat().st_size > 0:
                df = pd.read_csv(csv_path, encoding="utf-8")
                df["source_file"] = csv_path.name
                dataframes.append(df)
        except Exception as e:
            logger.error(f"Failed to read {csv_path}: {e}")

    if not dataframes:
        return pd.DataFrame()

    return pd.concat(dataframes, ignore_index=True)


def export_excel(
    df: pd.DataFrame,
    excel_path: Path,
    hidden_cols: list,
    widths: dict,
) -> None:
    """Export DataFrame to Excel with formatting."""
    with pd.ExcelWriter(
        excel_path,
        engine="xlsxwriter",
        datetime_format="yyyy-mm-dd",
        date_format="yyyy-mm-dd",
    ) as writer:
        df.to_excel(writer, sheet_name="Data", index=False)
        ws = writer.sheets["Data"]
        ws.freeze_panes(1, 0)
        for idx, col_name in enumerate(df.columns):
            width = widths.get(col_name, 12)
            options = {"hidden": True} if col_name in hidden_cols else {}
            ws.set_column(idx, idx, width, None, options)

    logger.info(f"Saved Excel: {excel_path}")


# ========================================
# Helpers
# ========================================


def _extract_document_date(data_dict: dict, pdf_stem: str) -> str | None:
    """Extract document date from extraction data or filename.

    Args:
        data_dict: Dict containing collection_date or report_date fields
        pdf_stem: PDF filename stem (without extension) to extract date from

    Returns:
        Date string in YYYY-MM-DD format, or None if not found
    """
    doc_date = data_dict.get("collection_date") or data_dict.get("report_date")
    if doc_date == "0000-00-00":
        doc_date = None
    if not doc_date:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", pdf_stem)
        if match:
            doc_date = match.group(1)
    return doc_date


def _get_csv_path(pdf_path: Path, output_path: Path) -> Path:
    """Get the output CSV path for a given PDF file."""
    return output_path / pdf_path.stem / f"{pdf_path.stem}.csv"


def _update_json_with_standardized_values(all_results: list[dict], doc_out_dir: Path) -> None:
    """Update JSON files with standardized lab names and units."""
    # Group results by page number to minimize file I/O
    results_by_page: dict[int, list[dict]] = {}
    for result in all_results:
        page_num = result.get("page_number")
        if page_num is not None:
            results_by_page.setdefault(page_num, []).append(result)

    # Update each JSON file
    for page_num, page_results in results_by_page.items():
        # Find the JSON file for this page
        json_files = list(doc_out_dir.glob(f"*.{page_num:03d}.json"))
        if not json_files:
            continue
        json_path = json_files[0]

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))

            # Update each result by result_index
            for result in page_results:
                idx = result.get("result_index")
                if idx is not None and 0 <= idx < len(data.get("lab_results", [])):
                    data["lab_results"][idx]["lab_name_standardized"] = result.get(
                        "lab_name_standardized"
                    )
                    data["lab_results"][idx]["lab_unit_standardized"] = result.get(
                        "lab_unit_standardized"
                    )

            json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to update JSON {json_path}: {e}")


REQUIRED_CSV_COLS = ["result_index", "page_number", "source_file"]


def _is_csv_valid(csv_path: Path, required_cols: list[str] = REQUIRED_CSV_COLS) -> bool:
    """Check if CSV exists and has all required columns."""
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path, nrows=0)
        return all(col in df.columns for col in required_cols)
    except Exception:
        return False


def _init_worker_logging(log_dir: Path):
    """Initialize logging in worker processes."""
    setup_logging(log_dir, clear_logs=False)


def _process_pdf_wrapper(args):
    """Wrapper function for multiprocessing.

    Returns tuple of (csv_path, failed_pages) from process_single_pdf.
    """
    return process_single_pdf(*args)


def _find_empty_extractions(
    output_path: Path, matching_stems: set[str]
) -> list[tuple[Path, list[Path]]]:
    """Find all PDFs that have empty extraction JSONs.

    Only considers output directories that match the input file pattern.
    """
    empty_by_pdf = []

    for pdf_dir in output_path.iterdir():
        if not pdf_dir.is_dir():
            continue
        if pdf_dir.name.startswith("."):
            continue
        # Only check directories that match the input pattern
        if pdf_dir.name not in matching_stems:
            continue

        empty_jsons = []
        for json_path in pdf_dir.glob("*.json"):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if (
                    isinstance(data, dict)
                    and not data.get("lab_results")
                    and data.get("page_has_lab_data") is not False
                ):
                    empty_jsons.append(json_path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        if empty_jsons:
            empty_by_pdf.append((pdf_dir, sorted(empty_jsons)))

    return sorted(empty_by_pdf, key=lambda x: x[0].name)


def _prompt_reprocess_empty(output_path: Path, matching_stems: set[str]) -> list[Path]:
    """Check for empty extractions and prompt user to reprocess each one."""
    empty_extractions = _find_empty_extractions(output_path, matching_stems)

    if not empty_extractions:
        return []

    print(f"\nFound {len(empty_extractions)} PDF(s) with empty extraction pages:")
    for pdf_dir, empty_jsons in empty_extractions:
        print(f"  - {pdf_dir.name}: {len(empty_jsons)} empty page(s)")

    pdfs_to_reprocess = []

    for pdf_dir, empty_jsons in empty_extractions:
        print(f"\n{pdf_dir.name}:")
        for json_path in empty_jsons:
            print(f"  - {json_path.name}")

        response = input(f"Reprocess {pdf_dir.name}? [y/N/a(ll)/q(uit)]: ").strip().lower()

        if response == "q":
            print("Skipping remaining files.")
            break
        elif response == "a":
            pdfs_to_reprocess.append(pdf_dir)
            for remaining_pdf_dir, _ in empty_extractions[
                empty_extractions.index((pdf_dir, empty_jsons)) + 1 :
            ]:
                pdfs_to_reprocess.append(remaining_pdf_dir)
            break
        elif response == "y":
            pdfs_to_reprocess.append(pdf_dir)

    if pdfs_to_reprocess:
        print(f"\nDeleting empty extractions for {len(pdfs_to_reprocess)} PDF(s)...")
        for pdf_dir in pdfs_to_reprocess:
            for json_path in pdf_dir.glob("*.json"):
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                    if (
                        isinstance(data, dict)
                        and not data.get("lab_results")
                        and data.get("page_has_lab_data") is not False
                    ):
                        json_path.unlink()
                        logger.info(f"Deleted empty JSON: {json_path.name}")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

            csv_path = pdf_dir / f"{pdf_dir.name}.csv"
            if csv_path.exists():
                csv_path.unlink()
                logger.info(f"Deleted CSV: {csv_path.name}")

    return pdfs_to_reprocess


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
        "--list-profiles", action="store_true", help="List available profiles and exit"
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


def build_config(args) -> ExtractionConfig:
    """Build ExtractionConfig from args and env."""
    # Load profile (required)
    profile_path = ProfileConfig.find_path(args.profile)

    if not profile_path:
        print(f"Error: Profile '{args.profile}' not found")
        print("Use --list-profiles to see available profiles.")
        sys.exit(1)

    profile = ProfileConfig.from_file(profile_path)
    logger.info(f"Using profile: {profile.name}")

    # Validate profile has required paths
    if not profile.input_path:
        print(f"Error: Profile '{args.profile}' has no input_path defined.")
        sys.exit(1)
    if not profile.output_path:
        print(f"Error: Profile '{args.profile}' has no output_path defined.")
        sys.exit(1)

    # Get API key from environment (still required)
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    # Get model IDs from environment (required)
    extract_model_id = os.getenv("EXTRACT_MODEL_ID")
    if not extract_model_id:
        print("Error: EXTRACT_MODEL_ID environment variable not set")
        sys.exit(1)
    self_consistency_model_id = os.getenv("SELF_CONSISTENCY_MODEL_ID")
    if not self_consistency_model_id:
        print("Error: SELF_CONSISTENCY_MODEL_ID environment variable not set")
        sys.exit(1)

    # Build config from profile
    config = ExtractionConfig(
        input_path=profile.input_path,
        output_path=profile.output_path,
        openrouter_api_key=openrouter_api_key,
        extract_model_id=extract_model_id,
        self_consistency_model_id=self_consistency_model_id,
        input_file_regex=profile.input_file_regex or "*.pdf",
        n_extractions=int(os.getenv("N_EXTRACTIONS", "1")),
        max_workers=profile.workers or int(os.getenv("MAX_WORKERS", "0")) or (os.cpu_count() or 1),
    )

    # Override from CLI args (highest priority)
    if args.model:
        config.extract_model_id = args.model
        config.self_consistency_model_id = args.model
    if args.workers:
        config.max_workers = args.workers
    if args.pattern:
        config.input_file_regex = args.pattern

    # Validate input path exists
    if not config.input_path.exists():
        print(f"Error: Input path does not exist: {config.input_path}")
        sys.exit(1)

    # Ensure output directory exists
    config.output_path.mkdir(parents=True, exist_ok=True)

    return config


# ========================================
# Main Pipeline
# ========================================


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
        # Just check that the server is reachable with a lightweight request
        # Try models.list first, but don't validate model existence
        client.models.list(timeout=timeout)
        return True, "Server is available"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return (
                False,
                f"Authentication failed - check your OPENROUTER_API_KEY: {error_msg}",
            )
        elif "timeout" in error_msg.lower():
            return False, f"Server timeout after {timeout}s - server may be unreachable"
        elif "404" in error_msg:
            # Server responded with 404 - endpoint not implemented (e.g., local servers)
            # This is a valid response, so server is available
            return True, "Server is available (models endpoint not implemented)"
        elif (
            "Connection" in error_msg
            or "refused" in error_msg.lower()
            or "reset" in error_msg.lower()
            or "Name or service not known" in error_msg
            or "getaddrinfo" in error_msg
        ):
            return False, f"Cannot connect to server: {error_msg}"
        else:
            # For any other error, assume server is unavailable to be safe
            return False, f"Server check failed: {error_msg}"


def run_for_profile(args, profile_name: str) -> bool:
    """Run extraction pipeline for a single profile.

    Returns True if successful, False otherwise.
    """
    # Temporarily set args.profile for build_config
    original_profile = args.profile
    args.profile = profile_name

    try:
        config = build_config(args)
    finally:
        args.profile = original_profile

    # Setup logging to output folder for later review
    global logger
    log_dir = config.output_path / "logs"
    logger = setup_logging(log_dir, clear_logs=True)

    # Check server availability before processing
    logger.info("Checking OpenRouter server availability...")
    is_available, message = check_server_availability(client, config.extract_model_id)
    if not is_available:
        logger.error(f"Server check failed: {message}")
        print(f"\nError: Cannot start extraction - {message}")
        print("Please check your internet connection and API key, then try again.")
        return False
    logger.info(f"Server check passed: {message}")

    logger.info(f"Input: {config.input_path}")
    logger.info(f"Output: {config.output_path}")
    logger.info(f"Model: {config.extract_model_id}")

    # Load lab specs
    lab_specs = LabSpecsConfig()

    # Copy lab specs to output folder for reproducibility
    if lab_specs.exists:
        lab_specs_dest = config.output_path / "lab_specs.json"
        shutil.copy2(lab_specs.config_path, lab_specs_dest)
        logger.info(f"Copied lab specs to output: {lab_specs_dest}")

    # Build standardization section for inline standardization during extraction
    standardization_section: str | None = None
    if lab_specs.exists:
        standardization_section = _build_standardized_names_section(
            lab_specs.standardized_names, lab_specs._specs
        )
        logger.info(
            f"Built standardization section: {len(lab_specs.standardized_names)} lab names"
        )

    # Get column configuration
    export_cols, hidden_cols, widths, dtypes = get_column_lists(COLUMN_SCHEMA)

    # Find PDFs to process
    pdf_files = sorted(config.input_path.glob(config.input_file_regex))
    matching_stems = {p.stem for p in pdf_files}
    logger.info(f"Found {len(pdf_files)} PDF(s) matching '{config.input_file_regex}'")

    if not pdf_files:
        logger.warning("No PDF files found.")
        return False

    # Check for empty extractions and prompt user to reprocess
    _prompt_reprocess_empty(config.output_path, matching_stems)

    # Filter out PDFs that already have a valid CSV
    pdfs_to_process = []
    skipped_count = 0
    for pdf_path in pdf_files:
        csv_path = _get_csv_path(pdf_path, config.output_path)
        if _is_csv_valid(csv_path):
            skipped_count += 1
        else:
            if csv_path.exists():
                logger.warning(f"Re-processing {pdf_path.name}: CSV missing required columns")
            pdfs_to_process.append(pdf_path)

    logger.info(f"Skipping {skipped_count} already-processed PDF(s)")
    logger.info(f"Processing {len(pdfs_to_process)} PDF(s)")

    all_failed_pages = []  # Aggregate extraction failures across all PDFs

    if not pdfs_to_process:
        logger.info("All PDFs already processed. Moving to merge step...")
        csv_paths = [
            p for pdf in pdf_files if _is_csv_valid(p := _get_csv_path(pdf, config.output_path))
        ]
        pdfs_failed = 0
    else:
        # Process PDFs in parallel
        n_workers = min(config.max_workers, len(pdfs_to_process))
        logger.info(f"Using {n_workers} worker(s) for PDF processing")

        tasks = [
            (pdf, config.output_path, config, lab_specs, standardization_section)
            for pdf in pdfs_to_process
        ]

        with Pool(n_workers, initializer=_init_worker_logging, initargs=(log_dir,)) as pool:
            results = []
            with tqdm(total=len(tasks), desc="Processing PDFs", unit="pdf") as pbar:
                for result in pool.imap(_process_pdf_wrapper, tasks):
                    results.append(result)
                    pbar.update(1)

        # Unpack results: each is (csv_path, failed_pages)
        pdfs_failed = sum(1 for csv_path, _ in results if csv_path is None)
        for _, failed_pages in results:
            all_failed_pages.extend(failed_pages)

        csv_paths = [
            p for pdf in pdf_files if _is_csv_valid(p := _get_csv_path(pdf, config.output_path))
        ]

    if not csv_paths:
        logger.error("No PDFs successfully processed.")
        return False

    logger.info(f"Successfully processed {len(csv_paths)} PDFs")

    # Merge all CSVs
    logger.info("Merging CSV files...")
    merged_df = merge_csv_files(csv_paths)
    rows_after_merge = len(merged_df)
    logger.info(f"Merged data: {rows_after_merge} rows")

    if merged_df.empty:
        logger.error("No data to process")
        return False

    # Apply normalizations (no demographics - moved to review tool)
    logger.info("Applying normalizations...")
    merged_df = apply_normalizations(merged_df, lab_specs, client, config.self_consistency_model_id)

    # Filter out non-lab-test rows
    unknown_mask = merged_df["lab_name_standardized"] == UNKNOWN_VALUE
    if unknown_mask.any():
        unknown_count = unknown_mask.sum()
        logger.error(f"Filtering {unknown_count} rows with unknown lab names")
        merged_df = merged_df[~unknown_mask].reset_index(drop=True)

    # Deduplicate
    if lab_specs.exists:
        logger.info("Deduplicating results...")
        merged_df = deduplicate_results(merged_df, lab_specs)
        logger.info(f"After deduplication: {len(merged_df)} rows")

    # Rename columns to simplified schema
    column_renames = {
        "lab_name_standardized": "lab_name",
        "value_primary": "value",
        "lab_unit_primary": "unit",
        "lab_unit_raw": "unit_raw",
        "reference_min_primary": "reference_min",
        "reference_max_primary": "reference_max",
    }
    merged_df = merged_df.rename(columns=column_renames)

    # Run value-based validation
    logger.info("Running value-based validation...")
    validator = ValueValidator(lab_specs)
    merged_df = validator.validate(merged_df)
    validation_stats = validator.validation_stats
    if validation_stats.get("rows_flagged", 0) > 0:
        logger.info(f"Validation flagged {validation_stats['rows_flagged']} rows for review")
        for reason, count in validation_stats.get("flags_by_reason", {}).items():
            logger.info(f"  - {reason}: {count}")

    # Add confidence column (default to 1.0)
    merged_df["confidence"] = 1.0
    logger.debug(
        f"After setting confidence=1.0: NaN count = {merged_df['confidence'].isna().sum()}"
    )

    # Select final columns
    final_cols = [col for col in export_cols if col in merged_df.columns]
    merged_df = merged_df[final_cols]
    logger.debug(
        f"After column filtering: confidence NaN count = {merged_df['confidence'].isna().sum() if 'confidence' in merged_df.columns else 'column missing'}"
    )

    # Apply dtype conversions
    logger.info("Applying data type conversions...")
    merged_df = apply_dtype_conversions(merged_df, dtypes)
    logger.debug(
        f"After dtype conversion: confidence NaN count = {merged_df['confidence'].isna().sum()}"
    )

    # Save merged CSV
    logger.info("Saving merged CSV...")
    csv_path = config.output_path / "all.csv"
    merged_df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved merged CSV: {csv_path}")

    # Export Excel
    logger.info("Exporting to Excel...")
    excel_path = config.output_path / "all.xlsx"
    export_excel(merged_df, excel_path, hidden_cols, widths)

    # Final summary
    logger.info("=" * 50)
    logger.info("Pipeline completed")
    logger.info(f"  PDFs processed: {len(csv_paths)}")
    logger.info(f"  Output: {csv_path}")

    # Report extraction failures
    if all_failed_pages:
        logger.warning(f"  Pages with extraction failures: {len(all_failed_pages)}")
        for failure in all_failed_pages:
            logger.warning(f"    - {failure['page']}: {failure['reason']}")
        print(f"\n⚠️  Extraction failures detected ({len(all_failed_pages)} pages):")
        for failure in all_failed_pages:
            print(f"    - {failure['page']}: {failure['reason']}")
    else:
        logger.info("  Extraction failures: 0")

    return True


def main():
    """Main pipeline orchestration."""
    args = parse_args()

    # Handle --list-profiles
    if args.list_profiles:
        profiles = ProfileConfig.list_profiles()
        if profiles:
            print("Available profiles:")
            for name in profiles:
                print(f"  - {name}")
        else:
            print("No profiles found. Create profiles in the 'profiles/' directory.")
        return

    # Determine which profiles to run
    if args.profile:
        profiles_to_run = [args.profile]
    else:
        profiles_to_run = ProfileConfig.list_profiles()
        if not profiles_to_run:
            print("No profiles found. Create profiles in the 'profiles/' directory.")
            print("Or use --profile to specify one.")
            sys.exit(1)
        print(f"Running all profiles: {', '.join(profiles_to_run)}")

    # Run extraction for each profile
    results = {}
    for profile_name in profiles_to_run:
        print(f"\n{'=' * 60}")
        print(f"Processing profile: {profile_name}")
        print(f"{'=' * 60}")
        try:
            success = run_for_profile(args, profile_name)
            results[profile_name] = "success" if success else "failed"
        except Exception as e:
            print(f"Error processing profile {profile_name}: {e}")
            results[profile_name] = f"error: {e}"

    # Summary if multiple profiles
    if len(profiles_to_run) > 1:
        print(f"\n{'=' * 60}")
        print("Summary:")
        print(f"{'=' * 60}")
        for profile_name, status in results.items():
            print(f"  {profile_name}: {status}")


if __name__ == "__main__":
    main()
