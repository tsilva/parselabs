"""Main entry point for lab results extraction and processing."""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import sys
import re
import json
import shutil
import logging
import argparse
import pandas as pd
import pdf2image
from pathlib import Path
from multiprocessing import Pool
from openai import OpenAI
from tqdm import tqdm

# Local imports
from config import ExtractionConfig, LabSpecsConfig, ProfileConfig, UNKNOWN_VALUE
from utils import preprocess_page_image, setup_logging, ensure_columns
from extraction import (
    LabResult, HealthLabReport, extract_labs_from_page_image, self_consistency
)
from standardization import standardize_lab_names, standardize_lab_units
from normalization import apply_normalizations, deduplicate_results, apply_dtype_conversions

# Module-level logger (file handlers added after config is loaded)
logger = logging.getLogger(__name__)

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Default model for extraction
DEFAULT_MODEL = "google/gemini-2.5-flash"


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
    "verified": {"dtype": "boolean", "excel_width": 10},

    # Internal (hidden in Excel)
    "lab_type": {"dtype": "str", "excel_width": 10, "excel_hidden": True},
    "result_index": {"dtype": "Int64", "excel_width": 10, "excel_hidden": True},
}


def get_column_lists(schema: dict):
    """Extract ordered lists from schema."""
    ordered = [
        # Core columns in logical order
        "date", "source_file", "page_number",
        "lab_name", "value", "unit",
        "reference_min", "reference_max",
        "lab_name_raw", "value_raw", "unit_raw",
        "confidence", "verified",
        "lab_type", "result_index",
    ]
    export_cols = [k for k in ordered if k in schema]
    hidden_cols = [col for col, props in schema.items() if props.get("excel_hidden")]
    widths = {col: props["excel_width"] for col, props in schema.items() if "excel_width" in props}
    dtypes = {col: props["dtype"] for col, props in schema.items() if "dtype" in props}

    return export_cols, hidden_cols, widths, dtypes


# ========================================
# PDF Processing
# ========================================

def correct_percentage_lab_names(results: list[dict], lab_specs: LabSpecsConfig) -> list[dict]:
    """Correct lab names when unit is % but name doesn't end with (%)."""
    corrected_count = 0
    for result in results:
        std_name = result.get("lab_name_standardized")
        std_unit = result.get("lab_unit_standardized")

        if std_unit == "%" and std_name and not std_name.endswith("(%)"):
            percentage_variant = lab_specs.get_percentage_variant(std_name)
            if percentage_variant:
                logger.debug(f"Correcting lab name '{std_name}' -> '{percentage_variant}' (unit is %)")
                result["lab_name_standardized"] = percentage_variant
                corrected_count += 1

    if corrected_count > 0:
        logger.info(f"Corrected {corrected_count} percentage lab names")

    return results


def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    config: ExtractionConfig,
    lab_specs: LabSpecsConfig
) -> Path | None:
    """Process a single PDF file: extract, standardize, and save results."""
    pdf_stem = pdf_path.stem
    doc_out_dir = output_dir / pdf_stem
    doc_out_dir.mkdir(exist_ok=True, parents=True)
    csv_path = doc_out_dir / f"{pdf_stem}.csv"

    try:
        logger.info(f"[{pdf_stem}] Processing...")

        # Copy PDF to output directory
        copied_pdf = doc_out_dir / pdf_path.name
        if not copied_pdf.exists() or copied_pdf.stat().st_size != pdf_path.stat().st_size:
            shutil.copy2(pdf_path, copied_pdf)

        # Convert PDF to images
        try:
            pil_pages = pdf2image.convert_from_path(str(copied_pdf))
        except Exception as e:
            logger.error(f"[{pdf_stem}] Failed to convert PDF: {e}")
            return None

        # Process each page
        all_results = []
        doc_date = None

        logger.info(f"[{pdf_stem}] Processing {len(pil_pages)} page(s)...")
        for page_idx, page_image in enumerate(pil_pages):
            page_name = f"{pdf_stem}.{page_idx+1:03d}"
            jpg_path = doc_out_dir / f"{page_name}.jpg"
            json_path = doc_out_dir / f"{page_name}.json"

            logger.info(f"[{page_name}] Processing page {page_idx+1}/{len(pil_pages)}...")

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
                        client
                    )
                    logger.info(f"[{page_name}] Extraction completed")

                    # Post-extraction verification (simplified - cross-model only)
                    if config.enable_verification and page_data.get("lab_results"):
                        logger.info(f"[{page_name}] Running cross-model verification...")
                        try:
                            from verification import verify_page_extraction
                            page_data, verification_summary = verify_page_extraction(
                                image_path=jpg_path,
                                extracted_data=page_data,
                                client=client,
                                primary_model=config.extract_model_id,
                                verification_model=config.verification_model_id,
                            )
                            logger.info(
                                f"[{page_name}] Verification: {verification_summary.get('verified', 0)} verified, "
                                f"{verification_summary.get('corrected', 0)} corrected, "
                                f"avg confidence: {verification_summary.get('avg_confidence', 0):.2f}"
                            )
                        except Exception as ve:
                            logger.error(f"[{page_name}] Verification failed: {ve}")

                    json_path.write_text(json.dumps(page_data, indent=2, ensure_ascii=False), encoding='utf-8')
                except Exception as e:
                    logger.error(f"[{page_name}] Extraction failed: {e}")
                    page_data = HealthLabReport(lab_results=[]).model_dump(mode='json')
            else:
                logger.info(f"[{page_name}] Loading cached extraction data")
                page_data = json.loads(json_path.read_text(encoding='utf-8'))

            # Extract date from first page
            if page_idx == 0:
                doc_date = page_data.get("collection_date") or page_data.get("report_date")
                if doc_date == "0000-00-00":
                    doc_date = None
                if not doc_date:
                    # Try to extract from filename
                    match = re.search(r"(\d{4}-\d{2}-\d{2})", pdf_stem)
                    if match:
                        doc_date = match.group(1)

            # Add page metadata and result index to results
            for result_idx, result in enumerate(page_data.get("lab_results", [])):
                result["result_index"] = result_idx
                result["page_number"] = page_idx + 1
                result["source_file"] = page_name
                all_results.append(result)

        if not all_results:
            logger.warning(f"[{pdf_stem}] No results extracted")
            pd.DataFrame().to_csv(csv_path, index=False)
            return csv_path

        # Standardize lab names
        logger.info(f"[{pdf_stem}] Standardizing lab names...")
        raw_names = [r.get("lab_name_raw") for r in all_results if r.get("lab_name_raw")]
        if raw_names and lab_specs.exists:
            try:
                unique_names = list(set(raw_names))
                name_mapping = standardize_lab_names(
                    unique_names,
                    config.self_consistency_model_id,
                    lab_specs.standardized_names,
                    client
                )
                for result in all_results:
                    raw_name = result.get("lab_name_raw")
                    result["lab_name_standardized"] = name_mapping.get(raw_name, UNKNOWN_VALUE) if raw_name else UNKNOWN_VALUE
                logger.info(f"[{pdf_stem}] Standardized {len(unique_names)} unique test names")
            except Exception as e:
                logger.error(f"[{pdf_stem}] Name standardization failed: {e}")
                for result in all_results:
                    result["lab_name_standardized"] = UNKNOWN_VALUE
        else:
            for result in all_results:
                result["lab_name_standardized"] = UNKNOWN_VALUE

        # Standardize units
        logger.info(f"[{pdf_stem}] Standardizing units...")

        def normalize_raw_unit(raw_unit):
            """Normalize raw unit to handle null/nan/None values."""
            if raw_unit is None:
                return "null"
            raw_str = str(raw_unit).strip().lower()
            if raw_str in ("nan", "none", ""):
                return "null"
            return raw_unit

        unit_contexts = [
            (normalize_raw_unit(r.get("lab_unit_raw")), r.get("lab_name_standardized", ""))
            for r in all_results
        ]
        if unit_contexts and lab_specs.exists:
            try:
                unit_mapping = standardize_lab_units(
                    unit_contexts,
                    config.self_consistency_model_id,
                    lab_specs.standardized_units,
                    client,
                    lab_specs
                )
                for result in all_results:
                    raw_unit = normalize_raw_unit(result.get("lab_unit_raw"))
                    lab_name = result.get("lab_name_standardized", "")
                    standardized_unit = unit_mapping.get((raw_unit, lab_name), UNKNOWN_VALUE)

                    # Post-process: If LLM returned $UNKNOWN$ for a null unit, use lab spec primary unit
                    if standardized_unit == UNKNOWN_VALUE and raw_unit == "null" and lab_name != UNKNOWN_VALUE:
                        primary_unit = lab_specs.get_primary_unit(lab_name)
                        if primary_unit:
                            standardized_unit = primary_unit
                            logger.debug(f"[{pdf_stem}] Used primary unit '{primary_unit}' for null unit in '{lab_name}'")

                    result["lab_unit_standardized"] = standardized_unit
                logger.info(f"[{pdf_stem}] Standardized {len(set(r.get('lab_unit_raw') for r in all_results))} unique units")
            except Exception as e:
                logger.error(f"[{pdf_stem}] Unit standardization failed: {e}")
                for result in all_results:
                    result["lab_unit_standardized"] = UNKNOWN_VALUE
        else:
            for result in all_results:
                result["lab_unit_standardized"] = UNKNOWN_VALUE

        # Post-process: Correct percentage lab names
        if lab_specs.exists:
            all_results = correct_percentage_lab_names(all_results, lab_specs)

        # Update JSON files with standardized values
        _update_json_with_standardized_values(all_results, doc_out_dir)

        # Create DataFrame and save
        df = pd.DataFrame(all_results)
        df["date"] = doc_date

        # Ensure core columns exist
        core_cols = list(LabResult.model_fields.keys()) + ["date"]
        ensure_columns(df, core_cols, default=None)

        df = df[[col for col in core_cols if col in df.columns]]
        df.to_csv(csv_path, index=False, encoding='utf-8')

        logger.info(f"[{pdf_stem}] Completed successfully")
        return csv_path

    except Exception as e:
        logger.error(f"[{pdf_stem}] Unhandled exception: {e}", exc_info=True)
        return None


# ========================================
# Data Merging & Export
# ========================================

def merge_csv_files(csv_paths: list[Path]) -> pd.DataFrame:
    """Merge multiple CSV files into a single DataFrame."""
    dataframes = []
    for csv_path in csv_paths:
        try:
            if csv_path.stat().st_size > 0:
                df = pd.read_csv(csv_path, encoding='utf-8')
                df['source_file'] = csv_path.name
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
    with pd.ExcelWriter(excel_path, engine="xlsxwriter", datetime_format='yyyy-mm-dd', date_format='yyyy-mm-dd') as writer:
        df.to_excel(writer, sheet_name="Data", index=False)
        ws = writer.sheets["Data"]
        ws.freeze_panes(1, 0)
        for idx, col_name in enumerate(df.columns):
            width = widths.get(col_name, 12)
            options = {'hidden': True} if col_name in hidden_cols else {}
            ws.set_column(idx, idx, width, None, options)

    logger.info(f"Saved Excel: {excel_path}")


# ========================================
# Helpers
# ========================================

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
            data = json.loads(json_path.read_text(encoding='utf-8'))

            # Update each result by result_index
            for result in page_results:
                idx = result.get("result_index")
                if idx is not None and 0 <= idx < len(data.get("lab_results", [])):
                    data["lab_results"][idx]["lab_name_standardized"] = result.get("lab_name_standardized")
                    data["lab_results"][idx]["lab_unit_standardized"] = result.get("lab_unit_standardized")

            json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
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


def _process_pdf_wrapper(args):
    """Wrapper function for multiprocessing."""
    return process_single_pdf(*args)


def _find_empty_extractions(output_path: Path, matching_stems: set[str]) -> list[tuple[Path, list[Path]]]:
    """Find all PDFs that have empty extraction JSONs.

    Only considers output directories that match the input file pattern.
    """
    empty_by_pdf = []

    for pdf_dir in output_path.iterdir():
        if not pdf_dir.is_dir():
            continue
        if pdf_dir.name.startswith('.'):
            continue
        # Only check directories that match the input pattern
        if pdf_dir.name not in matching_stems:
            continue

        empty_jsons = []
        for json_path in pdf_dir.glob("*.json"):
            try:
                data = json.loads(json_path.read_text(encoding='utf-8'))
                if isinstance(data, dict) and not data.get("lab_results") and data.get("page_has_lab_data") is not False:
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

        if response == 'q':
            print("Skipping remaining files.")
            break
        elif response == 'a':
            pdfs_to_reprocess.append(pdf_dir)
            for remaining_pdf_dir, _ in empty_extractions[empty_extractions.index((pdf_dir, empty_jsons)) + 1:]:
                pdfs_to_reprocess.append(remaining_pdf_dir)
            break
        elif response == 'y':
            pdfs_to_reprocess.append(pdf_dir)

    if pdfs_to_reprocess:
        print(f"\nDeleting empty extractions for {len(pdfs_to_reprocess)} PDF(s)...")
        for pdf_dir in pdfs_to_reprocess:
            for json_path in pdf_dir.glob("*.json"):
                try:
                    data = json.loads(json_path.read_text(encoding='utf-8'))
                    if isinstance(data, dict) and not data.get("lab_results") and data.get("page_has_lab_data") is not False:
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
        description='Lab Results Parser - Extract lab results from PDFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using a profile (required):
  python main.py --profile tsilva

  # List available profiles:
  python main.py --list-profiles

  # Override settings:
  python main.py --profile tsilva --model google/gemini-2.5-pro --no-verify
        """
    )
    # Profile-based
    parser.add_argument(
        '--profile', '-p',
        type=str,
        help='Profile name (without extension)'
    )
    parser.add_argument(
        '--list-profiles',
        action='store_true',
        help='List available profiles and exit'
    )

    # Overrides
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Model ID for extraction (default: google/gemini-2.5-flash)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Disable cross-model verification'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default=None,
        help='Glob pattern for input files (overrides profile, default: *.pdf)'
    )

    return parser.parse_args()


def build_config(args) -> ExtractionConfig:
    """Build ExtractionConfig from args and env."""
    # Load profile (required)
    profile_path = None
    for ext in ('.yaml', '.yml', '.json'):
        p = Path(f"profiles/{args.profile}{ext}")
        if p.exists():
            profile_path = p
            break

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

    # Build config from profile
    config = ExtractionConfig(
        input_path=profile.input_path,
        output_path=profile.output_path,
        openrouter_api_key=openrouter_api_key,
        extract_model_id=profile.model or os.getenv("EXTRACT_MODEL_ID", DEFAULT_MODEL),
        self_consistency_model_id=profile.model or os.getenv("SELF_CONSISTENCY_MODEL_ID", DEFAULT_MODEL),
        input_file_regex=profile.input_file_regex or "*.pdf",
        n_extractions=int(os.getenv("N_EXTRACTIONS", "1")),
        max_workers=profile.workers or int(os.getenv("MAX_WORKERS", "0")) or (os.cpu_count() or 1),
        enable_verification=profile.verify if profile.verify is not None else os.getenv("ENABLE_VERIFICATION", "true").lower() == "true",
        verification_model_id=os.getenv("VERIFICATION_MODEL_ID") or None,
    )

    # Override from CLI args (highest priority)
    if args.model:
        config.extract_model_id = args.model
        config.self_consistency_model_id = args.model
    if args.no_verify:
        config.enable_verification = False
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

    # Profile is required for all other operations
    if not args.profile:
        print("Error: --profile is required.")
        print("Use --list-profiles to see available profiles.")
        print("Example: python main.py --profile tsilva")
        sys.exit(1)

    # Build configuration from env + profile + CLI args
    config = build_config(args)

    # Setup logging to output folder for later review
    global logger
    log_dir = config.output_path / "logs"
    logger = setup_logging(log_dir, clear_logs=True)

    logger.info(f"Input: {config.input_path}")
    logger.info(f"Output: {config.output_path}")
    logger.info(f"Model: {config.extract_model_id}")
    logger.info(f"Verification: {'enabled' if config.enable_verification else 'disabled'}")

    # Load lab specs
    lab_specs = LabSpecsConfig()

    # Get column configuration
    export_cols, hidden_cols, widths, dtypes = get_column_lists(COLUMN_SCHEMA)

    # Find PDFs to process
    pdf_files = sorted(list(config.input_path.glob(config.input_file_regex)))
    matching_stems = {p.stem for p in pdf_files}
    logger.info(f"Found {len(pdf_files)} PDF(s) matching '{config.input_file_regex}'")

    if not pdf_files:
        logger.warning("No PDF files found. Exiting.")
        return

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
                logger.info(f"Re-processing {pdf_path.name}: CSV missing required columns")
            pdfs_to_process.append(pdf_path)

    logger.info(f"Skipping {skipped_count} already-processed PDF(s)")
    logger.info(f"Processing {len(pdfs_to_process)} PDF(s)")

    if not pdfs_to_process:
        logger.info("All PDFs already processed. Moving to merge step...")
        csv_paths = [p for pdf in pdf_files if _is_csv_valid(p := _get_csv_path(pdf, config.output_path))]
        pdfs_failed = 0
    else:
        # Process PDFs in parallel
        n_workers = min(config.max_workers, len(pdfs_to_process))
        logger.info(f"Using {n_workers} worker(s) for PDF processing")

        tasks = [(pdf, config.output_path, config, lab_specs) for pdf in pdfs_to_process]

        with Pool(n_workers) as pool:
            results = []
            with tqdm(total=len(tasks), desc="Processing PDFs", unit="pdf") as pbar:
                for result in pool.imap(_process_pdf_wrapper, tasks):
                    results.append(result)
                    pbar.update(1)

        pdfs_failed = sum(1 for r in results if r is None)
        csv_paths = [p for pdf in pdf_files if _is_csv_valid(p := _get_csv_path(pdf, config.output_path))]

    if not csv_paths:
        logger.error("No PDFs successfully processed. Exiting.")
        return

    logger.info(f"Successfully processed {len(csv_paths)} PDFs")

    # Merge all CSVs
    logger.info("Merging CSV files...")
    merged_df = merge_csv_files(csv_paths)
    rows_after_merge = len(merged_df)
    logger.info(f"Merged data: {rows_after_merge} rows")

    if merged_df.empty:
        logger.error("No data to process")
        return

    # Apply normalizations (no demographics - moved to review tool)
    logger.info("Applying normalizations...")
    merged_df = apply_normalizations(merged_df, lab_specs, client, config.self_consistency_model_id)

    # Filter out non-lab-test rows
    unknown_mask = merged_df["lab_name_standardized"] == UNKNOWN_VALUE
    if unknown_mask.any():
        unknown_count = unknown_mask.sum()
        logger.info(f"Filtering {unknown_count} rows with unknown lab names")
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

    # Add confidence column (from verification or default)
    if "verification_confidence" in merged_df.columns:
        merged_df["confidence"] = merged_df["verification_confidence"]
    else:
        merged_df["confidence"] = 1.0

    # Add verified column
    if "verification_status" in merged_df.columns:
        merged_df["verified"] = merged_df["verification_status"].isin(["verified", "corrected"])
    else:
        merged_df["verified"] = False

    # Select final columns
    final_cols = [col for col in export_cols if col in merged_df.columns]
    merged_df = merged_df[final_cols]

    # Apply dtype conversions
    logger.info("Applying data type conversions...")
    merged_df = apply_dtype_conversions(merged_df, dtypes)

    # Save merged CSV
    logger.info("Saving merged CSV...")
    csv_path = config.output_path / "all.csv"
    merged_df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Saved merged CSV: {csv_path}")

    # Export Excel
    logger.info("Exporting to Excel...")
    excel_path = config.output_path / "all.xlsx"
    export_excel(merged_df, excel_path, hidden_cols, widths)

    logger.info("Pipeline completed successfully")
    logger.info(f"Output: {csv_path}")


if __name__ == "__main__":
    main()
