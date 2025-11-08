"""Main entry point for lab results extraction and processing."""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import json
import shutil
import logging
import pandas as pd
import pdf2image
from pathlib import Path
from multiprocessing import Pool
from openai import OpenAI
from tqdm import tqdm

# Local imports
from config import ExtractionConfig, LabSpecsConfig, UNKNOWN_VALUE
from utils import preprocess_page_image, setup_logging, clear_directory, ensure_columns
from extraction import (
    LabResult, HealthLabReport, extract_labs_from_page_image, self_consistency
)
from standardization import standardize_lab_names, standardize_lab_units
from normalization import apply_normalizations, deduplicate_results, apply_dtype_conversions
from plotting import LabPlotter

# Setup logging
LOG_DIR = Path("./logs")
logger = setup_logging(LOG_DIR, clear_logs=False)

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)


# ========================================
# Column Schema (kept here for now as it defines export structure)
# ========================================

COLUMN_SCHEMA = {
    "date": {"dtype": "datetime64[ns]", "excel_width": 13, "plotting_role": "date"},
    "lab_name_raw": {"dtype": "str", "excel_width": 35, "excel_hidden": False},
    "value_raw": {"dtype": "float64", "excel_width": 12, "excel_hidden": False},
    "lab_unit_raw": {"dtype": "str", "excel_width": 15, "excel_hidden": False},
    "reference_range": {"dtype": "str", "excel_width": 25, "excel_hidden": False},
    "reference_min_raw": {"dtype": "float64", "excel_width": 12, "excel_hidden": True},
    "reference_max_raw": {"dtype": "float64", "excel_width": 12, "excel_hidden": True},
    "is_abnormal": {"dtype": "boolean", "excel_width": 10, "excel_hidden": False},
    "comments": {"dtype": "str", "excel_width": 40, "excel_hidden": False},
    "source_text": {"dtype": "str", "excel_width": 50, "excel_hidden": True},
    "page_number": {"dtype": "Int64", "excel_width": 8},
    "source_file": {"dtype": "str", "excel_width": 25},
    "lab_type": {"dtype": "str", "excel_width": 10},
    "lab_name_standardized": {"dtype": "str", "excel_width": 35, "plotting_role": "group"},
    "lab_unit_standardized": {"dtype": "str", "excel_width": 15},
    "lab_name_slug": {"dtype": "str", "excel_width": 30, "excel_hidden": True},
    "value_primary": {"dtype": "float64", "excel_width": 14, "plotting_role": "value"},
    "lab_unit_primary": {"dtype": "str", "excel_width": 14, "plotting_role": "unit"},
    "reference_min_primary": {"dtype": "float64", "excel_width": 14},
    "reference_max_primary": {"dtype": "float64", "excel_width": 14},
    "is_out_of_reference": {"dtype": "boolean", "excel_width": 14},
    "healthy_range_min": {"dtype": "float64", "excel_width": 16},
    "healthy_range_max": {"dtype": "float64", "excel_width": 16},
    "is_in_healthy_range": {"dtype": "boolean", "excel_width": 18},
}


def get_column_lists(schema: dict):
    """Extract ordered lists from schema."""
    ordered = [
        # Metadata (when/where)
        "date", "source_file", "page_number",

        # Standardized lab data (main data people care about)
        "lab_name_standardized", "lab_unit_standardized", "lab_type",
        "value_primary", "lab_unit_primary",
        "is_in_healthy_range", "is_out_of_reference",

        # Reference/healthy ranges (for primary values)
        "healthy_range_min", "healthy_range_max",
        "reference_min_primary", "reference_max_primary",

        # Raw extraction (what was in PDF)
        "lab_name_raw", "value_raw", "lab_unit_raw",
        "reference_range", "is_abnormal", "comments",

        # Technical/internal fields
        "reference_min_raw", "reference_max_raw",
        "lab_name_slug", "source_text"
    ]
    export_cols = [k for k in ordered if k in schema]
    hidden_cols = [col for col, props in schema.items() if props.get("excel_hidden")]
    widths = {col: props["excel_width"] for col, props in schema.items() if "excel_width" in props}
    dtypes = {col: props["dtype"] for col, props in schema.items() if "dtype" in props}

    plotting_cols = {
        "date": next((col for col, props in schema.items() if props.get("plotting_role") == "date"), "date"),
        "value": next((col for col, props in schema.items() if props.get("plotting_role") == "value"), "value_normalized"),
        "group": next((col for col, props in schema.items() if props.get("plotting_role") == "group"), "lab_name"),
        "unit": next((col for col, props in schema.items() if props.get("plotting_role") == "unit"), "unit_normalized"),
    }

    return export_cols, hidden_cols, widths, dtypes, plotting_cols


# ========================================
# PDF Processing
# ========================================

def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    config: ExtractionConfig,
    lab_specs: LabSpecsConfig
) -> Path | None:
    """
    Process a single PDF file: extract, standardize, and save results.

    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory
        config: Extraction configuration
        lab_specs: Lab specifications configuration

    Returns:
        Path to output CSV, or None if processing failed
    """
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
                    json_path.write_text(json.dumps(page_data, indent=2, ensure_ascii=False), encoding='utf-8')
                    logger.info(f"[{page_name}] Extraction completed")
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
                    import re
                    match = re.search(r"(\d{4}-\d{2}-\d{2})", pdf_stem)
                    if match:
                        doc_date = match.group(1)

            # Add page metadata to results
            for result in page_data.get("lab_results", []):
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

        # Standardize units (with lab name context)
        logger.info(f"[{pdf_stem}] Standardizing units...")
        unit_contexts = [
            (r.get("lab_unit_raw") if r.get("lab_unit_raw") is not None else "null", r.get("lab_name_standardized", ""))
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
                    raw_unit = result.get("lab_unit_raw") if result.get("lab_unit_raw") is not None else "null"
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


def export_excel_with_sheets(
    df: pd.DataFrame,
    excel_path: Path,
    export_cols: list,
    hidden_cols: list,
    widths: dict,
    date_col: str,
    group_col: str
) -> None:
    """Export DataFrame to Excel with AllData and MostRecentByEnum sheets."""
    # Sort for most recent extraction
    if date_col in df.columns and group_col in df.columns:
        df = df.sort_values(by=[date_col, group_col], ascending=[False, True])
        most_recent_df = df.drop_duplicates(subset=[group_col], keep='first').copy()
    else:
        most_recent_df = pd.DataFrame()

    with pd.ExcelWriter(excel_path, engine="xlsxwriter", datetime_format='yyyy-mm-dd', date_format='yyyy-mm-dd') as writer:
        # AllData sheet
        df.to_excel(writer, sheet_name="AllData", index=False)
        ws_all = writer.sheets["AllData"]
        ws_all.freeze_panes(1, 0)
        for idx, col_name in enumerate(df.columns):
            width = widths.get(col_name, 12)
            options = {'hidden': True} if col_name in hidden_cols else {}
            ws_all.set_column(idx, idx, width, None, options)

        # MostRecentByEnum sheet
        if not most_recent_df.empty:
            most_recent_df.to_excel(writer, sheet_name="MostRecentByEnum", index=False)
            ws_recent = writer.sheets["MostRecentByEnum"]
            ws_recent.freeze_panes(1, 0)
            for idx, col_name in enumerate(most_recent_df.columns):
                width = widths.get(col_name, 12)
                options = {'hidden': True} if col_name in hidden_cols else {}
                ws_recent.set_column(idx, idx, width, None, options)
        else:
            pd.DataFrame().to_excel(writer, sheet_name="MostRecentByEnum", index=False)

    logger.info(f"Saved Excel with AllData and MostRecentByEnum sheets: {excel_path}")


# ========================================
# Main Pipeline
# ========================================

def _process_pdf_wrapper(args):
    """Wrapper function for multiprocessing with progress tracking."""
    return process_single_pdf(*args)


def main():
    """Main pipeline orchestration."""
    # Clear logs at start of run
    global logger
    logger = setup_logging(LOG_DIR, clear_logs=True)

    # Load configuration
    config = ExtractionConfig.from_env()
    lab_specs = LabSpecsConfig()

    # Get column configuration
    export_cols, hidden_cols, widths, dtypes, plotting_cols = get_column_lists(COLUMN_SCHEMA)

    # Find PDFs to process
    pdf_files = sorted(list(config.input_path.glob(config.input_file_regex)))
    logger.info(f"Found {len(pdf_files)} PDF(s) matching '{config.input_file_regex}'")

    if not pdf_files:
        logger.warning("No PDF files found. Exiting.")
        return

    # Filter out PDFs that already have their CSV
    pdfs_to_process = []
    skipped_count = 0
    for pdf_path in pdf_files:
        csv_path = config.output_path / pdf_path.stem / f"{pdf_path.stem}.csv"
        if csv_path.exists():
            skipped_count += 1
        else:
            pdfs_to_process.append(pdf_path)

    logger.info(f"Skipping {skipped_count} already-processed PDF(s)")
    logger.info(f"Processing {len(pdfs_to_process)} PDF(s)")

    if not pdfs_to_process:
        logger.info("All PDFs already processed. Moving to merge step...")
        # Still need to get CSV paths for merging
        csv_paths = [config.output_path / pdf.stem / f"{pdf.stem}.csv"
                     for pdf in pdf_files
                     if (config.output_path / pdf.stem / f"{pdf.stem}.csv").exists()]
    else:
        # Process PDFs in parallel
        n_workers = min(config.max_workers, len(pdfs_to_process))
        logger.info(f"Using {n_workers} worker(s) for PDF processing")

        tasks = [(pdf, config.output_path, config, lab_specs) for pdf in pdfs_to_process]

        # Use progress bar for PDF processing
        with Pool(n_workers) as pool:
            results = []
            with tqdm(total=len(tasks), desc="Processing PDFs", unit="pdf") as pbar:
                for result in pool.imap(_process_pdf_wrapper, tasks):
                    results.append(result)
                    pbar.update(1)

        csv_paths = [path for path in results if path and path.exists()]

    if not csv_paths:
        logger.error("No PDFs successfully processed. Exiting.")
        return

    logger.info(f"Successfully processed {len(csv_paths)} PDFs")

    # Merge all CSVs
    logger.info("Merging CSV files...")
    merged_df = merge_csv_files(csv_paths)
    logger.info(f"Merged data: {len(merged_df)} rows")

    if merged_df.empty:
        logger.error("No data to process")
        return

    # Apply normalizations
    logger.info("Applying normalizations...")
    merged_df = apply_normalizations(merged_df, lab_specs)

    # Deduplicate
    if lab_specs.exists:
        logger.info("Deduplicating results...")
        merged_df = deduplicate_results(merged_df, lab_specs)
        logger.info(f"After deduplication: {len(merged_df)} rows")

    # Reorder columns
    logger.info("Finalizing column order...")
    final_cols = [col for col in export_cols if col in merged_df.columns]
    final_cols += [col for col in merged_df.columns if col not in export_cols]
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
    export_excel_with_sheets(
        merged_df, excel_path, export_cols, hidden_cols, widths,
        plotting_cols["date"], plotting_cols["group"]
    )

    logger.info("All data processing and file exports finished")

    # Generate plots
    logger.info("Generating plots...")
    plots_base_dir = Path("plots")
    plots_base_dir.mkdir(exist_ok=True)
    clear_directory(plots_base_dir)

    output_plots_dir = config.output_path / "plots"
    output_plots_dir.mkdir(exist_ok=True)
    clear_directory(output_plots_dir)

    plotter = LabPlotter(
        date_col=plotting_cols["date"],
        value_col=plotting_cols["value"],
        group_col=plotting_cols["group"],
        unit_col=plotting_cols["unit"]
    )

    plotter.generate_all_plots(merged_df, [plots_base_dir, output_plots_dir])

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
