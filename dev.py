#!/usr/bin/env python3
"""
Development script for iterative PDF extraction testing.
Usage: python dev.py <path_to_pdf>
"""

import sys
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

# Import from main
from main import (
    process_single_pdf,
    load_env_config,
    setup_logging,
    HealthLabReport
)

def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 80}")
    print(f" {title}")
    print(f"{char * 80}\n")

def print_lab_result(result: dict, index: int):
    """Pretty print a single lab result."""
    print(f"  [{index + 1}] {result.get('test_name', 'N/A')}")

    # Show standardized name if available
    standardized = result.get('lab_name_standardized')
    if standardized:
        if standardized == '$UNKNOWN$':
            print(f"      → ❌ {standardized} (no match found)")
        else:
            print(f"      → ✅ {standardized}")

    # Show value with both raw and standardized unit
    raw_unit = result.get('unit', '')
    standardized_unit = result.get('lab_unit_standardized', '')

    if standardized_unit and standardized_unit != '$UNKNOWN$' and standardized_unit != raw_unit:
        print(f"      Value: {result.get('value', 'N/A')} {raw_unit} → {standardized_unit}")
    else:
        print(f"      Value: {result.get('value', 'N/A')} {raw_unit}")
    if result.get('reference_range'):
        print(f"      Reference: {result.get('reference_range')}")
    if result.get('is_abnormal'):
        print(f"      ⚠️  ABNORMAL")
    if result.get('comments'):
        print(f"      Comments: {result.get('comments')}")
    print(f"      Source: {result.get('source_text', 'N/A')[:100]}...")
    print()

def display_extraction_results(pdf_path: Path, output_dir: Path):
    """Display extraction results for all pages."""
    import pandas as pd
    pdf_stem = pdf_path.stem
    doc_out_dir = output_dir / pdf_stem

    print_section(f"Extraction Results: {pdf_path.name}")

    # Read from CSV to get standardized names
    csv_path = doc_out_dir / f"{pdf_stem}.csv"
    if not csv_path.exists():
        print("❌ No CSV results found!")
        return

    try:
        df = pd.read_csv(csv_path)
        all_results = df.to_dict('records')
        total_results = len(all_results)

        # Group by page for display
        pages = {}
        for result in all_results:
            page_num = result.get('page_number', 0)
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(result)

        # Display by page
        for page_num in sorted(pages.keys()):
            results = pages[page_num]
            print(f"📄 Page {page_num:03d}: {len(results)} results")

            # Get dates from first result of the page
            if results:
                print(f"   Date: {results[0].get('date', 'N/A')}")
            print()

            for i, result in enumerate(results):
                print_lab_result(result, i)

        print_section(f"Summary: {total_results} total lab results extracted")

    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Check for common issues
    print("\n🔍 Quality Checks:")

    # Check for missing values
    missing_values = sum(1 for r in all_results if r.get('value') is None)
    if missing_values:
        print(f"   ⚠️  {missing_values} results missing values")

    # Check for missing units
    missing_units = sum(1 for r in all_results if r.get('unit') is None)
    if missing_units:
        print(f"   ⚠️  {missing_units} results missing units")

    # Check for missing test names
    missing_names = sum(1 for r in all_results if not r.get('test_name'))
    if missing_names:
        print(f"   ⚠️  {missing_names} results missing test names")

    # Check for unknown standardized names
    unknown_standardized_names = sum(1 for r in all_results if r.get('lab_name_standardized') == '$UNKNOWN$')
    if unknown_standardized_names:
        print(f"   ⚠️  {unknown_standardized_names} results with $UNKNOWN$ standardized names")
        print(f"       (these need manual review or config updates)")

    standardized_names_count = sum(1 for r in all_results if r.get('lab_name_standardized') and r.get('lab_name_standardized') != '$UNKNOWN$')
    if standardized_names_count:
        print(f"   ✅ {standardized_names_count} lab names successfully standardized")

    # Check for unknown standardized units
    unknown_standardized_units = sum(1 for r in all_results if r.get('lab_unit_standardized') == '$UNKNOWN$')
    if unknown_standardized_units:
        print(f"   ⚠️  {unknown_standardized_units} results with $UNKNOWN$ standardized units")
        print(f"       (these need manual review or config updates)")

    standardized_units_count = sum(1 for r in all_results if r.get('lab_unit_standardized') and r.get('lab_unit_standardized') != '$UNKNOWN$')
    if standardized_units_count:
        print(f"   ✅ {standardized_units_count} lab units successfully standardized")

    if not (missing_values or missing_units or missing_names or unknown_standardized_names or unknown_standardized_units):
        print("   ✅ All results have complete data and are fully standardized")

    print()

def main():
    # Default to test/test.pdf if no argument provided
    if len(sys.argv) == 1:
        pdf_path = Path(__file__).parent / "test" / "test.pdf"
        if not pdf_path.exists():
            print("❌ Error: test/test.pdf not found")
            print("\nUsage: python dev.py [path_to_pdf]")
            print("\nExample:")
            print("  python dev.py                    # Uses test/test.pdf")
            print("  python dev.py '/path/to/lab.pdf' # Uses custom PDF")
            sys.exit(1)
    elif len(sys.argv) == 2:
        pdf_path_str = sys.argv[1]
        pdf_path = Path(pdf_path_str)
    else:
        print("Usage: python dev.py [path_to_pdf]")
        print("\nExample:")
        print("  python dev.py                    # Uses test/test.pdf")
        print("  python dev.py '/path/to/lab.pdf' # Uses custom PDF")
        sys.exit(1)

    if not pdf_path.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    if not pdf_path.suffix.lower() == '.pdf':
        print(f"❌ Error: File is not a PDF: {pdf_path}")
        sys.exit(1)

    print_section(f"Dev Mode: Processing {pdf_path.name}", "=")

    # Setup logging
    setup_logging(clear_logs=True)

    # Load config
    try:
        config = load_env_config()
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

    # Override output directory to test/outputs
    output_dir = Path(__file__).parent / "test" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_stem = pdf_path.stem
    doc_out_dir = output_dir / pdf_stem

    # Clean existing outputs for this PDF
    if doc_out_dir.exists():
        print(f"🧹 Cleaning previous outputs: {doc_out_dir}")
        shutil.rmtree(doc_out_dir)

    print(f"📂 Output directory: {doc_out_dir}")
    print(f"🤖 Transcription model: {config['transcribe_model_id']}")
    print(f"🤖 Extraction model: {config['extract_model_id']}")
    print(f"🔄 Self-consistency: {config['n_transcriptions']} transcriptions, {config['n_extractions']} extractions")
    print()

    # Process the PDF
    print_section("Processing PDF...", "-")

    try:
        result_csv = process_single_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            self_consistency_model_id=config["self_consistency_model_id"],
            transcribe_model_id=config["transcribe_model_id"],
            n_transcribe=config["n_transcriptions"],
            extract_model_id=config["extract_model_id"],
            n_extract=config["n_extractions"]
        )

        if result_csv and result_csv.exists():
            print(f"✅ Processing complete!")
            print(f"📄 CSV saved to: {result_csv}")
        else:
            print("❌ Processing failed - no CSV generated")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Display results
    display_extraction_results(pdf_path, output_dir)

    # Show file locations
    print_section("Output Files", "-")
    print(f"📁 All outputs: {doc_out_dir}")

    # List key files
    page_images = sorted(doc_out_dir.glob("*.jpg"))
    page_txts = sorted(doc_out_dir.glob("*.txt"))
    page_jsons = sorted(doc_out_dir.glob("*.json"))

    print(f"\n   {len(page_images)} page images (.jpg)")
    print(f"   {len(page_txts)} transcriptions (.txt)")
    print(f"   {len(page_jsons)} extractions (.json)")
    print(f"   1 summary CSV (.csv)")

    print("\n💡 Tips:")
    print("   - Review transcriptions (.txt) to check OCR quality")
    print("   - Review extractions (.json) to check structured data")
    print("   - Compare with original PDF side-by-side")
    print("   - Run again to test consistency (files will be regenerated)")
    print()

if __name__ == "__main__":
    main()
