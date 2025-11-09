"""
Extraction Quality Assessment Script

Samples documents, runs extraction, and rigorously validates results against source images.
Aims to verify 100% accuracy of extraction.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import random

# Import from main pipeline
from main import process_single_pdf
from config import ExtractionConfig, LabSpecsConfig
from extraction import extract_labs_from_page_image
from openai import OpenAI

load_dotenv()

# Sample documents from different time periods
SAMPLE_DOCS = [
    "2003-07-07 - analises.pdf",  # Early format
    "2015-09-12 - analises.pdf",  # Mid period
    "2024-11-20 - analises.pdf",  # Recent format
]

def verify_extraction_against_image(
    image_path: str,
    extracted_results: List[Dict],
    client
) -> Dict:
    """
    Verify extracted lab results against the source image using creative validation.

    Returns a dict with:
    - total_values: int
    - verified_correct: int
    - errors: List[Dict] with details of discrepancies
    """

    # Build a detailed prompt that asks the model to verify specific values
    verification_prompt = """You are a medical lab report validation expert.

I will show you a lab report image and a list of extracted values. Your task is to verify each extracted value against what you see in the image.

EXTRACTED VALUES:
"""

    for i, result in enumerate(extracted_results, 1):
        verification_prompt += f"\n{i}. Lab: {result.get('lab_name_raw', 'N/A')}"
        verification_prompt += f"\n   Value: {result.get('value_raw', 'N/A')}"
        verification_prompt += f"\n   Unit: {result.get('lab_unit_raw', 'N/A')}"
        verification_prompt += f"\n   Reference Range: {result.get('reference_range', 'N/A')}"
        verification_prompt += "\n"

    verification_prompt += """
For EACH extracted value above, please verify:
1. Is the lab name correct as it appears in the image?
2. Is the value correct (exact number/text)?
3. Is the unit correct?
4. Is the reference range correct?

Respond in JSON format with this structure:
{
  "verifications": [
    {
      "index": 1,
      "lab_name_correct": true/false,
      "value_correct": true/false,
      "unit_correct": true/false,
      "reference_range_correct": true/false,
      "issues": "description of any discrepancies found",
      "correct_values": {
        "lab_name": "what you see in image if different",
        "value": "what you see in image if different",
        "unit": "what you see in image if different",
        "reference_range": "what you see in image if different"
      }
    }
  ],
  "missing_labs": "list any lab results visible in the image that were NOT extracted",
  "extra_labs": "list any extracted results that are NOT in the image"
}

Be EXTREMELY precise. Even tiny differences (like spacing, capitalization, decimal places) should be noted.
"""

    # Call vision model to verify
    try:
        # Read image
        import base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        response = client.chat.completions.create(
            model=os.getenv('EXTRACT_MODEL_ID'),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": verification_prompt
                        }
                    ]
                }
            ],
            temperature=0.2
        )

        # Get the response content
        content = response.choices[0].message.content

        # Try to parse JSON - handle markdown code blocks
        try:
            verification_result = json.loads(content)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code block
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                verification_result = json.loads(json_match.group(1))
            else:
                # Try finding any JSON object in the response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    verification_result = json.loads(json_match.group(0))
                else:
                    raise ValueError(f"Could not parse JSON from response: {content[:500]}")

        # Count errors
        total_values = len(extracted_results)
        errors = []

        for v in verification_result.get('verifications', []):
            if not all([
                v.get('lab_name_correct', True),
                v.get('value_correct', True),
                v.get('unit_correct', True),
                v.get('reference_range_correct', True)
            ]):
                errors.append({
                    'index': v.get('index'),
                    'issues': v.get('issues'),
                    'correct_values': v.get('correct_values', {})
                })

        # Add missing/extra labs as errors
        if verification_result.get('missing_labs'):
            errors.append({
                'type': 'missing_labs',
                'details': verification_result['missing_labs']
            })

        if verification_result.get('extra_labs'):
            errors.append({
                'type': 'extra_labs',
                'details': verification_result['extra_labs']
            })

        verified_correct = total_values - len([e for e in errors if e.get('index')])

        return {
            'total_values': total_values,
            'verified_correct': verified_correct,
            'errors': errors,
            'raw_verification': verification_result
        }

    except Exception as e:
        print(f"❌ Verification failed for {image_path}: {str(e)}")
        return {
            'total_values': len(extracted_results),
            'verified_correct': 0,
            'errors': [{'error': str(e)}],
            'raw_verification': None
        }


def assess_quality():
    """Main quality assessment function"""

    print("=" * 80)
    print("EXTRACTION QUALITY ASSESSMENT")
    print("=" * 80)
    print()

    # Load config
    config = ExtractionConfig.from_env()
    lab_specs = LabSpecsConfig()
    input_path = config.input_path
    output_path = config.output_path

    # Initialize client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    print(f"📊 Assessing quality on {len(SAMPLE_DOCS)} sampled documents:")
    for doc in SAMPLE_DOCS:
        print(f"   - {doc}")
    print()

    # Process each document
    all_results = []

    for doc_name in SAMPLE_DOCS:
        pdf_path = input_path / doc_name

        if not pdf_path.exists():
            print(f"❌ Document not found: {pdf_path}")
            continue

        print(f"\n{'=' * 80}")
        print(f"Processing: {doc_name}")
        print(f"{'=' * 80}")

        # Run extraction (this will create output directory with images and JSON)
        try:
            csv_path = process_single_pdf(pdf_path, output_path, config, lab_specs)
            print(f"✅ Extraction complete: {csv_path}")
        except Exception as e:
            print(f"❌ Extraction failed: {str(e)}")
            continue

        # Now validate each page
        doc_stem = pdf_path.stem
        doc_dir = output_path / doc_stem

        # Find all page JSONs
        page_jsons = sorted(doc_dir.glob(f"{doc_stem}.*.json"))

        print(f"\n🔍 Validating {len(page_jsons)} pages...")

        doc_stats = {
            'document': doc_name,
            'pages': [],
            'total_values': 0,
            'verified_correct': 0,
            'total_errors': 0
        }

        for page_json in page_jsons:
            # Get corresponding image
            page_num = page_json.stem.split('.')[-1]
            page_image = doc_dir / f"{doc_stem}.{page_num}.jpg"

            if not page_image.exists():
                print(f"⚠️  Image not found for {page_json}")
                continue

            # Load extracted results
            with open(page_json, 'r') as f:
                page_data = json.load(f)

            lab_results = page_data.get('lab_results', [])

            if not lab_results:
                print(f"   Page {page_num}: No lab results extracted")
                continue

            print(f"\n   Page {page_num}: Verifying {len(lab_results)} lab results...")

            # Verify
            verification = verify_extraction_against_image(
                str(page_image),
                lab_results,
                client
            )

            # Update stats
            page_stats = {
                'page': page_num,
                'total_values': verification['total_values'],
                'verified_correct': verification['verified_correct'],
                'errors': verification['errors']
            }

            doc_stats['pages'].append(page_stats)
            doc_stats['total_values'] += verification['total_values']
            doc_stats['verified_correct'] += verification['verified_correct']
            doc_stats['total_errors'] += len(verification['errors'])

            # Print page summary
            accuracy = (verification['verified_correct'] / verification['total_values'] * 100) if verification['total_values'] > 0 else 0
            print(f"      ✓ Correct: {verification['verified_correct']}/{verification['total_values']} ({accuracy:.1f}%)")

            if verification['errors']:
                print(f"      ❌ Errors: {len(verification['errors'])}")
                for error in verification['errors'][:3]:  # Show first 3
                    if 'issues' in error:
                        print(f"         - {error['issues']}")

        # Document summary
        doc_accuracy = (doc_stats['verified_correct'] / doc_stats['total_values'] * 100) if doc_stats['total_values'] > 0 else 0
        print(f"\n📈 Document Summary:")
        print(f"   Total values: {doc_stats['total_values']}")
        print(f"   Verified correct: {doc_stats['verified_correct']}")
        print(f"   Errors: {doc_stats['total_errors']}")
        print(f"   Accuracy: {doc_accuracy:.2f}%")

        all_results.append(doc_stats)

    # Overall summary
    print(f"\n\n{'=' * 80}")
    print("OVERALL QUALITY ASSESSMENT")
    print(f"{'=' * 80}")

    total_values = sum(r['total_values'] for r in all_results)
    total_correct = sum(r['verified_correct'] for r in all_results)
    total_errors = sum(r['total_errors'] for r in all_results)

    overall_accuracy = (total_correct / total_values * 100) if total_values > 0 else 0

    print(f"\n📊 Aggregated Statistics:")
    print(f"   Documents assessed: {len(all_results)}")
    print(f"   Total values extracted: {total_values}")
    print(f"   Verified correct: {total_correct}")
    print(f"   Total errors: {total_errors}")
    print(f"   Overall accuracy: {overall_accuracy:.2f}%")

    # Per-document breakdown
    print(f"\n📋 Per-Document Accuracy:")
    for result in all_results:
        doc_acc = (result['verified_correct'] / result['total_values'] * 100) if result['total_values'] > 0 else 0
        print(f"   {result['document']}: {doc_acc:.2f}% ({result['verified_correct']}/{result['total_values']})")

    # Error analysis
    print(f"\n🔍 Error Analysis:")
    all_errors = []
    for result in all_results:
        for page in result['pages']:
            for error in page['errors']:
                all_errors.append({
                    'document': result['document'],
                    'page': page['page'],
                    'error': error
                })

    if all_errors:
        print(f"   Total errors: {len(all_errors)}")
        print(f"\n   Sample errors:")
        for i, err in enumerate(all_errors[:10], 1):
            print(f"   {i}. {err['document']} (page {err['page']})")
            if 'issues' in err['error']:
                print(f"      {err['error']['issues']}")
    else:
        print(f"   🎉 No errors found! 100% accuracy achieved!")

    # Save detailed report
    report_path = output_path / "quality_assessment_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            'summary': {
                'documents_assessed': len(all_results),
                'total_values': total_values,
                'verified_correct': total_correct,
                'total_errors': total_errors,
                'overall_accuracy': overall_accuracy
            },
            'document_results': all_results,
            'all_errors': all_errors
        }, f, indent=2)

    print(f"\n💾 Detailed report saved to: {report_path}")

    # Final verdict
    print(f"\n{'=' * 80}")
    if overall_accuracy >= 99.5:
        print("✅ EXCELLENT: Extraction quality meets the 100% accuracy target!")
    elif overall_accuracy >= 95:
        print("⚠️  GOOD: Extraction quality is high but has room for improvement")
    else:
        print("❌ NEEDS IMPROVEMENT: Extraction quality is below acceptable threshold")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    assess_quality()
