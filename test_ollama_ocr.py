#!/usr/bin/env python3
"""Test GLM-OCR single-pass structured extraction.

Uses GLM-OCR's Information Extraction mode to output structured JSON directly.
"""

import base64
import json
import re
import time
from pathlib import Path
from openai import OpenAI

from extraction import HealthLabReport

# Configuration
MODEL = "glm-ocr"
IMAGE_PATH = Path("/Users/tsilva/Library/CloudStorage/GoogleDrive-eng.tiago.silva.sync@gmail.com/My Drive/labsparser-tiago/2001-12-27 - analises/2001-12-27 - analises.001.jpg")

# Ollama client
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


def normalize_date(date_str: str) -> str | None:
    """Convert various date formats to YYYY-MM-DD."""
    if not date_str:
        return None

    # Already correct format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    # DD/MM/YY or DD/MM/YYYY
    match = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", date_str)
    if match:
        day, month, year = match.groups()
        if len(year) == 2:
            year = "20" + year if int(year) < 50 else "19" + year
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    return None


def parse_reference_bounds(result: dict) -> dict:
    """Parse reference_min_raw and reference_max_raw from various formats.

    Handles cases where the model puts full range text into these fields,
    e.g., "4.5 a 5.9" instead of separate numeric values.
    """
    ref_min = result.get("reference_min_raw")
    ref_max = result.get("reference_max_raw")
    ref_range = result.get("reference_range", "")

    # If both are already valid numbers, return as-is
    if isinstance(ref_min, (int, float)) and isinstance(ref_max, (int, float)):
        return result

    # Source of truth for parsing: prefer existing ref_range, fallback to ref_min/ref_max values
    range_text = ref_range or str(ref_min or ref_max or "")

    parsed_min = None
    parsed_max = None

    # Try range pattern: "X a Y", "X - Y", "X-Y"
    match = re.search(r"(\d+\.?\d*)\s*[-aA]\s*(\d+\.?\d*)", range_text)
    if match:
        parsed_min = float(match.group(1))
        parsed_max = float(match.group(2))
    else:
        # Try "< X" pattern
        match = re.search(r"<\s*(\d+\.?\d*)", range_text)
        if match:
            parsed_max = float(match.group(1))
        else:
            # Try "> X" pattern
            match = re.search(r">\s*(\d+\.?\d*)", range_text)
            if match:
                parsed_min = float(match.group(1))

    result["reference_min_raw"] = parsed_min
    result["reference_max_raw"] = parsed_max
    return result


def clean_lab_results(result_dict: dict) -> dict:
    """Clean and fix common model output issues before Pydantic validation."""
    if "lab_results" not in result_dict:
        return result_dict

    cleaned_results = []
    for result in result_dict["lab_results"]:
        if isinstance(result, dict):
            # Parse reference bounds
            result = parse_reference_bounds(result)
            cleaned_results.append(result)

    result_dict["lab_results"] = cleaned_results
    return result_dict


def get_extraction_prompt() -> str:
    """Generate extraction prompt from Pydantic model schema."""
    schema = HealthLabReport.model_json_schema()
    defs = schema.get("$defs", {})

    # Get HealthLabReport fields (top-level)
    report_props = schema.get("properties", {})

    # Get LabResult fields from $defs
    lab_result_schema = defs.get("LabResult", {})
    lab_result_props = lab_result_schema.get("properties", {})

    # Fields to include in extraction (exclude internal/pipeline fields)
    report_fields = ["collection_date", "lab_facility", "page_has_lab_data", "lab_results"]
    lab_result_fields = [
        "lab_name_raw", "value_raw", "lab_unit_raw", "reference_range",
        "reference_min_raw", "reference_max_raw", "is_abnormal"
    ]

    # Build report field descriptions
    report_desc = []
    for field in report_fields:
        if field in report_props:
            desc = report_props[field].get("description", "")
            report_desc.append(f"- {field}: {desc}")

    # Build lab result field descriptions
    lab_result_desc = []
    for field in lab_result_fields:
        if field in lab_result_props:
            desc = lab_result_props[field].get("description", "")
            lab_result_desc.append(f"- {field}: {desc}")

    return f"""Information Extraction:

Extract ALL lab test results from this document into JSON format.

Required JSON structure:
{chr(10).join(report_desc)}

Each lab_results item must have:
{chr(10).join(lab_result_desc)}

IMPORTANT: Extract EVERY test visible in the document. Do not skip any.
Return ONLY valid JSON, no other text."""


def extract_structured(image_path: Path) -> HealthLabReport:
    """Single-pass extraction using GLM-OCR's Information Extraction mode."""
    print(f"Loading image: {image_path.name}")
    with open(image_path, "rb") as f:
        img_b64 = base64.standard_b64encode(f.read()).decode()

    extraction_prompt = get_extraction_prompt()

    print(f"Sending to {MODEL}...")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": extraction_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]}
        ],
        temperature=0.0,
        max_tokens=16384
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response content")

    print(f"Raw response: {len(content)} chars")

    # Try to extract JSON from the response (may be wrapped in markdown)
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        result_dict = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Content around error:\n{content[max(0, e.pos-100):e.pos+100]}")
        raise

    # Normalize date format if needed
    if result_dict.get("collection_date"):
        result_dict["collection_date"] = normalize_date(result_dict["collection_date"])
    if result_dict.get("report_date"):
        result_dict["report_date"] = normalize_date(result_dict["report_date"])

    # Clean lab results before Pydantic validation
    result_dict = clean_lab_results(result_dict)

    report = HealthLabReport(**result_dict)
    report.normalize_empty_optionals()

    return report


def main():
    start_time = time.time()
    report = extract_structured(IMAGE_PATH)
    elapsed_time = time.time() - start_time

    # Display results
    print(f"\nExtracted {len(report.lab_results)} lab results in {elapsed_time:.2f}s:")
    print("-" * 90)

    if report.collection_date:
        print(f"Collection Date: {report.collection_date}")
    if report.lab_facility:
        print(f"Lab Facility: {report.lab_facility}")

    print(f"\n{'Test Name':<40} {'Value':<15} {'Unit':<10} {'Reference'}")
    print("-" * 90)

    for result in report.lab_results:
        name = (result.lab_name_raw or "")[:38]
        value = (result.value_raw or "")[:13]
        unit = (result.lab_unit_raw or "")[:8]

        ref = ""
        if result.reference_min_raw is not None and result.reference_max_raw is not None:
            ref = f"{result.reference_min_raw} - {result.reference_max_raw}"
        elif result.reference_max_raw is not None:
            ref = f"< {result.reference_max_raw}"
        elif result.reference_min_raw is not None:
            ref = f"> {result.reference_min_raw}"
        elif result.reference_range:
            ref = result.reference_range[:20]

        abnormal = " *" if result.is_abnormal else ""
        print(f"{name:<40} {value:<15} {unit:<10} {ref}{abnormal}")

    # Full JSON output
    print("\n" + "=" * 60)
    print("FULL JSON OUTPUT:")
    print("=" * 60)
    print(json.dumps(report.model_dump(mode='json'), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
