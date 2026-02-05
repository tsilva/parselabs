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


def extract_structured(image_path: Path) -> HealthLabReport:
    """Single-pass extraction using GLM-OCR's Information Extraction mode."""
    print(f"Loading image: {image_path.name}")
    with open(image_path, "rb") as f:
        img_b64 = base64.standard_b64encode(f.read()).decode()

    # Simplified prompt - show only the structure without placeholders
    extraction_prompt = """Information Extraction:

Extract all lab test results from this medical document into JSON.

Return JSON like this example:
{"collection_date":"2001-12-27","lab_facility":"Lab Name","page_has_lab_data":true,"lab_results":[{"lab_name_raw":"Glucose","value_raw":"95","lab_unit_raw":"mg/dL","reference_range":"70-100","reference_min_raw":70,"reference_max_raw":100,"is_abnormal":false}]}

For each test found in the image, add an object to lab_results with:
- lab_name_raw: exact test name from image
- value_raw: exact result value from image
- lab_unit_raw: exact unit from image (or null)
- reference_range: exact reference text from image (or null)
- reference_min_raw: lower bound as number (or null)
- reference_max_raw: upper bound as number (or null)
- is_abnormal: true if flagged abnormal, else false

Date format must be YYYY-MM-DD. Extract every test you see."""

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

    result_dict = json.loads(content)

    # Normalize date format if needed
    if result_dict.get("collection_date"):
        result_dict["collection_date"] = normalize_date(result_dict["collection_date"])
    if result_dict.get("report_date"):
        result_dict["report_date"] = normalize_date(result_dict["report_date"])

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
