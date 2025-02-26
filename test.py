import os
import re
import pandas as pd
import anthropic
import base64
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
import json

def find_lab_file_pairs(output_dir):
    """Find all JPG/CSV file pairs recursively in the output directory."""
    file_pairs = []
    
    # Walk through all directories
    for root, _, files in os.walk(output_dir):
        # Group files by their base name without extension
        file_groups = {}
        
        for file in files:
            match = re.match(r"(.+\.\d{3})\.(jpg|csv)$", file, re.IGNORECASE)
            if match:
                base_name, ext = match.groups()
                if base_name not in file_groups:
                    file_groups[base_name] = {}
                file_groups[base_name][ext.lower()] = os.path.join(root, file)
        
        # Find pairs
        for base_name, files_dict in file_groups.items():
            if 'jpg' in files_dict and 'csv' in files_dict:
                file_pairs.append((files_dict['jpg'], files_dict['csv']))
    
    return file_pairs

def read_csv_data(csv_path):
    """Read lab data from CSV file."""
    try:
        # Read CSV with semicolon separator
        df = pd.read_csv(csv_path, sep=';')
        return df
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None

def encode_image(image_path):
    """Encode image as base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def parse_claude_response(response_text):
    """Parse Claude's verification response into structured data."""
    result = {
        "assessment": "invalid",
        "confidence": 0,
        "mismatches": []
    }
    
    # Extract verification result (YES/NO)
    verification_match = re.search(r"Verification Result:.*?\b(YES|NO)\b", response_text, re.IGNORECASE | re.DOTALL)
    if verification_match:
        result["assessment"] = "valid" if verification_match.group(1).upper() == "YES" else "invalid"
    
    # Extract confidence level
    confidence_match = re.search(r"Confidence Level:.*?(\d+)%", response_text, re.IGNORECASE | re.DOTALL)
    if confidence_match:
        result["confidence"] = int(confidence_match.group(1))
    
    # Extract discrepancies/mismatches
    if "No discrepancies found" not in response_text:
        # Look for the discrepancies section
        discrepancies_section = re.search(r"Discrepancies:(.*?)(?:Confidence Level:|$)", 
                                         response_text, re.IGNORECASE | re.DOTALL)
        if discrepancies_section:
            discrepancies_text = discrepancies_section.group(1).strip()
            # Extract individual discrepancies
            # This is a simple extraction that might need refinement based on Claude's actual output format
            discrepancy_items = re.findall(r'[\-\*]\s*(.*?)(?=[\-\*]|\Z)', discrepancies_text, re.DOTALL)
            for item in discrepancy_items:
                item = item.strip()
                if item:
                    # Try to extract image vs CSV values
                    mismatch_info = re.search(r'(.*?):\s*image shows\s*(.*?),\s*CSV shows\s*(.*?)(?:$|\.)', item, re.IGNORECASE)
                    if mismatch_info:
                        test_name = mismatch_info.group(1).strip()
                        image_value = mismatch_info.group(2).strip()
                        csv_value = mismatch_info.group(3).strip()
                        result["mismatches"].append({
                            "test_name": test_name,
                            "image_value": image_value,
                            "csv_value": csv_value
                        })
                    else:
                        # If specific format not found, store the raw mismatch text
                        result["mismatches"].append({"description": item})
    
    return result

def verify_with_claude(image_path, csv_data, client):
    """Use Claude to verify if the lab data was correctly extracted using function calling."""
    
    # Encode the image
    image_base64 = encode_image(image_path)
    if not image_base64:
        return False, "Failed to encode image", None

    TOOLS = [
        {
            "name": "record_verification_result",
            "description": "Record the verification result of lab test data",
            "input_schema": {
                "type": "object",
                "properties": {
                    "valid": {
                        "type": "boolean",
                        "description": "Whether all lab tests from the image are correctly represented in the CSV"
                    },
                    "confidence": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Confidence level as a percentage (0-100%)"
                    },
                    "mismatches": {
                        "type": "array",
                        "description": "List of any discrepancies between image and CSV",
                        "items": {
                            "type": "object",
                            "properties": {
                                "csv_lab_name": {
                                    "type": "string",
                                    "description": "Name of the lab test as shown in the CSV"
                                },
                                "image_lab_name": {
                                    "type": "string",
                                    "description": "Name of the lab test as shown in the image"
                                },
                                "csv_lab_value": {
                                    "type": "string",
                                    "description": "Value of the lab test as shown in the CSV (including unit if relevant)"
                                },
                                "image_lab_value": {
                                    "type": "string",
                                    "description": "Value of the lab test as shown in the image (including unit if relevant)"
                                },
                                "csv_lab_unit": {
                                    "type": "string",
                                    "description": "Unit of measurement as shown in the CSV (if applicable)"
                                },
                                "image_lab_unit": {
                                    "type": "string",
                                    "description": "Unit of measurement as shown in the image (if applicable)"
                                },
                                "explanation": {
                                    "type": "string",
                                    "description": "Explanation of the discrepancy between the image and CSV"
                                }
                            },
                            "required": ["csv_lab_name", "image_lab_name", "csv_lab_value", "image_lab_value", "csv_lab_unit", "image_lab_unit", "explanation"]
                        }
                    }
                },
                "required": ["valid", "confidence", "mismatches"]
            }
        }
    ]
    
    # Create prompt for Claude
    prompt = f"""
You are provided with a lab test result image and a CSV file extracted from this image. Your task is to verify whether all lab test values visible in the image are accurately represented in the CSV data.

**CSV Data Format**:  
The CSV data is semicolon-separated and includes columns for lab test names, values, units, and reference ranges (minimum and maximum values).

**Verification Instructions**:  
For each lab test visible in the image, check if there is a corresponding entry in the CSV that meets these criteria:  
1. **Lab Name**: The name may vary in spelling, casing, or phrasing but must refer to the same test. Use your knowledge to determine if two names represent the same test, accounting for synonyms, abbreviations, or formatting differences (e.g., 'Glucose' and 'Blood Sugar' are equivalent, 'HbA1c' and 'Hemoglobin A1c' are the same, 'WBC' and 'White Blood Cell Count' are identical, regardless of capitalization or spacing).  
2. **Lab Value**: The value must match numerically. Treat numbers as equal regardless of:  
- Trailing zeros or decimal places if the precision is effectively the same (e.g., '35.0' equals '35', '5.0' equals '5', '7.10' equals '7.1').  
- Decimal point vs. comma separators (e.g., '5.09' equals '5,09', '6.1' equals '6,1').  
- Special case: If the image shows 'negative' and the CSV shows `lab_unit = N/A; lab_min_range = 0; lab_max_range = 9999`, treat this as a match.  
3. **Unit**: The unit must represent the same measurement, even if the formatting differs slightly (e.g., 'mg/dL' and 'mg/dl' are equivalent).  
4. **Reference Range**: The minimum and maximum values of the reference range must match exactly as shown in the image.

**Additional Guidance**:  
- If a lab test in the image has no matching entry in the CSV, or if any of the above criteria are not met, record it as a discrepancy.  
- If multiple tests in the image have similar names (e.g., sub-tests in a panel), ensure each is matched correctly based on its value, unit, and reference range.  
- Ignore headers, footers, or non-test information in the image (e.g., patient details, dates, or flags like 'H' or 'L').  
- The CSV may include extra tests not in the image; these can be ignored for this verification.

For each discrepancy, provide:  
1. The test name  
2. What the image shows (value/unit/range)  
3. What the CSV shows (value/unit/range)  

Provide your confidence in this assessment as a percentage (0-100%), considering any ambiguities or potential errors.

**CSV Data (semicolon-separated)**:  
{csv_data}
""".strip()
    
    message = client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=2000,
        tools=TOOLS,
        tool_choice={
            "type": "tool", 
            "name": "record_verification_result"
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}}
                ]
            }
        ]
    )
    
    result = None
    for content in message.content:
        if not hasattr(content, "input"): continue
        result = content.input
        break
    
    assert result, "No response from Claude"

    if not result["valid"]:
        print(image_path)
        print(json.dumps(result, indent=2))
        errors_path = image_path.replace(".jpg", ".errors.json")
        with open(errors_path, 'w', 'utf8') as f:
            json.dump(result, f, indent=2)
        print(errors_path)

    return result

def main():
    parser = argparse.ArgumentParser(description="Verify lab test extraction using Claude")
    parser.add_argument("--api-key", help="Anthropic API key")
    parser.add_argument("--output-dir", default="./output", help="Directory to scan for lab test files")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Anthropic API key is required. Provide it with --api-key or set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)
    
    # Initialize Claude client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Find all JPG/CSV pairs
    output_dir = Path(args.output_dir)
    file_pairs = find_lab_file_pairs(output_dir)
    
    if not file_pairs:
        print(f"No matching JPG/CSV file pairs found in {output_dir}")
        return
    
    print(f"Found {len(file_pairs)} JPG/CSV file pairs")
    
    # Limit number of files if requested
    if args.limit and args.limit > 0:
        file_pairs = file_pairs[:args.limit]
        print(f"Processing first {args.limit} pairs")
    
    # Process each pair
    results = []
    for jpg_path, csv_path in tqdm(file_pairs, desc="Verifying lab tests"):
        with open(csv_path, "r") as f:
            csv_data = f.read()

        # Using the function call approach
        verify_with_claude(jpg_path, csv_data, client)
        
if __name__ == "__main__":
    main()
