"""Lab result extraction from images using vision models."""

import json
import re
import base64
import logging
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field
from openai import OpenAI, APIError

from utils import parse_llm_json_response

logger = logging.getLogger(__name__)


# ========================================
# Pydantic Models
# ========================================

class LabResult(BaseModel):
    """Single lab test result - optimized for extraction accuracy."""

    # Raw extraction (exactly as shown in PDF)
    lab_name_raw: str = Field(
        description="Test name EXACTLY as written in the PDF. Preserve all spacing, capitalization, symbols, and formatting."
    )
    value_raw: Optional[float] = Field(
        default=None,
        description="Numeric result value. For text results (Positive/Negative), use 1/0 or leave null and put in comments."
    )
    lab_unit_raw: Optional[str] = Field(
        default=None,
        description="Unit EXACTLY as written in PDF (preserve case, spacing, symbols)."
    )
    reference_range: Optional[str] = Field(
        default=None,
        description="Complete reference range text EXACTLY as shown."
    )
    reference_min_raw: Optional[float] = Field(
        default=None,
        description="Minimum reference value (extract number from reference_range if available)"
    )
    reference_max_raw: Optional[float] = Field(
        default=None,
        description="Maximum reference value (extract number from reference_range if available)"
    )
    is_abnormal: Optional[bool] = Field(
        default=None,
        description="Whether result is marked/flagged as abnormal in PDF"
    )
    comments: Optional[str] = Field(
        default=None,
        description="Any notes, comments, or qualitative results"
    )
    source_text: str = Field(
        description="Exact row or section from PDF containing this result"
    )

    # Internal fields (added by pipeline, not by LLM)
    page_number: Optional[int] = Field(default=None, ge=1, description="Page number in PDF")
    source_file: Optional[str] = Field(default=None, description="Source file identifier")
    lab_name_standardized: Optional[str] = Field(default=None, description="Standardized lab name")
    lab_unit_standardized: Optional[str] = Field(default=None, description="Standardized lab unit")


class HealthLabReport(BaseModel):
    """Document-level lab report metadata."""

    collection_date: Optional[str] = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Specimen collection date in YYYY-MM-DD format"
    )
    report_date: Optional[str] = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Report issue date in YYYY-MM-DD format"
    )
    lab_facility: Optional[str] = Field(
        default=None,
        description="Name of laboratory that performed tests"
    )
    lab_results: List[LabResult] = Field(
        default_factory=list,
        description="List of all lab test results extracted from this page/document"
    )
    source_file: Optional[str] = Field(default=None, description="Source PDF filename")

    def normalize_empty_optionals(self):
        """Convert empty strings to None for optional fields."""
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            field_info = self.model_fields[field_name]
            is_optional_type = field_info.is_required() is False
            if value == "" and is_optional_type:
                setattr(self, field_name, None)

        for lab_result in self.lab_results:
            for field_name in lab_result.model_fields:
                value = getattr(lab_result, field_name)
                field_info = lab_result.model_fields[field_name]
                is_optional_type = field_info.is_required() is False
                if value == "" and is_optional_type:
                    setattr(lab_result, field_name, None)


# Tool definition for function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_lab_results",
            "description": "Extracts lab results from medical report image",
            "parameters": HealthLabReport.model_json_schema()
        }
    }
]


# ========================================
# Extraction Prompts
# ========================================

EXTRACTION_SYSTEM_PROMPT = """
You are a medical lab report data extractor. Your PRIMARY goal is ACCURACY - extract exactly what you see in the lab report image.

CRITICAL RULES:
1. COPY, DON'T INTERPRET: Extract test names, values, and units EXACTLY as written in the image
   - Preserve capitalization, spacing, symbols, punctuation
   - Do NOT standardize, translate, or normalize
   - Example: If it says "Hemoglobina A1c", write "Hemoglobina A1c" (not "Hemoglobin A1c")
   - Example: If it says "mg/dl", write "mg/dl" (not "mg/dL")

2. COMPLETENESS: Extract ALL test results from the image
   - Process line by line
   - Don't skip any tests, including qualitative results
   - If a row has MULTIPLE numeric values with different units, extract each as a SEPARATE result

3. TEST NAMES WITH CONTEXT:
   - Include section headers as prefixes for clarity
   - Example: If you see "BILIRRUBINAS" as header and "Total" below it, use "BILIRRUBINAS - Total"

4. NUMERIC vs QUALITATIVE VALUES:
   - `value`: ONLY for numeric results (e.g., 5.0, 14.2, 1.74)
   - For TEXT-ONLY results (e.g., "AMARELA", "NAO CONTEM", "NORMAL", "POSITIVE", "NEGATIVE"):
     * Set `value` to null (None)
     * Put the text result in `comments`
   - For RANGE results (e.g., "1 a 2/campo"):
     * Set `value` to null (None)
     * Put the full text in `comments`
     * Extract any unit visible in the text (e.g., "/campo" from "1 a 2/campo")
   - `unit`: Extract the unit EXACTLY as shown in the document
     * Copy the unit symbol or abbreviation exactly
     * If NO unit is visible or implied in the document → use null
     * Do NOT infer or normalize units - just extract what you see

5. REFERENCE RANGES:
   - `reference_range`: Copy the complete reference range text
   - `reference_min` / `reference_max`: Parse numeric bounds from reference_range if possible

6. FLAGS & CONTEXT:
   - `is_abnormal`: Set to true if result is marked (H, L, *, ↑, ↓, "HIGH", "LOW", etc.)
   - `comments`: Capture any notes, qualitative results, or text values

7. TRACEABILITY:
   - `source_text`: Copy the exact row/line containing this result

8. DATES: Format as YYYY-MM-DD or leave null

SCHEMA FIELD NAMES:
- Use `lab_name_raw` (raw test name from PDF)
- Use `value_raw` (raw numeric value)
- Use `lab_unit_raw` (raw unit from PDF)

Remember: Your job is to be a perfect copier, not an interpreter. Extract EVERYTHING, even qualitative results.
""".strip()

EXTRACTION_USER_PROMPT = """
Please extract all lab test results from this medical lab report image.
Extract test names, values, units, and reference ranges exactly as they appear.
Pay special attention to preserving the exact formatting and symbols.
""".strip()


# ========================================
# Self-Consistency
# ========================================

def self_consistency(fn, model_id, n, *args, **kwargs):
    """
    Run a function multiple times and vote on the best result.

    Args:
        fn: Function to run
        model_id: Model to use for voting
        n: Number of times to run the function
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Tuple of (best_result, all_results)
    """
    if n == 1:
        result = fn(*args, **kwargs)
        return result, [result]

    results = []
    effective_kwargs = kwargs.copy()
    if 'temperature' in fn.__code__.co_varnames and 'temperature' not in effective_kwargs:
        effective_kwargs['temperature'] = 0.5

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(fn, *args, **effective_kwargs) for _ in range(n)]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Error during self-consistency task execution: {e}")
                for f_cancel in futures:
                    if not f_cancel.done():
                        f_cancel.cancel()
                raise

    if not results:
        raise RuntimeError("All self-consistency calls failed.")

    # If all results are identical, return the first
    if all(r == results[0] for r in results):
        return results[0], results

    # Vote on best result using LLM
    return vote_on_best_result(results, model_id, fn.__name__)


def vote_on_best_result(results: list, model_id: str, fn_name: str):
    """Use LLM to vote on the most consistent result."""
    from openai import OpenAI
    import os

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    system_prompt = (
        "You are an expert at comparing multiple outputs of the same extraction task. "
        "We have extracted several samples from the same prompt in order to average out any errors or inconsistencies. "
        "Your job is to select the output that is most consistent with the majority of the provided samples. "
        "Prioritize agreement on extracted content (test names, values, units, reference ranges, etc.). "
        "Ignore formatting, whitespace, and layout differences. "
        "Return ONLY the best output, verbatim, with no extra commentary. "
        "Do NOT include any delimiters, output numbers, or extra labels in your response."
    )

    prompt = "".join(
        f"--- Output {i+1} ---\n{json.dumps(v, ensure_ascii=False) if type(v) in [list, dict] else v}\n\n"
        for i, v in enumerate(results)
    )

    voted_raw = None
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        voted_raw = completion.choices[0].message.content.strip()

        if fn_name == 'extract_labs_from_page_image':
            voted_result = parse_llm_json_response(voted_raw, fallback=None)
            if voted_result:
                return voted_result, results
            else:
                logger.error("Failed to parse voted result as JSON")
                return results[0], results
        else:
            return voted_raw, results

    except Exception as e:
        logger.error(f"Error during self-consistency voting: {e}")
        return results[0], results


# ========================================
# Extraction Function
# ========================================

def extract_labs_from_page_image(
    image_path: Path,
    model_id: str,
    client: OpenAI,
    temperature: float = 0.3
) -> dict:
    """
    Extract lab results from a page image using vision model.

    Args:
        image_path: Path to the preprocessed page image
        model_id: Vision model to use for extraction
        client: OpenAI client instance
        temperature: Temperature for sampling

    Returns:
        Dictionary with extracted report data (validated by Pydantic)
    """
    with open(image_path, "rb") as img_file:
        img_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": EXTRACTION_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
                ]}
            ],
            temperature=temperature,
            max_tokens=8000,
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "extract_lab_results"}}
        )
    except APIError as e:
        logger.error(f"API Error during lab extraction from {image_path.name}: {e}")
        raise RuntimeError(f"Lab extraction failed for {image_path.name}: {e}")

    # Check for valid response structure
    if not completion or not completion.choices or len(completion.choices) == 0:
        logger.error(f"Invalid completion response structure")
        return HealthLabReport(lab_results=[]).model_dump(mode='json')

    if not completion.choices[0].message.tool_calls:
        logger.warning(f"No tool call by model for lab extraction from {image_path.name}")
        return HealthLabReport(lab_results=[]).model_dump(mode='json')

    tool_args_raw = completion.choices[0].message.tool_calls[0].function.arguments
    try:
        tool_result_dict = json.loads(tool_args_raw)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for tool args: {e}")
        return HealthLabReport(lab_results=[]).model_dump(mode='json')

    # Pre-process: Fix common LLM issues before Pydantic validation (using LLM for soft parsing)
    tool_result_dict = _fix_lab_results_format(tool_result_dict, client, model_id)

    # Validate with Pydantic
    try:
        report_model = HealthLabReport(**tool_result_dict)
        report_model.normalize_empty_optionals()
        return report_model.model_dump(mode='json')
    except Exception as e:
        num_results = len(tool_result_dict.get("lab_results", []))
        logger.error(f"Model validation error for report with {num_results} lab_results: {e}")
        # Try to salvage individual results
        return _salvage_lab_results(tool_result_dict)


def _normalize_date_format(date_str: Optional[str]) -> Optional[str]:
    """
    Normalize date strings to YYYY-MM-DD format.

    Handles common formats:
    - DD/MM/YYYY (e.g., 20/11/2024 -> 2024-11-20)
    - DD-MM-YYYY (e.g., 20-11-2024 -> 2024-11-20)
    - YYYY-MM-DD (already correct)

    Args:
        date_str: Date string in various formats

    Returns:
        Date string in YYYY-MM-DD format, or None if invalid/null
    """
    if not date_str or date_str == "0000-00-00":
        return None

    # Already in correct format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    # DD/MM/YYYY or DD-MM-YYYY format
    match = re.match(r"^(\d{2})[/-](\d{2})[/-](\d{4})$", date_str)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month}-{day}"

    # If we can't parse it, log and return None
    logger.warning(f"Unable to normalize date format: {date_str}")
    return None


def _fix_lab_results_format(tool_result_dict: dict, client: OpenAI, model_id: str) -> dict:
    """Fix common LLM formatting issues in lab_results and dates using LLM-based parsing."""
    # Fix date formats at report level
    for date_field in ["collection_date", "report_date"]:
        if date_field in tool_result_dict:
            tool_result_dict[date_field] = _normalize_date_format(tool_result_dict[date_field])

    if "lab_results" not in tool_result_dict or not isinstance(tool_result_dict["lab_results"], list):
        return tool_result_dict

    # Collect all string-based lab results that need parsing
    string_results = []
    string_indices = []
    parsed_lab_results = []

    for i, lr_data in enumerate(tool_result_dict["lab_results"]):
        if isinstance(lr_data, str):
            # Try to parse as JSON first (simple case)
            try:
                parsed_lr = json.loads(lr_data)
                parsed_lab_results.append(parsed_lr)
                continue
            except json.JSONDecodeError:
                pass

            # Collect for LLM-based parsing
            string_results.append(lr_data)
            string_indices.append(i)
            parsed_lab_results.append(None)  # Placeholder
        else:
            parsed_lab_results.append(lr_data)

    # If we have string results, use LLM to parse them
    if string_results:
        logger.info(f"Found {len(string_results)} lab results as strings, using LLM to parse them")
        parsed = _parse_string_results_with_llm(string_results, client, model_id)

        # Insert parsed results back
        skipped_count = 0
        for idx, parsed_result in zip(string_indices, parsed):
            if parsed_result:
                parsed_lab_results[idx] = parsed_result
            else:
                # Failed to parse, remove from results
                logger.debug(f"Failed to parse lab_result[{idx}], skipping. Data: {string_results[string_indices.index(idx)][:200]}")
                skipped_count += 1

        # Log summary if any failed
        if skipped_count > 0:
            logger.info(
                f"Skipped {skipped_count}/{len(string_results)} unparseable lab results. "
                f"This is normal during self-consistency voting - other extraction attempts may have proper structure."
            )

    # Remove None placeholders
    parsed_lab_results = [r for r in parsed_lab_results if r is not None]

    tool_result_dict["lab_results"] = parsed_lab_results
    return tool_result_dict


def _parse_string_results_with_llm(string_results: List[str], client: OpenAI, model_id: str) -> List[Optional[dict]]:
    """
    Use LLM to parse string-formatted lab results into structured format.

    This is a soft parser that uses AI to extract structure from any text format,
    avoiding hardcoded regex patterns.

    Args:
        string_results: List of lab result strings to parse
        client: OpenAI client instance
        model_id: Model to use for parsing

    Returns:
        List of parsed dictionaries (None for unparseable results)
    """
    if not string_results:
        return []

    # Build prompt for LLM to parse the strings
    prompt = f"""You are parsing lab test result strings into structured format.

For each string below, extract:
- lab_name_raw: The name of the lab test
- value_raw: Numeric value (null if text-only result like "Negative")
- lab_unit_raw: Unit of measurement (null if none)
- reference_range: Reference range text (null if none)
- reference_min_raw: Min reference value (null if not available)
- reference_max_raw: Max reference value (null if not available)
- comments: Any qualitative results or notes
- source_text: The original string

Input strings:
{json.dumps(string_results, indent=2, ensure_ascii=False)}

Return a JSON array of parsed lab results matching the LabResult schema.
Each result must have at minimum: lab_name_raw, value_raw, lab_unit_raw, and source_text fields."""

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a medical data parser. Parse lab result strings into structured JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        response_text = completion.choices[0].message.content
        parsed_json = json.loads(response_text)

        # Handle different possible response formats
        if isinstance(parsed_json, dict) and "results" in parsed_json:
            results = parsed_json["results"]
        elif isinstance(parsed_json, dict) and "lab_results" in parsed_json:
            results = parsed_json["lab_results"]
        elif isinstance(parsed_json, list):
            results = parsed_json
        else:
            logger.error(f"Unexpected LLM response format for string parsing: {parsed_json}")
            return [None] * len(string_results)

        # Validate we got the right number of results
        if len(results) != len(string_results):
            logger.warning(f"LLM returned {len(results)} results but expected {len(string_results)}")
            # Pad or trim to match
            while len(results) < len(string_results):
                results.append(None)
            results = results[:len(string_results)]

        return results

    except Exception as e:
        logger.error(f"Failed to parse string results with LLM: {e}")
        return [None] * len(string_results)


def _salvage_lab_results(tool_result_dict: dict) -> dict:
    """Try to salvage valid lab results even if report validation fails."""
    if "lab_results" not in tool_result_dict or not isinstance(tool_result_dict["lab_results"], list):
        return HealthLabReport(lab_results=[]).model_dump(mode='json')

    valid_results = []
    for i, lr_data in enumerate(tool_result_dict["lab_results"]):
        if isinstance(lr_data, str):
            logger.warning(f"Skipping string lab_result[{i}] in salvage: {lr_data[:100]}")
            continue
        try:
            lr_model = LabResult(**lr_data)
            valid_results.append(lr_model.model_dump(mode='json'))
        except Exception as e:
            logger.warning(f"Failed to validate lab_result[{i}] in salvage: {e}. Data: {lr_data}")

    logger.info(f"Salvaged {len(valid_results)}/{len(tool_result_dict['lab_results'])} lab results")
    return HealthLabReport(lab_results=valid_results).model_dump(mode='json')
