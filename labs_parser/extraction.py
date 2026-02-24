"""Lab result extraction from images using vision models."""

import json
import re
import base64
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field, field_validator
from openai import OpenAI, APIError

from labs_parser.utils import parse_llm_json_response

logger = logging.getLogger(__name__)

# Sentinel for unknown standardized values (imported from config to avoid circular import)
_UNKNOWN_VALUE = "$UNKNOWN$"


# ========================================
# Pydantic Models
# ========================================


class LabResult(BaseModel):
    """Single lab test result - optimized for extraction accuracy."""

    # Raw extraction (exactly as shown in PDF)
    lab_name_raw: str = Field(
        description="Test name ONLY as written in the PDF. Must contain ONLY the test name - "
        "DO NOT include values, units, reference ranges, or field labels. "
        "WRONG: 'Glucose, value_raw: 100' CORRECT: 'Glucose'"
    )
    value_raw: str | None = Field(
        default=None,
        description="Result value ONLY. Must contain ONLY the numeric or text result - "
        "DO NOT include test names, units, or field labels. "
        "Examples: '5.2', '14.8', 'NEGATIVO', 'POSITIVO'",
    )
    lab_unit_raw: str | None = Field(
        default=None,
        description="Unit ONLY as written in PDF. Must contain ONLY the unit symbol - "
        "DO NOT include values or test names. Examples: 'mg/dL', '%', 'U/L'",
    )
    reference_range: str | None = Field(
        default=None, description="Complete reference range text EXACTLY as shown."
    )
    reference_notes: str | None = Field(
        default=None,
        description="Any notes, comments, or additional context about the reference range. "
        "Examples: 'Confirmado por duplo ensaio', 'Criança<400', 'valores podem variar'. "
        "Put methodology notes, population-specific ranges, or validation comments HERE.",
    )
    reference_min_raw: float | None = Field(
        default=None,
        description="Minimum reference value as a PLAIN NUMBER ONLY. Parse from reference_range. "
        "Put any comments or notes in reference_notes instead. "
        "Examples: '< 40' → null, '150 - 400' → 150, '26.5-32.6' → 26.5",
    )
    reference_max_raw: float | None = Field(
        default=None,
        description="Maximum reference value as a PLAIN NUMBER ONLY. Parse from reference_range. "
        "Put any comments or notes in reference_notes instead. "
        "Examples: '< 40' → 40, '150 - 400' → 400, '26.5-32.6' → 32.6",
    )
    is_abnormal: bool | None = Field(
        default=None, description="Whether result is marked/flagged as abnormal in PDF"
    )
    comments: str | None = Field(
        default=None,
        description="Additional notes or remarks about the test (NOT the test result itself). Only use for extra information like methodology notes or special conditions.",
    )
    source_text: str | None = Field(
        default="", description="Exact row or section from PDF containing this result"
    )

    # Internal fields (added by pipeline, not by LLM)
    page_number: int | None = Field(
        default=None, ge=1, description="Page number in PDF"
    )
    source_file: str | None = Field(default=None, description="Source file identifier")
    lab_name_standardized: str | None = Field(
        default=None, description="Standardized lab name"
    )
    lab_unit_standardized: str | None = Field(
        default=None, description="Standardized lab unit"
    )

    # Review tracking fields (all prefixed with review_)
    result_index: int | None = Field(
        default=None,
        description="Index of this result in the source JSON lab_results array",
    )
    review_needed: bool | None = Field(
        default=False,
        description="Whether this result needs human review (auto-flagged)",
    )
    review_reason: str | None = Field(
        default=None, description="Reason why review is needed (auto-generated)"
    )
    review_confidence: float | None = Field(
        default=1.0, description="Confidence score 0-1 (auto-generated)"
    )
    review_status: str | None = Field(
        default=None, description="Human review status: 'accepted', 'rejected', or null"
    )
    review_completed_at: str | None = Field(
        default=None, description="ISO timestamp when review was completed"
    )

    @field_validator("value_raw", mode="before")
    @classmethod
    def coerce_value_raw_to_string(cls, v):
        """Coerce numeric values to strings - LLMs often return floats instead of strings."""
        if v is None:
            return v
        if isinstance(v, (int, float)):
            # Format without unnecessary decimal places for integers
            return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
        return v

    @field_validator("lab_name_raw", mode="before")
    @classmethod
    def validate_lab_name_raw(cls, v):
        """Reject malformed lab names with embedded metadata.

        Previously this cleaned malformed entries. Now it fails fast
        to force investigation when the model misbehaves.
        """
        if v is None:
            return v

        v_str = str(v)

        # Check for embedded metadata patterns
        malformed_patterns = [
            "value_raw:",
            "lab_unit_raw:",
            "reference_range:",
            "source_text:",
            "reference_min_raw:",
            "reference_max_raw:",
        ]
        v_lower = v_str.lower()

        if any(pattern in v_lower for pattern in malformed_patterns):
            raise ValueError(
                f"MALFORMED OUTPUT: lab_name_raw contains embedded field data. "
                f"Model returned: '{v_str[:100]}...'. "
                f"This indicates the extraction prompt needs improvement."
            )

        return v_str


class HealthLabReport(BaseModel):
    """Document-level lab report metadata."""

    collection_date: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Specimen collection date in YYYY-MM-DD format",
    )
    report_date: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Report issue date in YYYY-MM-DD format",
    )
    lab_facility: str | None = Field(
        default=None, description="Name of laboratory that performed tests"
    )
    page_has_lab_data: bool | None = Field(
        default=None,
        description="True if page contains lab test results, False if page is cover/instructions/administrative with no lab data",
    )
    lab_results: list[LabResult] = Field(
        default_factory=list,
        description="List of all lab test results extracted from this page/document",
    )
    source_file: str | None = Field(default=None, description="Source PDF filename")

    @staticmethod
    def _clear_empty_strings(model: BaseModel):
        """Set empty string optional fields to None on a Pydantic model."""
        for field_name in model.model_fields:
            value = getattr(model, field_name)
            field_info = model.model_fields[field_name]
            if value == "" and not field_info.is_required():
                setattr(model, field_name, None)

    def normalize_empty_optionals(self):
        """Convert empty strings to None for optional fields."""
        self._clear_empty_strings(self)
        for lab_result in self.lab_results:
            self._clear_empty_strings(lab_result)


# ========================================
# LLM-Facing Schema (subset of full model)
# ========================================


class LabResultExtraction(BaseModel):
    """LLM-facing schema — only fields the model should populate.

    Excludes pipeline-internal fields (review_*, page_number, source_file,
    result_index, lab_name_standardized, lab_unit_standardized) to reduce
    token usage and avoid confusing the model.
    """

    lab_name_raw: str = Field(
        description="Test name ONLY as written in the PDF. Must contain ONLY the test name - "
        "DO NOT include values, units, reference ranges, or field labels. "
        "WRONG: 'Glucose, value_raw: 100' CORRECT: 'Glucose'"
    )
    lab_name: str | None = Field(
        default=None,
        description="Standardized lab name from the provided list. "
        "Match the raw name to the CLOSEST entry. Use '$UNKNOWN$' if no match.",
    )
    value_raw: str | None = Field(
        default=None,
        description="Result value ONLY. Must contain ONLY the numeric or text result - "
        "DO NOT include test names, units, or field labels. "
        "Examples: '5.2', '14.8', 'NEGATIVO', 'POSITIVO'",
    )
    lab_unit_raw: str | None = Field(
        default=None,
        description="Unit ONLY as written in PDF. Must contain ONLY the unit symbol - "
        "DO NOT include values or test names. Examples: 'mg/dL', '%', 'U/L'",
    )
    unit: str | None = Field(
        default=None,
        description="Standardized unit from the provided list for this lab. "
        "Normalize FORMAT only (e.g., 'mg/dl' → 'mg/dL'). Do NOT convert units. "
        "Use '$UNKNOWN$' if no match.",
    )
    reference_range: str | None = Field(
        default=None, description="Complete reference range text EXACTLY as shown."
    )
    reference_notes: str | None = Field(
        default=None,
        description="Any notes, comments, or additional context about the reference range.",
    )
    reference_min_raw: float | None = Field(
        default=None,
        description="Minimum reference value as a PLAIN NUMBER ONLY. Parse from reference_range.",
    )
    reference_max_raw: float | None = Field(
        default=None,
        description="Maximum reference value as a PLAIN NUMBER ONLY. Parse from reference_range.",
    )
    is_abnormal: bool | None = Field(
        default=None, description="Whether result is marked/flagged as abnormal in PDF"
    )
    comments: str | None = Field(
        default=None,
        description="Additional notes or remarks about the test (NOT the test result itself).",
    )
    source_text: str | None = Field(
        default="", description="Exact row or section from PDF containing this result"
    )


class HealthLabReportExtraction(BaseModel):
    """LLM-facing schema for function calling.

    Excludes pipeline-internal fields from HealthLabReport. Uses LabResultExtraction
    to reduce schema size (~41% smaller) and avoid confusing the model.
    """

    collection_date: str | None = Field(
        default=None,
        description="Specimen collection date in YYYY-MM-DD format",
    )
    report_date: str | None = Field(
        default=None,
        description="Report issue date in YYYY-MM-DD format",
    )
    lab_facility: str | None = Field(
        default=None, description="Name of laboratory that performed tests"
    )
    page_has_lab_data: bool | None = Field(
        default=None,
        description="True if page contains lab test results, False if page is cover/instructions/administrative",
    )
    lab_results: list[LabResultExtraction] = Field(
        default_factory=list,
        description="List of all lab test results extracted from this page/document",
    )
    source_file: str | None = Field(default=None, description="Source PDF filename")


# Tool definition for function calling — uses LLM-facing schema to reduce token usage
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_lab_results",
            "description": "Extracts lab results from medical report image",
            "parameters": HealthLabReportExtraction.model_json_schema(),
        },
    }
]


def _build_standardized_names_section(standardized_names: list[str], lab_specs_data: dict) -> str:
    """Build the standardized names reference section for the extraction prompt.

    Args:
        standardized_names: Sorted list of all standardized lab names
        lab_specs_data: Raw lab specs dict for looking up primary units and lab types

    Returns:
        Formatted string to append to the system prompt
    """
    sections: dict[str, list[str]] = {"blood": [], "urine": [], "feces": [], "saliva": [], "other": []}

    for name in standardized_names:
        spec = lab_specs_data.get(name, {})
        primary_unit = spec.get("primary_unit", "")
        lab_type = spec.get("lab_type", "blood")
        entry = f"- {name} [{primary_unit}]"
        sections.setdefault(lab_type, []).append(entry)

    result = "\n\nSTANDARDIZED LAB NAMES AND UNITS:\n"
    for lab_type in ["blood", "urine", "feces", "saliva", "other"]:
        entries = sections.get(lab_type, [])
        if entries:
            result += f"\n### {lab_type.title()}\n" + "\n".join(entries) + "\n"

    return result


# ========================================
# Extraction Prompts
# ========================================

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
EXTRACTION_SYSTEM_PROMPT = (
    (_PROMPTS_DIR / "extraction_system.txt").read_text(encoding="utf-8").strip()
)
EXTRACTION_USER_PROMPT = (
    (_PROMPTS_DIR / "extraction_user.txt").read_text(encoding="utf-8").strip()
)


def _empty_report() -> dict:
    """Return a fresh empty lab report dict."""
    return HealthLabReport(lab_results=[]).model_dump(mode="json")


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

    # Fixed temperature for i.i.d. sampling (aligned with self-consistency research)
    # T=0.5 provides good diversity without being too creative
    SELF_CONSISTENCY_TEMPERATURE = 0.5

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = []
        for i in range(n):
            effective_kwargs = kwargs.copy()
            # Use fixed temperature if function accepts it and not already set
            if "temperature" in fn.__code__.co_varnames and "temperature" not in kwargs:
                effective_kwargs["temperature"] = SELF_CONSISTENCY_TEMPERATURE
            futures.append(executor.submit(fn, *args, **effective_kwargs))

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
    import os

    client = OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
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
        f"--- Output {i + 1} ---\n{json.dumps(v, ensure_ascii=False) if type(v) in [list, dict] else v}\n\n"
        for i, v in enumerate(results)
    )

    voted_raw = None
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        voted_raw = completion.choices[0].message.content.strip()

        if fn_name == "extract_labs_from_page_image":
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
    temperature: float = 0.0,
    max_retries: int = 3,
    temperature_step: float = 0.2,
    standardization_section: str | None = None,
) -> dict:
    """
    Extract lab results from a page image using vision model.

    Uses temperature escalation retry strategy for malformed outputs.
    When validation fails due to malformed LLM output (e.g., all fields
    concatenated into a single string), retries with higher temperature
    to get the model to produce properly structured output.

    Args:
        image_path: Path to the preprocessed page image
        model_id: Vision model to use for extraction
        client: OpenAI client instance
        temperature: Initial temperature for sampling (default: 0.0)
        max_retries: Maximum retry attempts with escalating temperature (default: 3)
        temperature_step: Temperature increment per retry (default: 0.2)
        standardization_section: Optional standardized names/units list to append to system prompt.
            When provided, the model also populates lab_name and unit fields inline.

    Returns:
        Dictionary with extracted report data (validated by Pydantic).
        On failure, includes _extraction_failed=True and _failure_reason.
    """
    with open(image_path, "rb") as img_file:
        img_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

    effective_system_prompt = (
        EXTRACTION_SYSTEM_PROMPT + standardization_section
        if standardization_section
        else EXTRACTION_SYSTEM_PROMPT
    )

    current_temp = temperature
    last_error = None

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        if attempt > 0:
            current_temp = temperature + (attempt * temperature_step)
            logger.info(
                f"[{image_path.name}] Retry {attempt}/{max_retries} with temperature={current_temp:.1f} "
                f"(previous error: malformed output)"
            )

        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": effective_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": EXTRACTION_USER_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_data}"
                                },
                            },
                        ],
                    },
                ],
                temperature=current_temp,
                max_tokens=16384,
                tools=TOOLS,
                tool_choice={
                    "type": "function",
                    "function": {"name": "extract_lab_results"},
                },
            )
        except APIError as e:
            logger.error(f"API Error during lab extraction from {image_path.name}: {e}")
            raise RuntimeError(f"Lab extraction failed for {image_path.name}: {e}")

        # Check for valid response structure
        if not completion or not completion.choices:
            logger.error(f"Invalid completion response structure")
            return _empty_report()

        if not completion.choices[0].message.tool_calls:
            logger.warning(
                f"No tool call by model for lab extraction from {image_path.name}"
            )
            return _empty_report()

        tool_args_raw = completion.choices[0].message.tool_calls[0].function.arguments
        try:
            tool_result_dict = json.loads(tool_args_raw)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for tool args: {e}")
            return _empty_report()

        # Pre-process: Fix common LLM issues before Pydantic validation (using LLM for soft parsing)
        tool_result_dict = _fix_lab_results_format(tool_result_dict, client, model_id)

        # Validate with Pydantic
        try:
            report_model = HealthLabReport(**tool_result_dict)
            report_model.normalize_empty_optionals()

            # Check for extraction quality - warn if most values are null
            if report_model.lab_results:
                null_count = sum(
                    1 for r in report_model.lab_results if r.value_raw is None
                )
                total_count = len(report_model.lab_results)
                null_pct = (null_count / total_count * 100) if total_count > 0 else 0

                if null_pct > 50:
                    logger.warning(
                        f"Extraction quality issue: {null_count}/{total_count} ({null_pct:.0f}%) lab results have null values. "
                        f"This suggests the model failed to extract numeric values from the image.\n"
                        f"\t- {image_path}"
                    )
            else:
                if report_model.page_has_lab_data is False:
                    logger.debug(
                        f"Page confirmed to have no lab data:\n\t- {image_path}"
                    )
                else:
                    logger.warning(
                        f"Extraction returned 0 lab results. "
                        f"This may indicate a model extraction failure - image should be manually reviewed.\n"
                        f"\t- {image_path}"
                    )

            # Success - return the validated result
            result = report_model.model_dump(mode="json")
            if attempt > 0:
                logger.info(
                    f"[{image_path.name}] Extraction succeeded on retry {attempt} with temp={current_temp:.1f}"
                )
            return result

        except ValueError as e:
            error_msg = str(e)
            # Check if this is a malformed output error that warrants retry
            if "MALFORMED OUTPUT" in error_msg:
                last_error = error_msg
                num_results = len(tool_result_dict.get("lab_results", []))
                logger.warning(
                    f"[{image_path.name}] Malformed output detected ({num_results} results), "
                    f"attempt {attempt + 1}/{max_retries + 1}"
                )
                # Continue to next retry attempt
                continue
            else:
                # Non-retryable validation error - try to salvage
                num_results = len(tool_result_dict.get("lab_results", []))
                logger.error(
                    f"Model validation error for report with {num_results} lab_results: {e}"
                )
                return _salvage_lab_results(tool_result_dict)

        except Exception as e:
            num_results = len(tool_result_dict.get("lab_results", []))
            logger.error(
                f"Model validation error for report with {num_results} lab_results: {e}"
            )
            # Try to salvage individual results
            return _salvage_lab_results(tool_result_dict)

    # All retries exhausted - return failure marker
    logger.error(
        f"[{image_path.name}] Extraction failed after {max_retries + 1} attempts. "
        f"Last error: {last_error[:200] if last_error else 'unknown'}"
    )
    result = _empty_report()
    result["_extraction_failed"] = True
    result["_failure_reason"] = f"Malformed output after {max_retries + 1} attempts"
    result["_retry_count"] = max_retries + 1
    return result


def extract_labs_from_text(
    text: str,
    model_id: str,
    client: OpenAI,
    temperature: float = 0.0,
    standardization_section: str | None = None,
) -> dict:
    """
    Extract lab results from text using a text-only LLM (no vision).

    This is a cost-optimized extraction path for PDFs with embedded text.
    Uses the same prompts and tool schema as vision extraction.

    Args:
        text: Extracted text from PDF (via pdftotext)
        model_id: Model to use for extraction
        client: OpenAI client instance
        temperature: Temperature for sampling
        standardization_section: Optional standardized names/units list to append to system prompt.

    Returns:
        Dictionary with extracted report data (validated by Pydantic)
    """
    effective_system_prompt = (
        EXTRACTION_SYSTEM_PROMPT + standardization_section
        if standardization_section
        else EXTRACTION_SYSTEM_PROMPT
    )

    # Use same prompts but adapted for text input
    std_reminder = (
        "\nAlso set lab_name (standardized name from the provided list) and unit (standardized unit format) for each result.\n"
        if standardization_section
        else ""
    )
    user_prompt = f"""Please extract ALL lab test results from this medical lab report text.

CRITICAL: For EACH lab test you find, you MUST extract:
1. lab_name_raw - The test name EXACTLY as shown (required)
2. value_raw - The result value EXACTLY as shown (ALWAYS PUT THE RESULT HERE - whether numeric or text)
3. lab_unit_raw - The unit EXACTLY as shown (extract what you see, can be null if no unit)
4. reference_range - The reference range text (if visible)
5. reference_min_raw and reference_max_raw - Parse the numeric bounds from the reference range
{std_reminder}
Extract test names, values, units, and reference ranges EXACTLY as they appear.
Pay special attention to preserving the exact formatting and symbols.

CRITICAL: Extract EVERY lab test you see, including:
- Numeric results → Put in value_raw (examples: "5.2", "14.8", "0.75")
- Text-based qualitative results → Put in value_raw (examples: "NEGATIVO", "POSITIVO", "NORMAL", "AMARELA", "NAO CONTEM", "AUSENTE", "PRESENTE")
- Range results → Put in value_raw (examples: "1 a 2", "1-5 / campo", "0-3 / campo")

IMPORTANT: The value_raw field should contain the ACTUAL TEST RESULT, whether it's a number or text.

Also set page_has_lab_data:
- true if this document contains lab test results
- false if this is a cover page, instructions, or administrative content with no lab tests

--- DOCUMENT TEXT ---
{text}
--- END OF DOCUMENT ---"""

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=16384,
            tools=TOOLS,
            tool_choice={
                "type": "function",
                "function": {"name": "extract_lab_results"},
            },
        )
    except APIError as e:
        logger.error(f"API Error during text-based lab extraction: {e}")
        raise RuntimeError(f"Text-based lab extraction failed: {e}")

    # Check for valid response structure
    if not completion or not completion.choices:
        logger.error("Invalid completion response structure for text extraction")
        return HealthLabReport(lab_results=[]).model_dump(mode="json")

    if not completion.choices[0].message.tool_calls:
        logger.warning("No tool call by model for text-based lab extraction")
        return HealthLabReport(lab_results=[]).model_dump(mode="json")

    tool_args_raw = completion.choices[0].message.tool_calls[0].function.arguments
    try:
        tool_result_dict = json.loads(tool_args_raw)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for text extraction tool args: {e}")
        return HealthLabReport(lab_results=[]).model_dump(mode="json")

    # Pre-process: Fix common LLM issues before Pydantic validation
    tool_result_dict = _fix_lab_results_format(tool_result_dict, client, model_id)

    # Validate with Pydantic
    try:
        report_model = HealthLabReport(**tool_result_dict)
        report_model.normalize_empty_optionals()

        if report_model.lab_results:
            logger.info(
                f"Text extraction successful: {len(report_model.lab_results)} lab results"
            )
        else:
            if report_model.page_has_lab_data is False:
                logger.debug("Document confirmed to have no lab data (text extraction)")
            else:
                logger.warning("Text extraction returned 0 lab results")

        return report_model.model_dump(mode="json")
    except Exception as e:
        num_results = len(tool_result_dict.get("lab_results", []))
        logger.error(
            f"Model validation error for text extraction with {num_results} lab_results: {e}"
        )
        return _salvage_lab_results(tool_result_dict)


def _normalize_date_format(date_str: str | None) -> str | None:
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


# Known LabResult field names for detecting flattened key-value format
_LAB_RESULT_FIELDS = {
    "lab_name_raw",
    "lab_name",   # NEW: standardized name from LLM
    "value_raw",
    "lab_unit_raw",
    "unit",       # NEW: standardized unit from LLM
    "reference_range",
    "reference_min_raw",
    "reference_max_raw",
    "is_abnormal",
    "comments",
    "source_text",
}


def _parse_labresult_repr(s: str) -> dict | None:
    """
    Parse Python repr() format of LabResult objects.

    Some LLMs return strings like:
    "LabResult(lab_name_raw='Glucose', value_raw='100', lab_unit_raw='mg/dL', ...)"

    Returns a dict if parseable, None otherwise.
    """
    if not s.startswith("LabResult("):
        return None

    try:
        # Extract the content inside LabResult(...)
        content = s[10:-1]  # Remove 'LabResult(' and ')'

        # Parse key=value pairs
        result = {}
        # Use regex to find key=value patterns, handling quoted strings and None values
        pattern = r"(\w+)=(?:'([^']*)'|\"([^\"]*)\"|(\d+\.?\d*)|None|(True|False))"
        for match in re.finditer(pattern, content):
            key = match.group(1)
            # Get the matched value from whichever group matched
            if match.group(2) is not None:  # Single-quoted string
                value = match.group(2)
            elif match.group(3) is not None:  # Double-quoted string
                value = match.group(3)
            elif match.group(4) is not None:  # Number
                value = (
                    float(match.group(4))
                    if "." in match.group(4)
                    else int(match.group(4))
                )
            elif match.group(5) is not None:  # True/False
                value = match.group(5) == "True"
            else:  # None
                value = None
            result[key] = value

        return result if "lab_name_raw" in result else None
    except Exception:
        return None


def _reassemble_flattened_key_values(items: list) -> list:
    """
    Detect and reassemble flattened key-value strings into proper objects.

    Some LLMs return lab_results as flattened key-value pairs:
    ['lab_name_raw: Glucose', 'value_raw: 100', 'lab_unit_raw: mg/dL', ...]

    This function detects this pattern and reassembles them into proper dicts.
    Returns the original list if the pattern is not detected.
    """
    if not items:
        return items

    # Check if this looks like flattened key-value format
    # Pattern: strings like "field_name: value" where field_name is a known LabResult field
    kv_pattern = re.compile(r"^(\w+):\s*(.*)$")

    # Count how many items match the pattern with known field names
    kv_matches = 0
    for item in items[: min(10, len(items))]:  # Check first 10 items
        if isinstance(item, str):
            match = kv_pattern.match(item)
            if match and match.group(1) in _LAB_RESULT_FIELDS:
                kv_matches += 1

    # If less than 50% match the pattern, not flattened format
    if kv_matches < len(items[: min(10, len(items))]) * 0.5:
        return items

    logger.info(f"Detected flattened key-value format, reassembling {len(items)} items")

    # Reassemble into objects - lab_name_raw starts a new record
    reassembled = []
    current_obj = {}

    for item in items:
        if not isinstance(item, str):
            # Non-string item, save current and add as-is
            if current_obj:
                reassembled.append(current_obj)
                current_obj = {}
            reassembled.append(item)
            continue

        match = kv_pattern.match(item)
        if not match:
            continue

        key, value = match.group(1), match.group(2).strip()

        # lab_name_raw starts a new record
        if key == "lab_name_raw" and current_obj:
            reassembled.append(current_obj)
            current_obj = {}

        # Convert 'null' strings to None, parse numbers for reference fields
        if value.lower() == "null" or value == "":
            value = None
        elif key in ("reference_min_raw", "reference_max_raw"):
            try:
                value = float(value) if value else None
            except (ValueError, TypeError):
                value = None
        elif key == "is_abnormal":
            value = value.lower() == "true" if value else None

        current_obj[key] = value

    # Don't forget the last object
    if current_obj:
        reassembled.append(current_obj)

    logger.info(
        f"Reassembled {len(items)} key-value items into {len(reassembled)} lab results"
    )
    return reassembled


def _clean_numeric_field(value) -> float | None:
    """Strip embedded metadata from numeric fields.

    Some LLMs embed extra field data into numeric reference fields:
    - '1.20, comments: Confirmado por duplo ensaio (mesma amostra)'
    - '457.0, is_abnormal: True'

    This function extracts just the numeric value.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        # Strip embedded field markers
        for pattern in [
            ", comments:",
            ", is_abnormal:",
            ", source_text:",
            ", reference_notes:",
        ]:
            if pattern.lower() in value.lower():
                value = value.split(",")[0].strip()
                break
        try:
            # Handle Portuguese decimal format (comma as decimal separator)
            # But only if there's a single comma and no dot
            if "," in value and "." not in value:
                value = value.replace(",", ".")
            return float(value)
        except ValueError:
            logger.warning(f"Could not parse numeric field value: {value[:50]}")
            return None
    return None


def _fix_lab_results_format(
    tool_result_dict: dict, client: OpenAI, model_id: str
) -> dict:
    """Fix common LLM formatting issues in lab_results and dates using LLM-based parsing."""
    # Fix date formats at report level
    for date_field in ["collection_date", "report_date"]:
        if date_field in tool_result_dict:
            tool_result_dict[date_field] = _normalize_date_format(
                tool_result_dict[date_field]
            )

    if "lab_results" not in tool_result_dict or not isinstance(
        tool_result_dict["lab_results"], list
    ):
        return tool_result_dict

    # Map LLM-populated standardization fields to internal fields
    # lab_name (standardized name) → lab_name_standardized
    # unit (standardized unit) → lab_unit_standardized
    for item in tool_result_dict["lab_results"]:
        if isinstance(item, dict):
            if "lab_name" in item:
                std_name = item.pop("lab_name")
                item["lab_name_standardized"] = std_name if std_name else _UNKNOWN_VALUE
            if "unit" in item:
                std_unit = item.pop("unit")
                # Map $UNKNOWN$ to None so unit inference in normalization can handle it
                item["lab_unit_standardized"] = (
                    std_unit
                    if (std_unit and std_unit != _UNKNOWN_VALUE)
                    else None
                )

    # First, try to reassemble flattened key-value format (common LLM issue)
    # e.g., ['lab_name_raw: Glucose', 'value_raw: 100', ...] -> [{'lab_name_raw': 'Glucose', 'value_raw': '100', ...}]
    tool_result_dict["lab_results"] = _reassemble_flattened_key_values(
        tool_result_dict["lab_results"]
    )

    # Clean numeric reference fields (strip embedded metadata like ", comments: ...")
    for item in tool_result_dict["lab_results"]:
        if isinstance(item, dict):
            if "reference_min_raw" in item:
                item["reference_min_raw"] = _clean_numeric_field(
                    item["reference_min_raw"]
                )
            if "reference_max_raw" in item:
                item["reference_max_raw"] = _clean_numeric_field(
                    item["reference_max_raw"]
                )

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

            # Try to parse LabResult repr format: "LabResult(lab_name_raw='...', ...)"
            parsed_repr = _parse_labresult_repr(lr_data)
            if parsed_repr:
                parsed_lab_results.append(parsed_repr)
                continue

            # Collect for LLM-based parsing
            string_results.append(lr_data)
            string_indices.append(i)
            parsed_lab_results.append(None)  # Placeholder
        else:
            parsed_lab_results.append(lr_data)

    # If we have string results, use LLM to parse them
    if string_results:
        logger.info(
            f"Found {len(string_results)} lab results as strings, using LLM to parse them"
        )
        parsed = _parse_string_results_with_llm(string_results, client, model_id)

        # Insert parsed results back
        skipped_count = 0
        for idx, parsed_result in zip(string_indices, parsed):
            if parsed_result:
                parsed_lab_results[idx] = parsed_result
            else:
                # Failed to parse, remove from results
                logger.debug(
                    f"Failed to parse lab_result[{idx}], skipping. Data: {string_results[string_indices.index(idx)][:200]}"
                )
                skipped_count += 1

        # Log summary if any failed
        if skipped_count > 0:
            logger.warning(
                f"Skipped {skipped_count}/{len(string_results)} unparseable lab results. "
                f"This is normal during self-consistency voting - other extraction attempts may have proper structure."
            )

    # Remove None placeholders
    parsed_lab_results = [r for r in parsed_lab_results if r is not None]

    tool_result_dict["lab_results"] = parsed_lab_results
    return tool_result_dict


def _parse_string_results_with_llm(
    string_results: list[str], client: OpenAI, model_id: str
) -> list[dict | None]:
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
                {
                    "role": "system",
                    "content": "You are a medical data parser. Parse lab result strings into structured JSON format.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
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
            logger.error(
                f"Unexpected LLM response format for string parsing: {parsed_json}"
            )
            return [None] * len(string_results)

        # Validate we got the right number of results
        if len(results) != len(string_results):
            logger.warning(
                f"LLM returned {len(results)} results but expected {len(string_results)}"
            )
            # Pad or trim to match
            while len(results) < len(string_results):
                results.append(None)
            results = results[: len(string_results)]

        return results

    except Exception as e:
        logger.error(f"Failed to parse string results with LLM: {e}")
        return [None] * len(string_results)


def _salvage_lab_results(tool_result_dict: dict) -> dict:
    """Try to salvage valid lab results even if report validation fails."""
    if "lab_results" not in tool_result_dict or not isinstance(
        tool_result_dict["lab_results"], list
    ):
        return HealthLabReport(lab_results=[]).model_dump(mode="json")

    valid_results = []
    for i, lr_data in enumerate(tool_result_dict["lab_results"]):
        if isinstance(lr_data, str):
            logger.warning(
                f"Skipping string lab_result[{i}] in salvage: {lr_data[:100]}"
            )
            continue
        try:
            lr_model = LabResult(**lr_data)
            valid_results.append(lr_model.model_dump(mode="json"))
        except Exception as e:
            logger.error(
                f"Failed to validate lab_result[{i}] in salvage: {e}. Data: {lr_data}"
            )

    logger.warning(
        f"Salvaged {len(valid_results)}/{len(tool_result_dict['lab_results'])} lab results"
    )
    return HealthLabReport(lab_results=valid_results).model_dump(mode="json")
