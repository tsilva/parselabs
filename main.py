from dotenv import load_dotenv
load_dotenv(override=True)

import glob
from enum import Enum
import logging
import os
import json
import shutil
import base64
import pandas as pd
import pdf2image
import unicodedata
import re
import hashlib
from multiprocessing import Pool
from PIL import Image
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Any, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add OpenAI import for OpenRouter
from openai import OpenAI, APIError

########################################
# Centralized Column Schema
########################################

COLUMN_SCHEMA = {
    # Core columns
    "date": {"dtype": "datetime64[ns]", "excel_width": 13, "plotting_role": "date"},

    # === RAW EXTRACTION FIELDS (exactly as in PDF) ===
    "test_name": {"dtype": "str", "excel_width": 35, "excel_hidden": False},
    "value": {"dtype": "float64", "excel_width": 12, "excel_hidden": False},
    "unit": {"dtype": "str", "excel_width": 15, "excel_hidden": False},
    "reference_range": {"dtype": "str", "excel_width": 25, "excel_hidden": False},
    "reference_min": {"dtype": "float64", "excel_width": 12, "excel_hidden": True},
    "reference_max": {"dtype": "float64", "excel_width": 12, "excel_hidden": True},
    "is_abnormal": {"dtype": "boolean", "excel_width": 10, "excel_hidden": False},
    "comments": {"dtype": "str", "excel_width": 40, "excel_hidden": False},

    # Traceability
    "source_text": {"dtype": "str", "excel_width": 50, "excel_hidden": True},
    "page_number": {"dtype": "Int64", "excel_width": 8},
    "source_file": {"dtype": "str", "excel_width": 25},

    # === NORMALIZED FIELDS (added in post-processing) ===
    "lab_type": {"dtype": "str", "excel_width": 10, "derivation_logic": "infer_from_normalized_name"},
    "lab_name": {"dtype": "str", "excel_width": 35, "derivation_logic": "normalize_test_name", "plotting_role": "group"},
    "lab_unit": {"dtype": "str", "excel_width": 15, "derivation_logic": "normalize_unit"},
    "lab_name_slug": {"dtype": "str", "excel_width": 30, "excel_hidden": True, "derivation_logic": "slugify_test_name"},

    # Unit conversion
    "value_normalized": {"dtype": "float64", "excel_width": 14, "derivation_logic": "convert_to_primary_unit", "plotting_role": "value"},
    "unit_normalized": {"dtype": "str", "excel_width": 14, "derivation_logic": "convert_to_primary_unit", "plotting_role": "unit"},
    "reference_min_normalized": {"dtype": "float64", "excel_width": 14, "derivation_logic": "convert_to_primary_unit"},
    "reference_max_normalized": {"dtype": "float64", "excel_width": 14, "derivation_logic": "convert_to_primary_unit"},

    # Health status
    "is_out_of_reference": {"dtype": "boolean", "excel_width": 14, "derivation_logic": "compute_vs_reference"},
    "healthy_range_min": {"dtype": "float64", "excel_width": 16, "derivation_logic": "get_healthy_range"},
    "healthy_range_max": {"dtype": "float64", "excel_width": 16, "derivation_logic": "get_healthy_range"},
    "is_in_healthy_range": {"dtype": "boolean", "excel_width": 18, "derivation_logic": "compute_vs_healthy_range"},
}

# Helper functions to derive lists/dicts from COLUMN_SCHEMA
def get_export_columns_from_schema(schema: dict) -> list:
    """Returns an ordered list of columns for the main export."""
    ordered_keys = [
        # Core
        "date",
        # Raw extraction
        "test_name", "value", "unit", "reference_range",
        "reference_min", "reference_max", "is_abnormal", "comments",
        # Normalized
        "lab_type", "lab_name", "lab_unit", "lab_name_slug",
        "value_normalized", "unit_normalized",
        "reference_min_normalized", "reference_max_normalized",
        # Health status
        "is_out_of_reference",
        "healthy_range_min", "healthy_range_max", "is_in_healthy_range",
        # Traceability
        "source_text", "page_number", "source_file"
    ]
    return [key for key in ordered_keys if key in schema]

def get_hidden_excel_columns_from_schema(schema: dict) -> list:
    """Returns a list of columns to be hidden in the main Excel export."""
    return [col for col, props in schema.items() if props.get("excel_hidden")]

def get_excel_widths_from_schema(schema: dict) -> dict:
    """Returns a dictionary of {column_name: width} for Excel."""
    return {col: props["excel_width"] for col, props in schema.items() if "excel_width" in props}

def get_dtype_map_from_schema(schema: dict) -> dict:
    """Returns a dictionary of {column_name: dtype_string} for pandas type conversion."""
    dtype_map = {}
    for col, props in schema.items():
        if "dtype" in props:
            dtype_map[col] = props["dtype"]
    return dtype_map

PLOTTING_DATE_COL = next((col for col, props in COLUMN_SCHEMA.items() if props.get("plotting_role") == "date"), "date")
PLOTTING_VALUE_COL = next((col for col, props in COLUMN_SCHEMA.items() if props.get("plotting_role") == "value"), "lab_value_final")
PLOTTING_GROUP_COL = next((col for col, props in COLUMN_SCHEMA.items() if props.get("plotting_role") == "group"), "lab_name_enum")
PLOTTING_UNIT_COL = next((col for col, props in COLUMN_SCHEMA.items() if props.get("plotting_role") == "unit"), "lab_unit_final")


########################################
# Config / Logging
########################################

UNKNOWN_VALUE = "$UNKNOWN$"
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
INFO_LOG_PATH = LOG_DIR / "info.log"
ERROR_LOG_PATH = LOG_DIR / "error.log"

def setup_logging(clear_logs: bool = False) -> logging.Logger:
    """Configure file logging, optionally clearing existing logs."""
    if clear_logs:
        for log_file in (INFO_LOG_PATH, ERROR_LOG_PATH):
            if log_file.exists():
                log_file.write_text("", encoding="utf-8")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    info_handler = logging.FileHandler(INFO_LOG_PATH, encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    error_handler = logging.FileHandler(ERROR_LOG_PATH, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    return logger

logger = setup_logging(clear_logs=False)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def load_env_config():
    input_path = os.getenv("INPUT_PATH")
    input_file_regex = os.getenv("INPUT_FILE_REGEX")
    output_path = os.getenv("OUTPUT_PATH")
    self_consistency_model_id = os.getenv("SELF_CONSISTENCY_MODEL_ID")
    extract_model_id = os.getenv("EXTRACT_MODEL_ID")
    n_extractions = int(os.getenv("N_EXTRACTIONS", 1)) # Default to 1 if not set
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    max_workers_str = os.getenv("MAX_WORKERS", "1") # Default to "1" as string

    # Ensure max_workers is correctly parsed as int
    try:
        max_workers = int(max_workers_str)
        if max_workers < 1: max_workers = 1 # Ensure at least one worker
    except ValueError:
        logger.warning(f"MAX_WORKERS environment variable ('{max_workers_str}') is not a valid integer. Defaulting to 1.")
        max_workers = 1


    if not self_consistency_model_id: raise ValueError("SELF_CONSISTENCY_MODEL_ID not set")
    if not extract_model_id: raise ValueError("EXTRACT_MODEL_ID not set")
    if not input_path or not Path(input_path).exists(): raise ValueError(f"INPUT_PATH ('{input_path}') not set or does not exist.")
    if not input_file_regex: raise ValueError("INPUT_FILE_REGEX not set")
    if not output_path: raise ValueError("OUTPUT_PATH not set")
    if not openrouter_api_key: raise ValueError("OPENROUTER_API_KEY not set")

    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    return {
        "input_path" : Path(input_path),
        "input_file_regex" : input_file_regex,
        "output_path" : output_path_obj,
        "self_consistency_model_id" : self_consistency_model_id,
        "extract_model_id" : extract_model_id,
        "n_extractions": n_extractions,
        "openrouter_api_key": openrouter_api_key,
        "max_workers": max_workers
    }

def clear_directory(dir_path: Path) -> None:
    """Remove all contents of a directory without deleting the directory itself."""
    if dir_path.exists():
        for item in dir_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            else:
                shutil.rmtree(item)

########################################
# LLM Tools / Pydantic Models
########################################

# Enums removed - using plain strings during extraction for simplicity and accuracy
# Normalization happens in post-processing using mapping files

class LabResult(BaseModel):
    """Single lab test result - optimized for extraction accuracy.

    Primary goal: Extract exactly what appears in the PDF without interpretation.
    Normalization happens in post-processing.
    """

    # === RAW EXTRACTION (exactly as shown in PDF) ===
    test_name: str = Field(
        description=(
            "Test name EXACTLY as written in the PDF. "
            "Preserve all spacing, capitalization, symbols, and formatting. "
            "Examples: 'Hemoglobina A1c', 'ALT (TGP)', 'Vitamina D 25-hidroxi', 'Glicose'. "
            "Do NOT standardize or translate - copy the exact text."
        )
    )
    value: Optional[float] = Field(
        default=None,
        description="Numeric result value. For text results (Positive/Negative), use 1/0 or leave null and put in comments."
    )
    unit: Optional[str] = Field(
        default=None,
        description=(
            "Unit EXACTLY as written in PDF (preserve case, spacing, symbols). "
            "Examples: '%', 'mg/dL', 'mg/dl', 'U/L', 'g/dL', '10³/µL'. "
            "Copy exactly - do NOT standardize."
        )
    )
    reference_range: Optional[str] = Field(
        default=None,
        description=(
            "Complete reference range text EXACTLY as shown. "
            "Examples: '4.5-6.0', '<5.7', '70-100', '≤ 34', '3.5 - 5.5 mg/dL'. "
            "Include all text, symbols, and units if present."
        )
    )

    # === PARSED REFERENCE VALUES ===
    reference_min: Optional[float] = Field(
        default=None,
        description="Minimum reference value (extract number from reference_range if available)"
    )
    reference_max: Optional[float] = Field(
        default=None,
        description="Maximum reference value (extract number from reference_range if available)"
    )

    # === FLAGS & CONTEXT ===
    is_abnormal: Optional[bool] = Field(
        default=None,
        description="Whether result is marked/flagged as abnormal in PDF (H/L, arrows, asterisks, etc.)"
    )
    comments: Optional[str] = Field(
        default=None,
        description="Any notes, comments, or qualitative results"
    )

    # === TRACEABILITY ===
    source_text: str = Field(
        description="Exact row or section from PDF containing this result (helps verify accuracy)"
    )

    # Internal fields (added by pipeline, not by LLM)
    page_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="Page number in PDF (added by pipeline)"
    )
    source_file: Optional[str] = Field(default=None, description="Source file identifier (added by pipeline)")
    lab_name_standardized: Optional[str] = Field(default=None, description="Standardized lab name (added by standardization step)")
    lab_unit_standardized: Optional[str] = Field(default=None, description="Standardized lab unit (added by standardization step)")

class HealthLabReport(BaseModel):
    """Document-level lab report metadata.

    Focus on essential information needed for organizing results.
    """

    # Essential dates
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

    # Lab facility (optional but helpful)
    lab_facility: Optional[str] = Field(
        default=None,
        description="Name of laboratory that performed tests"
    )

    # The actual results
    lab_results: List[LabResult] = Field(
        default_factory=list,
        description="List of all lab test results extracted from this page/document"
    )

    # Internal tracking
    source_file: Optional[str] = Field(default=None, description="Source PDF filename")

    def normalize_empty_optionals(self):
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            field_info = self.model_fields[field_name]
            # Check if field is Optional and value is empty string
            is_optional_type = field_info.is_required() is False # Simplified check
            if value == "" and is_optional_type:
                 setattr(self, field_name, None)
        
        for lab_result in self.lab_results:
            for field_name in lab_result.model_fields:
                value = getattr(lab_result, field_name)
                field_info = lab_result.model_fields[field_name]
                is_optional_type = field_info.is_required() is False
                if value == "" and is_optional_type:
                    setattr(lab_result, field_name, None)

TOOLS = [{"type": "function", "function": {"name": "extract_lab_results", "description": "Extracts lab results...", "parameters": HealthLabReport.model_json_schema()}}] # Truncated for brevity

########################################
# Helper Functions
########################################

def hash_file(file_path: Path, length=4) -> str:
    with open(file_path, "rb") as f:
        h = hashlib.md5()
        for chunk in iter(lambda: f.read(4096), b""): h.update(chunk)
    return h.hexdigest()[:length]

def preprocess_page_image(image: Image.Image) -> Image.Image:
    from PIL import ImageEnhance # Moved import inside
    gray_image = image.convert('L')
    MAX_WIDTH = 1200
    if gray_image.width > MAX_WIDTH:
        ratio = MAX_WIDTH / gray_image.width
        new_height = int(gray_image.height * ratio)
        gray_image = gray_image.resize((MAX_WIDTH, new_height), Image.Resampling.LANCZOS)
    return ImageEnhance.Contrast(gray_image).enhance(2.0)

def slugify(value: Any) -> str:
    """Create a normalized slug for mapping/debugging purposes."""
    if pd.isna(value):
        return ""
    value = str(value).strip().lower().replace('µ', 'micro').replace('μ', 'micro').replace('%', 'percent')
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_]+", "-", value).strip('-')
    value = value.replace("-", "")
    return value

def self_consistency(fn, model_id, n, *args, **kwargs):
    if n == 1:
        result = fn(*args, **kwargs)
        return result, [result]
    
    results = []
    # Add temperature to kwargs if the function expects it and it's not already set by the caller of self_consistency
    effective_kwargs = kwargs.copy()
    if 'temperature' in fn.__code__.co_varnames and 'temperature' not in effective_kwargs:
        effective_kwargs['temperature'] = 0.5

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(fn, *args, **effective_kwargs) for _ in range(n)]
        for future in as_completed(futures):
            try: results.append(future.result())
            except Exception as e:
                logger.error(f"Error during self-consistency task execution: {e}")
                for f_cancel in futures: 
                    if not f_cancel.done(): f_cancel.cancel()
                raise
    if not results: raise RuntimeError("All self-consistency calls failed.")
    if all(r == results[0] for r in results): return results[0], results

    system_prompt = (
        "You are an expert at comparing multiple outputs of the same extraction task. "
        "We have extracted several samples from the same prompt in order to average out any errors or inconsistencies that may appear in individual outputs. "
        "Your job is to select the output that is most consistent with the majority of the provided samples—"
        "in other words, the output that best represents the 'average' or consensus among all outputs. "
        "Prioritize agreement on extracted content (test names, values, units, reference ranges, etc.). "
        "Ignore formatting, whitespace, and layout differences. "
        "Return ONLY the best output, verbatim, with no extra commentary. "
        "Do NOT include any delimiters, output numbers, or extra labels in your response—return only the raw content of the best output."
    )
    prompt = "".join(f"--- Output {i+1} ---\n{json.dumps(v, ensure_ascii=False) if type(v) in [list, dict] else v}\n\n" for i, v in enumerate(results))

    voted_raw = None
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        )
        voted_raw = completion.choices[0].message.content.strip()

        # TODO: hack
        # Try to parse the voted result back to the expected type
        if fn.__name__ == 'extract_labs_from_page_image':
            # For extract function, we expect a dictionary
            try:
                # Strip markdown code fences if present
                if voted_raw.startswith("```"):
                    # Remove opening fence (```json or just ```)
                    lines = voted_raw.split('\n')
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    # Remove closing fence
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    voted_raw = '\n'.join(lines).strip()

                voted_result = json.loads(voted_raw)
                return voted_result, results
            except json.JSONDecodeError:
                logger.error(f"Failed to parse voted result as JSON for extract function. Raw: '{voted_raw[:200]}...'")
                return results[0], results  # Fallback to first result
        else:
            # For other functions, return the string as-is
            return voted_raw, results
            
    except Exception as e:
        logger.error(f"Error during self-consistency voting logic. Raw: '{voted_raw if voted_raw else 'N/A'}'. Error: {e}")

        # Fallback: Pick result with highest average confidence if available
        if fn.__name__ == 'extract_labs_from_page_image':
            try:
                best_result = max(results, key=lambda r:
                    sum(lr.get('confidence', 0.5) for lr in r.get('lab_results', [])) /
                    max(len(r.get('lab_results', [])), 1)
                )
                logger.info(f"Voting failed. Selected extraction result with highest average confidence.")
                return best_result, results
            except Exception as fallback_error:
                logger.error(f"Confidence-based fallback also failed: {fallback_error}")
                return results[0], results

        # For other functions, return first result
        return results[0], results

def extract_labs_from_page_image(image_path: Path, model_id: str, temperature: float = 0.3) -> dict:
    system_prompt = """
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
   - Example: "Neutrofilos  73.0  4.307  2.1 a 7.6" should create TWO results:
     * "Neutrofilos (%)" with value 73.0, unit "%"
     * "Neutrofilos (absolute)" with value 4.307, unit "10^9/L", reference "2.1 a 7.6"

3. TEST NAMES WITH CONTEXT:
   - Include section headers as prefixes for clarity
   - Example: If you see "BILIRRUBINAS" as header and "Total" below it, use "BILIRRUBINAS - Total"
   - Example: If you see "URINA" header and "Cor" below, use "URINA - Cor"

4. NUMERIC vs QUALITATIVE VALUES:
   - `value`: ONLY for numeric results (e.g., 5.0, 14.2, 1.74)
   - For TEXT-ONLY results (e.g., "AMARELA", "NAO CONTEM", "NORMAL", "POSITIVE", "NEGATIVE"):
     * Set `value` to null (None)
     * Put the text result in `comments`
   - For RANGE results (e.g., "1 a 2/campo", "0 a 1/campo"):
     * Set `value` to null (None)
     * Put the full text in `comments`
     * Extract any unit visible in the text (e.g., "/campo" from "1 a 2/campo")
   - `unit`: Extract the unit EXACTLY as shown in the document
     * Copy the unit symbol or abbreviation exactly (e.g., "mg/dl", "g/dl", "%", "U/L", "/campo")
     * If NO unit is visible or implied in the document → use null
     * Do NOT infer or normalize units - just extract what you see

5. REFERENCE RANGES:
   - `reference_range`: Copy the complete reference range text (e.g., "4.5-6.0", "<5.7", "70-100 mg/dL")
   - `reference_min` / `reference_max`: Parse numeric bounds from reference_range if possible

6. FLAGS & CONTEXT:
   - `is_abnormal`: Set to true if result is marked (H, L, *, ↑, ↓, "HIGH", "LOW", etc.)
   - `comments`: Capture any notes, qualitative results, or text values

7. TRACEABILITY:
   - `source_text`: Copy the exact row/line containing this result

8. DATES: Format as YYYY-MM-DD or leave null

SCHEMA FIELD NAMES:
- Use `test_name` (NOT lab_name)
- Use `value` (NOT lab_value)
- Use `unit` (NOT lab_unit)
- Use `reference_range`, `reference_min`, `reference_max`
- Use `is_abnormal` (NOT is_flagged)
- Use `comments` (NOT lab_comments)

EXAMPLES:
✅ CORRECT:
  {"test_name": "URINA - Cor", "value": null, "unit": null, "comments": "AMARELA"}
  {"test_name": "URINA - Proteinas", "value": null, "unit": null, "comments": "NAO CONTEM"}
  {"test_name": "URINA - Glicose", "value": null, "unit": null, "comments": "NAO CONTEM"}
  {"test_name": "URINA - Urobilinogenio", "value": null, "unit": null, "comments": "NORMAL"}
  {"test_name": "Hemoglobina", "value": 14.2, "unit": "g/dl", "comments": null}
  {"test_name": "URINA - pH", "value": 5.0, "unit": null, "comments": null}
  {"test_name": "URINA - Densidade a 15º C", "value": 1.022, "unit": null, "comments": null}
  {"test_name": "EXAME MICROSCOPICO - CELULAS EPITELIAIS", "value": null, "unit": "/campo", "comments": "1 a 2/campo"}
  {"test_name": "EXAME MICROSCOPICO - LEUCOCITOS", "value": null, "unit": "/campo", "comments": "0 a 1/campo"}

❌ WRONG:
  {"test_name": "Cor", "value": "AMARELA", ...}  # Text in value field!
  {"test_name": "Total", ...}  # Missing context (should be "BILIRRUBINAS - Total")
  {"test_name": "CELULAS EPITELIAIS", "value": null, "unit": null, "comments": "1 a 2/campo"}  # Missing /campo unit from range text!

Remember: Your job is to be a perfect copier, not an interpreter. Extract EVERYTHING, even qualitative results.
""".strip()

    user_prompt = """
Please extract all lab test results from this medical lab report image.
Extract test names, values, units, and reference ranges exactly as they appear.
Pay special attention to preserving the exact formatting and symbols.
""".strip()

    with open(image_path, "rb") as img_file:
        img_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
                ]}
            ],
            temperature=temperature, max_tokens=8000, tools=TOOLS, tool_choice={"type": "function", "function": {"name": "extract_lab_results"}}
        )
    except APIError as e:
        logger.error(f"API Error during lab extraction from {image_path.name}: {e}")
        raise RuntimeError(f"Lab extraction failed for {image_path.name}: {e}")

    # Check for valid response structure
    if not completion or not completion.choices or len(completion.choices) == 0:
        logger.error(f"Invalid completion response structure. Completion: {completion}")
        return HealthLabReport(lab_results=[]).model_dump(mode='json')

    if not completion.choices[0].message.tool_calls:
        logger.warning(f"No tool call by model for lab extraction from {image_path.name}")
        return HealthLabReport(lab_results=[]).model_dump(mode='json')

    tool_args_raw = completion.choices[0].message.tool_calls[0].function.arguments
    try: tool_result_dict = json.loads(tool_args_raw)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for tool args: {e}. Raw: '{tool_args_raw[:500]}'")
        return HealthLabReport(lab_results=[]).model_dump(mode='json') # Fallback

    # Pre-process: Fix common LLM issues before Pydantic validation

    # Fix 1: Convert lab_results strings to dicts (LLM sometimes returns JSON strings or Python repr strings)
    if "lab_results" in tool_result_dict and isinstance(tool_result_dict["lab_results"], list):
        parsed_lab_results = []
        for i, lr_data in enumerate(tool_result_dict["lab_results"]):
            # If it's a string, try to parse it
            if isinstance(lr_data, str):
                # Try JSON first
                try:
                    parsed_lr = json.loads(lr_data)
                    parsed_lab_results.append(parsed_lr)
                    logger.debug(f"Parsed lab_result[{i}] from JSON string")
                    continue
                except json.JSONDecodeError:
                    pass

                # Try Python repr string parser for LabResult(...) format
                if lr_data.startswith("LabResult("):
                    try:
                        # Parse LabResult(key=value, key=value, ...) format
                        # Extract content between LabResult( and )
                        content = lr_data[len("LabResult("):-1]

                        # Simple parser for key=value pairs
                        parsed_dict = {}
                        # Match key='value' or key="value" or key=number or key=null
                        pattern = r"(\w+)=(?:'([^']*)'|\"([^\"]*)\"|(\d+\.?\d*)|(\w+))"
                        matches = re.findall(pattern, content)

                        for match in matches:
                            key = match[0]
                            # Get the value from whichever group matched
                            value = match[1] or match[2] or match[3] or match[4]

                            # Convert types
                            if value == 'null' or value == 'None':
                                parsed_dict[key] = None
                            elif value == 'True':
                                parsed_dict[key] = True
                            elif value == 'False':
                                parsed_dict[key] = False
                            elif match[3]:  # Numeric match
                                parsed_dict[key] = float(value) if '.' in value else int(value)
                            else:
                                parsed_dict[key] = value

                        if parsed_dict:
                            parsed_lab_results.append(parsed_dict)
                            logger.debug(f"Parsed lab_result[{i}] from Python repr string")
                            continue
                    except Exception as e:
                        logger.warning(f"Failed to parse lab_result[{i}] Python repr string: {e}. Raw: {lr_data[:200]}")
                        continue

                # Unknown format
                logger.warning(f"Failed to parse lab_result[{i}] - unknown format. Skipping. Raw: {lr_data[:200]}")
                continue
            else:
                parsed_lab_results.append(lr_data)
        tool_result_dict["lab_results"] = parsed_lab_results

    # Field names now match schema directly (test_name, value, unit, etc.)
    # No mapping needed

    try:
        report_model = HealthLabReport(**tool_result_dict)
        report_model.normalize_empty_optionals()
        return report_model.model_dump(mode='json')
    except Exception as e: # Pydantic validation or other error
        logger.error(f"Model validation error post-extraction: {e}. Data keys: {tool_result_dict.keys()}, lab_results count: {len(tool_result_dict.get('lab_results', []))}")
        # Attempt to salvage results if main report fails
        if "lab_results" in tool_result_dict and isinstance(tool_result_dict["lab_results"], list):
            valid_results = []
            for i, lr_data in enumerate(tool_result_dict["lab_results"]):
                try:
                    # Skip if it's still a string (shouldn't happen after preprocessing)
                    if isinstance(lr_data, str):
                        logger.warning(f"lab_result[{i}] is still a string after preprocessing, skipping")
                        continue
                    lr_model = LabResult(**lr_data)
                    valid_results.append(lr_model.model_dump(mode='json'))
                except Exception as lr_error:
                    logger.debug(f"Failed to validate lab_result[{i}]: {lr_error}")
                    pass # Ignore individual failures
            logger.info(f"Salvaged {len(valid_results)}/{len(tool_result_dict['lab_results'])} lab results after validation error")
            return HealthLabReport(lab_results=valid_results).model_dump(mode='json') # Return with what could be salvaged
        return HealthLabReport(lab_results=[]).model_dump(mode='json') # Final fallback


########################################
# Lab Name Standardization
########################################

def load_standardized_lab_names() -> list:
    """Load the list of valid standardized lab names from config."""
    config_path = Path("config/lab_specs.json")
    if not config_path.exists():
        logger.warning("lab_specs.json not found, standardization will return $UNKNOWN$")
        return []

    with open(config_path, 'r', encoding='utf-8') as f:
        specs = json.load(f)

    # Extract standardized names from the keys
    standardized_names = sorted(specs.keys())
    logger.info(f"Loaded {len(standardized_names)} standardized lab names from lab_specs.json")
    return standardized_names

def load_standardized_lab_units() -> list:
    """Load the list of valid standardized lab units from config."""
    config_path = Path("config/lab_specs.json")
    if not config_path.exists():
        logger.warning("lab_specs.json not found, unit standardization will return $UNKNOWN$")
        return []

    with open(config_path, 'r', encoding='utf-8') as f:
        specs = json.load(f)

    # Collect all unique units from primary_unit and alternatives
    all_units = set()
    for lab_name, spec in specs.items():
        primary = spec.get('primary_unit')
        if primary:
            all_units.add(primary)

        for alt in spec.get('alternatives', []):
            unit = alt.get('unit')
            if unit:
                all_units.add(unit)

    standardized_units = sorted(all_units)
    logger.info(f"Loaded {len(standardized_units)} standardized lab units")
    return standardized_units

def load_lab_type_mapping() -> dict[str, str]:
    """Load mapping of standardized lab name -> lab type from config."""
    config_path = Path("config/lab_specs.json")
    if not config_path.exists():
        logger.warning("lab_specs.json not found, lab type lookup will fail")
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        specs = json.load(f)

    # Build mapping: standardized_lab_name -> lab_type
    lab_type_map = {}
    for lab_name, spec in specs.items():
        lab_type = spec.get('lab_type', 'blood')  # default to blood
        lab_type_map[lab_name] = lab_type

    logger.info(f"Loaded lab type mappings for {len(lab_type_map)} tests")
    return lab_type_map

def standardize_lab_names(raw_test_names: list[str], model_id: str, standardized_names: list[str]) -> dict[str, str]:
    """
    Map raw test names to standardized lab names using LLM.

    Args:
        raw_test_names: List of raw test names from extraction
        model_id: Model to use for standardization
        standardized_names: List of valid standardized lab names

    Returns:
        Dictionary mapping raw_test_name -> standardized_name
    """
    if not raw_test_names:
        return {}

    if not standardized_names:
        logger.warning("No standardized names available, returning $UNKNOWN$ for all")
        return {name: UNKNOWN_VALUE for name in raw_test_names}

    # Build the prompt
    system_prompt = f"""You are a medical lab test name standardization expert.

Your task: Map raw test names from lab reports to standardized lab names from a predefined list.

CRITICAL RULES:
1. Choose the BEST MATCH from the standardized names list
2. Consider semantic similarity and medical terminology
3. Account for language variations (Portuguese/English)
4. If NO good match exists, use exactly: "{UNKNOWN_VALUE}"
5. Return a JSON object mapping each raw name to its standardized name

STANDARDIZED NAMES LIST ({len(standardized_names)} names):
{json.dumps(standardized_names, ensure_ascii=False, indent=2)}

EXAMPLES:
- "Hemoglobina" → "Blood - Hemoglobin"
- "GLICOSE -jejum-" → "Blood - Glucose"
- "URINA - pH" → "Urine Type II - pH"
- "Some Unknown Test" → "{UNKNOWN_VALUE}"
"""

    user_prompt = f"""Map these raw test names to standardized names:

RAW TEST NAMES:
{json.dumps(raw_test_names, ensure_ascii=False, indent=2)}

Return a JSON object with the mapping:
{{
  "raw_name_1": "Standardized Name 1",
  "raw_name_2": "Standardized Name 2",
  ...
}}
"""

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=4000
        )

        if not completion or not completion.choices or len(completion.choices) == 0:
            logger.error("Invalid completion response for standardization")
            return {name: UNKNOWN_VALUE for name in raw_test_names}

        response_text = completion.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if response_text.startswith("```"):
            lines = response_text.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = '\n'.join(lines).strip()

        # Parse the JSON mapping
        mapping = json.loads(response_text)

        # Validate that all raw names are in the mapping
        result = {}
        for raw_name in raw_test_names:
            if raw_name in mapping:
                standardized = mapping[raw_name]
                # Validate it's either $UNKNOWN$ or in the list
                if standardized == UNKNOWN_VALUE or standardized in standardized_names:
                    result[raw_name] = standardized
                else:
                    logger.warning(f"LLM returned invalid standardized name: '{standardized}' for '{raw_name}', using $UNKNOWN$")
                    result[raw_name] = UNKNOWN_VALUE
            else:
                logger.warning(f"LLM didn't return mapping for '{raw_name}', using $UNKNOWN$")
                result[raw_name] = UNKNOWN_VALUE

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse standardization response as JSON: {e}")
        return {name: UNKNOWN_VALUE for name in raw_test_names}
    except Exception as e:
        logger.error(f"Error during standardization: {e}")
        return {name: UNKNOWN_VALUE for name in raw_test_names}

def standardize_lab_units(unit_contexts: list[tuple[str, str]], model_id: str, standardized_units: list[str]) -> dict[str, str]:
    """
    Map raw lab units to standardized units using LLM with lab name context.

    Args:
        unit_contexts: List of (raw_unit, standardized_lab_name) tuples for context
        model_id: Model to use for standardization
        standardized_units: List of valid standardized units

    Returns:
        Dictionary mapping raw_unit -> standardized_unit
    """
    if not unit_contexts:
        return {}

    if not standardized_units:
        logger.warning("No standardized units available, returning $UNKNOWN$ for all")
        return {unit: UNKNOWN_VALUE for unit, _ in unit_contexts}

    # Get unique (raw_unit, lab_name) pairs for context-aware mapping
    unique_pairs = list(set(unit_contexts))

    # Build the prompt with context
    system_prompt = f"""You are a medical laboratory unit standardization expert.

Your task: Map (raw_unit, lab_name) pairs to standardized units from a predefined list.

CRITICAL RULES:
1. Choose the BEST MATCH from the standardized units list
2. Handle case variations (e.g., "mg/dl" → "mg/dL", "u/l" → "IU/L")
3. Handle symbol variations (e.g., "µ" vs "μ", superscripts)
4. Handle spacing variations (e.g., "mg / dl" → "mg/dL")
5. For null/missing units, infer from lab name:
   - "Urine Type II - Color", "Urine Type II - Density" → "unitless"
   - "Urine Type II - pH" → "pH"
   - "Urine Type II - Proteins", "Urine Type II - Glucose", "Urine Type II - Blood", etc. → "boolean"
   - "Blood - Red Blood Cell Morphology" (qualitative) → "{UNKNOWN_VALUE}"
6. If NO good match exists, use exactly: "{UNKNOWN_VALUE}"
7. Return a JSON array with objects: {{"raw_unit": "...", "lab_name": "...", "standardized_unit": "..."}}

STANDARDIZED UNITS LIST ({len(standardized_units)} units):
{json.dumps(standardized_units, ensure_ascii=False, indent=2)}

EXAMPLES:
- {{"raw_unit": "mg/dl", "lab_name": "Blood - Glucose", "standardized_unit": "mg/dL"}}
- {{"raw_unit": "U/L", "lab_name": "Blood - AST", "standardized_unit": "IU/L"}}
- {{"raw_unit": "fl", "lab_name": "Blood - MCV", "standardized_unit": "fL"}}
- {{"raw_unit": "null", "lab_name": "Urine Type II - Color", "standardized_unit": "unitless"}}
- {{"raw_unit": "null", "lab_name": "Urine Type II - pH", "standardized_unit": "pH"}}
- {{"raw_unit": "null", "lab_name": "Urine Type II - Proteins", "standardized_unit": "boolean"}}
"""

    # Build list of pairs to map
    pairs_to_map = [
        {"raw_unit": raw_unit, "lab_name": lab_name}
        for raw_unit, lab_name in unique_pairs
    ]

    user_prompt = f"""Map these (raw_unit, lab_name) pairs to standardized units:

{json.dumps(pairs_to_map, ensure_ascii=False, indent=2)}

Return a JSON array with the standardized_unit added to each object."""

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=2000
        )

        if not completion or not completion.choices or len(completion.choices) == 0:
            logger.error("Invalid completion response for unit standardization")
            return {pair: UNKNOWN_VALUE for pair in unique_pairs}

        response_text = completion.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if response_text.startswith("```"):
            lines = response_text.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = '\n'.join(lines).strip()

        # Parse the JSON array response
        mappings_array = json.loads(response_text)

        # Convert array to dict mapping (raw_unit, lab_name) -> standardized_unit
        result = {}
        for item in mappings_array:
            raw_unit = item.get("raw_unit")
            lab_name = item.get("lab_name")
            standardized = item.get("standardized_unit")

            if raw_unit is None or lab_name is None:
                continue

            # Validate standardized unit
            if standardized and (standardized == UNKNOWN_VALUE or standardized in standardized_units):
                result[(raw_unit, lab_name)] = standardized
            else:
                logger.warning(f"LLM returned invalid standardized unit: '{standardized}' for ({raw_unit}, {lab_name}), using $UNKNOWN$")
                result[(raw_unit, lab_name)] = UNKNOWN_VALUE

        # Ensure all unique pairs have a mapping
        for pair in unique_pairs:
            if pair not in result:
                logger.warning(f"LLM didn't return mapping for {pair}, using $UNKNOWN$")
                result[pair] = UNKNOWN_VALUE

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse unit standardization response as JSON: {e}")
        return {pair: UNKNOWN_VALUE for pair in unique_pairs}
    except Exception as e:
        logger.error(f"Error during unit standardization: {e}")
        return {pair: UNKNOWN_VALUE for pair in unique_pairs}


########################################
# The Single-PDF Processor
########################################

def process_single_pdf(
    pdf_path: Path, output_dir: Path, self_consistency_model_id: str,
    extract_model_id: str, n_extract: int
) -> Optional[Path]:
    pdf_stem = pdf_path.stem
    doc_out_dir = output_dir / pdf_stem
    doc_out_dir.mkdir(exist_ok=True, parents=True)
    pdf_level_csv_path = doc_out_dir / f"{pdf_stem}.csv"

    # Load standardized lab names and units for this PDF processing
    standardized_names = load_standardized_lab_names()
    standardized_units = load_standardized_lab_units()

    try:
        logger.info(f"[{pdf_stem}] - processing...")
        copied_pdf_path = doc_out_dir / pdf_path.name
        if not copied_pdf_path.exists() or copied_pdf_path.stat().st_size != pdf_path.stat().st_size:
            shutil.copy2(pdf_path, copied_pdf_path)

        try: pil_pages = pdf2image.convert_from_path(str(copied_pdf_path))
        except Exception as e:
            logger.error(f"[{pdf_stem}] - Failed to convert PDF: {e}"); return None

        all_page_lab_results = []
        first_page_report_data = {}
        resolved_doc_date_for_pdf = None

        for page_idx, page_image in enumerate(pil_pages):
            page_file_name = f"{pdf_stem}.{page_idx+1:03d}"
            page_jpg_path = doc_out_dir / f"{page_file_name}.jpg"

            if not page_jpg_path.exists():
                processed_image = preprocess_page_image(page_image)
                processed_image.save(page_jpg_path, "JPEG", quality=95)

            page_json_path = doc_out_dir / f"{page_file_name}.json"
            current_page_json_data = None
            if not page_json_path.exists():
                try:
                    page_json_dict, _ = self_consistency(
                        extract_labs_from_page_image, self_consistency_model_id, n_extract,
                        page_jpg_path, extract_model_id # fn, model_id, n, *args for fn
                    )
                    current_page_json_data = page_json_dict # Already validated dict from extract_labs
                    page_json_path.write_text(json.dumps(current_page_json_data, indent=2, ensure_ascii=False), encoding='utf-8')
                except Exception as e:
                    logger.error(f"[{page_file_name}] Extract. failed: {e}")
                    current_page_json_data = HealthLabReport(lab_results=[]).model_dump(mode='json')
            else:
                try:
                    current_page_json_data = json.loads(page_json_path.read_text(encoding='utf-8'))
                except Exception as e:
                    logger.error(f"[{page_file_name}] Load JSON failed: {e}")
                    current_page_json_data = HealthLabReport(lab_results=[]).model_dump(mode='json')

            # Ensure current_page_json_data is always a dictionary
            if not isinstance(current_page_json_data, dict):
                logger.error(f"[{page_file_name}] current_page_json_data is not a dict: {type(current_page_json_data)}")
                current_page_json_data = HealthLabReport(lab_results=[]).model_dump(mode='json')

            if current_page_json_data:
                if page_idx == 0: # First page processing for report-level data
                    first_page_report_data = {k: v for k, v in current_page_json_data.items() if k != "lab_results"}
                    resolved_doc_date_for_pdf = current_page_json_data.get("collection_date") or current_page_json_data.get("report_date")
                    if resolved_doc_date_for_pdf == "0000-00-00": resolved_doc_date_for_pdf = None
                    if not resolved_doc_date_for_pdf:
                        match = re.search(r"(\d{4}-\d{2}-\d{2})", pdf_stem)
                        if match: resolved_doc_date_for_pdf = match.group(1)
                    if not resolved_doc_date_for_pdf: logger.warning(f"[{pdf_stem}] No date found for PDF.")

                for lab_result_dict in current_page_json_data.get("lab_results", []):
                    lab_result_dict["page_number"] = page_idx + 1
                    lab_result_dict["source_file"] = page_file_name # Page specific source
                    all_page_lab_results.append(lab_result_dict)

        if not all_page_lab_results:
            logger.warning(f"[{pdf_stem}] - No lab results extracted."); pd.DataFrame().to_csv(pdf_level_csv_path, index=False); return pdf_level_csv_path

        # Standardize lab names
        logger.info(f"[{pdf_stem}] - Standardizing lab names...")
        raw_test_names = [result.get("test_name") for result in all_page_lab_results if result.get("test_name")]
        if raw_test_names and standardized_names:
            try:
                # Get unique raw names to avoid duplicate API calls
                unique_raw_names = list(set(raw_test_names))
                name_mapping = standardize_lab_names(unique_raw_names, self_consistency_model_id, standardized_names)

                # Apply standardized names to all results
                for result in all_page_lab_results:
                    raw_name = result.get("test_name")
                    if raw_name:
                        result["lab_name_standardized"] = name_mapping.get(raw_name, UNKNOWN_VALUE)
                    else:
                        result["lab_name_standardized"] = UNKNOWN_VALUE

                logger.info(f"[{pdf_stem}] - Standardized {len(unique_raw_names)} unique test names")
            except Exception as e:
                logger.error(f"[{pdf_stem}] - Standardization failed: {e}, using $UNKNOWN$ for all")
                for result in all_page_lab_results:
                    result["lab_name_standardized"] = UNKNOWN_VALUE
        else:
            logger.warning(f"[{pdf_stem}] - Skipping standardization (no raw names or standardized names unavailable)")
            for result in all_page_lab_results:
                result["lab_name_standardized"] = UNKNOWN_VALUE

        # Standardize lab units (with lab name context for better inference)
        logger.info(f"[{pdf_stem}] - Standardizing lab units...")
        # Build context pairs: (raw_unit, standardized_lab_name) for all results
        # Convert None to "null" string for consistency
        unit_contexts = [
            (result.get("unit") if result.get("unit") is not None else "null",
             result.get("lab_name_standardized", ""))
            for result in all_page_lab_results
        ]

        if unit_contexts and standardized_units:
            try:
                unit_mapping = standardize_lab_units(unit_contexts, self_consistency_model_id, standardized_units)

                # Apply standardized units to all results using (raw_unit, lab_name) lookup
                for result in all_page_lab_results:
                    raw_unit = result.get("unit") if result.get("unit") is not None else "null"
                    lab_name = result.get("lab_name_standardized", "")
                    result["lab_unit_standardized"] = unit_mapping.get((raw_unit, lab_name), UNKNOWN_VALUE)

                unique_raw_units = list(set(result.get("unit") for result in all_page_lab_results))
                logger.info(f"[{pdf_stem}] - Standardized {len(unique_raw_units)} unique units")
            except Exception as e:
                logger.error(f"[{pdf_stem}] - Unit standardization failed: {e}, using $UNKNOWN$ for all")
                for result in all_page_lab_results:
                    result["lab_unit_standardized"] = UNKNOWN_VALUE
        else:
            logger.warning(f"[{pdf_stem}] - Skipping unit standardization (no raw units or standardized units unavailable)")
            for result in all_page_lab_results:
                result["lab_unit_standardized"] = UNKNOWN_VALUE

        pdf_df = pd.DataFrame(all_page_lab_results)
        if resolved_doc_date_for_pdf: pdf_df["date"] = resolved_doc_date_for_pdf
        else: pdf_df["date"] = None

        # Ensure core LabResult columns exist
        core_cols = list(LabResult.model_fields.keys()) + ["date"]
        for col_name in core_cols:
            if col_name not in pdf_df.columns: pdf_df[col_name] = None

        pdf_df = pdf_df[[col for col in core_cols if col in pdf_df.columns]] # Select relevant known columns
        pdf_df.to_csv(pdf_level_csv_path, index=False, encoding='utf-8')
        logger.info(f"[{pdf_stem}] - processing finished. CSV: {pdf_level_csv_path}")
        return pdf_level_csv_path
    except Exception as e:
        logger.error(f"[{pdf_stem}] - Unhandled exception during processing: {e}", exc_info=True)
        return None

########################################
# The Main Function
########################################

def plot_lab_enum(args):
    lab_name_enum_val, merged_df_path_str, plots_dir_str, output_plots_dir_str = args
    import pandas as pd # Keep imports inside for multiprocessing safety
    import matplotlib.pyplot as plt
    import numpy as np
    import re
    from pathlib import Path

    date_col, value_col, group_col, unit_col = PLOTTING_DATE_COL, PLOTTING_VALUE_COL, PLOTTING_GROUP_COL, PLOTTING_UNIT_COL
    try:
        merged_df = pd.read_csv(Path(merged_df_path_str))
        if date_col not in merged_df.columns or value_col not in merged_df.columns or group_col not in merged_df.columns:
            print(f"Plotting Warning: Missing required columns in {merged_df_path_str}"); return
        merged_df[date_col] = pd.to_datetime(merged_df[date_col], errors="coerce")
        df_lab = merged_df[merged_df[group_col] == lab_name_enum_val].copy()
        if df_lab.empty or df_lab[date_col].isnull().all() or len(df_lab[df_lab[value_col].notna()]) < 2: return # Need at least 2 valid points
        
        df_lab = df_lab.sort_values(date_col, ascending=True)
        unit_str = next((u for u in df_lab[unit_col].dropna().astype(str).unique() if u), "") if unit_col in df_lab.columns else ""
        y_label = f"Value ({unit_str})" if unit_str else "Value"
        title = f"{lab_name_enum_val} " + (f" [{unit_str}]" if unit_str else "")
        
        plt.figure(figsize=(12, 6))
        plt.plot(df_lab[date_col], df_lab[value_col], marker='o', linestyle='-')

        # Ensure the first and last year are always shown on the x-axis
        import matplotlib.dates as mdates
        start_date = df_lab[date_col].min()
        end_date = df_lab[date_col].max()
        ax = plt.gca()
        ax.set_xlim(start_date, end_date)
        year_ticks = pd.date_range(start_date, end_date, freq="YS").to_pydatetime().tolist()
        ticks = []
        if start_date not in year_ticks:
            ticks.append(start_date)
        ticks.extend(year_ticks)
        if end_date not in year_ticks:
            ticks.append(end_date)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # Add light green band for reference range if available
        if "healthy_range_min" in df_lab.columns and "healthy_range_max" in df_lab.columns:
            min_vals = df_lab["healthy_range_min"].dropna()
            max_vals = df_lab["healthy_range_max"].dropna()
            if not min_vals.empty and not max_vals.empty:
                y_min_mode = float(min_vals.mode()[0])
                y_max_mode = float(max_vals.mode()[0])
                x_start = df_lab[date_col].min()
                x_end = df_lab[date_col].max()
                #light_green = "#d8f5d0"
                #light_red = "#f5d8d8"  # same tone as green
                light_green = "#b7e6a1"  # slightly darker than before
                light_red = "#e6b7b7"
                plt.fill_between(
                    [x_start, x_end],
                    y_min_mode,
                    y_max_mode,
                    color=light_green,
                    alpha=0.6,
                    label="Reference Range",
                )
                cur_ymin, cur_ymax = ax.get_ylim()
                ax.set_ylim(min(cur_ymin, y_min_mode), max(cur_ymax, y_max_mode))
                cur_ymin, cur_ymax = ax.get_ylim()
                plt.fill_between(
                    [x_start, x_end],
                    cur_ymin,
                    y_min_mode,
                    color=light_red,
                    alpha=0.3,
                    label="Below Range",
                )
                plt.fill_between(
                    [x_start, x_end],
                    y_max_mode,
                    cur_ymax,
                    color=light_red,
                    alpha=0.3,
                    label="Above Range",
                )

        # Optional: Add reference range lines (mode)
        if "healthy_range_min" in df_lab.columns and df_lab["healthy_range_min"].notna().any():
            y_min_line = float(df_lab["healthy_range_min"].mode()[0])
            plt.axhline(y=y_min_line, color='gray', linestyle='--', label='Ref Min (mode)')
            cur_ymin, cur_ymax = ax.get_ylim()
            ax.set_ylim(min(cur_ymin, y_min_line), cur_ymax)
        if "healthy_range_max" in df_lab.columns and df_lab["healthy_range_max"].notna().any():
             y_max_line = float(df_lab["healthy_range_max"].mode()[0])
             plt.axhline(y=y_max_line, color='gray', linestyle='--', label='Ref Max (mode)')
             cur_ymin, cur_ymax = ax.get_ylim()
             ax.set_ylim(cur_ymin, max(cur_ymax, y_max_line))
        if ("healthy_range_min" in df_lab.columns and df_lab["healthy_range_min"].notna().any()) or \
           ("healthy_range_max" in df_lab.columns and df_lab["healthy_range_max"].notna().any()):
            plt.legend()

        plt.title(title, fontsize=16); plt.xlabel("Date", fontsize=12); plt.ylabel(y_label, fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        # Convert '(%)' to '(percentage)' before sanitizing the filename
        lab_name_for_filename = str(lab_name_enum_val).replace('(%)', '(percentage)')
        # Allow parentheses in filenames by including them in the allowed characters
        safe_lab_name = re.sub(r'[^\w\-_. ()]', '_', lab_name_for_filename)
        plt.savefig(Path(plots_dir_str) / f"{safe_lab_name}.png")
        plt.savefig(Path(output_plots_dir_str) / f"{safe_lab_name}.png")
        plt.close()
    except Exception as e: print(f"Plotting Error for {lab_name_enum_val}: {e}")


def main():
    # Clear previous logs once per run and reconfigure file handlers
    setup_logging(clear_logs=True)

    config = load_env_config()
    output_dir = config["output_path"]

    pdf_files = sorted(list(config["input_path"].glob(config["input_file_regex"])))
    logger.info(f"Found {len(pdf_files)} PDF(s) matching '{config['input_file_regex']}' in '{config['input_path']}'")
    if not pdf_files: logger.warning("No PDF files found. Exiting."); return

    n_workers = min(config["max_workers"], len(pdf_files))
    logger.info(f"Using up to {n_workers} worker(s) for PDF processing.")
    tasks = [(pdf, output_dir, config["self_consistency_model_id"],
              config["extract_model_id"], config["n_extractions"]) for pdf in pdf_files]

    processed_pdf_csv_paths = []
    with Pool(n_workers) as pool:
        results = pool.starmap(process_single_pdf, tasks)
        for path in results: 
            if path and path.exists(): processed_pdf_csv_paths.append(path)
    
    if not processed_pdf_csv_paths: logger.error("No PDFs successfully processed. Exiting."); return
    logger.info(f"Successfully processed {len(processed_pdf_csv_paths)} PDFs.")

    dataframes = []
    for csv_path in processed_pdf_csv_paths:
        try:
            if csv_path.stat().st_size > 0:
                df = pd.read_csv(csv_path, encoding='utf-8')
                df['source_file'] = csv_path.name # PDF's CSV filename (e.g., doc_stem.csv)
                dataframes.append(df)
        except Exception as e: logger.error(f"Failed to read/process {csv_path}: {e}")

    if not dataframes: logger.error("No data loaded from CSVs. Merged file will be empty."); merged_df = pd.DataFrame()
    else: merged_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Merged data: {len(merged_df)} rows.")

    # --------- Add normalized columns (post-processing) ---------

    # Normalize unit mapping (case-insensitive, symbol normalization)
    unit_normalization_map = {
        # Percentage
        '%': '%', 'percent': '%', 'per cent': '%',
        # mg/dL variations
        'mg/dl': 'mg/dL', 'mg/dL': 'mg/dL', 'mg/deciliter': 'mg/dL',
        # g/dL variations
        'g/dl': 'g/dL', 'g/dL': 'g/dL',
        # U/L variations
        'u/l': 'U/L', 'U/L': 'U/L', 'UI/L': 'U/L',
        # Add more as needed
    }

    def normalize_unit(raw_unit: str) -> str:
        """Normalize unit to standard form."""
        if pd.isna(raw_unit):
            return None
        normalized = unit_normalization_map.get(str(raw_unit).strip(), str(raw_unit).strip())
        return normalized

    if "unit" in merged_df.columns:
        merged_df["lab_unit"] = merged_df["unit"].apply(normalize_unit)
    else:
        merged_df["lab_unit"] = None

    config_path = Path("config")
    lab_specs_exists = (config_path / "lab_specs.json").exists()
    if not lab_specs_exists:
        logger.error(f"Missing lab_specs.json in '{config_path}'. Normalized columns will be incomplete.")
        # Initialize columns to prevent KeyErrors
        derived_cols_to_init = [
            "lab_type", "lab_name", "lab_name_slug",
            "value_normalized", "unit_normalized",
            "reference_min_normalized", "reference_max_normalized",
            "is_out_of_reference", "healthy_range_min", "healthy_range_max", "is_in_healthy_range"
        ]
        for col_key in derived_cols_to_init:
            if col_key not in merged_df.columns:
                merged_df[col_key] = None
    else:
        with open(config_path / "lab_specs.json", "r", encoding="utf-8") as f:
            lab_specs = json.load(f)

        # Load lab_type mapping from lab_specs
        lab_type_map = {name: spec.get('lab_type', 'blood') for name, spec in lab_specs.items()}

        # Use lab_name_standardized if available (new standardization approach)
        # Otherwise fall back to empty lab_name column
        if "lab_name_standardized" in merged_df.columns:
            merged_df["lab_name"] = merged_df["lab_name_standardized"]
        elif "test_name" in merged_df.columns:
            # No standardized names available, just copy test_name
            merged_df["lab_name"] = merged_df["test_name"]
        else:
            merged_df["lab_name"] = None

        # Look up lab_type from lab_specs config instead of parsing prefix
        def lookup_lab_type(lab_name: str) -> str:
            """Look up lab type from config mapping."""
            if pd.isna(lab_name) or lab_name == UNKNOWN_VALUE:
                return "blood"  # default
            # Look up in lab_type_map
            return lab_type_map.get(lab_name, "blood")  # default to blood if not found

        if "lab_name" in merged_df.columns:
            merged_df["lab_type"] = merged_df["lab_name"].apply(lookup_lab_type)

            # Create slug for tracking
            merged_df["lab_name_slug"] = merged_df.apply(
                lambda row: f"{row['lab_type']}-{slugify(row.get('test_name', ''))}",
                axis=1
            )
        else:
            merged_df["lab_name"] = UNKNOWN_VALUE
            merged_df["lab_type"] = "blood"
            merged_df["lab_name_slug"] = ""

        # Convert to primary units
        def convert_to_primary_unit(row):
            """Convert values to primary unit defined in lab_specs."""
            lab_name = row.get("lab_name", "")
            lab_unit = row.get("lab_unit", "")
            val = row.get("value")
            r_min = row.get("reference_min")
            r_max = row.get("reference_max")

            # Default: no conversion
            v_f, r_min_f, r_max_f, u_f = val, r_min, r_max, lab_unit

            # Check if we have a spec for this lab
            if not lab_name or pd.isna(lab_name) or lab_name not in lab_specs:
                return pd.Series([v_f, r_min_f, r_max_f, u_f])

            spec = lab_specs[lab_name]
            prim_unit = spec.get("primary_unit")

            if not prim_unit:
                return pd.Series([v_f, r_min_f, r_max_f, u_f])

            # If already in primary unit, no conversion needed
            if lab_unit == prim_unit:
                return pd.Series([val, r_min, r_max, prim_unit])

            # Find conversion factor
            factor = next(
                (alt.get("factor") for alt in spec.get("alternatives", []) if alt.get("unit") == lab_unit),
                None
            )

            if factor:
                try:
                    v_f = float(val) * float(factor) if pd.notnull(val) else val
                except:
                    pass
                try:
                    r_min_f = float(r_min) * float(factor) if pd.notnull(r_min) else r_min
                except:
                    pass
                try:
                    r_max_f = float(r_max) * float(factor) if pd.notnull(r_max) else r_max
                except:
                    pass
                u_f = prim_unit

            return pd.Series([v_f, r_min_f, r_max_f, u_f])

        if all(c in merged_df.columns for c in ["lab_name", "lab_unit", "value", "reference_min", "reference_max"]):
            final_cols_df = merged_df.apply(convert_to_primary_unit, axis=1)
            final_cols_df.columns = ["value_normalized", "reference_min_normalized", "reference_max_normalized", "unit_normalized"]
            merged_df = pd.concat([merged_df, final_cols_df], axis=1)
        else:
            merged_df["value_normalized"] = merged_df.get("value")
            merged_df["unit_normalized"] = merged_df.get("lab_unit")
            merged_df["reference_min_normalized"] = merged_df.get("reference_min")
            merged_df["reference_max_normalized"] = merged_df.get("reference_max")

        # Compute if value is outside reference range
        def compute_is_out_of_reference(row):
            """Check if value is outside the reference range from PDF."""
            val = row.get("value_normalized")
            minv = row.get("reference_min_normalized")
            maxv = row.get("reference_max_normalized")

            if pd.isna(val):
                return None
            try:
                val_f = float(val)
            except:
                return None

            is_low = pd.notna(minv) and val_f < float(minv)
            is_high = pd.notna(maxv) and val_f > float(maxv)
            return is_low or is_high if (pd.notna(minv) or pd.notna(maxv)) else None

        if all(c in merged_df.columns for c in ["value_normalized", "reference_min_normalized", "reference_max_normalized"]):
            merged_df["is_out_of_reference"] = merged_df.apply(compute_is_out_of_reference, axis=1)
        else:
            merged_df["is_out_of_reference"] = None

        # Get healthy range from lab specs
        def get_healthy_range(row):
            """Get healthy range from lab specs config."""
            lab_name = row.get("lab_name", "")
            if not lab_name or pd.isna(lab_name) or lab_name not in lab_specs:
                return pd.Series([None, None])
            healthy = lab_specs[lab_name].get("ranges", {}).get("healthy")
            return pd.Series([healthy.get("min"), healthy.get("max")]) if healthy else pd.Series([None, None])

        if "lab_name" in merged_df.columns:
            merged_df[["healthy_range_min", "healthy_range_max"]] = merged_df.apply(get_healthy_range, axis=1)
        else:
            merged_df["healthy_range_min"] = None
            merged_df["healthy_range_max"] = None

        # Check if value is in healthy range
        def compute_is_in_healthy_range(row):
            """Check if value is within healthy range from lab specs."""
            val = row.get("value_normalized")
            min_h = row.get("healthy_range_min")
            max_h = row.get("healthy_range_max")

            if pd.isna(val):
                return None
            try:
                val_f = float(val)
            except:
                return None

            if pd.isna(min_h) and pd.isna(max_h):
                return None  # No healthy range defined

            too_low = pd.notna(min_h) and val_f < float(min_h)
            too_high = pd.notna(max_h) and val_f > float(max_h)
            return not (too_low or too_high)

        if all(c in merged_df.columns for c in ["value_normalized", "healthy_range_min", "healthy_range_max"]):
            merged_df["is_in_healthy_range"] = merged_df.apply(compute_is_in_healthy_range, axis=1)
        else:
            merged_df["is_in_healthy_range"] = None

    # --------- Deduplicate by (date, lab_name) keeping best match ---------
    # Only if lab_specs is loaded and required columns exist
    if lab_specs_exists and "date" in merged_df.columns and "lab_name" in merged_df.columns and "lab_unit" in merged_df.columns:
        def pick_best_dupe(group):
            """Pick best duplicate: prefer primary unit if multiple entries exist."""
            lab_name = group.iloc[0]["lab_name"]
            primary_unit = None
            if lab_name and lab_name in lab_specs:
                primary_unit = lab_specs[lab_name].get("primary_unit")
            if primary_unit and (group["lab_unit"] == primary_unit).any():
                return group[group["lab_unit"] == primary_unit].iloc[0]
            else:
                return group.iloc[0]

        merged_df = (
            merged_df
            .groupby(["date", "lab_name"], dropna=False, as_index=False)
            .apply(pick_best_dupe, include_groups=True)
            .reset_index(drop=True)
        )

    export_cols_ordered = get_export_columns_from_schema(COLUMN_SCHEMA)
    final_select_cols = [col for col in export_cols_ordered if col in merged_df.columns] + \
                        [col for col in merged_df.columns if col not in export_cols_ordered] # Keep all columns
    merged_df = merged_df[final_select_cols]

    date_col_name = PLOTTING_DATE_COL # Use schema defined date column
    if date_col_name in merged_df.columns:
        merged_df[date_col_name] = pd.to_datetime(merged_df[date_col_name], errors="coerce")
        merged_df = merged_df.sort_values(by=[date_col_name, PLOTTING_GROUP_COL], ascending=[False, True]) # Sort for most_recent
    
    dtype_map_from_schema = get_dtype_map_from_schema(COLUMN_SCHEMA)
    for col, target_dtype in dtype_map_from_schema.items():
        if col in merged_df.columns:
            try:
                if target_dtype == "datetime64[ns]": merged_df[col] = pd.to_datetime(merged_df[col], errors="coerce")
                elif target_dtype == "boolean": merged_df[col] = merged_df[col].astype("boolean")
                elif target_dtype == "Int64": merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").astype("Int64")
                elif "float" in target_dtype: merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")
                # other types like string are often fine as 'object'
            except Exception as e: logger.warning(f"Dtype conversion failed for {col} to {target_dtype}: {e}")

    merged_csv_path = output_dir / "all.csv"
    merged_df.to_csv(merged_csv_path, index=False, encoding='utf-8')
    logger.info(f"Saved main merged CSV: {merged_csv_path}")

    # --------- Export Enhanced Excel file (all.xlsx) with two tabs ---------
    excel_path = output_dir / "all.xlsx"
    excel_hidden_cols = get_hidden_excel_columns_from_schema(COLUMN_SCHEMA)
    excel_col_widths = get_excel_widths_from_schema(COLUMN_SCHEMA)

    # Prepare data for the second sheet: Most Recent By Enum
    # Ensure merged_df is sorted by date descending for `drop_duplicates` to pick the most recent
    if date_col_name in merged_df.columns and PLOTTING_GROUP_COL in merged_df.columns:
        # Sorting is already done: by=[date_col_name, PLOTTING_GROUP_COL], ascending=[False, True]
        most_recent_by_enum_df = merged_df.drop_duplicates(subset=[PLOTTING_GROUP_COL], keep='first').copy()
    else:
        logger.warning(f"Cannot generate 'MostRecentByEnum' sheet due to missing columns: '{date_col_name}' or '{PLOTTING_GROUP_COL}'.")
        most_recent_by_enum_df = pd.DataFrame() # Empty DF if columns missing

    with pd.ExcelWriter(excel_path, engine="xlsxwriter", datetime_format='yyyy-mm-dd', date_format='yyyy-mm-dd') as writer:
        # First sheet: AllData
        merged_df.to_excel(writer, sheet_name="AllData", index=False)
        worksheet_all = writer.sheets["AllData"]
        worksheet_all.freeze_panes(1, 0)
        for idx, col_name in enumerate(merged_df.columns):
            width = excel_col_widths.get(col_name, 12)
            options = {'hidden': True} if col_name in excel_hidden_cols else {}
            worksheet_all.set_column(idx, idx, width, None, options)
        
        # Second sheet: MostRecentByEnum
        if not most_recent_by_enum_df.empty:
            most_recent_by_enum_df.to_excel(writer, sheet_name="MostRecentByEnum", index=False)
            worksheet_recent = writer.sheets["MostRecentByEnum"]
            worksheet_recent.freeze_panes(1, 0)
            for idx, col_name in enumerate(most_recent_by_enum_df.columns): # Use columns from this df
                width = excel_col_widths.get(col_name, 12)
                options = {'hidden': True} if col_name in excel_hidden_cols else {}
                worksheet_recent.set_column(idx, idx, width, None, options)
        else:
            # Create an empty sheet if no data for it, or skip
             pd.DataFrame().to_excel(writer, sheet_name="MostRecentByEnum", index=False)


    logger.info(f"Saved enhanced Excel with AllData and MostRecentByEnum tabs to: {excel_path}")

    # Removed generation of all.final.* and all.final.blood-most-recent.* files

    logger.info("All data processing and file exports finished.")

    # --------- Plotting Section ---------
    if PLOTTING_GROUP_COL in merged_df.columns and \
       date_col_name in merged_df.columns and \
       PLOTTING_VALUE_COL in merged_df.columns:
        
        plots_base_dir = Path("plots")
        plots_base_dir.mkdir(exist_ok=True)
        clear_directory(plots_base_dir)

        output_plots_dir = output_dir / "plots"
        output_plots_dir.mkdir(exist_ok=True)
        clear_directory(output_plots_dir)

        logger.info(
            f"Starting plot generation into '{plots_base_dir}' and '{output_plots_dir}'."
        )
        
        if not pd.api.types.is_datetime64_any_dtype(merged_df[date_col_name]): # Ensure date is datetime for plotting
            merged_df[date_col_name] = pd.to_datetime(merged_df[date_col_name], errors="coerce")

        unique_lab_enums = merged_df[merged_df[PLOTTING_VALUE_COL].notna()][PLOTTING_GROUP_COL].dropna().unique()
        plot_args_list = [(lab_enum, str(merged_csv_path), str(plots_base_dir), str(output_plots_dir)) 
                          for lab_enum in unique_lab_enums 
                          if len(merged_df[(merged_df[PLOTTING_GROUP_COL] == lab_enum) & merged_df[date_col_name].notna() & merged_df[PLOTTING_VALUE_COL].notna()]) >= 2]

        if plot_args_list:
            plot_workers = min(max(1, (os.cpu_count() or 1) -1 ), len(plot_args_list)) 
            logger.info(f"Plotting for {len(plot_args_list)} lab enums using {plot_workers} worker(s).")
            with Pool(processes=plot_workers) as plot_pool:
                plot_pool.map(plot_lab_enum, plot_args_list)
            logger.info("Plotting finished.")
        else: logger.info("No lab enums with sufficient data for plotting.")
    else: logger.warning("Essential columns for plotting missing. Skipping plotting.")

if __name__ == "__main__":
    main()