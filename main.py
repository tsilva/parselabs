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
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add OpenAI import for OpenRouter
from openai import OpenAI, APIError

########################################
# Centralized Column Schema
########################################

COLUMN_SCHEMA = {
    # Fields from LabResult, potentially modified or used directly
    # 'dtype' specifies the target pandas dtype.
    # 'excel_width' is the default column width in Excel.
    # 'excel_hidden' flags if the column should be hidden in the main Excel export.
    # 'final_export' flags if the column is part of the "final" summarized export.
    # 'plotting_role' identifies columns for specific roles in plotting (date, value, group, unit).

    "date": {"dtype": "datetime64[ns]", "excel_width": 13, "final_export": True, "plotting_role": "date"},
    "lab_type": {"dtype": "str", "excel_width": 10, "final_export": True},
    "lab_name": {"dtype": "str", "excel_width": 35, "excel_hidden": True}, # Increased width
    "lab_code": {"dtype": "str", "excel_width": 15},
    "lab_value": {"dtype": "float64", "excel_width": 12, "excel_hidden": True},
    "lab_unit": {"dtype": "str", "excel_width": 15, "excel_hidden": True}, # Increased width
    "lab_method": {"dtype": "str", "excel_width": 20},
    "lab_range_min": {"dtype": "float64", "excel_width": 12, "excel_hidden": True},
    "lab_range_max": {"dtype": "float64", "excel_width": 12, "excel_hidden": True},
    "reference_range_text": {"dtype": "str", "excel_width": 25},
    "is_flagged": {"dtype": "boolean", "excel_width": 10, "excel_hidden": True},
    "lab_comments": {"dtype": "str", "excel_width": 40},
    "confidence": {"dtype": "float64", "excel_width": 10, "excel_hidden": True},
    "lack_of_confidence_reason": {"dtype": "str", "excel_width": 30},
    "source_text": {"dtype": "str", "excel_width": 50}, # Increased width
    "page_number": {"dtype": "Int64", "excel_width": 8}, # Pandas nullable integer
    "source_file": {"dtype": "str", "excel_width": 25}, # Increased width, refers to per-PDF CSV name in merged DF

    # Derived columns in main()
    "lab_name_slug": {"dtype": "str", "excel_width": 30, "excel_hidden": True, "derivation_logic": "map_lab_name_slug"},
    "lab_name_enum": {"dtype": "str", "excel_width": 30, "final_export": True, "derivation_logic": "map_lab_name_enum", "plotting_role": "group"},
    "lab_unit_enum": {"dtype": "str", "excel_width": 15, "derivation_logic": "map_lab_unit_enum"},

    "lab_value_final": {"dtype": "float64", "excel_width": 14, "final_export": True, "derivation_logic": "convert_to_primary_unit", "plotting_role": "value"},
    "lab_unit_final": {"dtype": "str", "excel_width": 14, "final_export": True, "derivation_logic": "convert_to_primary_unit", "plotting_role": "unit"},
    "lab_range_min_final": {"dtype": "float64", "excel_width": 14, "final_export": True, "derivation_logic": "convert_to_primary_unit"},
    "lab_range_max_final": {"dtype": "float64", "excel_width": 14, "final_export": True, "derivation_logic": "convert_to_primary_unit"},

    "is_flagged_final": {"dtype": "boolean", "excel_width": 14, "derivation_logic": "compute_is_flagged_final"},
    "healthy_range_min": {"dtype": "float64", "excel_width": 16, "derivation_logic": "get_healthy_range"}, # Increased width
    "healthy_range_max": {"dtype": "float64", "excel_width": 16, "derivation_logic": "get_healthy_range"}, # Increased width
    "is_in_healthy_range": {"dtype": "boolean", "excel_width": 18, "derivation_logic": "compute_is_in_healthy_range"}, # Increased width
}

# Helper functions to derive lists/dicts from COLUMN_SCHEMA
def get_export_columns_from_schema(schema: dict) -> list:
    """Returns an ordered list of columns for the main export."""
    # Define the comprehensive order for all.csv and all.xlsx
    ordered_keys = [
        "date", "lab_type", "lab_name", "lab_name_enum", "lab_name_slug",
        "lab_value", "lab_unit", "lab_unit_enum",
        "lab_range_min", "lab_range_max", "reference_range_text",
        "lab_value_final", "lab_unit_final",
        "lab_range_min_final", "lab_range_max_final",
        "is_flagged", "is_flagged_final",
        "healthy_range_min", "healthy_range_max", "is_in_healthy_range",
        "confidence", "lab_code", "lab_method", "lab_comments",
        "lack_of_confidence_reason", "source_text", "page_number", "source_file"
    ]
    return [key for key in ordered_keys if key in schema]

def get_final_export_columns_ordered(schema: dict) -> list:
    """Returns an ordered list of columns for the final summarized export."""
    ordered_final_keys = [
        "date", "lab_type", "lab_name_enum", "lab_value_final",
        "lab_unit_final", "lab_range_min_final", "lab_range_max_final"
    ]
    return [
        key for key in ordered_final_keys
        if key in schema and schema[key].get("final_export")
    ]

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

# Constants for plotting roles, derived from the schema
PLOTTING_DATE_COL = next((col for col, props in COLUMN_SCHEMA.items() if props.get("plotting_role") == "date"), "date")
PLOTTING_VALUE_COL = next((col for col, props in COLUMN_SCHEMA.items() if props.get("plotting_role") == "value"), "lab_value_final")
PLOTTING_GROUP_COL = next((col for col, props in COLUMN_SCHEMA.items() if props.get("plotting_role") == "group"), "lab_name_enum")
PLOTTING_UNIT_COL = next((col for col, props in COLUMN_SCHEMA.items() if props.get("plotting_role") == "unit"), "lab_unit_final")


########################################
# Config / Logging
########################################

UNKNOWN_VALUE = "$UNKNOWN$"

# Create logs directory if it doesn't exist
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Truncate log files (clear contents)
INFO_LOG_PATH = LOG_DIR / "info.log"
ERROR_LOG_PATH = LOG_DIR / "error.log"
for log_file in [INFO_LOG_PATH, ERROR_LOG_PATH]:
    if log_file.exists(): log_file.write_text("", encoding='utf-8')

# Create file handlers with UTF-8 encoding
info_handler = logging.FileHandler(INFO_LOG_PATH, encoding='utf-8')
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(ERROR_LOG_PATH, encoding='utf-8')
error_handler.setLevel(logging.ERROR)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# Create logger and add handlers
logger = logging.getLogger(__name__)
logger.addHandler(info_handler)
logger.addHandler(error_handler)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def load_env_config():
    """
    Load environment variables and return as a dict.
    """
    
    input_path = os.getenv("INPUT_PATH")
    input_file_regex = os.getenv("INPUT_FILE_REGEX")
    output_path = os.getenv("OUTPUT_PATH")
    self_consistency_model_id = os.getenv("SELF_CONSISTENCY_MODEL_ID")
    transcribe_model_id = os.getenv("TRANSCRIBE_MODEL_ID")
    n_transcriptions = int(os.getenv("N_TRANSCRIPTIONS"))
    extract_model_id = os.getenv("EXTRACT_MODEL_ID")
    n_extractions = int(os.getenv("N_EXTRACTIONS"))
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    max_workers = int(os.getenv("MAX_WORKERS"))

    if not self_consistency_model_id: raise ValueError("SELF_CONSISTENCY_MODEL_ID not set")
    if not transcribe_model_id: raise ValueError("TRANSCRIBE_MODEL_ID not set")
    if not extract_model_id: raise ValueError("EXTRACT_MODEL_ID not set")
    if not input_path or not Path(input_path).exists(): raise ValueError(f"INPUT_PATH not set or does not exist: {input_path}")
    if not input_file_regex: raise ValueError("INPUT_FILE_REGEX not set")
    if not output_path or not Path(output_path).exists(): raise ValueError(f"OUTPUT_PATH not set or does not exist: {output_path}") # Check existence
    if not openrouter_api_key: raise ValueError("OPENROUTER_API_KEY not set")

    # Ensure output_path is a directory
    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)


    return {
        "input_path" : Path(input_path),
        "input_file_regex" : input_file_regex,
        "output_path" : output_path_obj, # Use Path object
        "self_consistency_model_id" : self_consistency_model_id,
        "transcribe_model_id" : transcribe_model_id,
        "n_transcriptions": n_transcriptions,
        "extract_model_id" : extract_model_id,
        "n_extractions": n_extractions,
        "openrouter_api_key": openrouter_api_key,
        "max_workers": max_workers
    }

########################################
# LLM Tools / Pydantic Models
########################################

class LabType(str, Enum):
    BLOOD = "blood"
    URINE = "urine"
    SALIVA = "saliva"
    FECES = "feces" # Added 'feces' as it was in description but not enum
    UNKNOWN = "unknown" # Added for cases where type isn't clear

class LabResult(BaseModel):
    lab_type: LabType = Field(
        default=LabType.UNKNOWN, # Default to unknown
        description="Type of laboratory test (must be one of: blood, urine, saliva, feces, unknown)"
    )
    lab_name: str = Field(
        min_length=1,
        description="Name of the laboratory test as extracted verbatim from the document"
    )
    lab_code: Optional[str] = Field(
        default=None,
        description="Standardized code for the laboratory test (e.g., LOINC, CPT), if available"
    )
    lab_value: Optional[float] = Field( # Made Optional to handle non-numeric/missing cleanly
        default=None,
        description="Quantitative result of the laboratory test (positive/negative should be 1/0 if possible, otherwise text)"
    )
    lab_unit: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Unit of measurement as extracted verbatim (e.g., mg/dL, mmol/L, IU/mL, boolean)"
    )
    lab_method: Optional[str] = Field(
        default=None,
        description="Analytical method or technique as extracted verbatim (e.g., ELISA, HPLC, Microscopy), if available"
    )
    lab_range_min: Optional[float] = Field(
        default=None,
        description="Lower bound of the reference range, if available"
    )
    lab_range_max: Optional[float] = Field(
        default=None,
        description="Upper bound of the reference range, if available"
    )
    reference_range_text: Optional[str] = Field(
        default=None,
        description="Reference range as shown in the document, verbatim (e.g., '4.0-10.0', 'Normal: <5')"
    )
    is_flagged: Optional[bool] = Field(
        default=None,
        description="True if the result is flagged as abnormal/high/low in the document, else False or None if not specified"
    )
    lab_comments: Optional[str] = Field(
        default=None,
        description="Additional notes or observations about this result, if available"
    )
    confidence: float = Field(
        default=0.5, # Provide a default
        ge=0.0, le=1.0,
        description="Confidence score of the extraction process, ranging from 0 to 1"
    )
    lack_of_confidence_reason: Optional[str] = Field(
        default=None,
        description="Reason for low extraction confidence"
    )
    source_text: Optional[str] = Field(
        default=None,
        description="The exact line or snippet from the document where this result was extracted"
    )
    page_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="Page number in the PDF where this result was found, if available"
    )
    source_file: Optional[str] = Field( # e.g. page filename
        default=None,
        description="The filename or identifier of the source file/page"
    )

class HealthLabReport(BaseModel):
    report_date: Optional[str] = Field(
        default=None, # Use None as default for optional fields
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date the laboratory report was issued (YYYY-MM-DD), if unavailable use null"
    )
    collection_date: Optional[str] = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date the specimen was collected (YYYY-MM-DD), if available (also called subscription date), if unavailable use null"
    )
    lab_facility: Optional[str] = Field(default=None, description="Name of the laboratory or facility that performed the tests, if available")
    lab_facility_address: Optional[str] = Field(default=None, description="Address of the laboratory or facility, if available")
    patient_name: Optional[str] = Field(default=None, description="Full name of the patient")
    patient_id: Optional[str] = Field(default=None, description="Patient identifier or medical record number, if available")
    patient_birthdate: Optional[str] = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Birthdate of the patient (YYYY-MM-DD), if available"
    )
    physician_name: Optional[str] = Field(default=None, description="Name of the requesting or reviewing physician, if available")
    physician_id: Optional[str] = Field(default=None, description="Identifier for the physician, if available")
    page_count: Optional[int] = Field(default=None, ge=1, description="Total number of pages in the report, if available")
    lab_results: List[LabResult] = Field(default_factory=list, description="List of individual laboratory test results in this report")
    source_file: Optional[str] = Field(default=None, description="The filename or identifier of the source PDF file") # e.g. original PDF filename

    def normalize_empty_optionals(self):
        """
        For all optional fields in this report and its LabResult entries,
        replace empty string values ("") with None.
        Pydantic v2 usually handles this well, but explicit conversion can be safer with LLM outputs.
        """
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if value == "" and not self.model_fields[field_name].is_required():
                 # Check if field is Optional (or Union with None)
                if self.model_fields[field_name].outer_type_ is Optional[self.model_fields[field_name].annotation] or \
                   str(self.model_fields[field_name].outer_type_).startswith("typing.Union") and type(None) in self.model_fields[field_name].outer_type_.__args__:
                    setattr(self, field_name, None)
        
        for lab_result in self.lab_results:
            for field_name in lab_result.model_fields:
                value = getattr(lab_result, field_name)
                if value == "" and not lab_result.model_fields[field_name].is_required():
                    if lab_result.model_fields[field_name].outer_type_ is Optional[lab_result.model_fields[field_name].annotation] or \
                       str(lab_result.model_fields[field_name].outer_type_).startswith("typing.Union") and type(None) in lab_result.model_fields[field_name].outer_type_.__args__:
                        setattr(lab_result, field_name, None)

TOOLS = [
    {
        "type": "function",
        "function" : {
            "name": "extract_lab_results",
            "description": f"""
Extract structured laboratory test results from medical documents with high precision.

Specific requirements:
1. Extract EVERY test result visible in the image, including variants with different units.
2. For boolean-like results (e.g., Positive/Negative, Detected/Not Detected), use 1 for true/positive/detected and 0 for false/negative/not detected for the 'lab_value' field if possible. If the result is textual and cannot be converted to 1/0, use the text itself and ensure 'lab_unit' reflects this (e.g., 'text' or 'qualitative').
3. Dates must be in ISO 8601 format (YYYY-MM-DD). If a date is truly unavailable, use null.
4. Units must match exactly as shown in the document.
5. Use the most precise schema possible for each field.
6. For each result, include the exact source text/line and page number if possible.
7. If a field is not present or its value is unknown in the document, use null (None). Do not use placeholder strings like '{UNKNOWN_VALUE}' for optional fields. For required fields that are missing, use a sensible default if specified in the schema, otherwise this indicates an issue with the source or extraction.
8. Populate all fields of the 'LabResult' and 'HealthLabReport' models as accurately as possible.
9. If 'report_date' or 'collection_date' is not found, it should be null.
10. 'lab_type' should be one of: blood, urine, saliva, feces, unknown.
""".strip(),
            "parameters": HealthLabReport.model_json_schema()
        }
    }
]

########################################
# Helper Functions
########################################

def hash_file(file_path: Path, length=4) -> str:
    with open(file_path, "rb") as f:
        h = hashlib.md5()
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()[:length]

def preprocess_page_image(image: Image.Image) -> Image.Image:
    """
    Optimize the page image by:
      - Converting to grayscale
      - Resizing only if width exceeds a higher threshold (1200 px)
      - Enhancing contrast for better text readability
      - Saving in a lossless format (PNG) for maximum clarity
    """
    from PIL import Image, ImageEnhance

    gray_image = image.convert('L')
    MAX_WIDTH = 1200
    if gray_image.width > MAX_WIDTH:
        ratio = MAX_WIDTH / gray_image.width
        new_height = int(gray_image.height * ratio)
        gray_image = gray_image.resize((MAX_WIDTH, new_height), Image.Resampling.LANCZOS)

    enhanced_image = ImageEnhance.Contrast(gray_image).enhance(2.0)
    # normalized_image = enhanced_image.quantize(colors=128).convert('L') # Optional
    return enhanced_image # Return enhanced, not quantized by default

def self_consistency(fn, model_id, n, *args, **kwargs):
    """
    Calls the function `fn` N times with the same arguments,
    then uses the LLM to select the most content-consistent result.
    If n == 1, just returns the single result.

    Returns a tuple: (voted_result, [all_versions])
    Stops and raises if any call fails.
    """
    if n == 1:
        result = fn(*args, **kwargs)
        return result, [result]

    def call_with_temp():
        # Only add temperature if it's not already a kwarg for the function
        # Some functions like transcription_from_page_image define their own default temp
        if 'temperature' in fn.__code__.co_varnames:
             return fn(*args, **kwargs, temperature=0.5)
        return fn(*args, **kwargs)


    results = []
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(call_with_temp) for _ in range(n)]
        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
            except APIError as oe:
                logger.error(f"OpenAI API Error during self-consistency task: {oe}")
                for f_cancel in futures:
                    if not f_cancel.done(): f_cancel.cancel()
                raise RuntimeError(f"OpenAI API Error in self-consistency task: {str(oe)}")
            except Exception as e:
                logger.error(f"Error during self-consistency task: {e}")
                for f_cancel in futures:
                    if not f_cancel.done(): f_cancel.cancel()
                raise

    if not results: # Handle case where all calls fail before this point
        raise RuntimeError("All self-consistency calls failed.")

    if all(r == results[0] for r in results):
        return results[0], results

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
    prompt = ""
    prompt += "".join(f"--- Output {i+1} ---\n{json.dumps(v, ensure_ascii=False) if type(v) in [list, dict] else v}\n\n" for i, v in enumerate(results))
    prompt += "Based on the outputs above, provide the most consistent and complete JSON output. Ensure all fields are correctly populated according to the descriptions and requirements given in the function schema if this were a function call. Return only the JSON object."
    
    voted_raw = None # Initialize for potential error logging
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        voted_raw = completion.choices[0].message.content.strip()
        
        # Attempt to parse if the first result was a dict (implies JSON expected)
        if isinstance(results[0], dict):
            try:
                # Clean potential markdown code block fences
                if voted_raw.startswith("```json"):
                    voted_raw = voted_raw[7:]
                if voted_raw.endswith("```"):
                    voted_raw = voted_raw[:-3]
                voted_raw = voted_raw.strip()
                voted = json.loads(voted_raw)
            except json.JSONDecodeError as e_json_parse:
                logger.error(f"Self-consistency voting returned non-JSON or malformed JSON when JSON was expected. Raw: '{voted_raw}'. Error: {e_json_parse}. Falling back to first result.")
                voted = results[0] # Fallback to the first result if parsing fails
        else: # Assumed to be string
            voted = voted_raw
            
        return voted, results
    except APIError as e:
        logger.error(f"OpenAI API Error during self-consistency voting: {e}")
        raise RuntimeError(f"Self-consistency voting failed due to API error: {str(e)}")
    except json.JSONDecodeError as e: # Should be caught by inner try-except if results[0] is dict
        logger.error(f"JSON decode error during self-consistency voting. Raw content: '{voted_raw}'")
        raise RuntimeError(f"Self-consistency voting failed due to JSON decode error: {str(e)}. Raw content: '{voted_raw}'")
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error during self-consistency voting. Raw content: '{voted_raw}'. Error: {e}")
        # Fallback to the first result in case of unexpected error during voting
        if results:
            logger.warning("Falling back to the first result due to an unexpected error in self-consistency voting.")
            return results[0], results
        else:
            raise RuntimeError(f"Self-consistency voting failed with an unexpected error and no results to fallback on: {str(e)}")


def transcription_from_page_image(
    image_path: Path, 
    model_id: str,
    temperature: float = 0.0 # Model-specific temperature
) -> str:
    """
    1) Read the image as base64
    2) Send to OpenRouter to transcribe exactly
    """
    system_prompt = """
You are a precise document transcriber for medical lab reports. Your task is to:
1. Write out ALL text visible in the image exactly as it appears
2. Preserve the document's layout and formatting as much as possible using spaces and newlines
3. Include ALL numbers, units, and reference ranges exactly as shown
4. Use the exact same text case (uppercase/lowercase) as the document
5. Do not interpret, summarize, or structure the content - just transcribe it
""".strip()
    
    user_prompt = """
Please transcribe this lab report exactly as it appears, preserving layout and all details. 
Pay special attention to numbers, units (e.g., mg/dL), and reference ranges.
""".strip()
    
    with open(image_path, "rb") as img_file:
        img_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
                    ]
                }
            ],
            temperature=temperature,
            max_tokens=4096 # Max tokens for transcription, was 8192, Llama3.1-405b context is 8192, models like Haiku have less
        )
    except APIError as e:
        logger.error(f"OpenAI API Error during transcription for {image_path.name}: {e}")
        raise RuntimeError(f"Transcription failed for {image_path.name} due to API error: {str(e)}")
    transcription = completion.choices[0].message.content.strip()
    return transcription

def extract_labs_from_page_transcription(
    transcription: str, 
    model_id: str,
    temperature: float = 0.0 # Model-specific temperature
) -> dict: # Returns a dict matching HealthLabReport structure
    """
    1) Ask OpenRouter to parse out labs from the transcription
    2) Return them as a dict.
    """
    system_prompt = """
You are a medical lab report analyzer. Your task is to extract information from the provided transcription and structure it according to the 'extract_lab_results' tool schema.
Follow these strict requirements:
1. COMPLETENESS: Extract ALL test results from the provided transcription.
2. ACCURACY: Values and units must match exactly as they appear in the transcription.
3. SCHEMA ADHERENCE: Populate ALL fields of the `HealthLabReport` and nested `LabResult` models. If information for an optional field is not present, use `null`.
4. DATES: Ensure `report_date` and `collection_date` are in YYYY-MM-DD format or `null`.
5. LAB VALUES: If a lab value is clearly boolean (e.g., "Positive", "Negative"), convert `lab_value` to 1 or 0 respectively. Otherwise, use the numerical value. If it's purely textual (e.g., "See comments"), `lab_value` should be null and the text captured in `lab_comments` or `source_text`.
6. THOROUGHNESS: Process the text line by line to ensure nothing is missed.
7. UNKNOWN VALUES: For `lab_type`, if not clearly blood, urine, saliva, or feces, use 'unknown'.
""".strip()

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcription}
            ],
            temperature=temperature,
            max_tokens=4000, # Adjusted from 20000, as output is structured JSON
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "extract_lab_results"}}
        )
    except APIError as e:
        logger.error(f"OpenAI API Error during lab extraction: {e}")
        raise RuntimeError(f"Lab extraction failed due to API error: {str(e)}")

    if not completion.choices[0].message.tool_calls:
        logger.error(f"No tool call was made by the model for lab extraction. Transcription: {transcription[:500]}")
        # Return a minimal valid HealthLabReport structure to avoid downstream errors
        # or re-raise an error indicating extraction failure.
        # For now, let's try to return a default structure.
        empty_report = HealthLabReport(lab_results=[]).model_dump()
        logger.warning("Returning an empty HealthLabReport due to no tool call from the LLM.")
        return empty_report
        # raise RuntimeError("Lab extraction failed: Model did not make the expected tool call.")


    tool_args_raw = completion.choices[0].message.tool_calls[0].function.arguments
    
    try:
        tool_result = json.loads(tool_args_raw)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for tool arguments: {e}. Raw arguments: '{tool_args_raw}'")
        raise RuntimeError(f"Lab extraction failed due to JSON decode error in tool arguments: {str(e)}")

    # Removed the block that defaulted lab_range_min/max to 0/9999.
    # Let Pydantic handle Optional[float] as None if missing.

    try: 
        temp_model = HealthLabReport(**tool_result)
        temp_model.normalize_empty_optionals() # Normalize "" to None for optional fields
        # Re-validate after normalization to ensure type correctness if "" was problematic
        validated_model_dict = HealthLabReport.model_validate(temp_model.model_dump()).model_dump()
    except Exception as e:
        logger.error(f"Pydantic model validation error after extraction and normalization: {e}. Data: {tool_result}")
        # Attempt to salvage lab_results if the main report fails validation
        if "lab_results" in tool_result and isinstance(tool_result["lab_results"], list):
            salvaged_results = []
            for lr_data in tool_result["lab_results"]:
                try:
                    lr_model = LabResult(**lr_data)
                    lr_model.normalize_empty_optionals() # Assuming LabResult also has this method or adapt
                    salvaged_results.append(lr_model.model_dump())
                except Exception as lr_e:
                    logger.warning(f"Could not validate individual lab result: {lr_data}. Error: {lr_e}")
            if salvaged_results:
                logger.warning("Salvaged some lab results despite overall HealthLabReport validation failure.")
                # Create a minimal valid HealthLabReport with salvaged results
                # and default/None for other fields.
                minimal_report = HealthLabReport(lab_results=salvaged_results).model_dump()
                # Log which fields might have caused the main validation error by comparing keys
                expected_report_keys = set(HealthLabReport.model_fields.keys())
                provided_report_keys = set(tool_result.keys())
                missing_keys = expected_report_keys - provided_report_keys
                extra_keys = provided_report_keys - expected_report_keys
                if missing_keys: logger.error(f"Missing keys in HealthLabReport: {missing_keys}")
                if extra_keys: logger.error(f"Extra keys in HealthLabReport: {extra_keys}")
                return minimal_report


        logger.error(f"Model validation failed: {str(e)}. Raw tool result: {tool_result}")
        # Fallback to a minimal valid structure if validation fails catastrophically
        # This helps prevent the entire pipeline from crashing for one bad extraction.
        # However, this might hide issues, so logging is crucial.
        # Consider if raising RuntimeError is better in some scenarios.
        return HealthLabReport(lab_results=[]).model_dump()

    return validated_model_dict


########################################
# The Single-PDF Processor
########################################

def process_single_pdf(
    pdf_path: Path,
    output_dir: Path, # This is the global output_dir, doc_out_dir is derived inside
    self_consistency_model_id: str,
    transcribe_model_id: str,
    n_transcribe: int,
    extract_model_id: str,
    n_extract: int
) -> Optional[Path]: # Returns path to the generated PDF-level CSV
    """
    High-level function that:
      1) Copies `pdf_path` to `output_dir/<stem>/`.
      2) Extracts pages to JPEG, preprocesses them.
      3) Transcribes each page & extracts labs.
      4) Combines them into a single DataFrame for this PDF.
      5) Saves the PDF-level CSV inside that directory.
    Returns the path to the generated PDF-level CSV file, or None if processing fails.
    """
    pdf_stem = pdf_path.stem
    # doc_out_dir is specific to this PDF, inside the global output_dir
    doc_out_dir = output_dir / pdf_stem 
    doc_out_dir.mkdir(exist_ok=True, parents=True)

    pdf_level_csv_path = doc_out_dir / f"{pdf_stem}.csv"
    # Basic skip logic: if the final per-PDF CSV exists, skip.
    # More robust skipping might check for all intermediate files.
    # if pdf_level_csv_path.exists():
    #     logger.info(f"[{pdf_stem}] - skipped, CSV already exists: {pdf_level_csv_path}")
    #     return pdf_level_csv_path
    
    logger.info(f"[{pdf_stem}] - processing...")

    copied_pdf_path = doc_out_dir / pdf_path.name
    if not copied_pdf_path.exists() or copied_pdf_path.stat().st_size != pdf_path.stat().st_size : # Ensure it's a full copy
        logger.info(f"[{pdf_stem}] - copying to: {copied_pdf_path}")
        shutil.copy2(pdf_path, copied_pdf_path)

    try:
        pil_pages = pdf2image.convert_from_path(str(copied_pdf_path))
    except Exception as e:
        logger.error(f"[{pdf_stem}] - Failed to convert PDF to images: {e}")
        return None
    
    pages_data = [] # To store (page_image, page_file_name, page_jpg_path)

    for idx, page_image in enumerate(pil_pages, start=1):
        page_file_name = f"{pdf_stem}.{idx:03d}"
        page_jpg_path = doc_out_dir / f"{page_file_name}.jpg"
        pages_data.append({"image": page_image, "name": page_file_name, "path": page_jpg_path})

        if not page_jpg_path.exists(): # Preprocess and save if not exists
            logger.info(f"[{page_file_name}] - preprocessing and saving page JPG")
            processed_image = preprocess_page_image(page_image)
            processed_image.save(page_jpg_path, "JPEG", quality=95)
        else:
            logger.info(f"[{page_file_name}] - JPG already exists, skipping preprocessing.")
    
    logger.info(f"[{pdf_stem}] - extracted and preprocessed {len(pages_data)} page(s).")
    
    all_page_lab_results = [] # List to hold all LabResult objects from all pages
    first_page_report_data = {} # To store report-level data from the first page

    for page_idx, page_info in enumerate(pages_data):
        page_file_name = page_info["name"]
        page_jpg_path = page_info["path"]

        page_txt_path = doc_out_dir / f"{page_file_name}.txt"
        if not page_txt_path.exists():
            logger.info(f"[{page_file_name}] - transcribing page JPG")
            try:
                voted_txt, all_txt_versions = self_consistency(
                    lambda **sc_kwargs: transcription_from_page_image(page_jpg_path, transcribe_model_id, **sc_kwargs),
                    self_consistency_model_id, n_transcribe
                )
                if n_transcribe > 1:
                    for i, txt in enumerate(all_txt_versions, 1):
                        (doc_out_dir / f"{page_file_name}.v{i}.txt").write_text(txt, encoding='utf-8')
                page_txt_path.write_text(voted_txt, encoding='utf-8')
                page_txt = voted_txt
            except Exception as e:
                logger.error(f"[{page_file_name}] - transcription failed: {e}")
                continue # Skip to next page or handle error appropriately
        else:
            logger.info(f"[{page_file_name}] - TXT already exists, loading.")
            page_txt = page_txt_path.read_text(encoding='utf-8')

        page_json_path = doc_out_dir / f"{page_file_name}.json"
        current_page_json_data = None
        if not page_json_path.exists():
            logger.info(f"[{page_file_name}] - extracting JSON from page TXT")
            try:
                page_json_dict, all_json_versions = self_consistency(
                    lambda **sc_kwargs: extract_labs_from_page_transcription(page_txt, extract_model_id, **sc_kwargs),
                    self_consistency_model_id, n_extract
                )
                if n_extract > 1:
                    for i, j_data in enumerate(all_json_versions, 1):
                        (doc_out_dir / f"{page_file_name}.v{i}.json").write_text(json.dumps(j_data, indent=2, ensure_ascii=False), encoding='utf-8')
                
                # Ensure the result is a HealthLabReport model dict for consistency
                if not isinstance(page_json_dict, dict) or "lab_results" not in page_json_dict:
                    logger.error(f"[{page_file_name}] - Extraction did not return a valid HealthLabReport structure. Got: {type(page_json_dict)}")
                    page_json_model = HealthLabReport(lab_results=[]) # Create a default empty one
                else:
                     page_json_model = HealthLabReport(**page_json_dict)
                
                page_json_model.normalize_empty_optionals()
                current_page_json_data = page_json_model.model_dump()
                page_json_path.write_text(json.dumps(current_page_json_data, indent=2, ensure_ascii=False), encoding='utf-8')

            except Exception as e:
                logger.error(f"[{page_file_name}] - JSON extraction failed: {e}")
                # Create a minimal JSON structure to avoid crashing, or skip
                current_page_json_data = HealthLabReport(source_file=page_file_name, lab_results=[]).model_dump()
                page_json_path.write_text(json.dumps(current_page_json_data, indent=2, ensure_ascii=False), encoding='utf-8') # Save minimal
        else:
            logger.info(f"[{page_file_name}] - JSON already exists, loading.")
            try:
                current_page_json_data = json.loads(page_json_path.read_text(encoding='utf-8'))
                # Validate loaded JSON against Pydantic model
                page_json_model = HealthLabReport(**current_page_json_data)
                page_json_model.normalize_empty_optionals()
                current_page_json_data = page_json_model.model_dump()

            except json.JSONDecodeError as e:
                logger.error(f"[{page_file_name}] - Failed to decode existing JSON: {e}")
                current_page_json_data = HealthLabReport(source_file=page_file_name, lab_results=[]).model_dump() # Fallback
            except Exception as e: # Pydantic validation error
                logger.error(f"[{page_file_name}] - Failed to validate existing JSON: {e}")
                current_page_json_data = HealthLabReport(source_file=page_file_name, lab_results=[]).model_dump() # Fallback


        # Process extracted data for the current page
        if current_page_json_data:
            if page_idx == 0: # First page
                first_page_report_data = {
                    k: v for k, v in current_page_json_data.items() if k != "lab_results"
                }
                # Try to get a document date (collection or report)
                doc_date_str = current_page_json_data.get("collection_date") or current_page_json_data.get("report_date")
                if doc_date_str == "0000-00-00": doc_date_str = None

                # Fallback to filename if date is missing
                if not doc_date_str:
                    match = re.search(r"(\d{4}-\d{2}-\d{2})", pdf_stem)
                    if match:
                        doc_date_str = match.group(1)
                        logger.info(f"[{pdf_stem}] - Using date from filename: {doc_date_str}")
                        # Populate report_date/collection_date if they were missing and filename provided one
                        if not first_page_report_data.get("collection_date") and doc_date_str : first_page_report_data["collection_date"] = doc_date_str
                        if not first_page_report_data.get("report_date") and doc_date_str: first_page_report_data["report_date"] = doc_date_str
                
                first_page_report_data["_document_date_for_results"] = doc_date_str # Store resolved date
                if not doc_date_str:
                     logger.warning(f"[{pdf_stem}] - Document date is missing from page 1 and not found in filename. Results from this PDF might lack a date.")
                else:
                    # Basic check: if a date was found, it should ideally be in the filename or match it
                    if doc_date_str not in pdf_stem:
                         logger.warning(f"[{pdf_stem}] - Extracted document date {doc_date_str} is not in filename {pdf_stem}.")


            page_lab_results = current_page_json_data.get("lab_results", [])
            for lab_result_dict in page_lab_results:
                # Add page-specific info to each lab result
                lab_result_dict["page_number"] = page_idx + 1
                lab_result_dict["source_file"] = page_file_name # Page specific source (e.g. pdf_stem.001)
                all_page_lab_results.append(lab_result_dict)

    if not all_page_lab_results:
        logger.warning(f"[{pdf_stem}] - No lab results extracted from any page.")
        # Create an empty CSV to signify processing attempt but no data
        pd.DataFrame().to_csv(pdf_level_csv_path, index=False)
        return pdf_level_csv_path # Return path to empty CSV

    # Create DataFrame from all collected LabResult dicts
    pdf_df = pd.DataFrame(all_page_lab_results)

    # Add report-level information to each row of the DataFrame
    # This is less ideal than storing report-level info once, but matches one interpretation of merging
    # A better approach might be to have a separate report-level table.
    # For now, we add the resolved document date to each lab result row.
    resolved_doc_date = first_page_report_data.get("_document_date_for_results")
    if resolved_doc_date:
        pdf_df["date"] = resolved_doc_date # This adds the 'date' column expected by downstream processing
    else:
        pdf_df["date"] = None # Ensure column exists even if no date found
        logger.warning(f"[{pdf_stem}] - Final PDF DataFrame is missing 'date' for its lab results.")


    # Fill in other report-level data if needed, though 'date' is primary for lab results context
    # Example: pdf_df['report_date_meta'] = first_page_report_data.get("report_date")
    # pdf_df['collection_date_meta'] = first_page_report_data.get("collection_date")
    # pdf_df['patient_name_meta'] = first_page_report_data.get("patient_name")
    # etc. These would be new columns. The current structure expects 'date' directly on lab results.

    # Ensure all columns from LabResult model are present, filling with None if missing
    # This is important if some pages had partial extractions
    for col_name in LabResult.model_fields.keys():
        if col_name not in pdf_df.columns:
            pdf_df[col_name] = None
    
    # Select and order columns for the per-PDF CSV based on a subset of COLUMN_SCHEMA
    # Prioritize core lab data for these intermediate CSVs.
    # 'date' is critical.
    desired_cols_for_pdf_csv = [
        "date", "lab_type", "lab_name", "lab_value", "lab_unit",
        "lab_range_min", "lab_range_max", "reference_range_text", "is_flagged",
        "lab_comments", "confidence", "page_number", "source_file", "source_text" # source_file here is page specific
    ]
    
    # Filter for columns that actually exist in pdf_df after processing
    cols_to_save = [col for col in desired_cols_for_pdf_csv if col in pdf_df.columns]
    # Add any other columns that might have been generated if not in the desired list,
    # to avoid data loss at this stage.
    for col in pdf_df.columns:
        if col not in cols_to_save:
            cols_to_save.append(col)
            
    pdf_df = pdf_df[cols_to_save]


    pdf_df.to_csv(pdf_level_csv_path, index=False, encoding='utf-8')
    logger.info(f"[{pdf_stem}] - processing finished. CSV saved to {pdf_level_csv_path}")
    return pdf_level_csv_path


########################################
# The Main Function
########################################

def plot_lab_enum(args):
    lab_name_enum_val, merged_df_path_str, plots_dir_str, output_plots_dir_str = args
    import pandas as pd # Keep imports inside for multiprocessing safety
    import matplotlib.pyplot as plt
    import re
    from pathlib import Path

    # Use constants derived from COLUMN_SCHEMA
    date_col = PLOTTING_DATE_COL
    value_col = PLOTTING_VALUE_COL
    group_col = PLOTTING_GROUP_COL
    unit_col = PLOTTING_UNIT_COL

    try:
        merged_df = pd.read_csv(Path(merged_df_path_str))
        if date_col in merged_df.columns:
            merged_df[date_col] = pd.to_datetime(merged_df[date_col], errors="coerce")
        else:
            # logger.warning(f"Plotting: Date column '{date_col}' not found in {merged_df_path_str}.") # Needs logger config for multiproc
            print(f"Plotting Warning: Date column '{date_col}' not found in {merged_df_path_str}.")
            return

        if group_col not in merged_df.columns or value_col not in merged_df.columns:
            # logger.warning(f"Plotting: Group ('{group_col}') or Value ('{value_col}') column not found.")
            print(f"Plotting Warning: Group ('{group_col}') or Value ('{value_col}') column not found.")
            return

        df_lab = merged_df[merged_df[group_col] == lab_name_enum_val].copy()
        
        if df_lab.empty or df_lab[date_col].isnull().all() or len(df_lab) < 2:
            # print(f"Plotting Info: Not enough data to plot for {lab_name_enum_val}.")
            return
        
        df_lab = df_lab.sort_values(date_col, ascending=True)
        
        unit_str = ""
        if unit_col in df_lab.columns:
            unit_str = next((u for u in df_lab[unit_col].dropna().astype(str).unique() if u), "")
        
        y_label = f"Value ({unit_str})" if unit_str else "Value"
        title = f"{lab_name_enum_val} " + (f" [{unit_str}]" if unit_str else "")
        
        plt.figure(figsize=(12, 6)) # Increased figure size
        plt.plot(df_lab[date_col], df_lab[value_col], marker='o', linestyle='-')
        
        # Add reference ranges if available and make sense for the plot
        # This requires lab_range_min_final and lab_range_max_final to be present and consistent for the enum
        if "lab_range_min_final" in df_lab.columns and "lab_range_max_final" in df_lab.columns:
            # Use a typical or median range if multiple exist for the same enum (though ideally they are harmonized)
            # For simplicity, just pick the first valid one. More robust would be to check consistency.
            min_range = df_lab["lab_range_min_final"].dropna().unique()
            max_range = df_lab["lab_range_max_final"].dropna().unique()
            if len(min_range) > 0 and pd.notna(min_range[0]):
                 plt.axhline(y=float(min_range[0]), color='gray', linestyle='--', label=f'Ref Min: {min_range[0]}')
            if len(max_range) > 0 and pd.notna(max_range[0]):
                 plt.axhline(y=float(max_range[0]), color='gray', linestyle='--', label=f'Ref Max: {max_range[0]}')
            if (len(min_range) > 0 and pd.notna(min_range[0])) or \
               (len(max_range) > 0 and pd.notna(max_range[0])):
                plt.legend(loc='best')


        plt.title(title, fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        safe_lab_name = re.sub(r'[^\w\-_. ]', '_', str(lab_name_enum_val))
        plot_path = Path(plots_dir_str) / f"{safe_lab_name}.png"
        output_plot_path = Path(output_plots_dir_str) / f"{safe_lab_name}.png"
        
        plt.savefig(plot_path)
        plt.savefig(output_plot_path)
        plt.close()
    except Exception as e:
        # logger.error(f"Failed to plot {lab_name_enum_val}: {e}") # Needs logger config for multiproc
        print(f"Plotting Error for {lab_name_enum_val}: {e}")


def main():
    config = load_env_config()
    self_consistency_model_id = config["self_consistency_model_id"]
    transcribe_model_id = config["transcribe_model_id"]
    extract_model_id = config["extract_model_id"]
    input_dir = config["input_path"]
    output_dir = config["output_path"] # This is the global output directory
    pattern = config["input_file_regex"]
    n_transcriptions = config["n_transcriptions"]
    n_extractions = config["n_extractions"]
    max_workers = config.get("max_workers", os.cpu_count() or 1)


    pdf_files = sorted(list(input_dir.glob(pattern)))
    logger.info(f"Found {len(pdf_files)} PDF(s) matching pattern '{pattern}' in '{input_dir}'")
    if not pdf_files:
        logger.warning("No PDF files found. Exiting.")
        return

    n_workers = min(max_workers, len(pdf_files))
    logger.info(f"Using up to {n_workers} worker(s) for PDF processing.")

    tasks = [(
        pdf_path, output_dir, self_consistency_model_id, transcribe_model_id, n_transcriptions, extract_model_id, n_extractions
    ) for pdf_path in pdf_files]

    processed_pdf_csv_paths = []
    # Use multiprocessing.Pool for parallel PDF processing
    # Consider ProcessPoolExecutor if GIL is an issue with ThreadPool for CPU-bound tasks in self-consistency's threads
    with Pool(n_workers) as pool:
        results = pool.starmap(process_single_pdf, tasks)
        for result_path in results:
            if result_path and result_path.exists(): # process_single_pdf returns path or None
                processed_pdf_csv_paths.append(result_path)
            else:
                logger.warning(f"A PDF processing task did not return a valid CSV path or the file does not exist.")
    
    if not processed_pdf_csv_paths:
        logger.error("No PDFs were successfully processed to CSV. Cannot proceed with merging and further analysis.")
        return

    logger.info(f"Successfully processed {len(processed_pdf_csv_paths)} PDFs to individual CSVs.")

    dataframes = []
    for csv_path in processed_pdf_csv_paths: # Use the list of paths returned by workers
        try:
            if csv_path.stat().st_size > 0: # Check if CSV is not empty
                df = pd.read_csv(csv_path, encoding='utf-8')
                # The 'source_file' column in per-PDF CSVs (e.g., 'doc_stem.001.csv')
                # should be the per-PDF CSV name (e.g., 'doc_stem.csv') for the merged DF.
                df['source_file'] = csv_path.name # This is the 'pdf_stem.csv'
                dataframes.append(df)
            else:
                logger.warning(f"Skipping empty CSV: {csv_path}")
        except pd.errors.EmptyDataError:
            logger.warning(f"Skipping empty or malformed CSV (EmptyDataError): {csv_path}")
        except Exception as e:
            logger.error(f"Failed to read or process CSV {csv_path}: {e}")


    if not dataframes:
        logger.error("No data loaded from individual PDF CSVs. Final merged file will be empty or not created.")
        # Create empty files to signify completion if that's desired, or just exit
        # For now, we'll proceed, and empty files will be created if merged_df is empty.
        merged_df = pd.DataFrame()
    else:
        merged_df = pd.concat(dataframes, ignore_index=True)

    logger.info(f"Merged data from {len(dataframes)} CSVs into a single DataFrame with {len(merged_df)} rows.")

    # --------- Add derived columns (slugs, enums, final values) ---------
    # Load mappings
    # Ensure config directory and files exist
    config_path = Path("config")
    lab_names_mapping_path = config_path / "lab_names_mappings.json"
    lab_units_mapping_path = config_path / "lab_units_mappings.json"
    lab_specs_path = config_path / "lab_specs.json"

    if not lab_names_mapping_path.exists() or \
       not lab_units_mapping_path.exists() or \
       not lab_specs_path.exists():
        logger.error(f"One or more configuration files are missing from '{config_path}'. "
                     "Cannot proceed with deriving columns (enums, final values).")
        # Depending on strictness, either return or proceed with only raw extracted data.
        # For now, let's log and proceed, derived columns will be mostly empty/None.
        # Add empty columns to prevent KeyErrors later if strict schema adherence is expected for export
        for col_key in ["lab_name_slug", "lab_name_enum", "lab_unit_enum", 
                        "lab_value_final", "lab_unit_final", "lab_range_min_final", "lab_range_max_final",
                        "is_flagged_final", "healthy_range_min", "healthy_range_max", "is_in_healthy_range"]:
            if col_key not in merged_df.columns: merged_df[col_key] = None # or appropriate default based on type
    else:
        with open(lab_names_mapping_path, "r", encoding="utf-8") as f:
            lab_names_mapping = json.load(f)
        with open(lab_units_mapping_path, "r", encoding="utf-8") as f:
            lab_units_mapping = json.load(f)
        with open(lab_specs_path, "r", encoding="utf-8") as f:
            lab_specs = json.load(f)

        def slugify(value):
            if pd.isna(value): return ""
            value = str(value).strip().lower()
            value = value.replace('µ', 'micro')
            value = value.replace('%', 'percent')
            value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
            value = re.sub(r"[^\w\s-]", "", value) # Keep hyphens
            value = re.sub(r"[\s_]+", "-", value) # Replace spaces/underscores with single hyphen
            value = value.strip('-')
            return value

        def map_lab_name_slug(row):
            lab_type = row.get("lab_type", "")
            lab_name = row.get("lab_name", "")
            if pd.isna(lab_type) or pd.isna(lab_name): return ""
            return f"{str(lab_type).lower()}-{slugify(lab_name)}"

        merged_df["lab_name_slug"] = merged_df.apply(map_lab_name_slug, axis=1)
        merged_df["lab_name_enum"] = merged_df["lab_name_slug"].apply(lambda x: lab_names_mapping.get(x, ""))
        merged_df["lab_unit_enum"] = merged_df.get("lab_unit", pd.Series(dtype='str')).apply(lambda x: lab_units_mapping.get(slugify(x), ""))


        def convert_to_primary_unit(row):
            lab_name_enum = row.get("lab_name_enum", "")
            lab_unit_enum = row.get("lab_unit_enum", "") # This is already slugified and mapped
            value = row.get("lab_value")
            range_min = row.get("lab_range_min")
            range_max = row.get("lab_range_max")
            
            # Default: return original values and the mapped unit enum
            value_final, range_min_final, range_max_final = value, range_min, range_max
            unit_final = lab_unit_enum # Default to the mapped enum unit

            if not lab_name_enum or lab_name_enum not in lab_specs or pd.isna(lab_name_enum):
                return pd.Series([value, range_min, range_max, lab_unit_enum if pd.notna(lab_unit_enum) else None])

            spec = lab_specs[lab_name_enum]
            primary_unit_enum = spec.get("primary_unit") # This should be an enum key from lab_units_mappings.json
            
            if not primary_unit_enum: # No primary unit defined for this lab
                return pd.Series([value, range_min, range_max, unit_final])

            unit_final = primary_unit_enum # Target unit is the primary unit

            if lab_unit_enum == primary_unit_enum: # Already in primary unit
                return pd.Series([value, range_min, range_max, primary_unit_enum])

            # Look for conversion factor
            factor = None
            for alt_spec in spec.get("alternatives", []):
                if alt_spec.get("unit") == lab_unit_enum: # Comparing mapped unit with mapped unit in spec
                    factor = alt_spec.get("factor")
                    break
            
            if factor is not None:
                try: value_final = float(value) * float(factor) if pd.notnull(value) else value
                except (ValueError, TypeError): pass # Keep original if conversion fails
                try: range_min_final = float(range_min) * float(factor) if pd.notnull(range_min) else range_min
                except (ValueError, TypeError): pass
                try: range_max_final = float(range_max) * float(factor) if pd.notnull(range_max) else range_max
                except (ValueError, TypeError): pass
            
            return pd.Series([value_final, range_min_final, range_max_final, unit_final])

        if all(c in merged_df.columns for c in ["lab_name_enum", "lab_unit_enum", "lab_value", "lab_range_min", "lab_range_max"]):
            final_cols_df = merged_df.apply(convert_to_primary_unit, axis=1)
            final_cols_df.columns = ["lab_value_final", "lab_range_min_final", "lab_range_max_final", "lab_unit_final"]
            merged_df = pd.concat([merged_df, final_cols_df], axis=1)
        else:
            logger.warning("One or more source columns for unit conversion are missing. Skipping 'final' value calculations.")
            for col in ["lab_value_final", "lab_range_min_final", "lab_range_max_final", "lab_unit_final"]:
                 if col not in merged_df.columns: merged_df[col] = None


        def compute_is_flagged_final(row):
            value = row.get("lab_value_final")
            minv = row.get("lab_range_min_final")
            maxv = row.get("lab_range_max_final")
            if pd.isna(value): return None # Cannot determine if value is missing
            try: value_f = float(value)
            except (ValueError, TypeError): return None # Value not numeric

            is_low = False
            if pd.notna(minv):
                try: minv_f = float(minv)
                except (ValueError, TypeError): minv_f = None
                if minv_f is not None and value_f < minv_f: is_low = True
            
            is_high = False
            if pd.notna(maxv):
                try: maxv_f = float(maxv)
                except (ValueError, TypeError): maxv_f = None
                if maxv_f is not None and value_f > maxv_f: is_high = True
            
            return is_low or is_high if (pd.notna(minv) or pd.notna(maxv)) else None


        if all(c in merged_df.columns for c in ["lab_value_final", "lab_range_min_final", "lab_range_max_final"]):
            merged_df["is_flagged_final"] = merged_df.apply(compute_is_flagged_final, axis=1)
        else:
             if "is_flagged_final" not in merged_df.columns: merged_df["is_flagged_final"] = None


        def get_healthy_range(row):
            lab_name_enum = row.get("lab_name_enum", "")
            if not lab_name_enum or pd.isna(lab_name_enum) or lab_name_enum not in lab_specs:
                return pd.Series([None, None], index=["healthy_range_min", "healthy_range_max"])
            
            healthy_spec = lab_specs[lab_name_enum].get("ranges", {}).get("healthy")
            if healthy_spec and isinstance(healthy_spec, dict):
                return pd.Series([healthy_spec.get("min"), healthy_spec.get("max")], index=["healthy_range_min", "healthy_range_max"])
            return pd.Series([None, None], index=["healthy_range_min", "healthy_range_max"])

        if "lab_name_enum" in merged_df.columns:
            healthy_range_df = merged_df.apply(get_healthy_range, axis=1)
            merged_df[["healthy_range_min", "healthy_range_max"]] = healthy_range_df
        else:
            if "healthy_range_min" not in merged_df.columns: merged_df["healthy_range_min"] = None
            if "healthy_range_max" not in merged_df.columns: merged_df["healthy_range_max"] = None


        def compute_is_in_healthy_range(row):
            value = row.get("lab_value_final")
            minv = row.get("healthy_range_min")
            maxv = row.get("healthy_range_max")

            if pd.isna(value): return None
            try: value_f = float(value)
            except (ValueError, TypeError): return None

            # If neither min nor max healthy range is defined, cannot determine status
            if pd.isna(minv) and pd.isna(maxv): return None

            is_too_low = False
            if pd.notna(minv):
                try: minv_f = float(minv)
                except (ValueError, TypeError): minv_f = None # Treat unparsable range as undefined for this check
                if minv_f is not None and value_f < minv_f: is_too_low = True
            
            is_too_high = False
            if pd.notna(maxv):
                try: maxv_f = float(maxv)
                except (ValueError, TypeError): maxv_f = None # Treat unparsable range as undefined for this check
                if maxv_f is not None and value_f > maxv_f: is_too_high = True

            return not (is_too_low or is_too_high)

        if all(c in merged_df.columns for c in ["lab_value_final", "healthy_range_min", "healthy_range_max"]):
            merged_df["is_in_healthy_range"] = merged_df.apply(compute_is_in_healthy_range, axis=1)
        else:
            if "is_in_healthy_range" not in merged_df.columns: merged_df["is_in_healthy_range"] = None

    # --------- Filter and Sort Merged DataFrame ---------
    export_cols_ordered = get_export_columns_from_schema(COLUMN_SCHEMA)
    # Ensure only columns present in merged_df are selected, and all columns in merged_df are kept if not in schema.
    final_select_cols = [col for col in export_cols_ordered if col in merged_df.columns]
    for col in merged_df.columns: # Add any extra columns not in schema, to avoid data loss
        if col not in final_select_cols:
            final_select_cols.append(col)
    merged_df = merged_df[final_select_cols]


    if "date" in merged_df.columns:
        merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")
        merged_df = merged_df.sort_values("date", ascending=False)

    # --------- Assign Data Types Before Export ---------
    dtype_map_from_schema = get_dtype_map_from_schema(COLUMN_SCHEMA)
    for col, target_dtype_str in dtype_map_from_schema.items():
        if col in merged_df.columns:
            try:
                current_dtype = merged_df[col].dtype
                if target_dtype_str == "datetime64[ns]":
                    if not pd.api.types.is_datetime64_any_dtype(current_dtype):
                        merged_df[col] = pd.to_datetime(merged_df[col], errors="coerce")
                elif target_dtype_str == "boolean": # Pandas nullable boolean
                    if str(current_dtype).lower() != "boolean": # Check before converting
                         merged_df[col] = merged_df[col].astype("boolean") # Handles NA
                elif target_dtype_str == "Int64": # Pandas nullable integer
                     if str(current_dtype).lower() != "int64":
                        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").astype("Int64")
                elif "float" in target_dtype_str:
                    if not pd.api.types.is_float_dtype(current_dtype):
                        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")
                elif "int" in target_dtype_str and target_dtype_str != "Int64": # Standard int, careful with NaNs
                    if not pd.api.types.is_integer_dtype(current_dtype):
                        # Coercing to numeric first, then int, can raise error if NaNs exist and not Int64
                        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")
                        if merged_df[col].isnull().any():
                            logger.warning(f"Column {col} has NaNs, cannot convert to standard int, using float or Int64 instead.")
                            if merged_df[col].dtype != "Int64": merged_df[col] = merged_df[col].astype("float64") # or Int64
                        else:
                             merged_df[col] = merged_df[col].astype(target_dtype_str)
                # String types generally don't need explicit conversion if already object/string
                # else: # For other types like string, ensure it's not something weird
                #    if not pd.api.types.is_string_dtype(current_dtype) and not pd.api.types.is_object_dtype(current_dtype):
                #        merged_df[col] = merged_df[col].astype(str)
            except Exception as e:
                logger.error(f"Failed to convert column '{col}' to dtype '{target_dtype_str}': {e}")
    
    # Save merged CSV
    merged_csv_path = output_dir / "all.csv"
    merged_df.to_csv(merged_csv_path, index=False, encoding='utf-8')
    logger.info(f"Saved main merged CSV to: {merged_csv_path}")

    # --------- Export Excel file (all.xlsx) ---------
    excel_path = output_dir / "all.xlsx"
    excel_hidden_cols = get_hidden_excel_columns_from_schema(COLUMN_SCHEMA)
    excel_col_widths = get_excel_widths_from_schema(COLUMN_SCHEMA)

    with pd.ExcelWriter(excel_path, engine="xlsxwriter", datetime_format='yyyy-mm-dd', date_format='yyyy-mm-dd') as writer:
        merged_df.to_excel(writer, sheet_name="AllData", index=False)
        worksheet = writer.sheets["AllData"]
        worksheet.freeze_panes(1, 0)
        for idx, col_name in enumerate(merged_df.columns):
            width = excel_col_widths.get(col_name, 12) # Default width 12 if not in schema
            options = {'hidden': True} if col_name in excel_hidden_cols else {}
            worksheet.set_column(idx, idx, width, None, options)
    logger.info(f"Saved main merged Excel to: {excel_path}")

    # --------- Export final reduced file (all.final.csv/xlsx) ---------
    final_export_cols_ordered = get_final_export_columns_ordered(COLUMN_SCHEMA)
    # Ensure only columns present in merged_df are selected for final_df
    final_df_cols = [col for col in final_export_cols_ordered if col in merged_df.columns]
    if not all(col in merged_df.columns for col in final_df_cols):
        logger.warning(f"Not all expected columns for final_df found in merged_df. Required: {final_df_cols}. Found: {list(merged_df.columns)}")
    
    final_df = merged_df[final_df_cols].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Re-apply dtypes for final_df just to be safe, especially if columns were missing or added
    for col, target_dtype_str in dtype_map_from_schema.items():
        if col in final_df.columns:
            try:
                if target_dtype_str == "datetime64[ns]":
                    final_df.loc[:, col] = pd.to_datetime(final_df[col], errors="coerce")
                elif target_dtype_str == "boolean":
                    final_df.loc[:, col] = final_df[col].astype("boolean")
                elif target_dtype_str == "Int64":
                     final_df.loc[:, col] = pd.to_numeric(final_df[col], errors="coerce").astype("Int64")
                elif "float" in target_dtype_str : # Includes float64
                    final_df.loc[:, col] = pd.to_numeric(final_df[col], errors="coerce")
            except Exception as e:
                logger.debug(f"Could not convert column {col} in final_df to {target_dtype_str}: {e}")


    final_df.to_csv(output_dir / "all.final.csv", index=False, encoding='utf-8')
    final_df.to_excel(output_dir / "all.final.xlsx", sheet_name="FinalData", index=False)
    logger.info("Saved final reduced CSV and Excel files.")

    # --------- Export most recent value per blood test ---------
    if "lab_type" in final_df.columns and PLOTTING_GROUP_COL in final_df.columns and PLOTTING_DATE_COL in final_df.columns:
        # Ensure 'lab_type' is string for comparison
        final_df_lt_str = final_df["lab_type"].astype(str).str.lower()
        blood_df = final_df[final_df_lt_str == "blood"].copy()
        if not blood_df.empty:
            blood_df = blood_df.sort_values(PLOTTING_DATE_COL, ascending=False)
            most_recent_blood = blood_df.drop_duplicates(subset=[PLOTTING_GROUP_COL], keep="first")
            most_recent_blood.to_csv(output_dir / "all.final.blood-most-recent.csv", index=False, encoding='utf-8')
            most_recent_blood.to_excel(output_dir / "all.final.blood-most-recent.xlsx", sheet_name="RecentBlood", index=False)
            logger.info("Saved most recent blood test values.")
        else:
            logger.info("No blood type tests found in final_df, skipping 'most-recent blood' export.")
    else:
        logger.warning("Required columns for 'most-recent blood' export not found. Skipping.")
    
    logger.info("All data processing and file exports finished.")

    # --------- Plotting Section ---------
    if PLOTTING_GROUP_COL in merged_df.columns and \
       PLOTTING_DATE_COL in merged_df.columns and \
       PLOTTING_VALUE_COL in merged_df.columns:
        
        plots_base_dir = Path("plots") # Local plots directory
        plots_base_dir.mkdir(exist_ok=True)
        output_plots_dir = output_dir / "plots" # Plots in output directory
        output_plots_dir.mkdir(exist_ok=True)

        logger.info(f"Starting plot generation. Plots will be saved in '{plots_base_dir}' and '{output_plots_dir}'.")
        
        # Ensure merged_df for plotting has datetime properly converted if not already
        if not pd.api.types.is_datetime64_any_dtype(merged_df[PLOTTING_DATE_COL]):
            merged_df[PLOTTING_DATE_COL] = pd.to_datetime(merged_df[PLOTTING_DATE_COL], errors="coerce")

        unique_lab_enums = merged_df[PLOTTING_GROUP_COL].dropna().unique()
        
        # Filter out enums where there isn't enough data to plot
        plot_args_list = []
        for lab_enum in unique_lab_enums:
            df_enum_check = merged_df[
                (merged_df[PLOTTING_GROUP_COL] == lab_enum) &
                (merged_df[PLOTTING_DATE_COL].notna()) &
                (merged_df[PLOTTING_VALUE_COL].notna())
            ]
            if len(df_enum_check) >= 2:
                 plot_args_list.append((lab_enum, str(merged_csv_path), str(plots_base_dir), str(output_plots_dir)))
            # else:
            #    logger.debug(f"Skipping plot for {lab_enum} due to insufficient data points (<2).")


        if plot_args_list:
            # Determine number of workers for plotting, can be different from PDF processing
            plot_workers = min(max(1, (os.cpu_count() or 1) -1 ), len(plot_args_list)) 
            logger.info(f"Plotting for {len(plot_args_list)} lab enums using {plot_workers} worker(s).")
            
            # Use multiprocessing for plotting
            # Ensure plot_lab_enum is self-contained or pickles correctly
            with Pool(processes=plot_workers) as plot_pool:
                plot_pool.map(plot_lab_enum, plot_args_list)
            logger.info("Plotting finished.")
        else:
            logger.info("No lab enums with sufficient data for plotting.")
    else:
        logger.warning(f"One or more essential columns for plotting ({PLOTTING_GROUP_COL}, {PLOTTING_DATE_COL}, {PLOTTING_VALUE_COL}) are missing. Skipping plotting.")

if __name__ == "__main__":
    main()