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
    # 'final_export' flags if the column is part of the "final" summarized export (no longer used for separate files).
    # 'plotting_role' identifies columns for specific roles in plotting (date, value, group, unit).

    "date": {"dtype": "datetime64[ns]", "excel_width": 13, "plotting_role": "date"},
    "lab_type": {"dtype": "str", "excel_width": 10},
    "lab_name": {"dtype": "str", "excel_width": 35, "excel_hidden": True},
    "lab_code": {"dtype": "str", "excel_width": 15},
    "lab_value": {"dtype": "float64", "excel_width": 12, "excel_hidden": True},
    "lab_unit": {"dtype": "str", "excel_width": 15, "excel_hidden": True},
    "lab_method": {"dtype": "str", "excel_width": 20},
    "lab_range_min": {"dtype": "float64", "excel_width": 12, "excel_hidden": True},
    "lab_range_max": {"dtype": "float64", "excel_width": 12, "excel_hidden": True},
    "reference_range_text": {"dtype": "str", "excel_width": 25},
    "is_flagged": {"dtype": "boolean", "excel_width": 10, "excel_hidden": True},
    "lab_comments": {"dtype": "str", "excel_width": 40},
    "confidence": {"dtype": "float64", "excel_width": 10, "excel_hidden": True},
    "lack_of_confidence_reason": {"dtype": "str", "excel_width": 30},
    "source_text": {"dtype": "str", "excel_width": 50},
    "page_number": {"dtype": "Int64", "excel_width": 8},
    "source_file": {"dtype": "str", "excel_width": 25},

    # Derived columns in main()
    "lab_name_slug": {"dtype": "str", "excel_width": 30, "excel_hidden": True, "derivation_logic": "map_lab_name_slug"},
    "lab_unit_slug": {"dtype": "str", "excel_width": 15, "excel_hidden": True, "derivation_logic": "map_lab_unit_slug"},
    "lab_name_enum": {"dtype": "str", "excel_width": 30, "derivation_logic": "map_lab_name_enum", "plotting_role": "group"},
    "lab_unit_enum": {"dtype": "str", "excel_width": 15, "derivation_logic": "map_lab_unit_enum"},

    "lab_value_final": {"dtype": "float64", "excel_width": 14, "derivation_logic": "convert_to_primary_unit", "plotting_role": "value"},
    "lab_unit_final": {"dtype": "str", "excel_width": 14, "derivation_logic": "convert_to_primary_unit", "plotting_role": "unit"},
    "lab_range_min_final": {"dtype": "float64", "excel_width": 14, "derivation_logic": "convert_to_primary_unit"},
    "lab_range_max_final": {"dtype": "float64", "excel_width": 14, "derivation_logic": "convert_to_primary_unit"},

    "is_flagged_final": {"dtype": "boolean", "excel_width": 14, "derivation_logic": "compute_is_flagged_final"},
    "healthy_range_min": {"dtype": "float64", "excel_width": 16, "derivation_logic": "get_healthy_range"},
    "healthy_range_max": {"dtype": "float64", "excel_width": 16, "derivation_logic": "get_healthy_range"},
    "is_in_healthy_range": {"dtype": "boolean", "excel_width": 18, "derivation_logic": "compute_is_in_healthy_range"},
}

# Helper functions to derive lists/dicts from COLUMN_SCHEMA
def get_export_columns_from_schema(schema: dict) -> list:
    """Returns an ordered list of columns for the main export."""
    ordered_keys = [
        "date", "lab_type", "lab_name", "lab_name_enum", "lab_name_slug",
        "lab_value", "lab_unit", "lab_unit_slug", "lab_unit_enum",
        "lab_range_min", "lab_range_max", "reference_range_text",
        "lab_value_final", "lab_unit_final",
        "lab_range_min_final", "lab_range_max_final",
        "is_flagged", "is_flagged_final",
        "healthy_range_min", "healthy_range_max", "is_in_healthy_range",
        "confidence", "lab_code", "lab_method", "lab_comments",
        "lack_of_confidence_reason", "source_text", "page_number", "source_file"
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
    transcribe_model_id = os.getenv("TRANSCRIBE_MODEL_ID")
    n_transcriptions = int(os.getenv("N_TRANSCRIPTIONS", 1)) # Default to 1 if not set
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
    if not transcribe_model_id: raise ValueError("TRANSCRIBE_MODEL_ID not set")
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
        "transcribe_model_id" : transcribe_model_id,
        "n_transcriptions": n_transcriptions,
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

class LabType(str, Enum):
    BLOOD = "blood"
    URINE = "urine"
    SALIVA = "saliva"
    FECES = "feces"
    UNKNOWN = "unknown"

class LabResult(BaseModel):
    lab_type: LabType = Field(default=LabType.UNKNOWN, description="Type of laboratory test")
    lab_name: str = Field(description="Name of the laboratory test (only lab name, don't include lab method)")
    lab_code: Optional[str] = Field(default=None, description="Standardized code for the test")
    lab_value: Optional[float] = Field(default=None, description="Quantitative result") # Allow string for non-numeric if strictly needed by source
    lab_unit: Optional[str] = Field(default=None, description="Unit of measurement")
    lab_method: Optional[str] = Field(default=None, description="Method used for the test, if applicable")
    lab_range_min: Optional[float] = Field(default=None, description="Lower bound of reference range")
    lab_range_max: Optional[float] = Field(default=None, description="Upper bound of reference range")
    reference_range_text: Optional[str] = Field(default=None, description="Reference range as text")
    is_flagged: Optional[bool] = Field(default=None, description="Is result flagged abnormal")
    lab_comments: Optional[str] = Field(default=None, description="Additional notes")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Extraction confidence")
    lack_of_confidence_reason: Optional[str] = Field(default=None, description="Reason for low confidence")
    source_text: Optional[str] = Field(default=None, description="Exact source snippet")
    page_number: Optional[int] = Field(default=None, ge=1, description="Page number in PDF")
    source_file: Optional[str] = Field(default=None, description="Source file/page identifier")

class HealthLabReport(BaseModel):
    report_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Report issue date")
    collection_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Specimen collection date")
    lab_facility: Optional[str] = Field(default=None, description="Performing lab name")
    lab_facility_address: Optional[str] = Field(default=None, description="Lab address")
    patient_name: Optional[str] = Field(default=None, description="Patient's full name")
    patient_id: Optional[str] = Field(default=None, description="Patient ID/MRN")
    patient_birthdate: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Patient birthdate")
    physician_name: Optional[str] = Field(default=None, description="Requesting physician")
    physician_id: Optional[str] = Field(default=None, description="Physician ID")
    page_count: Optional[int] = Field(default=None, ge=1, description="Total pages in report")
    lab_results: List[LabResult] = Field(default_factory=list, description="List of lab results")
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
        if fn.__name__ == 'extract_labs_from_page_transcription':
            # For extract function, we expect a dictionary
            try:
                voted_result = json.loads(voted_raw)
                return voted_result, results
            except json.JSONDecodeError:
                logger.error(f"Failed to parse voted result as JSON for extract function. Raw: '{voted_raw[:200]}...'")
                return results[0], results  # Fallback to first result
        else:
            # For other functions (like transcription), return the string as-is
            return voted_raw, results
            
    except Exception as e:
        logger.error(f"Error during self-consistency voting logic. Raw: '{voted_raw if voted_raw else 'N/A'}'. Error: {e}")

        # Fallback: Pick result with highest average confidence if available
        if fn.__name__ == 'extract_labs_from_page_transcription':
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

        # For transcription or other functions, return first result
        return results[0], results

def transcription_from_page_image(image_path: Path, model_id: str, temperature: float = 0.3) -> str:
    system_prompt = """
You are a precise document transcriber for medical lab reports. Your task is to:
1. Write out ALL text visible in the image exactly as it appears
2. Preserve the document's layout and formatting as much as possible; use markdown tables for lab results.
3. Include ALL numbers, units, and reference ranges exactly as shown
4. Use the exact same text case (uppercase/lowercase) as the document
5. Do not interpret, summarize, or structure the content - just transcribe it
""".strip()
    
    user_prompt = """
Please transcribe this lab report exactly as it appears, preserving layout and all details. 
Pay special attention to numbers, units (e.g., mg/dL), and reference ranges.
""".strip()
    
    with open(image_path, "rb") as img_file: img_data = base64.standard_b64encode(img_file.read()).decode("utf-8")
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}]}],
            temperature=temperature, max_tokens=4096
        )
        return completion.choices[0].message.content.strip()
    except APIError as e:
        logger.error(f"API Error during transcription for {image_path.name}: {e}")
        raise RuntimeError(f"Transcription failed for {image_path.name}: {e}")

def extract_labs_from_page_transcription(transcription: str, model_id: str, temperature: float = 0.3) -> dict:
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
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": transcription}],
            temperature=temperature, max_tokens=4000, tools=TOOLS, tool_choice={"type": "function", "function": {"name": "extract_lab_results"}}
        )
    except APIError as e:
        logger.error(f"API Error during lab extraction: {e}")
        raise RuntimeError(f"Lab extraction failed: {e}")

    if not completion.choices[0].message.tool_calls:
        logger.warning(f"No tool call by model for lab extraction. Transcription snippet: {transcription[:200]}")
        return HealthLabReport(lab_results=[]).model_dump()

    tool_args_raw = completion.choices[0].message.tool_calls[0].function.arguments
    try: tool_result_dict = json.loads(tool_args_raw)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for tool args: {e}. Raw: '{tool_args_raw}'")
        return HealthLabReport(lab_results=[]).model_dump() # Fallback

    try:
        report_model = HealthLabReport(**tool_result_dict)
        report_model.normalize_empty_optionals()
        return report_model.model_dump()
    except Exception as e: # Pydantic validation or other error
        logger.error(f"Model validation error post-extraction: {e}. Data: {tool_result_dict}")
        # Attempt to salvage results if main report fails
        if "lab_results" in tool_result_dict and isinstance(tool_result_dict["lab_results"], list):
            valid_results = []
            for lr_data in tool_result_dict["lab_results"]:
                try:
                    lr_model = LabResult(**lr_data)
                    # lr_model.normalize_empty_optionals() # Assume LabResult doesn't have this, or adapt
                    valid_results.append(lr_model.model_dump())
                except: pass # Ignore individual failures
            return HealthLabReport(lab_results=valid_results).model_dump() # Return with what could be salvaged
        return HealthLabReport(lab_results=[]).model_dump() # Final fallback


########################################
# The Single-PDF Processor
########################################

def process_single_pdf(
    pdf_path: Path, output_dir: Path, self_consistency_model_id: str,
    transcribe_model_id: str, n_transcribe: int, extract_model_id: str, n_extract: int
) -> Optional[Path]:
    pdf_stem = pdf_path.stem
    doc_out_dir = output_dir / pdf_stem 
    doc_out_dir.mkdir(exist_ok=True, parents=True)
    pdf_level_csv_path = doc_out_dir / f"{pdf_stem}.csv"

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

        page_txt_path = doc_out_dir / f"{page_file_name}.txt"
        page_txt = ""
        if not page_txt_path.exists():
            try:
                voted_txt, _ = self_consistency(
                    transcription_from_page_image, self_consistency_model_id, n_transcribe,
                    page_jpg_path, transcribe_model_id # fn, model_id, n, *args for fn
                )
                page_txt_path.write_text(voted_txt, encoding='utf-8'); page_txt = voted_txt
            except Exception as e: logger.error(f"[{page_file_name}] Transcr. failed: {e}"); continue
        else: page_txt = page_txt_path.read_text(encoding='utf-8')

        page_json_path = doc_out_dir / f"{page_file_name}.json"
        current_page_json_data = None
        if not page_json_path.exists():
            try:
                page_json_dict, _ = self_consistency(
                    extract_labs_from_page_transcription, self_consistency_model_id, n_extract,
                    page_txt, extract_model_id # fn, model_id, n, *args for fn
                )
                current_page_json_data = page_json_dict # Already validated dict from extract_labs
                page_json_path.write_text(json.dumps(current_page_json_data, indent=2, ensure_ascii=False), encoding='utf-8')
            except Exception as e: 
                logger.error(f"[{page_file_name}] Extract. failed: {e}")
                current_page_json_data = HealthLabReport(lab_results=[]).model_dump()
        else:
            try: 
                current_page_json_data = json.loads(page_json_path.read_text(encoding='utf-8'))
            except Exception as e: 
                logger.error(f"[{page_file_name}] Load JSON failed: {e}")
                current_page_json_data = HealthLabReport(lab_results=[]).model_dump()

        # Ensure current_page_json_data is always a dictionary
        if not isinstance(current_page_json_data, dict):
            logger.error(f"[{page_file_name}] current_page_json_data is not a dict: {type(current_page_json_data)}")
            current_page_json_data = HealthLabReport(lab_results=[]).model_dump()

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
    tasks = [(pdf, output_dir, config["self_consistency_model_id"], config["transcribe_model_id"], 
              config["n_transcriptions"], config["extract_model_id"], config["n_extractions"]) for pdf in pdf_files]

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

    # --------- Add derived columns (slugs, enums, final values) ---------
    merged_df["lab_name_slug"] = merged_df.apply(
        lambda r: f"{str(r.get('lab_type', '')).lower()}-{slugify(r.get('lab_name', ''))}",
        axis=1,
    )
    merged_df["lab_unit_slug"] = merged_df.get("lab_unit", pd.Series(dtype="str")).apply(slugify)

    config_path = Path("config")
    paths_exist = all([(config_path / f).exists() for f in ["lab_names_mappings.json", "lab_units_mappings.json", "lab_specs.json"]])
    if not paths_exist:
        logger.error(f"Missing config files in '{config_path}'. Derived columns might be incomplete.")
        # Initialize columns to prevent KeyErrors if they are used later
        derived_cols_to_init = [
            "lab_name_slug", "lab_unit_slug", "lab_name_enum", "lab_unit_enum", "lab_value_final",
            "lab_unit_final", "lab_range_min_final", "lab_range_max_final",
            "is_flagged_final", "healthy_range_min", "healthy_range_max", "is_in_healthy_range"
        ]
        for col_key in derived_cols_to_init:
            if col_key not in merged_df.columns: merged_df[col_key] = None
    else:
        with open(config_path / "lab_names_mappings.json", "r", encoding="utf-8") as f: lab_names_mapping = json.load(f)
        with open(config_path / "lab_units_mappings.json", "r", encoding="utf-8") as f: lab_units_mapping = json.load(f)
        with open(config_path / "lab_specs.json", "r", encoding="utf-8") as f: lab_specs = json.load(f)

        def map_lab_name_enum(slug: str) -> str:
            mapped = lab_names_mapping.get(slug)
            if mapped is None:
                logger.error(f"Unmapped lab name slug '{slug}'")
                return slug
            return mapped

        def map_lab_unit_enum(value: Any) -> str:
            slug = slugify(value)
            mapped = lab_units_mapping.get(slug)
            if mapped is None or mapped == "":
                logger.error(f"Unmapped lab unit '{value}' (slug '{slug}')")
                return ""
            return mapped

        merged_df["lab_name_enum"] = merged_df["lab_name_slug"].apply(map_lab_name_enum)
        merged_df["lab_unit_enum"] = merged_df.get("lab_unit", pd.Series(dtype="str")).apply(map_lab_unit_enum)

        def convert_to_primary_unit(row):
            name_enum, unit_enum = row.get("lab_name_enum",""), row.get("lab_unit_enum","")
            val, r_min, r_max = row.get("lab_value"), row.get("lab_range_min"), row.get("lab_range_max")
            v_f, r_min_f, r_max_f, u_f = val, r_min, r_max, (unit_enum if pd.notna(unit_enum) else None)
            if not name_enum or pd.isna(name_enum) or name_enum not in lab_specs: return pd.Series([v_f, r_min_f, r_max_f, u_f])
            spec, prim_unit = lab_specs[name_enum], lab_specs[name_enum].get("primary_unit")
            if not prim_unit: return pd.Series([v_f, r_min_f, r_max_f, u_f])
            u_f = prim_unit
            if unit_enum == prim_unit: return pd.Series([v_f, r_min_f, r_max_f, u_f])
            factor = next((alt.get("factor") for alt in spec.get("alternatives",[]) if alt.get("unit")==unit_enum), None)
            if factor:
                try: v_f = float(val) * float(factor) if pd.notnull(val) else val
                except: pass
                try: r_min_f = float(r_min) * float(factor) if pd.notnull(r_min) else r_min
                except: pass
                try: r_max_f = float(r_max) * float(factor) if pd.notnull(r_max) else r_max
                except: pass
            return pd.Series([v_f, r_min_f, r_max_f, u_f])
        
        if all(c in merged_df.columns for c in ["lab_name_enum", "lab_unit_enum", "lab_value", "lab_range_min", "lab_range_max"]):
            final_cols_df = merged_df.apply(convert_to_primary_unit, axis=1)
            final_cols_df.columns = ["lab_value_final", "lab_range_min_final", "lab_range_max_final", "lab_unit_final"]
            merged_df = pd.concat([merged_df, final_cols_df], axis=1)

        def compute_is_flagged_final(row):
            val, minv, maxv = row.get("lab_value_final"), row.get("lab_range_min_final"), row.get("lab_range_max_final")
            if pd.isna(val): return None
            try: val_f = float(val)
            except: return None
            # This logic is reversed, should be val_f < minv_f or val_f > maxv_f
            is_low = pd.notna(minv) and val_f < float(minv)
            is_high = pd.notna(maxv) and val_f > float(maxv)
            return is_low or is_high if (pd.notna(minv) or pd.notna(maxv)) else None # Corrected logic
        if all(c in merged_df.columns for c in ["lab_value_final", "lab_range_min_final", "lab_range_max_final"]):
            merged_df["is_flagged_final"] = merged_df.apply(compute_is_flagged_final, axis=1)

        def get_healthy_range(row):
            name_enum = row.get("lab_name_enum", "")
            if not name_enum or pd.isna(name_enum) or name_enum not in lab_specs: return pd.Series([None, None])
            healthy = lab_specs[name_enum].get("ranges",{}).get("healthy")
            return pd.Series([healthy.get("min"), healthy.get("max")]) if healthy else pd.Series([None,None])
        if "lab_name_enum" in merged_df.columns:
            merged_df[["healthy_range_min", "healthy_range_max"]] = merged_df.apply(get_healthy_range, axis=1)
        
        def compute_is_in_healthy_range(row):
            val, min_h, max_h = row.get("lab_value_final"), row.get("healthy_range_min"), row.get("healthy_range_max")
            if pd.isna(val): return None
            try: val_f = float(val)
            except: return None
            if pd.isna(min_h) and pd.isna(max_h): return None # No healthy range defined
            too_low = pd.notna(min_h) and val_f < float(min_h)
            too_high = pd.notna(max_h) and val_f > float(max_h)
            return not (too_low or too_high)
        if all(c in merged_df.columns for c in ["lab_value_final", "healthy_range_min", "healthy_range_max"]):
            merged_df["is_in_healthy_range"] = merged_df.apply(compute_is_in_healthy_range, axis=1)

    # --------- Deduplicate by (date, lab_name_enum) keeping best match ---------
    # Only if lab_specs is loaded and required columns exist
    if paths_exist and "date" in merged_df.columns and "lab_name_enum" in merged_df.columns and "lab_unit_enum" in merged_df.columns:
        def pick_best_dupe(group):
            # group: DataFrame with same (date, lab_name_enum)
            name_enum = group.iloc[0]["lab_name_enum"]
            primary_unit = None
            if name_enum and name_enum in lab_specs:
                primary_unit = lab_specs[name_enum].get("primary_unit")
            if primary_unit and (group["lab_unit_enum"] == primary_unit).any():
                return group[group["lab_unit_enum"] == primary_unit].iloc[0]
            else:
                return group.iloc[0]
        merged_df = (
            merged_df
            .groupby(["date", "lab_name_enum"], dropna=False, as_index=False)
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