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
    if log_file.exists(): log_file.write_text("", encoding='utf-8')  # Clears the file by overwriting with an empty string
        
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
    if not output_path or not Path(output_path).exists(): raise ValueError("OUTPUT_PATH not set")
    if not openrouter_api_key: raise ValueError("OPENROUTER_API_KEY not set")

    return {
        "input_path" : Path(input_path),
        "input_file_regex" : input_file_regex,
        "output_path" : Path(output_path),
        "self_consistency_model_id" : self_consistency_model_id,
        "transcribe_model_id" : transcribe_model_id,
        "n_transcriptions": n_transcriptions,
        "extract_model_id" : extract_model_id,
        "n_extractions": n_extractions,
        "openrouter_api_key": openrouter_api_key,
        "max_workers": max_workers
    }

########################################
# LLM Tools
########################################

class LabType(str, Enum):
    BLOOD = "blood"
    URINE = "urine"
    SALIVA = "saliva"
    FECES = "feces"

class LabResult(BaseModel):
    lab_type: LabType = Field(
        description="Type of laboratory test (must be one of: blood, urine, saliva)"
    )
    lab_name: str = Field(
        min_length=1,
        description="Name of the laboratory test as extracted verbatim from the document"
    )
    lab_code: Optional[str] = Field(
        description="Standardized code for the laboratory test (e.g., LOINC, CPT), if available"
    )
    lab_value: float = Field(
        description="Quantitative result of the laboratory test (positive/negative should be 1/0)"
    )
    lab_unit: Optional[str] = Field(
        min_length=1,
        description="Unit of measurement as extracted verbatim (e.g., mg/dL, mmol/L, IU/mL, boolean)"
    )
    lab_method: Optional[str] = Field(
        description="Analytical method or technique as extracted verbatim (e.g., ELISA, HPLC, Microscopy), if available"
    )
    lab_range_min: Optional[float] = Field(
        description="Lower bound of the reference range, if available"
    )
    lab_range_max: Optional[float] = Field(
        description="Upper bound of the reference range, if available"
    )
    reference_range_text: Optional[str] = Field(
        description="Reference range as shown in the document, verbatim (e.g., '4.0-10.0', 'Normal: <5')"
    )
    is_flagged: Optional[bool] = Field(
        description="True if the result is flagged as abnormal/high/low in the document, else False"
    )
    lab_comments: Optional[str] = Field(
        description="Additional notes or observations about this result, if available"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score of the extraction process, ranging from 0 to 1"
    )
    lack_of_confidence_reason: Optional[str] = Field(
        description="Reason for low extraction confidence"
    )
    source_text: Optional[str] = Field(
        description="The exact line or snippet from the document where this result was extracted"
    )
    page_number: Optional[int] = Field(
        ge=1,
        description="Page number in the PDF where this result was found, if available"
    )
    source_file: Optional[str] = Field(
        description="The filename or identifier of the source file/page"
    )

class HealthLabReport(BaseModel):
    report_date: Optional[str] = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date the laboratory report was issued (YYYY-MM-DD), if unavailable use 0000-00-00"
    )
    collection_date: Optional[str] = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date the specimen was collected (YYYY-MM-DD), if available (also called subscription date), if unavailable use 0000-00-00"
    )
    lab_facility: Optional[str] = Field(
        description="Name of the laboratory or facility that performed the tests, if available"
    )
    lab_facility_address: Optional[str] = Field(
        description="Address of the laboratory or facility, if available"
    )
    patient_name: Optional[str] = Field(
        description="Full name of the patient"
    )
    patient_id: Optional[str] = Field(
        description="Patient identifier or medical record number, if available"
    )
    patient_birthdate: Optional[str] = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Birthdate of the patient (YYYY-MM-DD), if available"
    )
    physician_name: Optional[str] = Field(
        description="Name of the requesting or reviewing physician, if available"
    )
    physician_id: Optional[str] = Field(
        description="Identifier for the physician, if available"
    )
    page_count: Optional[int] = Field(
        ge=1,
        description="Total number of pages in the report, if available"
    )
    lab_results: List[LabResult] = Field(
        description="List of individual laboratory test results in this report"
    )
    source_file: Optional[str] = Field(
        description="The filename or identifier of the source file"
    )

    def normalize_empty_optionals(self):
        """
        For all optional fields in this report and its LabResult entries,
        replace empty string values ("") with None.
        """
        # Normalize HealthLabReport fields
        for field, field_info in self.model_fields.items():
            if field_info.is_required() or field == "lab_results":
                continue
            value = getattr(self, field)
            if value == "":
                setattr(self, field, None)
        # Normalize LabResult fields
        for lab in self.lab_results:
            for field, field_info in lab.model_fields.items():
                if field_info.is_required():
                    continue
                value = getattr(lab, field)
                if value == "":
                    setattr(lab, field, None)

TOOLS = [
    {
        "type": "function",
        "function" : {
            "name": "extract_lab_results",
            "description": f"""
Extract structured laboratory test results from medical documents with high precision.

Specific requirements:
1. Extract EVERY test result visible in the image, including variants with different units
2. Booleans should be converted to 0/1, where 0 = false/negative and 1 = true/positive
3. Dates must be in ISO 8601 format (YYYY-MM-DD)
4. Units must match exactly as shown in the document
5. Use the most precise schema possible for each field. 
6. For each result, include the exact source text/line and page number if possible.
7. If a field is not present in the document, use null or the value `{UNKNOWN_VALUE}` for required fields; for optional fields, use null.
8. NEVER skip or omit any non-optional field. Every required field must be present for every lab result, even if the value is unknown.
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

    # Convert to grayscale
    gray_image = image.convert('L')

    # Set a higher maximum width to preserve detail (only resize if necessary)
    MAX_WIDTH = 1200  # Increased from 800 to retain more detail
    if gray_image.width > MAX_WIDTH:
        ratio = MAX_WIDTH / gray_image.width
        new_height = int(gray_image.height * ratio)
        gray_image = gray_image.resize((MAX_WIDTH, new_height), Image.Resampling.LANCZOS)

    # Enhance contrast to make text stand out
    enhanced_image = ImageEnhance.Contrast(gray_image).enhance(2.0)  # Adjust contrast by 2x

    # Optional: Quantize to reduce noise while preserving readability (128 colors)
    # Comment this out if you want maximum fidelity without quantization
    normalized_image = enhanced_image.quantize(colors=128).convert('L')

    # Return the processed image (to be saved as PNG later for lossless quality)
    return normalized_image

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
        return fn(*args, **kwargs, temperature=0.5)

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
                    if not f_cancel.done():
                        f_cancel.cancel()
                raise RuntimeError(f"OpenAI API Error in self-consistency task: {str(oe)}")
            except Exception as e:
                for f_cancel in futures:
                    if not f_cancel.done():
                        f_cancel.cancel()
                raise

    if all(r == results[0] for r in results):
        return results[0], results

    system_prompt = (
        "You are an expert at comparing multiple outputs of the same extraction task. "
        "We have extracted several samples from the same prompt in order to average out any errors or inconsistencies that may appear in individual outputs. "
        "Your job is to select the output that is most consistent with the majority of the provided samples—"
        "in other words, the output that best represents the 'average' or consensus among all outputs. "
        "Prioritize agreement on extracted content (test names, values, units, reference ranges, etc). "
        "Ignore formatting, whitespace, and layout differences. "
        "Return ONLY the best output, verbatim, with no extra commentary. "
        "Do NOT include any delimiters, output numbers, or extra labels in your response—return only the raw content of the best output."
    )
    prompt = ""
    prompt += "".join(f"--- Output {i+1} ---\n{json.dumps(v) if type(v) in [list, dict] else v}\n\n" for i, v in enumerate(results))
    prompt += "Best output:"
    
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        voted_raw = completion.choices[0].message.content.strip()
        if type(results[0]) != str:
            voted = json.loads(voted_raw)
        else:
            voted = voted_raw
        return voted, results
    except APIError as e:
        logger.error(f"OpenAI API Error during self-consistency voting: {e}")
        raise RuntimeError(f"Self-consistency voting failed due to API error: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error during self-consistency voting. Raw content: '{voted_raw}'")
        raise RuntimeError(f"Self-consistency voting failed due to JSON decode error: {str(e)}. Raw content: '{voted_raw}'")

def transcription_from_page_image(
    image_path: Path, 
    model_id: str,
    temperature: float = 0.0
) -> str:
    """
    1) Read the image as base64
    2) Send to OpenRouter to transcribe exactly
    """

    # Define system prompt
    system_prompt = """
You are a precise document transcriber for medical lab reports. Your task is to:
1. Write out ALL text visible in the image exactly as it appears
2. Preserve the document's layout and formatting as much as possible using spaces and newlines
3. Include ALL numbers, units, and reference ranges exactly as shown
4. Use the exact same text case (uppercase/lowercase) as the document
5. Do not interpret, summarize, or structure the content - just transcribe it
""".strip()
    
    # Define user prompt
    user_prompt = """
Please transcribe this lab report exactly as it appears, preserving layout and all details. 
Pay special attention to numbers, units (e.g., mg/dL), and reference ranges.
""".strip()
    
    # Encode image as base64
    with open(image_path, "rb") as img_file:
        img_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

    # Prompt for image transcription
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}"
                            }
                        }
                    ]
                }
            ],
            temperature=temperature,
            max_tokens=8192
        )
    except APIError as e:
        logger.error(f"OpenAI API Error during transcription for {image_path.name}: {e}")
        raise RuntimeError(f"Transcription failed for {image_path.name} due to API error: {str(e)}")
    transcription = completion.choices[0].message.content.strip()
    return transcription

def extract_labs_from_page_transcription(
    transcription: str, 
    model_id: str,
    temperature: float = 0.0
) -> pd.DataFrame:
    """
    1) Ask OpenRouter to parse out labs from the transcription
    2) Return them as a DataFrame
    """

    system_prompt = """
You are a medical lab report analyzer with the following strict requirements:
1. COMPLETENESS: Extract ALL test results from the provided transcription
2. ACCURACY: Values and units must match exactly
3. VALIDATION: Verify each extraction matches the source text exactly
4. THOROUGHNESS: Process the text line by line to ensure nothing is missed
""".strip()

    # Extract structured data from transcription
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": transcription
                }
            ],
            temperature=temperature,
            max_tokens=20000,
            tools=TOOLS
        )
    except APIError as e:
        logger.error(f"OpenAI API Error during lab extraction: {e}")
        raise RuntimeError(f"Lab extraction failed due to API error: {str(e)}")

    tool_args = completion.choices[0].message.tool_calls[0].function.arguments
    tool_result = json.loads(tool_args)
    
    lab_results = tool_result.get("lab_results", [])
    for lab_result in lab_results:
        lab_range_min = lab_result.get("lab_range_min")
        lab_range_max = lab_result.get("lab_range_max")
        if lab_range_min is None: lab_result["lab_range_min"] = 0
        if lab_range_max is None: lab_result["lab_range_max"] = 9999
    tool_result["lab_results"] = lab_results

    tool_result = json.loads(tool_args)

    try: 
        # Normalize empty strings before validation
        temp_model = HealthLabReport(**tool_result)
        temp_model.normalize_empty_optionals()
        model = HealthLabReport.model_validate(temp_model.model_dump())
    except Exception as e:
        logger.error(f"Model validation error: {e}")
        raise RuntimeError(f"Model validation failed: {str(e)}")
    
    model_dict = model.model_dump()
    return model_dict

########################################
# The Single-PDF Processor
########################################

def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    self_consistency_model_id: str,
    transcribe_model_id: str,
    n_transcribe: int,
    extract_model_id: str,
    n_extract: int
) -> pd.DataFrame:
    """
    High-level function that:
      1) Copies `pdf_path` to `output_dir/<stem>/`.
      2) Extracts pages to JPEG, preprocesses them.
      3) Transcribes each page & extracts labs.
      4) Combines them into a single DataFrame for this PDF.
      5) Saves the PDF-level CSV inside that directory.
    Returns a single merged DataFrame for the entire PDF.
    """

    # 1) Set up subdirectory
    pdf_stem = pdf_path.stem
    doc_out_dir = output_dir / pdf_stem
    doc_out_dir.mkdir(exist_ok=True, parents=True)

    normalized_csv_path = os.path.join(doc_out_dir, f"{pdf_stem}.csv")
    #if os.path.exists(normalized_csv_path):
    #    logger.info(f"[{pdf_stem}] - skipped")
    #    return pd.read_csv(normalized_csv_path)
    
    logger.info(f"[{pdf_stem}] - processing...")

    # 2) Copy PDF to output subdirectory
    copied_pdf_path = doc_out_dir / pdf_path.name
    if not copied_pdf_path.exists(): 
        logger.info(f"[{pdf_stem}] - copying: {copied_pdf_path}")
        shutil.copy2(pdf_path, copied_pdf_path)

    # 3) Check if all expected page JPGs exist, else extract PDF pages
    # Try to find the number of pages by looking for existing JPGs
    existing_jpgs = sorted(doc_out_dir.glob(f"{pdf_stem}.*.jpg"))
    if existing_jpgs:
        # If any JPGs exist, assume all pages are already extracted
        pages = [Image.open(jpg_path) for jpg_path in existing_jpgs]
    else:
        # No JPGs found, extract pages from PDF and save images immediately
        pages = []
        pil_pages = pdf2image.convert_from_path(str(copied_pdf_path))
        for idx, page_image in enumerate(pil_pages, start=1):
            page_file_name = f"{pdf_stem}.{idx:03d}"
            page_jpg_path = doc_out_dir / f"{page_file_name}.jpg"
            page_image.save(page_jpg_path, "JPEG", quality=95)
            pages.append(page_image)
        logger.info(f"[{pdf_stem}] - extracted {len(pages)} page(s) from PDF")
    
    # 4) For each page: preprocess, transcribe, parse labs
    document_date = None
    report_date = None
    collection_date = None
    for page_number, page_image in enumerate(pages, start=1):
        page_file_name = f"{pdf_stem}.{page_number:03d}"

        # Preprocess
        page_jpg_path = doc_out_dir / f"{page_file_name}.jpg"
        if not page_jpg_path.exists():
            logger.info(f"[{page_file_name}] - preprocessing page JPG")

            # Preprocess the page image
            processed_image = preprocess_page_image(page_image)

            # Save the processed image as JPEG
            processed_image.save(page_jpg_path, "JPEG", quality=95)

        # Transcribe
        page_txt_path = doc_out_dir / f"{page_file_name}.txt"
        if not page_txt_path.exists():
            logger.info(f"[{page_file_name}] - extracting TXT from page JPG")

            # Transcribe the page image with self-consistency
            voted_txt, all_txt_versions = self_consistency(
                lambda **kwargs: transcription_from_page_image(
                    page_jpg_path,
                    transcribe_model_id,
                    **kwargs
                ), self_consistency_model_id, n_transcribe
            )

            # Only save versioned files if n_transcribe > 1
            if n_transcribe > 1:
                for idx, txt in enumerate(all_txt_versions, 1):
                    versioned_txt_path = doc_out_dir / f"{page_file_name}.v{idx}.txt"
                    versioned_txt_path.write_text(txt, encoding='utf-8')

            # Save the voted transcription as the main .txt
            page_txt_path.write_text(voted_txt, encoding='utf-8')
        
        # Extract labs
        page_json_path = doc_out_dir / f"{page_file_name}.json"
        if not page_json_path.exists():
            logger.info(f"[{page_file_name}] - extracting JSON from page TXT")

            # Parse labs with self-consistency
            page_txt = page_txt_path.read_text(encoding='utf-8')
            page_json, all_json_versions = self_consistency(
                lambda **kwargs: extract_labs_from_page_transcription(
                    page_txt,
                    extract_model_id,
                    **kwargs
                ), self_consistency_model_id, n_extract
            )

            # Only save versioned files if n_extract > 1
            if n_extract > 1:
                for idx, j in enumerate(all_json_versions, 1):
                    versioned_json_path = doc_out_dir / f"{page_file_name}.v{idx}.json"
                    versioned_json_path.write_text(json.dumps(j, indent=2, ensure_ascii=False), encoding='utf-8')

            # If this is the first page, save the report date
            if page_number == 1: 
                report_date = page_json.get("report_date")
                collection_date = page_json.get("collection_date")
                if report_date == "0000-00-00": report_date = None
                if collection_date == "0000-00-00": collection_date = None
                document_date = collection_date if collection_date else report_date

                # If document_date is missing, try to extract from pdf_stem
                if not document_date:
                    # Try to find a date in the pdf_stem (format: YYYY-MM-DD)
                    m = re.search(r"\d{4}-\d{2}-\d{2}", pdf_stem)
                    if m:
                        document_date = m.group(0)
                        # Propagate to missing fields
                        if not collection_date:
                            collection_date = document_date
                        if not report_date:
                            report_date = document_date
                    else:
                        raise AssertionError("Document date is missing and not found in filename")

                assert document_date, "Document date is missing"
                assert document_date in pdf_stem, f"Document date not in filename: {pdf_stem} vs {document_date}"

            page_json["report_date"] = report_date
            page_json["collection_date"] = collection_date
            page_json["source_file"] = page_file_name
            lab_results = page_json.get("lab_results", [])
            for lab_result in lab_results: 
                lab_result["date"] = document_date
            page_json["lab_results"] = lab_results

            # Save parsed labs
            page_json_path.write_text(json.dumps(page_json, indent=2, ensure_ascii=False), encoding='utf-8')
        else:
            # If JSON already exists, just load it
            page_json = json.loads(page_json_path.read_text(encoding='utf-8'))

        # Export to CSV
        page_csv_path = doc_out_dir / f"{page_file_name}.csv"
        if True: #not page_csv_path.exists():
            logger.info(f"[{page_file_name}] - converting JSON to CSV")

            # Load JSON and convert to DataFrame
            page_json = json.loads(page_json_path.read_text(encoding='utf-8'))
            df = pd.json_normalize(page_json["lab_results"])

            # Ensure 'date' is the first column if it exists
            if 'date' in df.columns:
                cols = ['date'] + [col for col in df.columns if col != 'date']
                df = df[cols]
            
            # Save DataFrame to CSV
            df.to_csv(page_csv_path, index=False)

    # Loop through files in the directory
    dataframes = []
    # Only merge per-page CSVs (e.g., .001.csv, .002.csv, etc.)
    for file in os.listdir(doc_out_dir):
        if not re.match(rf"^{re.escape(pdf_stem)}\.\d{{3}}\.csv$", file):
            continue
        file_path = os.path.join(doc_out_dir, file)
        contents = open(file_path, 'r', encoding='utf-8').read()
        if not contents.strip():
            continue
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate all dataframes and save to a single CSV
    merged_df = pd.concat(dataframes, ignore_index=True)

    merged_df.to_csv(normalized_csv_path, index=False)

    # --------- Export Excel file for Google Drive, freeze first row ---------
    excel_path = os.path.join(output_dir, "all.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        merged_df.to_excel(writer, index=False)
        worksheet = writer.sheets[merged_df.columns.name or writer.sheets.keys().__iter__().__next__()]
        worksheet.freeze_panes(1, 0)

    # --------- Export final reduced file ---------
    export_columns_final = [
        "date",
        "lab_type",
        "lab_name_enum",
        "lab_value_final",
        "lab_unit_final",
        "lab_range_min_final",
        "lab_range_max_final"
    ]
    final_df = merged_df[[col for col in export_columns_final if col in merged_df.columns]]
    final_df.to_csv(os.path.join(output_dir, "all.final.csv"), index=False)
    final_df.to_excel(os.path.join(output_dir, "all.final.xlsx"), index=False)

    # --------- Export most recent value per blood test ---------
    if "lab_type" in final_df.columns and "lab_name_enum" in final_df.columns and "date" in final_df.columns:
        blood_df = final_df[final_df["lab_type"].str.lower() == "blood"].copy()
        # Sort by date descending, then drop duplicates to keep the most recent per test
        blood_df = blood_df.sort_values("date", ascending=False)
        most_recent_blood = blood_df.drop_duplicates(subset=["lab_name_enum"], keep="first")
        most_recent_blood.to_csv(os.path.join(output_dir, "all.final.blood-most-recent.csv"), index=False)
        most_recent_blood.to_excel(os.path.join(output_dir, "all.final.blood-most-recent.xlsx"), index=False)

    logger.info(f"[{pdf_stem}] - processing finished successfully")

########################################
# The Main Function
########################################

def plot_lab_enum(args):
    lab_name_enum, merged_df_path, plots_dir_str = args
    import pandas as pd
    import re
    import matplotlib.pyplot as plt
    from pathlib import Path

    merged_df = pd.read_csv(merged_df_path)
    if "date" in merged_df.columns:
        merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")
    df_lab = merged_df[merged_df["lab_name_enum"] == lab_name_enum].copy()
    if df_lab.empty or df_lab["date"].isnull().all() or len(df_lab) < 2:
        return
    df_lab = df_lab.sort_values("date", ascending=True)
    # Get unit for this lab_name_enum (use first non-null unit)
    unit_str = next((u for u in df_lab["lab_unit_final"].dropna().astype(str).unique() if u), "")
    y_label = f"Value ({unit_str})" if unit_str else "Value"
    title = f"{lab_name_enum} " + (f" [{unit_str}]" if unit_str else "")
    plt.figure(figsize=(10, 5))
    plt.plot(df_lab["date"], df_lab["lab_value_final"], marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    safe_lab_name = re.sub(r'[^\w\-_. ]', '_', str(lab_name_enum))
    plot_path = Path(plots_dir_str) / f"{safe_lab_name}.png"
    plt.savefig(plot_path)
    plt.close()

def main():
    config = load_env_config()
    self_consistency_model_id = config["self_consistency_model_id"]
    transcribe_model_id = config["transcribe_model_id"]
    extract_model_id = config["extract_model_id"]
    input_dir = config["input_path"]
    output_dir = config["output_path"]
    pattern = config["input_file_regex"]
    n_transcriptions = config["n_transcriptions"]
    n_extractions = config["n_extractions"]
    max_workers = config.get("max_workers", 1)

    # Gather PDFs
    pdf_files = sorted([f for f in input_dir.glob("*") if re.search(pattern, f.name, re.IGNORECASE)])
    logger.info(f"Found {len(pdf_files)} PDF(s) matching pattern {pattern}")

    # Parallel process each PDF
    n_workers = min(max_workers, len(pdf_files))
    logger.info(f"Using up to {n_workers} worker(s)")

    # Prepare argument tuples for each PDF
    tasks = [(
        pdf_path, output_dir, self_consistency_model_id, transcribe_model_id, n_transcriptions, extract_model_id, n_extractions
    ) for pdf_path in pdf_files]

    # We’ll combine all results into a single DataFrame afterward
    with Pool(n_workers) as pool:
        for _ in pool.starmap(process_single_pdf, tasks): pass

    # Find all PDFs in the output directory (recursively)
    dataframes = []
    pdf_paths = glob.glob(os.path.join(output_dir, '**', '*.pdf'), recursive=True)
    for pdf_path in pdf_paths:
        pdf_dir = os.path.dirname(pdf_path)
        pdf_stem = Path(pdf_path).stem
        csv_path = os.path.join(pdf_dir, f"{pdf_stem}.csv")
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        df['source_file'] = os.path.basename(csv_path)
        dataframes.append(df)

    # Concatenate all dataframes and save to a single CSV
    merged_df = pd.concat(dataframes, ignore_index=True)

    # --------- Add lab_value_enum and lab_unit_enum columns ---------
    # Load mappings
    with open("config/lab_names_mappings.json", "r", encoding="utf-8") as f:
        lab_names_mapping = json.load(f)
    with open("config/lab_units_mappings.json", "r", encoding="utf-8") as f:
        lab_units_mapping = json.load(f)

    # Slugify function (add if not present)
    def slugify(value):
        value = str(value).strip().lower()
        value = value.replace('µ', 'micro')
        value = value.replace('%', 'percent')  # Replace % with "percent"
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
        value = re.sub(r"[^\w\s-]", "", value)
        value = re.sub(r"[\s_-]+", "", value)
        return value

    def map_lab_name_slug(row):
        lab_type = row["lab_type"]
        lab_name = row["lab_name"]
        lab_name_slug = f"{lab_type.lower()}-{slugify(lab_name)}"
        return lab_name_slug

    def map_lab_name_enum(row):
        lab_name_slug = map_lab_name_slug(row)
        lab_name_enum = lab_names_mapping.get(lab_name_slug, "")
        return lab_name_enum
    
    def map_lab_unit_enum(row):
        slug = slugify(row.get("lab_unit", ""))
        return lab_units_mapping.get(slug, "")

    merged_df["lab_name_slug"] = merged_df.apply(map_lab_name_slug, axis=1)
    merged_df["lab_name_enum"] = merged_df.apply(map_lab_name_enum, axis=1)
    merged_df["lab_unit_enum"] = merged_df.apply(map_lab_unit_enum, axis=1)

    # --------- Compute lab_value_final, lab_range_min_final, lab_range_max_final, lab_unit_final ---------
    # Load lab_specs.json for unit mapping
    with open("config/lab_specs.json", "r", encoding="utf-8") as f:
        lab_specs = json.load(f)

    def convert_to_primary_unit(row):
        lab_name_enum = row.get("lab_name_enum", "")
        lab_unit_enum = row.get("lab_unit_enum", "")
        value = row.get("lab_value")
        range_min = row.get("lab_range_min")
        range_max = row.get("lab_range_max")
        # Default: just copy values and unit
        value_final = value
        range_min_final = range_min
        range_max_final = range_max
        unit_final = lab_unit_enum

        if not lab_name_enum or lab_name_enum not in lab_specs:
            return pd.Series([value, range_min, range_max, lab_unit_enum])

        spec = lab_specs[lab_name_enum]
        primary_unit = spec.get("primary_unit")
        if not primary_unit:
            return pd.Series([value, range_min, range_max, lab_unit_enum])

        # If already in primary unit, just copy
        if lab_unit_enum == primary_unit:
            return pd.Series([value, range_min, range_max, primary_unit])

        # Otherwise, look for conversion factor
        factor = None
        for alt in spec.get("alternatives", []):
            if alt.get("unit") == lab_unit_enum:
                factor = alt.get("factor")
                break
        if factor is not None:
            try:
                value_final = float(value) * float(factor) if pd.notnull(value) else value
            except Exception:
                value_final = value
            try:
                range_min_final = float(range_min) * float(factor) if pd.notnull(range_min) else range_min
            except Exception:
                range_min_final = range_min
            try:
                range_max_final = float(range_max) * float(factor) if pd.notnull(range_max) else range_max
            except Exception:
                range_max_final = range_max
            unit_final = primary_unit
        # If no conversion found, just copy
        return pd.Series([value_final, range_min_final, range_max_final, unit_final])

    merged_df[["lab_value_final", "lab_range_min_final", "lab_range_max_final", "lab_unit_final"]] = merged_df.apply(convert_to_primary_unit, axis=1)

    # --------- Compute is_flagged_final ---------
    def compute_is_flagged_final(row):
        value = row.get("lab_value_final")
        minv = row.get("lab_range_min_final")
        maxv = row.get("lab_range_max_final")
        try:
            value = float(value)
        except Exception:
            return None
        try:
            minv = float(minv)
        except Exception:
            minv = None
        try:
            maxv = float(maxv)
        except Exception:
            maxv = None
        if minv is not None and value < minv:
            return True
        if maxv is not None and value > maxv:
            return True
        return False

    merged_df["is_flagged_final"] = merged_df.apply(compute_is_flagged_final, axis=1)

    # Only keep the specified columns for all.csv
    export_columns = [
        "date",
        "lab_type",
        "lab_name",
        "lab_name_enum",
        "lab_name_slug",
        "lab_value",
        "lab_unit",
        "lab_unit_enum",
        "lab_range_min",
        "lab_range_max",
        "lab_value_final",
        "lab_unit_final",
        "lab_range_min_final",
        "lab_range_max_final",
        "is_flagged",
        "confidence",
        "source_file",
        "is_flagged_final"
    ]
    
    merged_df = merged_df[[col for col in export_columns if col in merged_df.columns]]

    # Sort by date (recent to oldest)
    if "date" in merged_df.columns:
        merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")
        merged_df = merged_df.sort_values("date", ascending=False)

    merged_df.to_csv(os.path.join(output_dir, "all.csv"), index=False)

    # --------- Export Excel file for Google Drive, freeze first row ---------
    excel_path = os.path.join(output_dir, "all.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        merged_df.to_excel(writer, index=False)
        worksheet = writer.sheets[merged_df.columns.name or writer.sheets.keys().__iter__().__next__()]
        worksheet.freeze_panes(1, 0)

    # --------- Export final reduced file ---------
    export_columns_final = [
        "date",
        "lab_type",
        "lab_name_enum",
        "lab_value_final",
        "lab_unit_final",
        "lab_range_min_final",
        "lab_range_max_final",
        "is_flagged_final"
    ]
    final_df = merged_df[[col for col in export_columns_final if col in merged_df.columns]]
    final_df.to_csv(os.path.join(output_dir, "all.final.csv"), index=False)
    final_df.to_excel(os.path.join(output_dir, "all.final.xlsx"), index=False)

    # --------- Export most recent value per blood test ---------
    if "lab_type" in final_df.columns and "lab_name_enum" in final_df.columns and "date" in final_df.columns:
        blood_df = final_df[final_df["lab_type"].str.lower() == "blood"].copy()
        blood_df = blood_df.sort_values("date", ascending=False)
        most_recent_blood = blood_df.drop_duplicates(subset=["lab_name_enum"], keep="first")
        most_recent_blood.to_csv(os.path.join(output_dir, "all.final.blood-most-recent.csv"), index=False)
        most_recent_blood.to_excel(os.path.join(output_dir, "all.final.blood-most-recent.xlsx"), index=False)

    logger.info("All PDFs processed.")

    # --------- Plotting Section ---------
    import matplotlib.pyplot as plt
    import multiprocessing

    # Ensure plots directory exists
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    merged_df_path = os.path.join(output_dir, "all.csv")
    merged_df = pd.read_csv(merged_df_path)
    if "date" in merged_df.columns:
        merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")

    # --------- Parallelized plot for each lab_name_enum ---------
    if "lab_name_enum" in merged_df.columns and "date" in merged_df.columns and "lab_value_final" in merged_df.columns:
        unique_lab_enums = merged_df["lab_name_enum"].dropna().unique()
        n_workers = max(1, multiprocessing.cpu_count() - 1)
        # Pass merged_df_path and plots_dir as arguments to avoid pickling issues
        args_list = [(lab_name_enum, merged_df_path, str(plots_dir)) for lab_name_enum in unique_lab_enums]
        with multiprocessing.Pool(n_workers) as pool:
            pool.map(plot_lab_enum, args_list)

if __name__ == "__main__":
    main()