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
import re
import hashlib
from multiprocessing import Pool, cpu_count
from PIL import Image
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Any, Literal
from collections import Counter
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add OpenAI import for OpenRouter
from openai import OpenAI

########################################
# Config / Logging
########################################

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

    model_id = os.getenv("MODEL_ID")
    input_path = os.getenv("INPUT_PATH")
    input_file_regex = os.getenv("INPUT_FILE_REGEX")
    output_path = os.getenv("OUTPUT_PATH")
    n_transcriptions = int(os.getenv("N_TRANSCRIPTIONS"))
    n_extractions = int(os.getenv("N_EXTRACTIONS"))
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    max_workers = int(os.getenv("MAX_WORKERS"))

    if not model_id: raise ValueError("MODEL_ID not set")
    if not input_path or not Path(input_path).exists(): raise ValueError(f"INPUT_PATH not set or does not exist: {input_path}")
    if not input_file_regex: raise ValueError("INPUT_FILE_REGEX not set")
    if not output_path or not Path(output_path).exists(): raise ValueError("OUTPUT_PATH not set")
    if not openrouter_api_key: raise ValueError("OPENROUTER_API_KEY not set")

    return {
        "model_id" : model_id,
        "input_path" : Path(input_path),
        "input_file_regex" : input_file_regex,
        "output_path" : Path(output_path),
        "n_transcriptions": n_transcriptions,
        "n_extractions": n_extractions,
        "openrouter_api_key": openrouter_api_key,
        "max_workers": max_workers
    }

########################################
# LLM Tools
########################################

with open("config/lab_names.json", "r", encoding="utf-8") as f: LAB_NAMES = json.load(f)
with open("config/lab_methods.json", "r", encoding="utf-8") as f: LAB_METHODS = json.load(f)
with open("config/lab_units.json", "r", encoding="utf-8") as f: LAB_UNITS = json.load(f)

# Create dynamic enums for lab names and units
def create_dynamic_enum(name, data): return Enum(name, dict([(k, k) for k in data]), type=str)
LabTestNameEnum = create_dynamic_enum('LabTestNameEnum', LAB_NAMES)
LabMethodEnum = create_dynamic_enum('LabMethodEnum', LAB_METHODS)
LabTestUnitEnum = create_dynamic_enum('LabTestUnitEnum', LAB_UNITS)

class LabResult(BaseModel):
    """
    Represents an individual laboratory test result, including measurement details, reference ranges, and status interpretation.
    """
    lab_name: str = Field(
        description="Name of the laboratory test as extracted verbatim from the document"
    )
    standardized_lab_name: LabTestNameEnum = Field(
        description="Standardized name of the laboratory test using controlled vocabulary; when unsure output `$UNKNOWN$`",
    )
    lab_value: float = Field(
        description="Quantitative result of the laboratory test"
    )
    lab_unit: str = Field(
        description="Unit of measurement as extracted verbatim (e.g., mg/dL, mmol/L, IU/mL)."
    )
    standardized_lab_unit: LabTestUnitEnum = Field(
        description="Standardized unit of measurement; when unsure output `$UNKNOWN$`",
    )
    lab_method: Optional[str] = Field(
        description="Analytical method or technique as extracted verbatim (e.g., ELISA, HPLC, Microscopy), if available"
    )
    standardized_lab_method: Optional[LabMethodEnum] = Field(
        description="Standardized analytical method using controlled vocabulary; when unsure output `$UNKNOWN$`",
    )
    lab_range_min: float = Field(
        description="Lower bound of the reference range, 0 if not specified"
    )
    lab_range_max: float = Field(
        description="Upper bound of the reference range, 9999 if not specified"
    )
    lab_status: Optional[Literal["low", "normal", "high", "abnormal"]] = Field(
        description="Interpretation of the result relative to the reference range"
    )
    lab_comments: Optional[str] = Field(
        description="Additional notes or observations about this result, if available"
    )
    confidence: float = Field(
        description="Confidence score of the extraction process, ranging from 0 to 1"
    )
    lack_of_confidence_reason: Optional[str] = Field(
        description="Reason for low extraction confidence"
    )

class HealthLabReport(BaseModel):
    """
    Represents a complete laboratory report, including patient information, metadata, and a collection of test results.
    """
    report_date: Optional[str] = Field(
        description="Date the laboratory report was issued (YYYY-MM-DD)"
    )
    collection_date: Optional[str] = Field(
        description="Date the specimen was collected (YYYY-MM-DD), if available"
    )
    lab_facility: Optional[str] = Field(
        description="Name of the laboratory or facility that performed the tests, if available"
    )
    patient_name: Optional[str] = Field(
        description="Full name of the patient"
    )
    physician_name: Optional[str] = Field(
        description="Name of the requesting or reviewing physician, if available"
    )
    lab_results: List[LabResult] = Field(
        description="List of individual laboratory test results in this report"
    )

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
""".strip(),
            "parameters": HealthLabReport.model_json_schema()
        }
    }
]

########################################
# Helper Functions
########################################

def hash_file(file_path: Path, length=4) -> str:
    """Calculate MD5 hash of a file, return hex digest truncated to `length` characters."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()[:length]


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
    enhanced_image = ImageEnhance.Contrast(gray_image).enhance(1.5)  # Adjust contrast by 1.5x

    # Optional: Quantize to reduce noise while preserving readability (128 colors)
    # Comment this out if you want maximum fidelity without quantization
    final_image = enhanced_image.quantize(colors=128).convert('L')

    # Return the processed image (to be saved as PNG later for lossless quality)
    return final_image

def self_consistency(fn, n, *args, **kwargs):
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
            except Exception as e:
                for f in futures:
                    f.cancel()
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
    
    completion = client.chat.completions.create(
        model=os.getenv("MODEL_ID"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    voted = completion.choices[0].message.content.strip()
    if type(results[0]) != str: voted = json.loads(voted)
    return voted, results

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
    
    tool_args = completion.choices[0].message.tool_calls[0].function.arguments
    tool_result = json.loads(tool_args)

    model = HealthLabReport.model_validate(tool_result)
    model_dict = model.model_dump()

    return model_dict

########################################
# The Single-PDF Processor
########################################

def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    model_id: str,
    n_transcribe: int,
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
    logger.info(f"[{pdf_stem}] - processing: {pdf_path}")
    doc_out_dir = output_dir / pdf_stem
    doc_out_dir.mkdir(exist_ok=True, parents=True)

    final_csv_path = os.path.join(doc_out_dir, f"{pdf_stem}.csv")
    if os.path.exists(final_csv_path):
        logger.info(f"[{pdf_stem}] - already processed, skipping")
        return pd.read_csv(final_csv_path)

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
        # No JPGs found, extract pages from PDF
        pages = pdf2image.convert_from_path(str(copied_pdf_path))
        logger.info(f"[{pdf_stem}] - extracted {len(pages)} page(s) from PDF")
    
    # 4) For each page: preprocess, transcribe, parse labs
    document_date = None
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
                    model_id,
                    **kwargs
                ), n_transcribe
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
                    model_id,
                    **kwargs
                ), n_extract
            )

            # Only save versioned files if n_extract > 1
            if n_extract > 1:
                for idx, j in enumerate(all_json_versions, 1):
                    versioned_json_path = doc_out_dir / f"{page_file_name}.v{idx}.json"
                    versioned_json_path.write_text(json.dumps(j, indent=2), encoding='utf-8')

            # If this is the first page, save the report date
            if page_number == 1: 
                report_date = page_json.get("report_date")
                collection_date = page_json.get("collection_date")
                document_date = collection_date if collection_date else report_date
                assert document_date, "Document date is missing"
                assert document_date in pdf_stem, f"Document date not in filename: {pdf_stem} vs {document_date}"

            page_json["source_file"] = page_file_name
            lab_results = page_json.get("lab_results", [])
            for lab_result in lab_results: lab_result["date"] = document_date
            page_json["lab_results"] = lab_results

            # Save parsed labs
            page_json_path.write_text(json.dumps(page_json, indent=2), encoding='utf-8')
        else:
            # If JSON already exists, just load it
            page_json = json.loads(page_json_path.read_text(encoding='utf-8'))

        # Export to CSV
        page_csv_path = doc_out_dir / f"{page_file_name}.csv"
        if not page_csv_path.exists():
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
    for file in os.listdir(doc_out_dir):
        if not file.endswith('.csv'): continue
        file_path = os.path.join(doc_out_dir, file)
        contents = open(file_path, 'r', encoding='utf-8').read()
        if not contents.strip(): continue
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate all dataframes and save to a single CSV
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_csv(final_csv_path, index=False)

    logger.info(f"[{pdf_stem}] - processing finished successfully")

########################################
# The Main Function
########################################

def main():
    config = load_env_config()
    model_id = config["model_id"]
    input_dir = config["input_path"]
    output_dir = config["output_path"]
    pattern = config["input_file_regex"]
    n_transcriptions = config["n_transcriptions"]
    n_extractions = config["n_extractions"]
    max_workers = config.get("max_workers", 1)

    # Gather PDFs
    pdf_files = [f for f in input_dir.glob("*") if re.search(pattern, f.name, re.IGNORECASE)]
    logger.info(f"Found {len(pdf_files)} PDF(s) matching pattern {pattern}")

    # Parallel process each PDF
    n_workers = min(max_workers, len(pdf_files))
    logger.info(f"Using up to {n_workers} worker(s)")

    # Prepare argument tuples for each PDF
    tasks = [(pdf_path, output_dir, model_id, n_transcriptions, n_extractions) for pdf_path in pdf_files]

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
    merged_df.to_csv(os.path.join(output_dir, "all.csv"), index=False)

    logger.info("All PDFs processed.")

if __name__ == "__main__":
    main()
