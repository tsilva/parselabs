from dotenv import load_dotenv
load_dotenv(override=True)

from enum import Enum
import logging
import os
import json
import shutil
import anthropic
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

def load_env_config():
    """
    Load environment variables and return as a dict.
    """

    model_id = os.getenv("MODEL_ID")
    input_path = os.getenv("INPUT_PATH")
    input_file_regex = os.getenv("INPUT_FILE_REGEX")
    output_path = os.getenv("OUTPUT_PATH")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    if not model_id: raise ValueError("MODEL_ID not set")
    if not input_path or not Path(input_path).exists(): raise ValueError(f"INPUT_PATH not set or does not exist: {input_path}")
    if not input_file_regex: raise ValueError("INPUT_FILE_REGEX not set")
    if not output_path or not Path(output_path).exists(): raise ValueError("OUTPUT_PATH not set")
    if not anthropic_api_key: raise ValueError("ANTHROPIC_API_KEY not set")

    return {
        "model_id" : model_id,
        "input_path" : Path(input_path),
        "input_file_regex" : input_file_regex,
        "output_path" : Path(output_path)
    }

########################################
# LLM Tools
########################################

with open("config/lab_names.json", "r") as f: LAB_NAMES = json.load(f)
with open("config/lab_methods.json", "r") as f: LAB_METHODS = json.load(f)
with open("config/lab_units.json", "r") as f: LAB_UNITS = json.load(f)

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
        description="Standardized name of the laboratory test using controlled vocabulary"
    )
    lab_value: float = Field(
        description="Quantitative result of the laboratory test"
    )
    lab_unit: str = Field(
        description="Unit of measurement as extracted verbatim (e.g., mg/dL, mmol/L, IU/mL)"
    )
    standardized_lab_unit: LabTestUnitEnum = Field(
        description="Standardized unit of measurement"
    )
    lab_method: Optional[str] = Field(
        description="Analytical method or technique as extracted verbatim (e.g., ELISA, HPLC, Microscopy), if available"
    )
    standardized_lab_method: Optional[LabMethodEnum] = Field(
        description="Standardized analytical method using controlled vocabulary"
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
        "name": "extract_lab_results",
        "description": f"""
Extract structured laboratory test results from medical documents with high precision.

Specific requirements:
1. Extract EVERY test result visible in the image, including variants with different units
2. Booleans should be converted to 0/1, where 0 = false/negative and 1 = true/positive
3. Dates must be in ISO 8601 format (YYYY-MM-DD)
4. Units must match exactly as shown in the document
""".strip(),
        "input_schema": HealthLabReport.model_json_schema()
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


def transcription_from_page_image(
    image_path: Path, 
    model_id: str
) -> str:
    """
    1) Read the image as base64
    2) Send to Anthropic to transcribe exactly
    """

    # Create anthropic client
    client = anthropic.Anthropic()

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
    message = client.messages.create(
        model=model_id,
        max_tokens=8192,
        temperature=0.0,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_data
                        }
                    }
                ]
            }
        ]
    )
    
    transcription =  message.content[0].text.strip()
    return transcription


def extract_labs_from_page_transcription(
    transcription: str, 
    model_id: str
) -> pd.DataFrame:
    """
    1) Ask Anthropic to parse out labs from the transcription
    2) Return them as a DataFrame
    """
    client = anthropic.Anthropic()

    system_prompt = """
You are a medical lab report analyzer with the following strict requirements:
1. COMPLETENESS: Extract ALL test results from the provided transcription
2. ACCURACY: Values and units must match exactly
3. VALIDATION: Verify each extraction matches the source text exactly
4. THOROUGHNESS: Process the text line by line to ensure nothing is missed
""".strip()

    # Extract structured data from transcription
    response = client.messages.create(
        model=model_id,
        #max_tokens=8192,
        max_tokens=20000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        temperature=1,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {
                "role": "user",
                "content": transcription
            }
        ],
        tools=TOOLS
    )

    # Process response
    tool_result = None
    for content in response.content:
        if not hasattr(content, "input"): continue
        assert tool_result is None, "Multiple tools detected in message"
        tool_result = content.input

    try:
        model = HealthLabReport.model_validate(tool_result)
        model_dict = model.model_dump()
        return True, model_dict
    except ValidationError as e:
        error_list = e.errors()
        return False, error_list


########################################
# The Single-PDF Processor
########################################

def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    model_id: str
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

    try:
        # 1) Set up subdirectory
        pdf_stem = pdf_path.stem
        logger.info(f"[{pdf_stem}] - loaded: {pdf_path}")
        doc_out_dir = output_dir / pdf_stem
        doc_out_dir.mkdir(exist_ok=True, parents=True)

        # 2) Copy PDF to output subdirectory
        copied_pdf_path = doc_out_dir / pdf_path.name
        if not copied_pdf_path.exists(): 
            logger.info(f"[{pdf_stem}] - copying: {copied_pdf_path}")
            shutil.copy2(pdf_path, copied_pdf_path)

        # 3) Extract PDF pages
        pages = pdf2image.convert_from_path(str(copied_pdf_path))
        
        # 4) For each page: preprocess, transcribe, parse labs
        first_page_json = None
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

                # Transcribe the page image
                page_txt = transcription_from_page_image(
                    page_jpg_path,
                    model_id
                )

                # Save the transcription
                page_txt_path.write_text(page_txt, encoding='utf-8')
            
            # Extract labs
            page_json_path = doc_out_dir / f"{page_file_name}.json"
            if not page_json_path.exists():
                logger.info(f"[{page_file_name}] - extracting JSON from page TXT")

                # Parse labs
                page_txt = page_txt_path.read_text(encoding='utf-8')
                valid, result = extract_labs_from_page_transcription(
                    page_txt,
                    model_id
                )

                # If parsing failed, log the error
                if not valid:
                    logger.error(f"[{page_file_name}] - failed to extract: {json.dumps(result, indent=2)}")
                    continue

                # Inject page metadata
                page_json = result
                page_json["page_number"] = page_number

                # If this is the first page, save the metadata
                if page_number == 1:
                    first_page_json = page_json
                # Otherwise, copy metadata from the first page
                else:
                    keys_to_copy = ["report_date", "collection_date", "lab_facility", "patient_name", "physician_name"]
                    for key in keys_to_copy:
                        first_value = first_page_json.get(key)
                        if first_value is not None: page_json[key] = first_value
                        
                # Save parsed labs
                page_json_path.write_text(json.dumps(page_json, indent=2), encoding='utf-8')

            # Export to CSV
            page_csv_path = doc_out_dir / f"{page_file_name}.csv"
            if not page_csv_path.exists():
                logger.info(f"[{page_file_name}] - converting JSON to CSV")

                # Load JSON and convert to DataFrame
                page_json = json.loads(page_json_path.read_text(encoding='utf-8'))
                df = pd.json_normalize(page_json["lab_results"])

                # Save DataFrame to CSV
                df.to_csv(page_csv_path, index=False)

        logger.info(f"[{pdf_stem}] - processing finished successfully")
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return {}

    return {}

########################################
# The Main Function
########################################

def main():
    config = load_env_config()
    model_id = config["model_id"]
    input_dir = config["input_path"]
    output_dir = config["output_path"]
    pattern = config["input_file_regex"]

    # Gather PDFs
    pdf_files = [f for f in input_dir.glob("*") if re.search(pattern, f.name, re.IGNORECASE)]
    logger.info(f"Found {len(pdf_files)} PDF(s) matching pattern {pattern}")

    # Parallel process each PDF
    n_workers = min(cpu_count(), len(pdf_files))
    logger.info(f"Using up to {n_workers} worker(s)")

    # Prepare argument tuples for each PDF
    tasks = [(pdf_path, output_dir, model_id) for pdf_path in pdf_files]

    # We’ll combine all results into a single DataFrame afterward
    with Pool(n_workers) as pool:
        for _ in pool.starmap(process_single_pdf, tasks): pass
        
    logger.info("All PDFs processed.")

if __name__ == "__main__":
    main()
