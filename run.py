import os
import re
import json
import base64
import PyPDF2
import logging
from openai import OpenAI
from dotenv import load_dotenv
from pdf2image import convert_from_path
from utils import load_text, save_text, load_json, save_json, save_csv, load_paths, extract_text_from_image, create_completion, augment_lab_result

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define constants
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_CLIENT = OpenAI()

def convert_pdf_to_images(input_path: str, output_directory: str):
    # Read number of pages from pdf
    with open(input_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        number_pages = len(reader.pages)
    
    # In case all page images aready exist then skip
    page_file_paths = [os.path.basename(input_path).replace(".pdf", f".{i+1:03}.jpg") for i in range(number_pages)]
    page_file_paths = [os.path.join(output_directory, page_file_path) for page_file_path in page_file_paths]
    found_file_paths = [True for file_path in page_file_paths if os.path.exists(file_path)]
    if len(page_file_paths) == len(found_file_paths): return

    input_file_name = os.path.basename(input_path)
    images = convert_from_path(input_path, dpi=300, poppler_path='/usr/bin')
    for i, image in enumerate(images):
        output_file_name = input_file_name.replace(".pdf", f".{i+1:03}.jpg")
        output_path = os.path.join(output_directory, output_file_name)
        if os.path.exists(output_path): continue
        image.save(output_path, 'JPEG', quality=100)
        logging.info(f"Saved image: {output_path}")

def convert_image_to_text(input_path: str, output_directory: str):
    # Skip if output already exists
    output_file_name = os.path.basename(input_path).replace(".jpg", ".txt")
    output_path = os.path.join(output_directory, output_file_name)
    if os.path.exists(output_path): return

    # Read image
    with open(input_path, "rb") as image_file: base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Extract text from image
    content = extract_text_from_image(base64_image)

    # Ensure text was extracted successfully (at least 2 lines)
    lines = content.split("\n")
    lines = [line for line in lines if line.strip()]
    if len(lines) < 2: raise Exception(f"Could not extract text from image: {input_path}")
    if "I'm sorry,".lower() in content.lower(): raise Exception(f"Could not extract text from image: {input_path}")
    if "I don't have the capability to directly process images to extract text".lower() in content.lower(): raise Exception(f"Could not extract text from image: {input_path}")

    # Save the output
    save_text(output_path, content)

    logging.info(f"Saved text: {output_path}")

def convert_text_to_json(input_path: str, output_directory: str):
    # Skip if output already exists
    output_file_name = os.path.basename(input_path).replace(".txt", ".json")
    output_path = os.path.join(output_directory, output_file_name)
    if os.path.exists(output_path): return

    # Parse text using a function tool
    text = load_text(input_path)
    save_extract_blood_lab_results_tool = load_json("tools/save_extract_blood_lab_results.json")
    system_prompt = f"You are the best language model in the world at extracting blood lab results from text files. You are extremely capable at this and always get it right. Go for it!"
    user_prompt = f"Extract and save the following blood lab results:\n\n {text}"
    message = create_completion(user_prompt, model="gpt-4-1106-preview", system_prompt=system_prompt, tools=[save_extract_blood_lab_results_tool])

    # Extract the function payload
    results = []
    tool_calls = message.tool_calls
    if tool_calls:
        tool_call = tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        results = function_args["blood_lab_results"]

    # Save the result
    pattern = r'\b\d{4}-\d{2}-\d{2}\b'
    matches = re.findall(pattern, output_path)
    date_s = matches[0] if matches else None
    for result in results: result["date"] = date_s
    save_json(output_path, results)

    logging.info(f"Saved json: {output_path}")

def merge_page_jsons(json_paths: list, output_directory: str):
    def _group(file_paths):
        grouped_files = {}

        # Regular expression to match the base name and page number
        # Assuming the format is something like "base_name.001.extension"
        pattern = re.compile(r'(.*)\.\d{3}')

        for file_path in file_paths:
            match = pattern.match(file_path)
            if not match: continue
            base_name = match.group(1) + ".json"
            if not base_name in grouped_files: grouped_files[base_name] = []
            grouped_files[base_name].append(file_path)

        # Sorting the file paths for each base name
        for base_name in grouped_files:
            grouped_files[base_name].sort()

        return dict(grouped_files)

    paths = _group(json_paths)
    for base_path, page_paths in paths.items():
        output_file_name = os.path.basename(base_path)
        output_path = os.path.join(output_directory, output_file_name)
        if os.path.exists(output_path): continue
        
        results = []
        for page_path in page_paths:
            page_results = load_json(page_path)
            results.extend(page_results)

        save_json(output_path, results)

def merge_document_jsons(json_paths: list, output_path: str):
    results = []
    for json_path in json_paths:
        _results = load_json(json_path)
        results.extend(_results)
    results.sort(key=lambda x: x["date"])
    save_json(output_path, results)

def augment_labs_results(input_path: str, output_path: str):
    results = load_json(input_path)
    for result in results: augment_lab_result(result)
    save_json(output_path, results)

def build_final_labs_results(input_path: str, output_path: str):
    results = load_json(input_path)
    for result in results:
        result["name"] = result["_lab_spec"]["name"]
        keys = [key for key in result.keys() if key.startswith("_")]
        for key in keys: del result[key]
    save_json(output_path, results)

def build_labs_csv(input_path: str, output_path: str):
    results = load_json(input_path)
    save_csv(output_path, results)

def process_documents():
    # Convert PDF to images (one per page)
    pdf_paths = load_paths("inputs", lambda x: x.endswith(".pdf") and "analises" in x.lower() and "requisicao" not in x.lower())
    for pdf_path in pdf_paths: convert_pdf_to_images(pdf_path, "cache/docs/pages/images")

    # Transcribe page images to text
    image_paths = load_paths("cache/docs/pages/images", lambda x: x.endswith(".jpg") and "analises" in x.lower() and "requisicao" not in x.lower())
    for image_path in image_paths: 
        try: convert_image_to_text(image_path, f"cache/docs/pages/texts")
        except: continue

    # Parse page texts to json
    text_paths = load_paths("cache/docs/pages/texts", lambda x: x.endswith(".txt") and "analises" in x.lower() and "requisicao" not in x.lower())
    for text_path in text_paths: convert_text_to_json(text_path, "cache/docs/pages/jsons")

    # Merge page jsons into document jsons
    json_paths = load_paths("cache/docs/pages/jsons")
    merge_page_jsons(json_paths, "cache/docs/jsons")

    # Merge document jsons into final json
    json_paths = load_paths("cache/docs/jsons")
    merge_document_jsons(json_paths, "outputs/labs_results.json")

    # Augment the merged labs
    json_paths = load_paths("cache/docs/jsons")
    augment_labs_results("outputs/labs_results.json", "outputs/labs_results.augmented.json")

    # Build the final json file using augmentations
    build_final_labs_results("outputs/labs_results.augmented.json", "outputs/labs_results.final.json")

    # Save the final json file as csv
    build_labs_csv("outputs/labs_results.final.json", "outputs/labs_results.final.csv")

process_documents()
