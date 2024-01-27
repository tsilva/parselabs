import os
import re
import json
import PyPDF2
import logging
import concurrent
from openai import OpenAI
from dotenv import load_dotenv
from pdf2image import convert_from_path
from utils import load_text, save_text, load_json, save_json, save_csv, load_paths, extract_text_from_image, create_completion, augment_lab_result, prompt_visual, validate_lab_test_name_mappings

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

def convert_image_to_text(image_path: str, output_directory: str, validator=None):
    # Skip if output already exists
    output_file_name = os.path.basename(image_path).replace(".jpg", ".txt")
    output_path = os.path.join(output_directory, output_file_name)
    if os.path.exists(output_path): return

    # Try different temperatures until the content is valid
    valid = True
    for temperature in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # Extract text from image
        logging.info(f"Extracting text from image: {image_path} (temperature={temperature})")
        content = extract_text_from_image(image_path, temperature=temperature)

        # Invalid, continue to next temperature
        if content.startswith("I'm sorry, but I can't assist"): continue

        # Assert that content was extracted correctly before proceeding
        valid = validator(image_path, content) if validator else True
        if valid: break

    # Raise exception if content was not extracted correctly
    if not valid: raise Exception(f"Extracted text did not match image: {image_path}")
    
    # Save the output
    save_text(output_path, content)

    # Log success
    logging.info(f"Extract text from image: {output_path}")

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

def build_augmented_labs_results(input_path: str, output_path: str, max_workers=5):
    results = load_json(input_path)

    def _augment(result): augment_lab_result(result)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_augment, result) for result in results]
        concurrent.futures.wait(futures)

    save_json(output_path, results)

def build_lab_name_mappings(input_path: str, output_path: str):
    mappings = {}
    labs = load_json(input_path)
    for lab in labs:
        name = lab["name"]
        lab_spec = lab.get("_lab_spec")
        spec_name = lab_spec["name"] if lab_spec else None
        mappings[name] = spec_name
    save_json(output_path, mappings)

def build_lab_name_invalid_mappings(input_path: str, output_path: str):
    mappings = load_json(input_path)
    invalid_mappings = validate_lab_test_name_mappings(mappings)
    save_json(output_path, invalid_mappings)

def build_final_labs_results(input_path: str, output_path: str):
    results = load_json(input_path)
    for result in results:
        lab_spec = result.get("_lab_spec")
        alt_name = lab_spec["name"] if lab_spec else None
        if alt_name: result["name"] = alt_name
        keys = [key for key in result.keys() if key.startswith("_")]
        for key in keys: del result[key]
    save_json(output_path, results)

def build_labs_csv(input_path: str, output_path: str):
    results = load_json(input_path)
    save_csv(output_path, results)

def validate_lab_results_text_in_image(image_path: str, content: str):
    result = prompt_visual(f"""
{content}
---------------------------------------------
If the lab results above match the image then answer "1", otherwise answer "0".
""".strip(), image_path, max_tokens=1)
    result = result.strip()
    valid = result == "1"
    return valid

def extract_invalid_lab_results_text_from_image(image_path: str, content: str):
    result = prompt_visual(f"""
{content}
---------------------------------------------
Which lab result values above don't match the ones in the image?
""".strip(), image_path, max_tokens=1)
    result = result.strip()
    return result

def validate_processed_documents(pdf_paths):
    errors = {}

    # Validate each document
    for pdf_path in pdf_paths:
        _errors = errors.get(pdf_path, [])
        def _log_error(msg):
            logging.error(msg)
            _errors.append(msg)

        # Validate each page
        with open(pdf_path, 'rb') as file: number_pages = len(PyPDF2.PdfReader(file).pages)
        pdf_file_name = os.path.basename(pdf_path)
        image_file_names = [pdf_file_name.replace(".pdf", f".{i+1:03}.jpg") for i in range(number_pages)]
        for image_file_name in image_file_names:
            # Check if image for page exists
            image_file_path = os.path.join("cache/docs/pages/images", image_file_name)
            if not os.path.exists(image_file_path): 
                _log_error(f"Missing page image: {image_file_path}")
                continue

            # Check if text for page exists
            text_file_path = os.path.join("cache/docs/pages/texts", image_file_name.replace(".jpg", ".txt"))
            if not os.path.exists(text_file_path): 
                _log_error(f"Missing page text: {text_file_path}")
                continue
                
            # Check that text matches image
            text = load_text(text_file_path)
            valid = validate_lab_results_text_in_image(image_file_path, text)
            if not valid: 
                invalid_values = extract_invalid_lab_results_text_from_image(image_file_path, text)
                _log_error(f"Invalid page text: {text_file_path} - {invalid_values}")
                continue

            # Check if json for page exists
            json_file_path = os.path.join("cache/docs/pages/jsons", image_file_name.replace(".jpg", ".json"))
            if not os.path.exists(json_file_path):
                _log_error(f"Missing page json: {json_file_path}")
                continue
            
            # @tsilva TODO: check that json is valid
        
        # If no errors were found in the pages 
        # then check if the final document json exists
        if not _errors:
            json_file_name = pdf_file_name.replace(".pdf", ".json")
            json_file_path = os.path.join("cache/docs/jsons", json_file_name)
            if not os.path.exists(json_file_path): 
                _log_error(f"Missing document json: {json_file_path}")

        # If errors were found associate them with the pdf
        if _errors: errors[pdf_path] = _errors

    # If any errors were found then raise exception
    if errors:
        raise Exception(f"Missing files: {json.dumps(errors, indent=2)}")

def process_documents():
    # Convert PDF to images (one per page)
    logging.info("Converting PDF to images")
    pdf_paths = load_paths("inputs", lambda x: x.endswith(".pdf") and "analises" in x.lower() and "requisicao" not in x.lower())
    for pdf_path in pdf_paths: convert_pdf_to_images(pdf_path, "cache/docs/pages/images")
    logging.info("Converting PDF to images... DONE")

    # Transcribe page images to text
    logging.info("Transcribing page images to text")
    image_paths = load_paths("cache/docs/pages/images", lambda x: x.endswith(".jpg") and "analises" in x.lower() and "requisicao" not in x.lower())
    for image_path in image_paths: convert_image_to_text(image_path, f"cache/docs/pages/texts", validator=validate_lab_results_text_in_image)
    logging.info("Transcribing page images to text... DONE")

    # Parse page texts to json
    logging.info("Parsing page texts to json")
    text_paths = load_paths("cache/docs/pages/texts", lambda x: x.endswith(".txt") and "analises" in x.lower() and "requisicao" not in x.lower())
    for text_path in text_paths: convert_text_to_json(text_path, "cache/docs/pages/jsons")
    logging.info("Parsing page texts to json... DONE")

    # Merge page jsons into document jsons
    logging.info("Merging page jsons into document jsons")
    json_paths = load_paths("cache/docs/pages/jsons")
    merge_page_jsons(json_paths, "cache/docs/jsons")
    logging.info("Merging page jsons into document jsons... DONE")
    
    # Merge document jsons into final json
    logging.info("Merging document jsons into final json")
    json_paths = load_paths("cache/docs/jsons")
    merge_document_jsons(json_paths, "outputs/labs_results.json")
    logging.info("Merging document jsons into final json... DONE")

    # Augment the merged labs
    logging.info("Augmenting merged labs")
    json_paths = load_paths("cache/docs/jsons")
    build_augmented_labs_results("outputs/labs_results.json", "outputs/labs_results.augmented.json")
    logging.info("Augmenting merged labs... DONE")
    
    # Build the lab name mappings
    logging.info("Building lab name mappings")
    build_lab_name_mappings("outputs/labs_results.augmented.json", "outputs/labs_results.mappings.json")
    logging.info("Building lab name mappings... DONE")

    # Build the lab name invalid mappings
    logging.info("Building lab name invalid mappings")
    build_lab_name_invalid_mappings("outputs/labs_results.mappings.json", "outputs/labs_results.invalid_mappings.json")
    logging.info("Building lab name invalid mappings... DONE")

    # Build the final json file using augmentations
    logging.info("Building final json file")
    build_final_labs_results("outputs/labs_results.augmented.json", "outputs/labs_results.final.json")
    logging.info("Building final json file... DONE")

    # Save the final json file as csv
    logging.info("Saving final json file as CSV")
    build_labs_csv("outputs/labs_results.final.json", "outputs/labs_results.final.csv")
    logging.info("Saving final json file as CSV... DONE")

    # Validate that all documents were correctly processed
    logging.info("Validating processed documents")
    validate_processed_documents(pdf_paths)
    logging.info("Validating processed documents... DONE")

process_documents()
