import os
import re
import json
import base64
import logging
import requests 
#import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
#from pdf2image import convert_from_path
from utils import load_json, find_most_similar_lab_spec

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define constants
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_CLIENT = OpenAI()

def load_labs_specs():
    labs_specs = load_json("labs_specs.json")
    return labs_specs

def convert_pdfs_to_images(directory):
    for file in os.listdir(directory):
        if not file.endswith('.pdf'): continue
        input_path = os.path.join(directory, file)
        output_path_format = input_path.replace(".pdf", ".000.jpg")
        try: _convert_pdfs_to_images(input_path, output_path_format)
        except Exception as e: logging.error(f"Could not convert {input_path}: {e}")
        else: logging.info(f"Successfully extracted images from PDF: `{input_path}`.")

def _convert_pdfs_to_images(input_path: str, output_path_format: str):
    images = convert_from_path(input_path, dpi=300, poppler_path='/usr/bin')
    for i, image in enumerate(images):
        output_path = output_path_format.replace("000", f"{i+1:03}").replace("input/", "outputs/")
        image.save(output_path, 'JPEG', quality=100)

def convert_images_to_text(directory: str):
    for file in os.listdir(directory):
        if not file.endswith('.jpg') and not file.endswith(".png"): continue
        if not "analises" in file.lower(): continue
        
        input_path = os.path.join(directory, file)
        output_path = input_path.replace(".jpg", ".txt").replace("input/", "outputs/")
        if os.path.exists(output_path): continue

        try: _convert_images_to_text(input_path, output_path)
        except Exception as e: logging.error(f"Could not convert {input_path}: {e}")
        else: logging.info(f"Successfully extracted text from `{input_path}`.")

def _convert_images_to_text(input_path: str, output_path: str):
    # Getting the base64 string
    with open(input_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract text from image verbatim, preserve table formatting."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 3096
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    lines = content.split("\n")
    lines = [line for line in lines if line.strip()]
    if len(lines) < 2: raise Exception(f"Could not extract text from image: {input_path}")

    # Save the image as a JPEG with high quality
    with open(output_path, "w", encoding='utf-8') as text_file: text_file.write(content)

def convert_texts_to_json(directory):
    for file in os.listdir(directory):
        if not file.endswith('.txt'): continue
        if not "analises" in file.lower(): continue

        input_path = os.path.join(directory, file)
        output_path = input_path.replace(".txt", ".json").replace("input/", "output/")
        if os.path.exists(output_path): continue

        try: _convert_texts_to_json(input_path, output_path)
        except Exception as e: logging.error(f"Could not convert {input_path}: {e}")
        else: logging.info(f"Successfully extracted JSON from TXT: `{input_path}`.")

def _convert_texts_to_json(input_path: str, output_path: str):
    with open(input_path, "r", encoding='utf-8') as text_file: 
        text = text_file.read()

    with open("tools/save_extract_blood_lab_results.json", "r", encoding='utf-8') as function_file: 
        save_extract_blood_lab_results_tool = json.load(function_file)
    
    response = OPENAI_CLIENT.chat.completions.create(
        #model="gpt-3.5-turbo-0613",
        model="gpt-4-1106-preview",
        tools = [save_extract_blood_lab_results_tool],
        tool_choice="auto",
        messages=[
            {
                "role": "system",
                "content": f"You are the best language model in the world at extracting blood lab results from text files. You are extremely capable at this and always get it right. Go for it!"
            },
            {
                "role": "user",
                "content": f"Extract and save the following blood lab results:\n\n {text}"
            },
        ],
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if not tool_calls: raise Exception(f"No tool calls found: {input_path}")

    tool_call = tool_calls[0]
    function_args = json.loads(tool_call.function.arguments)

    results = function_args["blood_lab_results"]
    for result in results:
        for key, value in result.items():
            if value == "": raise Exception(f"Empty value for key `{key}`: {input_path}")

    with open(output_path, "w", encoding='utf-8') as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)

def validate_jsons(directory):

    def _is_valid_date(date_string):
        from datetime import datetime
        try: datetime.strptime(date_string, '%Y-%m-%d'); return True
        except ValueError: return False

    def _validate_result(result):
        errors = {}

        name = result.get("name")
        if not name: errors["name"] = "Missing name"

        value = result.get("value")
        if value in ["", None]: errors["value"] = "Missing value"
        elif not isinstance(value, int) and not isinstance(value, float): errors["value"] = "Invalid value"

        range_minimum = result.get("range_minimum")
        if range_minimum and not isinstance(value, int) and not isinstance(value, float): errors["range_minimum"] = "Invalid range_minimum"

        range_maximum = result.get("range_maximum")
        if range_maximum and not isinstance(value, int) and not isinstance(value, float): errors["range_maximum"] = "Invalid range_maximum"

        if not errors: return None
        return errors

    errors = {}
    for file in os.listdir(directory):
        if not file.endswith('.json'): continue
        json_path = os.path.join(directory, file)
        with open(json_path, "r", encoding='utf-8') as json_file: _results = json.load(json_file)
        for _result in _results: 
            _errors = _validate_result(_result)
            if not _errors: continue
            errors[json_path] = _errors
            break

    if errors:
        for json_path, _errors in errors.items():
            logging.error(f"Invalid JSON: {json_path} - {_errors}")
        raise Exception("Invalid JSONs")




def json_to_csv():
    with open("outputs/blood_labs.json", "r", encoding='utf-8') as file: labs = json.load(file)
    df = pd.DataFrame(labs)
    df.to_csv("outputs/blood_labs.csv", index=False)

#convert_pdfs_to_images("input") # @tsilva TODO: parallelize this
#convert_images_to_text("output")
#convert_texts_to_json("output")
#validate_jsons("output")
merge_jsons()
#json_to_csv()

# @tsilva TODO: validate the results
