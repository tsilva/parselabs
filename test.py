import os
import json
import logging    
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define constants
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_CLIENT = OpenAI()

def fix_name(name):
    response = OPENAI_CLIENT.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": f"Rewrite the given Portuguese lab test name using the most commonly used form in Portugal."
            },
            {
                "role": "user",
                "content": name
            },
        ],
    )
    response_message = response.choices[0].message.content
    return response_message

mappings = {}
files = os.listdir("output")
for file in files:
    if not file.endswith(".json"): continue
    file_path = os.path.join("output", file)
    with open(file_path, 'r', encoding='utf-8') as file: specs = json.load(file)

    def process_spec(spec):
        name = spec["name"]
        if name in mappings: return
        fixed_name = fix_name(name)
        mappings[name] = fixed_name
        print(f"{name} -> {fixed_name}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_spec, spec) for spec in specs]
        concurrent.futures.wait(futures)

with open("mappings.json", "w", encoding='utf-8') as file: json.dump(mappings, file, indent=2, ensure_ascii=False)