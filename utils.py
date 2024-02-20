import os
import csv
import json
import base64
import shutil
import functools
import numpy as np
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv
from hashlib import md5
from slugify import slugify

load_dotenv()

LAB_SPECS_FILE_NAME = "labs_specs.json"
CACHE_EMBEDDINGS_DIR = "/labs-parser-data/output/cache/embeddings"
OPENAI_CLIENT = OpenAI()

def load_labs_specs():
    labs_specs = load_json("labs_specs.json")
    return labs_specs

def load_paths(input_path: str, filter_function = None):
    paths = []
    for file in os.listdir(input_path):
        if filter_function and not filter_function(file): continue
        paths.append(os.path.join(input_path, file))
    return paths

def load_text(path):
    with open(path, "r", encoding="utf-8") as f: data = f.read()
    return data

def save_text(path, data):
    with open(path, "w", encoding="utf-8") as f: f.write(data)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    return data

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f: 
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_csv(path, data):
    keys = {}
    for row in data:
        for key in row.keys(): 
            keys[key] = True
    keys = list(keys.keys())

    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in data: writer.writerow(row)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if magnitude == 0: return 0
    return dot_product / magnitude

@functools.lru_cache(maxsize=None)
def create_embedding(text):
    response = OPENAI_CLIENT.embeddings.create(input=[text], model="text-embedding-ada-002")
    embedding = response.data[0].embedding
    return embedding

def create_embeddings(texts, max_workers=20):
    embeddings = {}
    def _embed(text): embeddings[text] = create_embedding(text)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_embed, text) for text in texts]
        concurrent.futures.wait(futures)
    return embeddings

def create_completion(user_prompt, model="gpt-3.5-turbo-0125", temperature=0.0, max_tokens=None, system_prompt=None, tools=[]):
    messages = ([{
        "role": "system",
        "content": system_prompt
    }] if system_prompt else []) + [{
        "role": "user",
        "content": user_prompt
    }]

    extra = {}
    if tools: extra["tools"] = tools
    if tools: extra["tool_choice"] = "auto"
    if max_tokens: extra["max_tokens"] = max_tokens

    response = OPENAI_CLIENT.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        timeout=60,
        **extra
    )
    message = response.choices[0].message
    return message

def extract_text_from_image(image_path: str, max_tokens=None, temperature=0.0):
    content = prompt_visual("Extract text from image verbatim, preserve table formatting.", image_path, max_tokens=max_tokens, temperature=temperature)
    return content

def prompt_visual(text_prompt: str, image_path: str, max_tokens=None, temperature=0.0):
    if not max_tokens: max_tokens = 2056

    with open(image_path, "rb") as image_file: base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = OPENAI_CLIENT.chat.completions.create(
        model="gpt-4-vision-preview", # gpt-4-0125-preview
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    }
                ]
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    content = content.strip()
    return content

@functools.lru_cache(maxsize=None)
def load_labs_specs():
    return load_json(LAB_SPECS_FILE_NAME)

@functools.lru_cache(maxsize=None)
def lab_spec_by_name(name):
    labs_specs = load_labs_specs()
    for spec in labs_specs:
        names = [spec["name"]] + spec.get("alternatives", [])
        names_l = [x.lower() for x in names]
        names_u = [x.upper() for x in names]
        names_uc = [x[0].upper() + x[1:] for x in names]
        names = names + names_l + names_u + names_uc
        if name in names: return spec
    return None

def load_embeddings_for_lab_spec_digest(digest):
    with open(f"embeddings/{digest}.json", "r", encoding="utf-8") as f: embeddings = json.load(f)
    return embeddings

@functools.lru_cache(maxsize=None)
def load_lab_spec_embeddings():
    embeddings = {}
    for filename in os.listdir(CACHE_EMBEDDINGS_DIR):
        with open(f"{CACHE_EMBEDDINGS_DIR}/{filename}", "r", encoding="utf-8") as f: _embeddings = json.load(f)
        for key, value in _embeddings.items(): embeddings[key] = value
    return embeddings

@functools.lru_cache(maxsize=None)
def find_similar_lab_names(name):
    matches = {}
    embedding = create_embedding(name)
    _embeddings = load_lab_spec_embeddings()
    for _name, _embedding in _embeddings.items():
        name_slug = slugify(name).replace("-", "").replace(".", "")
        _name_slug = slugify(_name).replace("-", "").replace(".", "")
        similarity = 1.0 if name_slug == _name_slug else cosine_similarity(embedding, _embedding)
        matches[_name] = similarity
    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    return sorted_matches

@functools.lru_cache(maxsize=None)
def find_most_similar_lab_name(name):
    matches = find_similar_lab_names(name)

    exact_matches = [match for match in matches if match[1] == 1.0]
    if exact_matches: return exact_matches[0][0]

    lab_names = [match[0] for match in matches[:3]]
    for lab_name in lab_names:
        valid = validate_lab_test_name_mapping(name, lab_name)
        if valid: return lab_name
    return None

@functools.lru_cache(maxsize=None)
def find_most_similar_lab_spec(name):
    similar_name = find_most_similar_lab_name(name)
    lab_spec = lab_spec_by_name(similar_name)
    return lab_spec

def augment_lab_result(result):
    result_name = result["name"]
    result_unit = result["unit"]
    result_value = result["value"]

    lab_spec = find_most_similar_lab_spec(result_name)
    if not lab_spec:
        result["_lab_spec"] = None
        return
    
    print(result_name + " -> " + lab_spec["name"] if lab_spec else "null")

    units = lab_spec.get("units", {})
    lab_spec_unit = units.get(result_unit, {})
    lab_spec_unit_range_min = lab_spec_unit.get("min")
    lab_spec_unit_range_max = lab_spec_unit.get("max")

    result_value_within_lab_spec_range = True
    if lab_spec_unit_range_min != None and result_value <= lab_spec_unit_range_min: result_value_within_lab_spec_range = False
    if lab_spec_unit_range_max != None and result_value >= lab_spec_unit_range_max: result_value_within_lab_spec_range = False

    result["_supported_lab_spec_unit"] = bool(lab_spec_unit)
    result["_within_lab_spec_range"] = result_value_within_lab_spec_range
    result["_lab_spec"] = lab_spec

def build_lab_spec_names_embeddings_cache():
    # Ensure that the lab name digests are up to date
    specs = load_json("labs_specs.json")
    for spec in specs:
        name = spec["name"]
        slug = slugify(name)
        spec["digest"] = md5(slug.encode()).hexdigest()
    save_json("labs_specs.json", specs)

    # Delete current cache
    if os.path.exists(CACHE_EMBEDDINGS_DIR): shutil.rmtree(CACHE_EMBEDDINGS_DIR)
    os.makedirs(CACHE_EMBEDDINGS_DIR)

    # For each spec, generate a file with the 
    # embeddings for the lab name and its alternatives
    labs_specs = load_labs_specs()
    for spec in labs_specs:
        name = spec["name"]
        digest = spec["digest"]
        alternatives = spec.get("alternatives", [])
        names = [name] + alternatives
        embeddings = create_embeddings(names)
        with open(f"{CACHE_EMBEDDINGS_DIR}/{digest}.json", "w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=4)

def generate_lab_name_mappings(labs):
    mappings = {}
    for lab in labs:
        name = lab["name"]
        lab_spec = find_most_similar_lab_spec(name)
        mapped_name = lab_spec["name"] if lab_spec else None
        mappings[name] = mapped_name
        print(f"{name} -> {mapped_name}")
    return mappings

@functools.lru_cache(maxsize=None)
def validate_lab_test_name_mapping(key, value):
    system_prompt = """
Your task is to determine whether two given lab test names, separated by an equals sign, refer to the same test. 
Respond with '1' for equivalent tests and '0' for non-equivalent tests.

Examples:

USER: Eosinófilos = Leucograma - Eosinófilos
ASSISTANT: 1
USER: Ferro = Ferritina
ASSISTANT: 0
USER: frro = Ferro
ASSISTANT: 1
USER: Uricémia = Homocisteína
ASSISTANT: 0
""".strip()
    user_prompt = f"{key} = {value}"
    message = create_completion(user_prompt, system_prompt=system_prompt, max_tokens=1)
    content = message.content
    content = content.strip()
    valid = content == "1"
    if not valid: print(f"{key} != {value}")
    return valid

def validate_lab_test_name_mappings(mappings, max_workers=10):
    invalid_mappings = {}
    def _validate(key, value): 
        valid = validate_lab_test_name_mapping(key, value)
        if not valid: invalid_mappings[key] = value
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_validate, key, value) for key, value in mappings.items()]
        concurrent.futures.wait(futures)
    return invalid_mappings
