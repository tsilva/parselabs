import os
import csv
import json
import base64
import shutil
from tqdm import tqdm
import functools
import numpy as np
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv
from hashlib import md5
from slugify import slugify

load_dotenv()

LAB_SPECS_FILE_NAME = "labs_specs.json"
CACHE_EMBEDDINGS_DIR = "cache/embeddings"
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

def create_embedding(text):
    response = OPENAI_CLIENT.embeddings.create(input=[text], model="text-embedding-3-small")
    embedding = response.data[0].embedding
    return embedding

def create_embeddings(texts, max_workers=20):
    embeddings = {}
    def _embed(text): embeddings[text] = create_embedding(text)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_embed, text) for text in texts]
        concurrent.futures.wait(futures)
    return embeddings

def create_completion(user_prompt, model="gpt-3.5-turbo-1106", temperature=0.0, system_prompt=None, tools=[]):
    messages = ([{
        "role": "system",
        "content": system_prompt
    }] if system_prompt else []) + [{
        "role": "user",
        "content": user_prompt
    }]
    response = OPENAI_CLIENT.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        tools=tools,
        tool_choice="auto"
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

def generate_units_for_lab_name(name):
    system_prompt = """
Given a lab test, what are the most 3 most common units used to measure that lab test (by default consider that it's a blood lab, unless specified otherwise). Output as CSV string, ordered from most to least common, with no other content.

eg: 
USER: Leucograma - Eosinófilos
ASSISTANT: %;células/µL;células/mm³
""".strip()
    message = create_completion(name, system_prompt=system_prompt)
    units_s = message.content
    units = units_s.split(";")
    units = [unit.strip() for unit in units]
    return units

def generate_range_for_lab_unit(name: str, unit: str):
    USER_GENDER = os.environ["USER_GENDER"]
    USER_RACE = os.environ["USER_RACE"]
    USER_AGE = os.environ["USER_AGE"]
    USER_LOCATION = os.environ["USER_LOCATION"]
    USER_HEIGHT = os.environ["USER_HEIGHT"]
    USER_WEIGHT = os.environ["USER_WEIGHT"]

    system_prompt = f"""
Given a lab test corresponding measurement unit, what is the min and max healthy range for a person with the following characteristics:

Gender: {USER_GENDER}
Age: {USER_AGE}
Race: {USER_RACE}
Location: {USER_LOCATION}
Height: {USER_HEIGHT}
Weight: {USER_WEIGHT}

Output just the range as CSV string with range min value followed by range max value. If there is no min output NULL in its place, if there is no max output NULL in its place. Don't output anything else.

Eg:
USER: Vitamina E [UNIT = mg/L]
ASSISTANT: 5.5;18.5
USER: Vitamina E  [UNIT = µg/mL]
ASSISTANT: 2.8;18.4
USER: Vitamina E  [UNIT = µmol/L]
ASSISTANT: 12;42
""".strip()
    user_prompt = f"{name} [UNIT = {unit}]"
    print(user_prompt)
    range_s = create_completion(user_prompt, model="gpt-4-1106-preview", system_prompt=system_prompt)
    range_values = range_s.replace("<", "").replace(">", "").replace(",", ".").split(";")
    range_values = [value.strip() for value in range_values]
    range_min, range_max = [float(value) if value != "NULL" else None for value in range_values]
    return range_min, range_max

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
        name_slug = slugify(name).replace("-", "")
        _name_slug = slugify(_name).replace("-", "")
        similarity = 1.0 if name_slug == _name_slug else cosine_similarity(embedding, _embedding)
        matches[_name] = similarity
    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    return sorted_matches

@functools.lru_cache(maxsize=None)
def find_most_similar_lab_name(name, threshold=0.40):
    matches = find_similar_lab_names(name)
    #print(matches[:3])
    filtered_matches = [(match[0], match[1]) for match in matches if match[1] >= threshold]
    if not filtered_matches: 
        print(name + " " + json.dumps(matches[0]))
        return None
    return filtered_matches[0][0]

@functools.lru_cache(maxsize=None)
def find_most_similar_lab_spec(name, threshold=0.40):
    similar_name = find_most_similar_lab_name(name, threshold=threshold)
    lab_spec = lab_spec_by_name(similar_name)
    return lab_spec

def generate_units_for_lab_specs(lab_specs):
    for spec in tqdm(lab_specs, total=len(lab_specs)):
        name = spec["name"]
        units = spec.get("units")
        if units: continue
        units = generate_units_for_lab_name(name)
        spec["units"] = dict([(unit, {}) for unit in units])
    return lab_specs

def generate_units_ranges_for_lab_specs(lab_specs):
    for spec in tqdm(lab_specs, total=len(lab_specs) * 3):
        lab_name = spec["name"]
        units = spec.get("units", {})
        for unit_name, values in units.items():
            if values: 
                continue
            try: 
                range_min, range_max = generate_range_for_lab_unit(lab_name, unit_name)
            except Exception as e: 
                print(e)
                continue
            spec["units"][unit_name] = {
                "min": range_min,
                "max": range_max
            }
            print(unit_name)
            print(spec["units"][unit_name])
    return lab_specs

def augment_lab_result(result):
    result_name = result["name"]
    result_unit = result["unit"]
    result_value = result["value"]

    lab_spec = find_most_similar_lab_spec(result_name)
    if not lab_spec:
        result["_lab_spec"] = None
        return

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

def generate_lab_spec_names_embeddings_cache():
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
