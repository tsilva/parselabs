import os
import json
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

def load_json(path):
    with open(path, "r", encoding="utf-8") as f: 
        data = json.load(f)
    return data

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f: 
        json.dump(data, f, indent=2, ensure_ascii=False)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if magnitude == 0: return 0
    return dot_product / magnitude

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

def create_completion(user_prompt, model="gpt-3.5-turbo", temperature=0.0, system_prompt=None):
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
    )
    response_message = response.choices[0].message.content
    return response_message

def generate_units_for_lab_name(name):
    system_prompt = """
Given a lab test, what are the most 3 most common units used to measure that lab test (by default consider that it's a blood lab, unless specified otherwise). Output as CSV string, ordered from most to least common, with no other content.

eg: 
USER: Leucograma - Eosinófilos
ASSISTANT: %;células/µL;células/mm³
""".strip()
    units_s = create_completion(name, system_prompt=system_prompt)
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
    print(range_s)
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
def find_most_similar_lab_name(name, threshold=0.80):
    matches = find_similar_lab_names(name)
    #print(matches[:3])
    filtered_matches = [(match[0], match[1]) for match in matches if match[1] >= threshold]
    if not filtered_matches: return None
    return filtered_matches[0][0]

@functools.lru_cache(maxsize=None)
def find_most_similar_lab_spec(name, threshold=0.80):
    similar_name = find_most_similar_lab_name(name, threshold=threshold)
    print(similar_name)
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

    lab_spec = find_most_similar_lab_spec(result_name) or {}
    units = lab_spec.get("units", {})
    lab_spec_name = lab_spec.get("name")
    lab_spec_unit = units.get(result_unit, {})
    lab_spec_unit_range_min = lab_spec_unit.get("min")
    lab_spec_unit_range_max = lab_spec_unit.get("max")

    result_value_within_lab_spec_range = True
    if lab_spec_unit_range_min != None and result_value <= lab_spec_unit_range_min: result_value_within_lab_spec_range = False
    if lab_spec_unit_range_max != None and result_value >= lab_spec_unit_range_max: result_value_within_lab_spec_range = False

    result["_name"] = lab_spec_name
    result["_valid_unit"] = bool(lab_spec_unit)
    result["_valid_range"] = result_value_within_lab_spec_range

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
