import os
import json
import shutil
from hashlib import md5
from slugify import slugify
from utils import load_labs_specs, create_embeddings

CACHE_EMBEDDINGS_DIR = "cache/embeddings"

def initialize_labs_specs():
    with open("labs_specs.json", "r", encoding="utf-8") as f: specs = json.load(f)
    for spec in specs:
        name = spec["name"]
        slug = slugify(name)
        spec["digest"] = md5(slug.encode()).hexdigest()
    with open("labs_specs.json", "w", encoding="utf-8") as f: json.dump(specs, f, ensure_ascii=False, indent=4)
    _initialize_labs_specs__embeddings_cache()

def _initialize_labs_specs__embeddings_cache():
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

def setup():
    initialize_labs_specs()

setup()