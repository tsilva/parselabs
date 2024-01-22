import os
import json
import functools
import numpy as np
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CACHE_EMBEDDINGS_DIR = "cache/embeddings"
OPENAI_CLIENT = OpenAI()

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

@functools.lru_cache(maxsize=None)
def load_labs_specs():
    with open("labs_specs.json", "r", encoding="utf-8") as f: labs_specs = json.load(f)
    return labs_specs

@functools.lru_cache(maxsize=None)
def lab_spec_by_name(name):
    labs_specs = load_labs_specs()
    for spec in labs_specs.values():
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
    embeddings = load_lab_spec_embeddings()
    for name, _embedding in embeddings.items():
        similarity = cosine_similarity(embedding, _embedding)
        matches[name] = similarity
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
    lab_spec = lab_spec_by_name(similar_name)
    return lab_spec
