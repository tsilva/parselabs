import json
import concurrent.futures
import numpy as np
from openai import OpenAI
from itertools import combinations
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client with API key
client = OpenAI()

def cosine_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def _get_embedding(text):
    response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    embedding = response.data[0].embedding
    return text, embedding

# Load strings from JSON file
with open("unique_labs.json", encoding='utf-8') as f: strings = json.load(f)

# Obtain embeddings using ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    future_to_text = {executor.submit(_get_embedding, string): string for string in strings}
    embeddings = {}
    for future in concurrent.futures.as_completed(future_to_text):
        text, embedding = future.result()
        if embedding is not None: embeddings[text] = embedding

# Calculate cosine similarity between each pair
similarity_pairs = []
for (str1, vec1), (str2, vec2) in combinations(embeddings.items(), 2):
    similarity = cosine_sim(np.array(vec1), np.array(vec2))
    similarity_pairs.append((str1, str2, similarity))

# Sorting pairs by similarity
similarity_pairs.sort(key=lambda x: x[2], reverse=True)

# Output the results
for pair in similarity_pairs:
    if pair[2] < 0.95: continue
    print(f"Pair: {pair[0]}, {pair[1]} - Cosine Similarity: {pair[2]}")
