import json
import os
from utils import find_most_similar_lab_spec

mappings = {}
for file in os.listdir("cache/docs"):
    if not file.endswith(".json"): continue
    results = json.load(open(f"cache/docs/{file}", "r", encoding="utf-8"))
    for result in results:
        name = result["name"]
        spec = find_most_similar_lab_spec(name)
        mappings[name] = spec['name'] if spec else None 
        print(f"{name} -> {spec['name'] if spec else None}")
with open("outputs/mappings.json", "w", encoding="utf-8") as f: json.dump(mappings, f, ensure_ascii=False, indent=4)
