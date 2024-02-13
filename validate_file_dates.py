import os
from utils import load_json

files = os.listdir("cache/docs/pages/jsons")
for file in files:
    if not file.endswith(".json"): continue
    date = file.split(" ")[0].strip()
    results = load_json(f"cache/docs/pages/jsons/{file}")
    for result in results:
        if result['date'] != date:
            print(f"Date mismatch: {date} vs {result['date']} in {file}")
            break
