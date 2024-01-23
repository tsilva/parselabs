import re
import os
import json
from utils import load_labs_specs, save_json

labs_specs = load_labs_specs()

results = []
for file in os.listdir("cache/docs"):
    if not file.endswith('.json'): continue
    json_path = os.path.join("cache/docs", file)
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    dates = re.findall(date_pattern, json_path)
    date = dates[0] if dates else None
    if not date: raise Exception(f"Could not find date in path: {json_path}")
    with open(json_path, "r", encoding='utf-8') as json_file: _results = json.load(json_file)
    for _result in _results:  
        _result["date"] = date
        results.append(_result)
results = sorted(results, key=lambda x: x["date"])

save_json("outputs/labs.json", results)
