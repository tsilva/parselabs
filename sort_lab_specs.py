import json

INPUT = "config/lab_specs.json"

with open(INPUT, encoding="utf-8") as f:
    data = json.load(f)

# Sort by keys
sorted_data = dict(sorted(data.items()))

with open(INPUT, "w", encoding="utf-8") as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=2)
