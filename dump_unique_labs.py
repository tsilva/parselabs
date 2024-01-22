import json
with open("mappings.json", "r", encoding='utf-8') as file: mappings = json.load(file)
values = mappings.values()
values = list(set(values))
values = sorted(values)
with open("unique_labs.json", "w", encoding='utf-8') as file: json.dump(values, file, indent=2, ensure_ascii=False)
