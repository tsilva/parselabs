import json
import os

def fix_percent_suffix(filepath):
    with open(filepath, encoding="utf-8") as f:
        mappings = json.load(f)
    changed = False
    for k, v in mappings.items():
        if "percent" in k and not str(v).strip().endswith("(%)"):
            mappings[k] = f"{v.strip()} (%)"
            changed = True
    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        print(f"Patched: {filepath}")
    else:
        print("No changes needed.")

if __name__ == "__main__":
    fix_percent_suffix("config/lab_names_mappings.json")
