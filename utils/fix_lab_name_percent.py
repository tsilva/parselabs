import os
import json

def fix_bilirrubina_urine_lab_results_in_file(filepath):
    changed = False
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    if "lab_results" in data:
        for lab in data["lab_results"]:
            if (
                str(lab.get("lab_type", "")).strip().lower() == "urine"
                and "citol".lower() in str(lab.get("lab_name", "")).strip().lower()
            ):
                    lab["lab_unit"] = "boolean"
                    lab["lab_range_min"] = 0
                    lab["lab_range_max"] = 1
                    changed = True
    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    for root, _, files in os.walk("output"):
        for fname in files:
            if fname.endswith(".json"):
                fpath = os.path.join(root, fname)
                try:
                    fix_bilirrubina_urine_lab_results_in_file(fpath)
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")

if __name__ == "__main__":
    main()

