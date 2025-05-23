import json

PREFIXES = {
    "blood-": "Blood - ",
    "urine-": "Urine - ",
    "feces-": "Feces - ",
}

def fix_prefixes(path="config/lab_names_mappings.json"):
    with open(path, encoding="utf-8") as f:
        mappings = json.load(f)
    changed = False
    for k in list(mappings.keys()):
        for prefix, label in PREFIXES.items():
            if k.startswith(prefix):
                v = mappings[k]
                if not v.startswith(label):
                    # Remove any existing prefix up to first ' - '
                    if " - " in v:
                        v = v.split(" - ", 1)[-1]
                    mappings[k] = label + v.lstrip()
                    changed = True
                break
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)
        print("Prefixes fixed and file updated.")
    else:
        print("No changes needed.")

if __name__ == "__main__":
    fix_prefixes()
