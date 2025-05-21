from dotenv import load_dotenv
load_dotenv(override=True)

import os
import json
import pandas as pd
from pathlib import Path
import re

# Load OUTPUT_DIR from environment or set manually
OUTPUT_DIR = os.getenv("OUTPUT_PATH")
OUTPUT_DIR = Path(OUTPUT_DIR)

LAB_NAMES_MAP_PATH = Path("config/lab_names_map.json")

# Read lab_names_map.json
with open(LAB_NAMES_MAP_PATH, "r", encoding="utf-8") as f:
    lab_names_map = json.load(f)

# Recursively find all .csv files not ending with .001.csv, .002.csv, etc.
csv_files = [
    p for p in OUTPUT_DIR.rglob("*.csv")
    if not p.name[-7:-4].isdigit()
]

# Track if we updated the mapping
updated = False

def slugify(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_-]+", "_", value)
    return value

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    if "lab_name" not in df.columns:
        continue
    # Add or update lab_name_enum column
    enum_col = []
    for lab_name in df["lab_name"]:
        slug = slugify(lab_name)
        mapped = lab_names_map.get(slug)
        if mapped is None:
            mapped = "$UNKNOWN$"
            lab_names_map[slug] = mapped
            updated = True
        enum_col.append(mapped)
    df["lab_name_enum"] = enum_col
    df.to_csv(csv_path, index=False)

# Write updated mapping if changed
if updated:
    with open(LAB_NAMES_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(lab_names_map, f, indent=4, ensure_ascii=False)
