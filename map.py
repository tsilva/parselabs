from dotenv import load_dotenv
load_dotenv(override=True)

import os
import json
import pandas as pd
from pathlib import Path
import re

# Step 1: Map lab_type-lab_name to enum values

OUTPUT_DIR = os.getenv("OUTPUT_PATH")
OUTPUT_DIR = Path(OUTPUT_DIR)

LAB_NAMES_MAP_PATH = Path("config/lab_names_map.json")
LABS_TXT_PATH = Path("config/all_labs.txt")

# Read lab_names_map.json
with open(LAB_NAMES_MAP_PATH, "r", encoding="utf-8") as f:
    lab_names_map = json.load(f)

# Recursively find all .csv files not ending with .001.csv, .002.csv, etc.
csv_files = [
    p for p in OUTPUT_DIR.rglob("*.csv")
    if not p.name[-7:-4].isdigit()
]

updated = False

def slugify(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_-]+", "_", value)
    return value

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    if "lab_name" not in df.columns or "lab_type" not in df.columns:
        continue
    enum_col = []
    for lab_type, lab_name in zip(df["lab_type"], df["lab_name"]):
        key = f"{slugify(lab_type)}-{slugify(lab_name)}"
        mapped = lab_names_map.get(key)
        if mapped is None:
            mapped = "$UNKNOWN$"
            lab_names_map[key] = mapped
            updated = True
        enum_col.append(mapped)
    df["lab_name_enum"] = enum_col
    df.to_csv(csv_path, index=False)

# Write updated mapping if changed
if updated:
    with open(LAB_NAMES_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(lab_names_map, f, indent=4, ensure_ascii=False)

# Step 2: Fill unknowns using LLM if any remain

unknown_keys = [k for k, v in lab_names_map.items() if v == "$UNKNOWN$"]
if unknown_keys:
    # Load canonical enum values
    with open(LABS_TXT_PATH, "r", encoding="utf-8") as f:
        lab_enum_values = set(line.strip() for line in f if line.strip())

    # Optionally use OpenAI or OpenRouter client as in main.py
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    MODEL_ID = "google/gemini-2.5-flash-preview-05-20"

    def batch(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    BATCH_SIZE = 30

    SYSTEM_PROMPT = f"""
You are an expert at mapping noisy laboratory test names to a canonical list of lab test names for data normalization.

Instructions:
- For each input lab name, select the closest match from the provided canonical list (below).
- Only use values from the canonical list. If there is no close match, pick the closest anyway.
- Output a JSON dictionary mapping each input to exactly one canonical value.
- Do not invent or skip any input.
- Canonical list:
{json.dumps(sorted(list(lab_enum_values)), ensure_ascii=False, indent=2)}
""".strip()

    def map_batch_with_llm(batch):
        user_prompt = (
            "Map the following lab names to the canonical list. Output a JSON dictionary mapping each input to a canonical value.\n\n"
            + json.dumps(batch, ensure_ascii=False, indent=2)
        )
        print(f"User prompt: {user_prompt}")
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=4096
        )
        content = completion.choices[0].message.content
        try:
            mapping = json.loads(content)
        except Exception:
            import re
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                mapping = json.loads(match.group(0))
            else:
                raise RuntimeError(f"Could not parse LLM output: {content}")
        return mapping

    remaining_keys = [k for k, v in lab_names_map.items() if v == "$UNKNOWN$"]
    updated = False
    progress = True

    while remaining_keys and progress:
        progress = False
        for batch_keys in batch(remaining_keys, BATCH_SIZE):
            print(f"Mapping batch: {batch_keys[0]} ... ({len(batch_keys)} items)")
            batch_mapping = map_batch_with_llm(batch_keys)
            invalid_keys = []
            for k, v in batch_mapping.items():
                if v in lab_enum_values:
                    if lab_names_map[k] != v:
                        lab_names_map[k] = v
                        updated = True
                        progress = True
                else:
                    print(f"Warning: LLM mapped '{k}' to unknown enum value: '{v}'")
                    invalid_keys.append(k)
        remaining_keys = [k for k, v in lab_names_map.items() if v == "$UNKNOWN$"]

    if updated:
        with open(LAB_NAMES_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(lab_names_map, f, indent=2, ensure_ascii=False)
        print("lab_names_map.json updated.")
    else:
        print("No updates made.")
    if remaining_keys:
        print(f"Unmapped keys remaining after all attempts: {remaining_keys}")
