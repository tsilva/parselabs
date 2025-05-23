from dotenv import load_dotenv
load_dotenv(override=True)

import os
import json
import pandas as pd
from pathlib import Path
import re
import unicodedata

MAPPING_MODEL_ID = os.getenv("MAPPING_MODEL_ID")
OUTPUT_DIR = os.getenv("OUTPUT_PATH")
OUTPUT_DIR = Path(OUTPUT_DIR)

# Generalized config for both labs and units
MAPPING_CONFIGS = [
    {
        "col_type": "lab_type",
        "col_value": "lab_name",
        "enum_col": "lab_name_enum",
        "map_path": Path("config/lab_names_mappings.json"),
        "all_values_path": Path("config/all_lab_names.txt"),
        "use_type_prefix": True,
    },
    {
        "col_type": "lab_type",
        "col_value": "lab_unit",
        "enum_col": "lab_unit_enum",
        "map_path": Path("config/lab_units_mappings.json"),
        "all_values_path": Path("config/all_lab_units.txt"),
        "use_type_prefix": False,
    }
]

LLM_BATCH_SIZE = 50

def slugify(value):
    value = str(value).strip().lower()
    value = value.replace('%', 'percent')  # Replace % with "percent"
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_-]+", "", value)
    return value

def process_mapping(config):
    # Read mapping file
    if not config["map_path"].exists():
        with open(config["map_path"], "w", encoding="utf-8") as f:
            json.dump({}, f, indent=4, ensure_ascii=False)
    with open(config["map_path"], "r", encoding="utf-8") as f:
        value_map = json.load(f)

    # Recursively find all .csv files not ending with .001.csv, .002.csv, etc.
    csv_files = [
        p for p in OUTPUT_DIR.rglob("*.csv")
        if not p.name[-7:-4].isdigit()
    ]

    updated = False

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if config["col_value"] not in df.columns or config["col_type"] not in df.columns:
            continue
        enum_col = []
        for col_type, col_value in zip(df[config["col_type"]], df[config["col_value"]]):
            if config.get("use_type_prefix", False):
                key = f"{slugify(col_type)}-{slugify(col_value)}"
            else:
                key = slugify(col_value)
            mapped = value_map.get(key)
            if mapped is None:
                mapped = "$UNKNOWN$"
                value_map[key] = mapped
                updated = True
            enum_col.append(mapped)
        df[config["enum_col"]] = enum_col
        df.to_csv(csv_path, index=False)

    # Write updated mapping if changed
    if updated:
        value_map = dict(sorted(value_map.items(), key=lambda item: item[1]))
        with open(config["map_path"], "w", encoding="utf-8") as f:
            json.dump(value_map, f, indent=4, ensure_ascii=False)

    # Step 2: Fill unknowns using LLM if any remain
    unknown_keys = [k for k, v in value_map.items() if v == "$UNKNOWN$"]
    if unknown_keys:
        with open(config["all_values_path"], "r", encoding="utf-8") as f:
            txt_values = set(line.strip() for line in f if line.strip())
        value_map_values = set(v for v in value_map.values() if v != "$UNKNOWN$")
        canonical_enum_values = sorted(txt_values | value_map_values)

        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

        def batch(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i+n]

        SYSTEM_PROMPT = f"""
You are an expert at mapping noisy laboratory {config['enum_col']}s to a canonical list for data normalization.

Instructions:
- For each input, select the closest match from the provided canonical list (below).
- Only use values from the canonical list. If there is no close match, pick the closest anyway.
- Output a JSON dictionary mapping each input to exactly one canonical value.
- Do not invent or skip any input.
- Canonical list:
{json.dumps(canonical_enum_values, ensure_ascii=False, indent=2)}
""".strip()

        def map_batch_with_llm(batch):
            user_prompt = (
                f"Map the following {config['enum_col']}s to the canonical list. Output a JSON dictionary mapping each input to a canonical value.\n\n"
                + json.dumps(batch, ensure_ascii=False, indent=2)
            )
            print(f"User prompt: {user_prompt}")
            completion = client.chat.completions.create(
                model=MAPPING_MODEL_ID,
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

        remaining_keys = [k for k, v in value_map.items() if v == "$UNKNOWN$"]
        updated = False
        progress = True

        while remaining_keys and progress:
            progress = False
            for batch_keys in batch(remaining_keys, LLM_BATCH_SIZE):
                print(f"Mapping batch: {batch_keys[0]} ... ({len(batch_keys)} items)")
                batch_mapping = map_batch_with_llm(batch_keys)
                invalid_keys = []
                for k, v in batch_mapping.items():
                    if v in canonical_enum_values:
                        if value_map[k] != v:
                            value_map[k] = v
                            updated = True
                            progress = True
                    else:
                        print(f"Warning: LLM mapped '{k}' to unknown enum value: '{v}'")
                        invalid_keys.append(k)
                # Save after each batch
                value_map_sorted = dict(sorted(value_map.items(), key=lambda item: item[1]))
                with open(config["map_path"], "w", encoding="utf-8") as f:
                    json.dump(value_map_sorted, f, indent=2, ensure_ascii=False)
            remaining_keys = [k for k, v in value_map.items() if v == "$UNKNOWN$"]

        if updated:
            value_map = dict(sorted(value_map.items(), key=lambda item: item[1]))
            with open(config["map_path"], "w", encoding="utf-8") as f:
                json.dump(value_map, f, indent=2, ensure_ascii=False)
            print(f"{config['map_path'].name} updated.")
        else:
            print("No updates made.")
        if remaining_keys:
            print(f"Unmapped keys remaining after all attempts: {remaining_keys}")

# Run for both labs and units
for config in MAPPING_CONFIGS:
    process_mapping(config)
