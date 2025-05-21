from dotenv import load_dotenv
load_dotenv(override=True)

import json
from pathlib import Path
import os

# Optionally use OpenAI or OpenRouter client as in main.py
from openai import OpenAI

LABS_TXT_PATH = Path("config/labs.txt")
LAB_NAMES_MAP_PATH = Path("config/lab_names_map.json")

# Load labs.txt as set of possible enum values
with open(LABS_TXT_PATH, "r", encoding="utf-8") as f:
    lab_enum_values = set(line.strip() for line in f if line.strip())

# Load lab_names_map.json
with open(LAB_NAMES_MAP_PATH, "r", encoding="utf-8") as f:
    lab_names_map = json.load(f)

# Find all unmapped (value == "$UNKNOWN$")
unknown_keys = [k for k, v in lab_names_map.items() if v == "$UNKNOWN$"]
if not unknown_keys:
    print("No unknowns to map.")
    exit(0)

# LLM setup
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
MODEL_ID = "google/gemini-2.5-flash-preview-05-20"

# Helper: batch unknowns for LLM (to avoid context overflow)
def batch(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

BATCH_SIZE = 30  # Tune as needed

# LLM prompt templates
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
    # Parse LLM output
    content = completion.choices[0].message.content
    try:
        mapping = json.loads(content)
    except Exception:
        # Try to extract JSON substring if LLM output is verbose
        import re
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            mapping = json.loads(match.group(0))
        else:
            raise RuntimeError(f"Could not parse LLM output: {content}")
    # Validate keys and values
    assert set(mapping.keys()) == set(batch), "LLM output keys mismatch"
    for v in mapping.values():
        if v not in lab_enum_values:
            raise ValueError(f"LLM mapped to unknown enum value: {v}")
    return mapping

# Main mapping loop
updated = False
for batch_keys in batch(unknown_keys, BATCH_SIZE):
    print(f"Mapping batch: {batch_keys[0]} ... ({len(batch_keys)} items)")
    batch_mapping = map_batch_with_llm(batch_keys)
    for k, v in batch_mapping.items():
        lab_names_map[k] = v
        updated = True

# Save updated mapping
if updated:
    with open(LAB_NAMES_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(lab_names_map, f, indent=2, ensure_ascii=False)
    print("lab_names_map.json updated.")
else:
    print("No updates made.")
