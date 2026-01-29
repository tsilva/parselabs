from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env.local"), override=True)

import os
import sys
import json
import re
from collections import defaultdict
from openai import OpenAI
import concurrent.futures


# Validate required environment variables
if not os.getenv("EXTRACT_MODEL_ID"):
    print("Error: EXTRACT_MODEL_ID environment variable not set")
    sys.exit(1)

client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def get_conversion_factor(lab_name, from_unit, to_unit, temperature=0.0):
    """
    Uses LLM to get the conversion factor from from_unit to to_unit for a given lab_name.
    """
    system_prompt = (
        "You are a medical laboratory assistant. "
        "Given a lab test name and two units, provide the numeric conversion factor to convert a value from the first unit to the second. "
        "Respond with only the numeric factor."
    )
    user_prompt = (
        f"Lab test: {lab_name}\n"
        f"Convert from: {from_unit}\n"
        f"Convert to: {to_unit}\n"
        "What is the numeric conversion factor? Respond with only the number."
    )
    completion = client.chat.completions.create(
        model=os.getenv("EXTRACT_MODEL_ID"),
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt}
                ]
            }
        ],
        temperature=temperature,
        max_tokens=16
    )
    # Extract the numeric factor from the response
    response = completion.choices[0].message.content.strip()
    try:
        factor = float(response)
    except Exception:
        factor = None
    return factor

def get_health_range(lab_name, primary_unit, user_stats, temperature=0.0):
    """
    Uses LLM to get the health range for a given lab_name in its primary unit, considering user stats.
    """
    system_prompt = (
        "You are a medical laboratory assistant. "
        "Given a lab test name, its primary unit, and user stats (gender, age, weight, height, activity level), "
        "provide the healthy reference range for the test in the primary unit. "
        "Respond with only the numeric range, e.g., '3.5-5.0' or '70-110'."
    )
    user_prompt = (
        f"Lab test: {lab_name}\n"
        f"Primary unit: {primary_unit}\n"
        f"User stats: {json.dumps(user_stats)}\n"
        "What is the healthy reference range for this test in the primary unit? Respond with only the numeric range."
    )
    completion = client.chat.completions.create(
        model=os.getenv("EXTRACT_MODEL_ID"),
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt}
                ]
            }
        ],
        temperature=temperature,
        max_tokens=32
    )
    response = completion.choices[0].message.content.strip()
    return response

def parse_range_string(range_str, primary_unit=None):
    """
    Parses a range string like '3.5-5.0' or '70-110' and returns a dict with min and max as floats.
    For boolean units, returns 0-0 (negative/absent) or 1-1 (positive/present).
    Returns None if parsing fails.
    """
    # Handle boolean units
    if primary_unit is not None and primary_unit.lower() in ["boolean"]:
        s = range_str.strip().lower()
        if s in ["negative", "absent", "none", "no", "false", "0", "0-0"]:
            return {"min": 0, "max": 0}
        if s in ["positive", "present", "yes", "true", "1", "1-1"]:
            return {"min": 1, "max": 1}
    match = re.match(r"^\s*([-\d\.]+)\s*[--]\s*([-\d\.]+)\s*$", range_str)
    if match:
        try:
            min_val = float(match.group(1))
            max_val = float(match.group(2))
            return {"min": min_val, "max": max_val}
        except Exception:
            return None
    # Try single value (e.g., "5.0")
    try:
        val = float(range_str.strip())
        return {"min": val, "max": val}
    except Exception:
        return None

def main():
    labs_specs_path = "temp_lab_specs.json"

    # Load existing labs_specs
    with open("config/lab_specs.json", "r", encoding="utf-8") as f:
        labs_specs = json.load(f)

    # Example user stats (replace with dynamic input as needed)
    with open("user_stats.json", "r", encoding="utf-8") as f:
        user_stats = json.load(f)

    # Prepare health range update tasks
    health_range_tasks = []
    for lab_name, spec in labs_specs.items():
        primary_unit = spec["primary_unit"]
        health_range_tasks.append((lab_name, primary_unit))

    # Update healthy ranges in parallel
    def health_range_task_fn(args):
        lab_name, primary_unit = args
        health_range = get_health_range(lab_name, primary_unit, user_stats)
        print(f"Health range for {lab_name} ({primary_unit}): {health_range}")
        parsed = parse_range_string(health_range, primary_unit)
        return (lab_name, parsed)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(health_range_task_fn, args) for args in health_range_tasks]
        for future in concurrent.futures.as_completed(futures):
            lab_name, parsed_range = future.result()
            labs_specs[lab_name]["ranges"]["healthy"] = parsed_range

    # Save updated labs_specs
    with open(labs_specs_path, "w", encoding="utf-8") as f:
        json.dump(labs_specs, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
