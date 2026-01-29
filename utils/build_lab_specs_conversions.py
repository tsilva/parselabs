from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env.local"), override=True)

import os
import sys
import csv
import json
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

def main():
    input_csv = "output/all.csv"
    output_json = "temp_labs_specs.json"
    lab_units = defaultdict(set)

    # Step 1: Read CSV and collect unique lab_name_enum and lab_unit_enum
    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lab_name = row["lab_name_enum"]
            lab_unit = row["lab_unit_enum"]
            lab_units[lab_name].add(lab_unit)

    # Step 2: Prepare all conversion tasks
    labs_specs = {}
    conversion_tasks = []
    task_map = {}  # (lab_name, unit, primary_unit) -> index in alternatives

    for lab_name, units in lab_units.items():
        units = list(units)
        primary_unit = units[0]
        alternatives = []
        for unit in units:
            if unit == primary_unit:
                continue
            # Store the task and its mapping for later assembly
            conversion_tasks.append((lab_name, unit, primary_unit))
            task_map[(lab_name, unit, primary_unit)] = (lab_name, unit, primary_unit)
            alternatives.append({
                "unit": unit,
                "factor": None  # Placeholder to be filled later
            })
        labs_specs[lab_name] = {
            "primary_unit": primary_unit,
            "alternatives": alternatives
        }

    # Step 3: Run conversion tasks in parallel
    def task_fn(args):
        lab_name, unit, primary_unit = args
        factor = get_conversion_factor(lab_name, unit, primary_unit)
        print(f"Conversion factor for {lab_name} from {unit} to {primary_unit}: {factor}")
        return (lab_name, unit, primary_unit, factor)

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(task_fn, args) for args in conversion_tasks]
        for future in concurrent.futures.as_completed(futures):
            lab_name, unit, primary_unit, factor = future.result()
            # Find the correct alternative entry and fill in the factor
            alternatives = labs_specs[lab_name]["alternatives"]
            for alt in alternatives:
                if alt["unit"] == unit:
                    alt["factor"] = factor
                    break

    # Step 4: Save as JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(labs_specs, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
