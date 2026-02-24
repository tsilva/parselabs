#!/usr/bin/env python3
"""
Lab Specifications Manager - Consolidated utility for managing lab_specs.json.

This script combines multiple lab spec maintenance operations:
  - sort: Sort lab_specs.json alphabetically by lab name
  - fix-encoding: Convert Unicode escape sequences to UTF-8 characters
  - build-conversions: Generate unit conversion factors from extracted data
  - build-ranges: Generate healthy reference ranges using LLM

Usage:
    python utils/lab_specs_manager.py <command> [options]

Commands:
    sort                    Sort lab_specs.json alphabetically
    fix-encoding            Fix Unicode encoding in lab_specs.json
    build-conversions       Build conversion factors from output/all.csv
    build-ranges            Build healthy ranges using LLM

Examples:
    python utils/lab_specs_manager.py sort
    python utils/lab_specs_manager.py fix-encoding
    python utils/lab_specs_manager.py build-conversions
    python utils/lab_specs_manager.py build-ranges
"""

import argparse
import concurrent.futures
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(
    Path(".env.local") if Path(".env.local").exists() else Path(".env"),
    override=True,
)

LAB_SPECS_PATH = Path("config/lab_specs.json")
TEMP_PATH = Path("temp_lab_specs.json")


def validate_env():
    """Validate required environment variables are set."""
    if not os.getenv("EXTRACT_MODEL_ID"):
        print("Error: EXTRACT_MODEL_ID environment variable not set")
        sys.exit(1)
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)


def get_openai_client():
    """Get configured OpenAI client for OpenRouter."""
    return OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


def cmd_sort(args):
    """Sort lab_specs.json alphabetically by lab name."""
    with open(LAB_SPECS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    sorted_data = dict(sorted(data.items()))

    with open(LAB_SPECS_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"✓ Sorted {len(data)} lab entries in {LAB_SPECS_PATH}")


def cmd_fix_encoding(args):
    """Fix Unicode encoding by converting escape sequences to UTF-8."""
    if not LAB_SPECS_PATH.exists():
        raise FileNotFoundError(f"File not found: {LAB_SPECS_PATH}")

    if args.backup:
        backup_path = LAB_SPECS_PATH.with_suffix(".json.backup")
        shutil.copy2(LAB_SPECS_PATH, backup_path)
        print(f"✓ Created backup: {backup_path}")

    with open(LAB_SPECS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(LAB_SPECS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Validate
    with open(LAB_SPECS_PATH, "r", encoding="utf-8") as f:
        validated_data = json.load(f)

    if data != validated_data:
        raise ValueError("Data integrity check failed!")

    print(f"✓ Fixed encoding in {LAB_SPECS_PATH}")


def get_conversion_factor(lab_name, from_unit, to_unit, client, temperature=0.0):
    """Use LLM to get conversion factor from from_unit to to_unit."""
    system_prompt = "You are a medical laboratory assistant. Given a lab test name and two units, provide the numeric conversion factor to convert a value from the first unit to the second. Respond with only the numeric factor."
    user_prompt = f"Lab test: {lab_name}\nConvert from: {from_unit}\nConvert to: {to_unit}\nWhat is the numeric conversion factor? Respond with only the number."

    completion = client.chat.completions.create(
        model=os.getenv("EXTRACT_MODEL_ID"),
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ],
        temperature=temperature,
        max_tokens=16,
    )

    response = completion.choices[0].message.content.strip()
    try:
        return float(response)
    except (ValueError, TypeError):
        # Response is not a valid number (e.g., empty string or non-numeric text)
        return None


def get_health_range(lab_name, primary_unit, user_stats, client, temperature=0.0):
    """Use LLM to get healthy reference range for a lab test."""
    system_prompt = (
        "You are a medical laboratory assistant. "
        "Given a lab test name, its primary unit, and user stats (gender, age, weight, height, activity level), "
        "provide the healthy reference range for the test in the primary unit. "
        "Respond with only the numeric range, e.g., '3.5-5.0' or '70-110'."
    )
    user_prompt = f"Lab test: {lab_name}\nPrimary unit: {primary_unit}\nUser stats: {json.dumps(user_stats)}\nWhat is the healthy reference range for this test in the primary unit? Respond with only the numeric range."

    completion = client.chat.completions.create(
        model=os.getenv("EXTRACT_MODEL_ID"),
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ],
        temperature=temperature,
        max_tokens=32,
    )

    return completion.choices[0].message.content.strip()


def parse_range_string(range_str, primary_unit=None):
    """Parse range string like '3.5-5.0' or '70-110' into min/max dict."""
    # Handle boolean units
    if primary_unit is not None and primary_unit.lower() in ["boolean"]:
        s = range_str.strip().lower()
        if s in ["negative", "absent", "none", "no", "false", "0", "0-0"]:
            return {"min": 0, "max": 0}
        if s in ["positive", "present", "yes", "true", "1", "1-1"]:
            return {"min": 1, "max": 1}

    # Try range pattern
    match = re.match(r"^\s*([-\d\.]+)\s*[--]\s*([-\d\.]+)\s*$", range_str)
    if match:
        try:
            return {"min": float(match.group(1)), "max": float(match.group(2))}
        except Exception:
            return None

    # Try single value
    try:
        val = float(range_str.strip())
        return {"min": val, "max": val}
    except Exception:
        return None


def cmd_build_conversions(args):
    """Build conversion factors from extracted CSV data."""
    validate_env()
    client = get_openai_client()

    input_csv = args.input or "output/all.csv"
    output_json = args.output or str(TEMP_PATH)

    lab_units = defaultdict(set)

    # Collect unique lab/unit combinations from CSV
    with open(input_csv, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lab_name = row.get("lab_name_enum") or row.get("lab_name")
            lab_unit = row.get("lab_unit_enum") or row.get("unit")
            if lab_name and lab_unit:
                lab_units[lab_name].add(lab_unit)

    # Prepare conversion tasks
    labs_specs = {}
    conversion_tasks = []

    for lab_name, units in lab_units.items():
        units = list(units)
        primary_unit = units[0]
        alternatives = []
        for unit in units[1:]:
            conversion_tasks.append((lab_name, unit, primary_unit))
            alternatives.append({"unit": unit, "factor": None})
        labs_specs[lab_name] = {
            "primary_unit": primary_unit,
            "alternatives": alternatives,
        }

    # Execute conversion tasks in parallel
    def task_fn(args):
        lab_name, unit, primary_unit = args
        factor = get_conversion_factor(lab_name, unit, primary_unit, client)
        print(f"  {lab_name}: {unit} → {primary_unit} = {factor}")
        return (lab_name, unit, primary_unit, factor)

    max_workers = args.workers or min(30, len(conversion_tasks))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task_fn, args) for args in conversion_tasks]
        for future in concurrent.futures.as_completed(futures):
            lab_name, unit, primary_unit, factor = future.result()
            for alt in labs_specs[lab_name]["alternatives"]:
                if alt["unit"] == unit:
                    alt["factor"] = factor
                    break

    # Save results
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(labs_specs, f, indent=2, ensure_ascii=False)

    print(f"✓ Generated conversion factors for {len(labs_specs)} labs → {output_json}")


def cmd_build_ranges(args):
    """Build healthy reference ranges using LLM."""
    validate_env()
    client = get_openai_client()

    # Load existing specs and user stats
    with open(LAB_SPECS_PATH, "r", encoding="utf-8") as f:
        labs_specs = json.load(f)

    user_stats_path = args.user_stats or "user_stats.json"
    with open(user_stats_path, "r", encoding="utf-8") as f:
        user_stats = json.load(f)

    # Prepare health range tasks
    health_range_tasks = [(lab_name, spec["primary_unit"]) for lab_name, spec in labs_specs.items()]

    def task_fn(args):
        lab_name, primary_unit = args
        health_range = get_health_range(lab_name, primary_unit, user_stats, client)
        parsed = parse_range_string(health_range, primary_unit)
        print(f"  {lab_name}: {health_range}")
        return (lab_name, parsed)

    max_workers = args.workers or min(10, len(health_range_tasks))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task_fn, args) for args in health_range_tasks]
        for future in concurrent.futures.as_completed(futures):
            lab_name, parsed_range = future.result()
            if "ranges" not in labs_specs[lab_name]:
                labs_specs[lab_name]["ranges"] = {}
            labs_specs[lab_name]["ranges"]["healthy"] = parsed_range

    # Save results
    output_path = args.output or str(TEMP_PATH)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labs_specs, f, indent=2, ensure_ascii=False)

    print(f"✓ Generated health ranges for {len(health_range_tasks)} labs → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Lab Specifications Manager - Manage lab_specs.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
    sort                    Sort lab_specs.json alphabetically
    fix-encoding            Fix Unicode encoding issues
    build-conversions       Build unit conversion factors from CSV data
    build-ranges            Build healthy reference ranges using LLM

Examples:
    python utils/lab_specs_manager.py sort
    python utils/lab_specs_manager.py fix-encoding --backup
    python utils/lab_specs_manager.py build-conversions --input output/all.csv
    python utils/lab_specs_manager.py build-ranges --user-stats user_stats.json
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # sort command
    subparsers.add_parser("sort", help="Sort lab_specs.json alphabetically")

    # fix-encoding command
    fix_parser = subparsers.add_parser("fix-encoding", help="Fix Unicode encoding")
    fix_parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup before modifying (default: True)",
    )
    fix_parser.add_argument(
        "--no-backup",
        action="store_true",
        dest="backup",
        help="Don't create backup",
    )

    # build-conversions command
    conv_parser = subparsers.add_parser("build-conversions", help="Build conversion factors from CSV")
    conv_parser.add_argument("--input", "-i", help="Input CSV file (default: output/all.csv)")
    conv_parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file (default: temp_lab_specs.json)",
    )
    conv_parser.add_argument("--workers", "-w", type=int, help="Number of parallel workers")

    # build-ranges command
    range_parser = subparsers.add_parser("build-ranges", help="Build healthy ranges using LLM")
    range_parser.add_argument("--user-stats", "-u", help="User stats JSON file")
    range_parser.add_argument("--output", "-o", help="Output JSON file")
    range_parser.add_argument("--workers", "-w", type=int, help="Number of parallel workers")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    commands = {
        "sort": cmd_sort,
        "fix-encoding": cmd_fix_encoding,
        "build-conversions": cmd_build_conversions,
        "build-ranges": cmd_build_ranges,
    }

    try:
        commands[args.command](args)
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
