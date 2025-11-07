#!/usr/bin/env python3
"""Test runner for single PDF processing"""

import os
import sys

# Override environment variables for test
os.environ["INPUT_PATH"] = "./test"
os.environ["INPUT_FILE_REGEX"] = "test.pdf"
os.environ["OUTPUT_PATH"] = "./test/outputs"
os.environ["SELF_CONSISTENCY_MODEL_ID"] = "google/gemini-2.5-flash"
os.environ["EXTRACT_MODEL_ID"] = "google/gemini-2.5-flash"
os.environ["N_EXTRACTIONS"] = "3"
os.environ["MAX_WORKERS"] = "1"

# Import and run main
from main import main

if __name__ == "__main__":
    print("Running test with simplified pipeline (no transcription step)...")
    print(f"Input: {os.environ['INPUT_PATH']}/{os.environ['INPUT_FILE_REGEX']}")
    print(f"Output: {os.environ['OUTPUT_PATH']}")
    print(f"Extract model: {os.environ['EXTRACT_MODEL_ID']}")
    print(f"N_EXTRACTIONS: {os.environ['N_EXTRACTIONS']}")
    print("-" * 60)
    main()
