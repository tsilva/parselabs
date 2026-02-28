"""CLI entry point for labs-parser."""

import os
import sys
from pathlib import Path


def main():
    # Ensure working directory and sys.path include the repo root
    # so `main.py` and relative paths (prompts/, config/) resolve correctly
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from main import main as _main

    _main()
