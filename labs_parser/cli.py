"""CLI entry point for labs-parser."""

import os
from pathlib import Path


def main():
    # Ensure working directory is the repo root so relative paths resolve
    os.chdir(Path(__file__).parent.parent)
    from main import main as _main

    _main()
