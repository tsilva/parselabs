"""CLI entry point for lab-spec schema validation."""

from __future__ import annotations

import argparse
import logging

from parselabs.lab_specs_validation import LabSpecsValidator


def main(argv: list[str] | None = None) -> int:
    """Validate the bundled lab specifications."""

    parser = argparse.ArgumentParser(description="Validate the bundled lab_specs.json schema and invariants")
    parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    validator = LabSpecsValidator()
    is_valid = validator.validate()
    validator.print_report()
    return 0 if is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
