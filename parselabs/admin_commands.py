"""Unified admin CLI for maintenance and migration utilities."""

from __future__ import annotations

import importlib
import sys

UTILITY_MODULES = {
    "validate-lab-specs": "utils.validate_lab_specs_schema",
    "analyze-unknowns": "utils.analyze_unknowns",
    "update-standardization-caches": "utils.update_standardization_caches",
    "regression": "utils.regression_cases",
    "review-artifacts": "utils.review_artifacts",
    "migrate-output-dirs": "utils.migrate_output_dirs",
    "migrate-raw-columns": "utils.migrate_raw_columns",
    "cleanup-removed-fields": "utils.cleanup_removed_fields",
    "lab-specs": "utils.lab_specs_manager",
}


def main(argv: list[str] | None = None) -> int:
    """Run the unified admin CLI."""

    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in {"-h", "--help"}:
        print(_help_text())
        return 0

    command, *rest = args
    return run_legacy_utility(command, rest)


def run_legacy_utility(command: str, argv: list[str] | None = None) -> int:
    """Dispatch one admin subcommand to its legacy implementation module."""

    module_name = UTILITY_MODULES.get(command)

    if module_name is None:
        print(f"Unknown admin command: {command}\n")
        print(_help_text())
        return 1

    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise RuntimeError(f"Utility module '{module_name}' does not expose main().")

    old_argv = sys.argv
    sys.argv = [f"parselabs admin {command}", *(argv or [])]

    try:
        result = module.main()
    finally:
        sys.argv = old_argv

    if result is None:
        return 0
    return int(result)


def _help_text() -> str:
    """Return the top-level admin help text."""

    commands = "\n".join(f"  {name}" for name in sorted(UTILITY_MODULES))
    return (
        "Usage: parselabs admin <command> [args]\n\n"
        "Commands:\n"
        f"{commands}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
