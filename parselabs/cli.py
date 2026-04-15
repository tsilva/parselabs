"""CLI entry points for parselabs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence


def _handle_ui_import_error(exc: ImportError) -> None:
    """Raise a concise remediation message for broken UI dependencies."""

    message = str(exc)

    # Guard: macOS sometimes blocks a broken or quarantined orjson wheel used by Gradio.
    if "orjson" in message:
        raise SystemExit(
            "UI dependencies failed to import because orjson is broken in the current venv.\n"
            "Run: uv pip install --force-reinstall --no-cache-dir orjson"
        ) from exc

    # Fallback: Surface a generic remediation path for any other UI import error.
    raise SystemExit(f"UI dependencies failed to import: {message}") from exc


def main(argv: Sequence[str] | None = None) -> None:
    """Run the unified parselabs CLI."""

    args = _coerce_argv(argv)

    if not args:
        _run_extract([])
        return

    command, *rest = args

    if command in {"-h", "--help", "help"}:
        print(_help_text())
        raise SystemExit(0)

    if command == "extract":
        _run_extract(rest, program_name="parselabs extract")
        return

    if command == "review":
        _run_review(rest, program_name="parselabs review")
        return

    if command == "admin":
        _run_admin(rest)
        return

    if command in {"viewer", "review-docs"}:
        raise SystemExit(f"Unsupported command '{command}'. Use 'parselabs review --tab results' or '--tab review'.")

    _run_extract(args)


def _run_extract(argv: Sequence[str], *, program_name: str = "parselabs") -> None:
    """Run the extraction pipeline with a temporary argv context."""

    from parselabs.pipeline import main as _main

    _run_with_argv(_main, argv, program_name)


def _run_review(argv: Sequence[str], *, program_name: str, default_tab: str = "results") -> None:
    """Run the combined review UI with tab-aware argument parsing."""

    try:
        from parselabs.ui import launch_app
    except ImportError as exc:
        _handle_ui_import_error(exc)

    args = _parse_review_args(argv, program_name=program_name, default_tab=default_tab)
    context = _load_ui_context(args)
    launch_app(context, default_tab=args.tab)


def _run_admin(argv: Sequence[str]) -> None:
    """Run the admin CLI with forwarded arguments."""

    from parselabs.admin_commands import main as _main

    raise SystemExit(_main(list(argv)))


def _run_with_argv(callback: Callable[[], None], argv: Sequence[str], program_name: str) -> None:
    """Call a CLI entry point after swapping in the desired argv."""

    old_argv = sys.argv
    sys.argv = [program_name, *argv]

    try:
        callback()
    finally:
        sys.argv = old_argv


def _parse_review_args(argv: Sequence[str], *, program_name: str, default_tab: str) -> argparse.Namespace:
    """Parse the review CLI arguments."""

    from parselabs.runtime import add_profile_arguments

    parser = add_profile_arguments(
        argparse.ArgumentParser(
            prog=program_name,
            description="Launch the combined Parselabs review UI.",
        ),
        profile_help="Profile name",
    )
    parser.add_argument(
        "--tab",
        choices=["results", "review"],
        default=default_tab,
        help="Default tab to open in the combined UI",
    )
    args = parser.parse_args(list(argv))

    # Guard: Review commands must target one explicit profile unless the user is listing choices.
    if not args.list_profiles and not args.profile:
        parser.error("--profile is required for review UI commands. Use --list-profiles to see available profiles.")

    return args


def _load_ui_context(args: argparse.Namespace):
    """Resolve the profile for the combined UI commands."""

    from parselabs.config import ProfileConfig
    from parselabs.runtime import load_ui_context

    if args.list_profiles:
        profiles = ProfileConfig.list_profiles()
        if profiles:
            for name in profiles:
                print(name)
        else:
            print(f"No profiles found. Create profile files in {ProfileConfig.get_profiles_dir()}.")
        raise SystemExit(0)

    return load_ui_context(args.profile)


def _coerce_argv(argv: Sequence[str] | None) -> list[str]:
    """Return a mutable argv list for top-level dispatch."""

    if argv is None:
        return list(sys.argv[1:])

    return list(argv)


def _help_text() -> str:
    """Return the top-level CLI help text."""

    return (
        "Usage: parselabs [extract-options]\n"
        "       parselabs extract [extract-options]\n"
        "       parselabs review --profile NAME [--tab {results,review}]\n"
        "       parselabs review --list-profiles\n"
        "       parselabs admin <command> [args]\n\n"
        "Commands:\n"
        "  extract   Run lab extraction across one or more configured profiles\n"
        "  review    Launch the combined review UI\n"
        "  admin     Run maintenance and migration utilities\n\n"
        "Examples:\n"
        "  parselabs --profile tsilva\n"
        "  parselabs extract --pattern \"2024-*.pdf\"\n"
        "  parselabs review --profile tsilva\n"
        "  parselabs review --profile tsilva --tab review\n"
        "  parselabs admin validate-lab-specs\n"
    )
