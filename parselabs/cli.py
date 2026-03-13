"""CLI entry points for parselabs."""

import argparse


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


def main() -> None:
    """Run the extraction CLI."""

    from parselabs.pipeline import main as _main

    _main()


def viewer() -> None:
    """Run the review viewer CLI."""

    try:
        from parselabs.ui import launch_app
    except ImportError as exc:
        _handle_ui_import_error(exc)

    context = _load_ui_context(_parse_ui_args())
    launch_app(context, default_tab="results")


def review_documents() -> None:
    """Run the processed document reviewer CLI."""

    try:
        from parselabs.ui import launch_app
    except ImportError as exc:
        _handle_ui_import_error(exc)

    context = _load_ui_context(_parse_ui_args())
    launch_app(context, default_tab="review")


def admin() -> None:
    """Run the unified admin CLI."""

    from parselabs.admin_commands import main as _main

    raise SystemExit(_main())


def _parse_ui_args() -> argparse.Namespace:
    """Parse the shared profile-selection arguments for UI commands."""

    from parselabs.profiles import add_profile_arguments

    parser = add_profile_arguments(
        argparse.ArgumentParser(),
        profile_help="Profile name (defaults to the first available profile)",
    )
    return parser.parse_args()


def _load_ui_context(args: argparse.Namespace):
    """Resolve the profile for the combined UI commands."""

    from parselabs.profiles import ProfileConfig, load_ui_context

    if args.list_profiles:
        profiles = ProfileConfig.list_profiles()
        if profiles:
            for name in profiles:
                print(name)
        else:
            print(f"No profiles found. Create profile files in {ProfileConfig.get_profiles_dir()}.")
        raise SystemExit(0)

    return load_ui_context(args.profile)
