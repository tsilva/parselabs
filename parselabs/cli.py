"""CLI entry points for parselabs."""


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
        from parselabs.app import launch_app
        from parselabs.runtime import RuntimeContext
    except ImportError as exc:
        _handle_ui_import_error(exc)

    context = _load_ui_context()
    launch_app(context, default_tab="results")


def review_documents() -> None:
    """Run the processed document reviewer CLI."""

    try:
        from parselabs.app import launch_app
        from parselabs.runtime import RuntimeContext
    except ImportError as exc:
        _handle_ui_import_error(exc)

    context = _load_ui_context()
    launch_app(context, default_tab="review")


def admin() -> None:
    """Run the unified admin CLI."""

    from parselabs.admin import main as _main

    raise SystemExit(_main())


def _load_ui_context():
    """Resolve the profile for the combined UI commands."""

    import argparse

    from parselabs.config import ProfileConfig
    from parselabs.paths import get_profiles_dir
    from parselabs.runtime import RuntimeContext

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", "-p", type=str)
    parser.add_argument("--list-profiles", action="store_true")
    args = parser.parse_args()

    if args.list_profiles:
        profiles = ProfileConfig.list_profiles()
        if profiles:
            for name in profiles:
                print(name)
        else:
            print(f"No profiles found. Create profile files in {get_profiles_dir()}.")
        raise SystemExit(0)

    profile_name = args.profile
    if not profile_name:
        available = [name for name in ProfileConfig.list_profiles() if not name.startswith("_")]
        if not available:
            raise SystemExit(f"No profiles found. Create profile files in {get_profiles_dir()}.")
        profile_name = available[0]

    return RuntimeContext.from_profile(
        profile_name,
        need_input=False,
        need_output=True,
        need_api=False,
        setup_logs=False,
    )
