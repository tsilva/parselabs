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

    from main import main as _main

    _main()


def viewer() -> None:
    """Run the review viewer CLI."""

    try:
        from viewer import main as _main
    except ImportError as exc:
        _handle_ui_import_error(exc)

    _main()


def review_documents() -> None:
    """Run the processed document reviewer CLI."""

    try:
        from review_documents import main as _main
    except ImportError as exc:
        _handle_ui_import_error(exc)

    _main()
