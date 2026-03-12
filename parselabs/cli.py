"""CLI entry points for parselabs."""


def main() -> None:
    """Run the extraction CLI."""

    from main import main as _main

    _main()


def viewer() -> None:
    """Run the review viewer CLI."""

    from viewer import main as _main

    _main()


def review_documents() -> None:
    """Run the processed document reviewer CLI."""

    from review_documents import main as _main

    _main()
