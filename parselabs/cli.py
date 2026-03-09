"""CLI entry points for parselabs."""


def main() -> None:
    """Run the extraction CLI."""

    from main import main as _main

    _main()


def viewer() -> None:
    """Run the review viewer CLI."""

    from viewer import main as _main

    _main()
