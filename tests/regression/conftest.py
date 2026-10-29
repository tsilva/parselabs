"""Local golden-suite selection; patient-specific selectors live outside Git."""


def pytest_addoption(parser):
    parser.addoption(
        "--golden-suite", choices=("full", "quick"), default="full",
        help="Use the complete private corpus or its explicitly curated local quick suite.",
    )
