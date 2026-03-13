"""Legacy compatibility wrapper for the package-native reviewer CLI."""

from parselabs import cli as _cli


if __name__ == "__main__":
    _cli.review_documents()
