"""Legacy compatibility wrapper for the package-native viewer CLI."""

from parselabs import cli as _cli


if __name__ == "__main__":
    _cli.viewer()
