"""Custom exceptions for the labs parser pipeline."""


class ConfigurationError(Exception):
    """Raised when profile/environment configuration is invalid."""

    pass


class PipelineError(Exception):
    """Raised when the extraction pipeline fails at runtime."""

    pass
