"""Legacy compatibility wrapper for the package-native pipeline module."""

from parselabs import pipeline as _pipeline

globals().update({name: getattr(_pipeline, name) for name in dir(_pipeline) if not name.startswith("__")})


if __name__ == "__main__":
    _pipeline.main()
