"""Compatibility wrapper for the package-native combined UI app."""

from parselabs import ui_app as _ui_app

globals().update({name: getattr(_ui_app, name) for name in dir(_ui_app) if not name.startswith("__")})
