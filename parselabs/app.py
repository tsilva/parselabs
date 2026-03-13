"""Compatibility wrapper for the package-native combined UI app."""

from parselabs import ui as _ui_module

globals().update({name: getattr(_ui_module, name) for name in dir(_ui_module) if not name.startswith("__")})
