"""Compatibility wrapper for the package-native admin command surface."""

from parselabs import admin_commands as _admin_commands

globals().update({name: getattr(_admin_commands, name) for name in dir(_admin_commands) if not name.startswith("__")})
