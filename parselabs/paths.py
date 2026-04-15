"""Runtime path helpers for package resources and user configuration."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "parselabs"

_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent


def get_project_root() -> Path:
    """Return the source tree root for this editable install."""

    return _PROJECT_ROOT


def get_user_config_dir() -> Path:
    """Return the directory for user-managed configuration files."""

    return Path.home() / ".config" / APP_NAME


def get_profiles_dir() -> Path:
    """Return the directory where profile YAML/JSON files are stored."""

    return get_user_config_dir() / "profiles"


def get_env_file() -> Path:
    """Return the user-managed dotenv path for shared runtime settings."""

    return get_user_config_dir() / ".env"


def get_prompts_dir() -> Path:
    """Return the directory containing prompt templates."""

    return get_project_root() / "prompts"


def get_lab_specs_path() -> Path:
    """Return the bundled lab_specs.json path."""

    return get_project_root() / "config" / "lab_specs.json"


def get_cache_dir() -> Path:
    """Return the persistent standardization cache directory."""

    return get_project_root() / "config" / "cache"


def get_static_dir() -> Path:
    """Return the directory containing viewer static assets."""

    return get_project_root() / "static"
