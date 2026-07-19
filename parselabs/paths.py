"""Runtime path helpers for package resources and user configuration."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "parselabs"

_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent
_PACKAGED_RESOURCES_ROOT = _PACKAGE_ROOT / "resources"


def get_project_root() -> Path:
    """Return the source tree root for this editable install."""

    return _PROJECT_ROOT


def get_bundled_resources_dir() -> Path:
    """Return the active immutable resource directory."""

    source_config = _PROJECT_ROOT / "config" / "lab_specs.json"
    if source_config.exists():
        return _PROJECT_ROOT
    return _PACKAGED_RESOURCES_ROOT


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

    return get_bundled_resources_dir() / "prompts"


def get_lab_specs_path() -> Path:
    """Return the bundled lab_specs.json path."""

    return get_bundled_resources_dir() / "config" / "lab_specs.json"


def get_source_lab_specs_path() -> Path:
    """Return the repository-owned lab specs or reject non-editable installs."""

    source_path = _PROJECT_ROOT / "config" / "lab_specs.json"
    if not source_path.is_file():
        raise RuntimeError("This lab-spec maintenance command requires an editable Parselabs source checkout.")
    return source_path


def get_bundled_cache_dir() -> Path:
    """Return the read-only standardization cache defaults."""

    return get_bundled_resources_dir() / "config" / "cache"


def get_cache_dir() -> Path:
    """Return the writable user standardization cache directory."""

    return get_user_config_dir() / "cache"


def get_static_dir() -> Path:
    """Return the directory containing viewer static assets."""

    return get_bundled_resources_dir() / "static"
