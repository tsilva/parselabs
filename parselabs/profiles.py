"""Shared profile and runtime helpers for CLI, UI, and utility entry points."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from openai import OpenAI

from parselabs.config import Demographics, ExtractionConfig, LabSpecsConfig, ProfileConfig
from parselabs.exceptions import ConfigurationError
from parselabs.utils import setup_logging


@dataclass(frozen=True)
class RuntimeContext:
    """Resolved runtime state for a single profile invocation."""

    profile_name: str
    profile: ProfileConfig
    extraction_config: ExtractionConfig | None
    input_path: Path | None
    output_path: Path | None
    demographics: Demographics | None
    lab_specs: LabSpecsConfig
    logger: logging.Logger
    openai_client: OpenAI | None = None

    @classmethod
    def from_profile(
        cls,
        profile_name: str,
        *,
        need_input: bool,
        need_output: bool,
        need_api: bool,
        overrides: Mapping[str, object] | None = None,
        create_output_dir: bool = False,
        setup_logs: bool = False,
        clear_logs: bool = False,
    ) -> "RuntimeContext":
        """Load and validate profile-backed runtime state."""

        profile_path = ProfileConfig.find_path(profile_name)

        # Guard: The requested profile must exist before any runtime can start.
        if not profile_path:
            raise ConfigurationError(f"Profile '{profile_name}' not found. Use --list-profiles to see available profiles.")

        profile = ProfileConfig.from_file(profile_path)
        overrides = overrides or {}
        input_path = profile.input_path
        output_path = profile.output_path

        # Guard: Extraction runs require an input path.
        if need_input and not input_path:
            raise ConfigurationError(f"Profile '{profile_name}' has no input_path defined.")

        # Guard: Review, export, and extraction flows require an output path.
        if need_output and not output_path:
            raise ConfigurationError(f"Profile '{profile_name}' has no output_path defined.")

        # Guard: API-backed flows require both the key and extraction model.
        if need_api and not profile.openrouter_api_key:
            raise ConfigurationError(f"Profile '{profile_name}' has no openrouter_api_key defined.")
        if need_api and not profile.extract_model_id and not overrides.get("model"):
            raise ConfigurationError(f"Profile '{profile_name}' has no extract_model_id defined.")

        # Guard: Input paths must exist before processing starts.
        if need_input and input_path and not input_path.exists():
            raise ConfigurationError(f"Input path does not exist: {input_path}")

        # Create extraction outputs lazily only for flows that are expected to write.
        if create_output_dir and output_path:
            output_path.mkdir(parents=True, exist_ok=True)

        # Guard: Read-only output flows still require the processed output directory to exist.
        if need_output and output_path and not create_output_dir and not output_path.exists():
            raise ConfigurationError(f"Output path does not exist: {output_path}")

        logger = logging.getLogger(__name__)

        # Configure per-profile logging only when the caller needs log files.
        if setup_logs and output_path:
            logger = setup_logging(output_path / "logs", clear_logs=clear_logs)

        extract_model_id = str(overrides.get("model") or profile.extract_model_id or "")
        input_file_regex = str(overrides.get("pattern") or profile.input_file_regex or "*.pdf")
        worker_override = overrides.get("workers")
        max_workers = int(worker_override) if worker_override is not None else int(profile.workers or (os.cpu_count() or 1))
        extraction_config: ExtractionConfig | None = None

        # Build the extraction config only for flows that actually need it.
        if need_input or need_api:
            extraction_config = ExtractionConfig(
                input_path=input_path if input_path is not None else Path(),
                output_path=output_path if output_path is not None else Path(),
                openrouter_api_key=str(profile.openrouter_api_key or ""),
                openrouter_base_url=profile.openrouter_base_url or "https://openrouter.ai/api/v1",
                extract_model_id=extract_model_id,
                input_file_regex=input_file_regex,
                max_workers=max_workers,
            )

        openai_client = None

        # Initialize the API client only for flows that use the LLM.
        if need_api and extraction_config is not None:
            openai_client = OpenAI(
                base_url=extraction_config.openrouter_base_url,
                api_key=extraction_config.openrouter_api_key,
            )

        return cls(
            profile_name=profile_name,
            profile=profile,
            extraction_config=extraction_config,
            input_path=input_path,
            output_path=output_path,
            demographics=profile.demographics,
            lab_specs=LabSpecsConfig(),
            logger=logger,
            openai_client=openai_client,
        )

    def copy_lab_specs_to_output(self) -> Path | None:
        """Copy the active lab specs file into the output directory when available."""

        # Guard: Reproducibility copies only make sense for writable output roots.
        if self.output_path is None or not self.lab_specs.exists:
            return None

        destination = self.output_path / "lab_specs.json"
        destination.write_bytes(self.lab_specs.config_path.read_bytes())
        return destination

    @classmethod
    def list_output_roots(cls) -> list[str]:
        """Return profile-backed filesystem roots that UI servers may expose."""

        output_roots: set[str] = set()

        # Read every configured profile once so the combined UI can serve assets safely.
        for profile_name in ProfileConfig.list_profiles():
            profile_path = ProfileConfig.find_path(profile_name)

            # Skip profiles that disappeared between discovery and resolution.
            if not profile_path:
                continue

            profile = ProfileConfig.from_file(profile_path)

            # Skip profiles without a processed-output root.
            if not profile.output_path:
                continue

            output_roots.add(str(profile.output_path))

            # Gradio also validates ancestor roots for served files.
            if profile.output_path.parent != profile.output_path:
                output_roots.add(str(profile.output_path.parent))

        return sorted(output_roots)


def add_profile_arguments(
    parser: argparse.ArgumentParser,
    *,
    profile_help: str,
    list_profiles_help: str = "List available profiles and exit",
) -> argparse.ArgumentParser:
    """Attach the shared profile-selection arguments to a parser."""

    parser.add_argument("--profile", "-p", type=str, help=profile_help)
    parser.add_argument("--list-profiles", action="store_true", help=list_profiles_help)
    return parser


def list_non_template_profiles() -> list[str]:
    """Return user-selectable profile names, excluding templates."""

    return [name for name in ProfileConfig.list_profiles() if not name.startswith("_")]


def resolve_profile_name(profile_name: str | None) -> str:
    """Resolve the active profile name or raise when none is available."""

    if profile_name:
        return profile_name

    available_profiles = list_non_template_profiles()

    # Guard: UI-backed flows need at least one configured profile.
    if not available_profiles:
        raise ConfigurationError(
            f"No profiles found. Create profile files in {ProfileConfig.get_profiles_dir()}."
        )

    return available_profiles[0]


def load_ui_context(profile_name: str | None) -> RuntimeContext:
    """Resolve a runtime context for the combined UI commands."""

    return RuntimeContext.from_profile(
        resolve_profile_name(profile_name),
        need_input=False,
        need_output=True,
        need_api=False,
        setup_logs=False,
    )
