"""Configuration management for lab parser."""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

UNKNOWN_VALUE = "$UNKNOWN$"


@dataclass
class ExtractionConfig:
    """Configuration for extraction pipeline."""
    input_path: Path
    input_file_regex: str
    output_path: Path
    self_consistency_model_id: str
    extract_model_id: str
    n_extractions: int
    openrouter_api_key: str
    max_workers: int

    # Verification settings
    enable_verification: bool = True
    verification_model_id: Optional[str] = None  # Auto-selected if None
    arbitration_model_id: Optional[str] = None   # Auto-selected if None
    enable_completeness_check: bool = True
    enable_character_verification: bool = True

    @classmethod
    def from_env(cls) -> 'ExtractionConfig':
        """Load configuration from environment variables."""
        input_path = os.getenv("INPUT_PATH")
        input_file_regex = os.getenv("INPUT_FILE_REGEX")
        output_path = os.getenv("OUTPUT_PATH")
        self_consistency_model_id = os.getenv("SELF_CONSISTENCY_MODEL_ID")
        extract_model_id = os.getenv("EXTRACT_MODEL_ID")
        n_extractions = int(os.getenv("N_EXTRACTIONS", 1))
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        max_workers_str = os.getenv("MAX_WORKERS", "1")

        # Verification settings
        enable_verification = os.getenv("ENABLE_VERIFICATION", "true").lower() == "true"
        verification_model_id = os.getenv("VERIFICATION_MODEL_ID") or None
        arbitration_model_id = os.getenv("ARBITRATION_MODEL_ID") or None
        enable_completeness_check = os.getenv("ENABLE_COMPLETENESS_CHECK", "true").lower() == "true"
        enable_character_verification = os.getenv("ENABLE_CHARACTER_VERIFICATION", "true").lower() == "true"

        # Validate required fields
        if not self_consistency_model_id:
            raise ValueError("SELF_CONSISTENCY_MODEL_ID not set")
        if not extract_model_id:
            raise ValueError("EXTRACT_MODEL_ID not set")
        if not input_path or not Path(input_path).exists():
            raise ValueError(f"INPUT_PATH ('{input_path}') not set or does not exist.")
        if not input_file_regex:
            raise ValueError("INPUT_FILE_REGEX not set")
        if not output_path:
            raise ValueError("OUTPUT_PATH not set")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        # Parse max_workers
        try:
            max_workers = max(1, int(max_workers_str))
        except ValueError:
            logger.warning(f"MAX_WORKERS ('{max_workers_str}') is not valid. Defaulting to 1.")
            max_workers = 1

        output_path_obj = Path(output_path)
        output_path_obj.mkdir(parents=True, exist_ok=True)

        return cls(
            input_path=Path(input_path),
            input_file_regex=input_file_regex,
            output_path=output_path_obj,
            self_consistency_model_id=self_consistency_model_id,
            extract_model_id=extract_model_id,
            n_extractions=n_extractions,
            openrouter_api_key=openrouter_api_key,
            max_workers=max_workers,
            enable_verification=enable_verification,
            verification_model_id=verification_model_id,
            arbitration_model_id=arbitration_model_id,
            enable_completeness_check=enable_completeness_check,
            enable_character_verification=enable_character_verification,
        )


class LabSpecsConfig:
    """Central configuration for lab specifications.

    Loads lab_specs.json once and provides multiple views to eliminate redundant file I/O.
    """

    def __init__(self, config_path: Path = Path("config/lab_specs.json")):
        """Load lab specs configuration."""
        self.config_path = config_path
        self._specs = {}
        self._standardized_names = []
        self._standardized_units = []
        self._lab_type_map = {}

        if not config_path.exists():
            logger.warning(f"lab_specs.json not found at {config_path}")
            return

        with open(config_path, 'r', encoding='utf-8') as f:
            self._specs = json.load(f)

        # Pre-compute all views
        self._standardized_names = sorted(self._specs.keys())

        # Collect unique units from primary_unit and alternatives
        all_units = set()
        for lab_name, spec in self._specs.items():
            primary = spec.get('primary_unit')
            if primary:
                all_units.add(primary)

            for alt in spec.get('alternatives', []):
                unit = alt.get('unit')
                if unit:
                    all_units.add(unit)

        self._standardized_units = sorted(all_units)

        # Build lab_type mapping
        self._lab_type_map = {
            lab_name: spec.get('lab_type', 'blood')
            for lab_name, spec in self._specs.items()
        }

        logger.info(f"Loaded {len(self._standardized_names)} lab specs, "
                    f"{len(self._standardized_units)} units")

    @property
    def exists(self) -> bool:
        """Check if config was successfully loaded."""
        return bool(self._specs)

    @property
    def standardized_names(self) -> list[str]:
        """Get list of valid standardized lab names."""
        return self._standardized_names

    @property
    def standardized_units(self) -> list[str]:
        """Get list of valid standardized units."""
        return self._standardized_units

    @property
    def specs(self) -> dict:
        """Get raw specs dictionary."""
        return self._specs

    def get_lab_type(self, lab_name: str) -> str:
        """Get lab type for a given lab name."""
        return self._lab_type_map.get(lab_name, "blood")

    def get_primary_unit(self, lab_name: str) -> Optional[str]:
        """Get primary unit for a lab."""
        if lab_name not in self._specs:
            return None
        return self._specs[lab_name].get('primary_unit')

    def get_conversion_factor(self, lab_name: str, from_unit: str) -> Optional[float]:
        """Get conversion factor from given unit to primary unit."""
        if lab_name not in self._specs:
            return None

        spec = self._specs[lab_name]
        primary_unit = spec.get('primary_unit')

        # Already in primary unit
        if from_unit == primary_unit:
            return 1.0

        # Find conversion factor in alternatives
        for alt in spec.get('alternatives', []):
            if alt.get('unit') == from_unit:
                return alt.get('factor')

        return None

    def get_healthy_range(self, lab_name: str) -> tuple[Optional[float], Optional[float]]:
        """Get healthy range (min, max) for a lab."""
        if lab_name not in self._specs:
            return (None, None)

        healthy = self._specs[lab_name].get('ranges', {}).get('healthy')
        if healthy:
            return (healthy.get('min'), healthy.get('max'))
        return (None, None)

    def get_percentage_variant(self, lab_name: str) -> Optional[str]:
        """Get the (%) variant of a lab name if it exists.

        Args:
            lab_name: The standardized lab name (e.g., "Blood - Basophils")

        Returns:
            The (%) variant if it exists (e.g., "Blood - Basophils (%)"), else None
        """
        if lab_name.endswith("(%)"):
            return None  # Already a percentage variant

        percentage_variant = f"{lab_name} (%)"
        if percentage_variant in self._specs:
            return percentage_variant
        return None
