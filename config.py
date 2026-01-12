"""Configuration management for lab parser."""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import date

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


@dataclass
class Demographics:
    """User demographic information for range calculation."""
    gender: Optional[str] = None  # "male", "female", "other"
    date_of_birth: Optional[str] = None  # ISO format: "YYYY-MM-DD"
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None

    @property
    def age(self) -> Optional[int]:
        """Calculate current age from date_of_birth."""
        if not self.date_of_birth:
            return None
        try:
            dob = date.fromisoformat(self.date_of_birth)
            today = date.today()
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except ValueError:
            return None


@dataclass
class ProfileConfig:
    """Configuration for a user profile."""
    name: str
    demographics: Demographics
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    input_file_regex: Optional[str] = None

    @classmethod
    def from_file(cls, profile_path: Path, env_config: Optional['ExtractionConfig'] = None) -> 'ProfileConfig':
        """Load profile from JSON file, inheriting from env_config where needed."""
        with open(profile_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Parse demographics
        demo_data = data.get('demographics', {})
        demographics = Demographics(
            gender=demo_data.get('gender'),
            date_of_birth=demo_data.get('date_of_birth'),
            height_cm=demo_data.get('height_cm'),
            weight_kg=demo_data.get('weight_kg'),
        )

        # Parse paths, with inheritance from env_config
        paths = data.get('paths', {})
        inherit = data.get('settings', {}).get('inherit_from_env', True)

        input_path = None
        if paths.get('input_path'):
            input_path = Path(paths['input_path'])
        elif inherit and env_config:
            input_path = env_config.input_path

        output_path = None
        if paths.get('output_path'):
            output_path = Path(paths['output_path'])
        elif inherit and env_config:
            output_path = env_config.output_path

        input_file_regex = paths.get('input_file_regex')
        if not input_file_regex and inherit and env_config:
            input_file_regex = env_config.input_file_regex

        return cls(
            name=data.get('name', profile_path.stem),
            demographics=demographics,
            input_path=input_path,
            output_path=output_path,
            input_file_regex=input_file_regex,
        )

    @classmethod
    def list_profiles(cls, profiles_dir: Path = Path("profiles")) -> list[str]:
        """List available profile names."""
        if not profiles_dir.exists():
            return []
        profiles = []
        for f in profiles_dir.glob("*.json"):
            if not f.name.startswith("_"):  # Skip templates like _template.json
                profiles.append(f.stem)
        return sorted(profiles)


class LabSpecsConfig:
    """Central configuration for lab specifications.

    Loads lab_specs.json once and provides multiple views to eliminate redundant file I/O.
    Supports demographic-aware healthy ranges embedded in the ranges section.
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
        """Get default healthy range (min, max) for a lab.

        Supports both new format (default: [min, max]) and legacy format (healthy: {min, max}).
        """
        if lab_name not in self._specs:
            return (None, None)

        ranges = self._specs[lab_name].get('ranges', {})

        # New format: "default": [min, max]
        default = ranges.get('default')
        if isinstance(default, list) and len(default) >= 2:
            return (default[0], default[1])

        # Legacy format: "healthy": {"min": X, "max": Y}
        healthy = ranges.get('healthy')
        if healthy:
            return (healthy.get('min'), healthy.get('max'))

        return (None, None)

    def get_healthy_range_for_demographics(
        self,
        lab_name: str,
        demographics: Optional['Demographics'] = None
    ) -> tuple[Optional[float], Optional[float]]:
        """Get healthy range with demographic-aware overrides.

        Resolution priority (first match wins):
        1. {gender}:{age_range} - gender + age match (e.g., "male:65+", "female:18-50")
        2. {gender} - gender match (e.g., "male", "female")
        3. default - applies to all
        4. healthy (legacy format) - fallback
        """
        if lab_name not in self._specs:
            return (None, None)

        ranges = self._specs[lab_name].get('ranges', {})
        gender = demographics.gender if demographics else None
        age = demographics.age if demographics else None

        # 1. Try gender:age_range (e.g., "male:18-64" or "male:65+")
        if gender and age is not None:
            for key, value in ranges.items():
                if key.startswith(f"{gender}:"):
                    age_part = key.split(":")[1]
                    try:
                        if "-" in age_part:
                            lo, hi = map(int, age_part.split("-"))
                            if lo <= age <= hi:
                                return (value[0], value[1]) if isinstance(value, list) else (None, None)
                        elif age_part.endswith("+"):
                            lo = int(age_part[:-1])
                            if age >= lo:
                                return (value[0], value[1]) if isinstance(value, list) else (None, None)
                    except (ValueError, IndexError):
                        continue

        # 2. Try gender (e.g., "male")
        if gender and gender in ranges:
            value = ranges[gender]
            if isinstance(value, list) and len(value) >= 2:
                return (value[0], value[1])

        # 3. Try default / healthy (fallback to get_healthy_range which handles both formats)
        return self.get_healthy_range(lab_name)

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
