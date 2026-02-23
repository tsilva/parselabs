"""Configuration management for lab parser."""
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

UNKNOWN_VALUE = "$UNKNOWN$"


@dataclass
class ExtractionConfig:
    """Configuration for extraction pipeline.

    Simplified to require only essential inputs with smart defaults.
    """
    input_path: Path
    output_path: Path
    openrouter_api_key: str

    # Model settings (required - set via .env)
    extract_model_id: str
    self_consistency_model_id: str

    # Processing settings
    input_file_regex: str = "*.pdf"
    n_extractions: int = 1
    max_workers: int = field(default_factory=lambda: os.cpu_count() or 1)


@dataclass
class ProfileConfig:
    """Profile configuration with optional demographics.

    Paths + optional setting overrides + demographics for personalized ranges.
    Supports both YAML and JSON formats.
    """
    name: str
    input_path: Path | None = None
    output_path: Path | None = None
    input_file_regex: str | None = None

    # Optional overrides
    workers: int | None = None

    # Demographics for personalized healthy ranges
    demographics: Demographics | None = None

    @classmethod
    def from_file(cls, profile_path: Path) -> 'ProfileConfig':
        """Load profile from YAML or JSON file."""
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")

        content = profile_path.read_text(encoding='utf-8')

        # Parse based on extension
        if profile_path.suffix in ('.yaml', '.yml'):
            import yaml
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)

        # Extract paths
        paths = data.get('paths', {})
        input_path_str = paths.get('input_path') or data.get('input_path')
        output_path_str = paths.get('output_path') or data.get('output_path')
        input_file_regex = paths.get('input_file_regex') or data.get('input_file_regex')

        # Extract optional overrides
        workers = data.get('workers')

        # Extract demographics (for personalized healthy ranges)
        demographics = None
        demo_data = data.get('demographics', {})
        if demo_data:
            demographics = Demographics(
                gender=demo_data.get('gender'),
                date_of_birth=demo_data.get('date_of_birth'),
                height_cm=demo_data.get('height_cm'),
                weight_kg=demo_data.get('weight_kg'),
            )

        return cls(
            name=data.get('name', profile_path.stem),
            input_path=Path(input_path_str) if input_path_str else None,
            output_path=Path(output_path_str) if output_path_str else None,
            input_file_regex=input_file_regex,
            workers=workers,
            demographics=demographics,
        )

    @classmethod
    def list_profiles(cls, profiles_dir: Path = Path("profiles")) -> list[str]:
        """List available profile names."""
        if not profiles_dir.exists():
            return []
        profiles = []
        for ext in ('*.json', '*.yaml', '*.yml'):
            for f in profiles_dir.glob(ext):
                if not f.name.startswith('_'):  # Skip templates
                    profiles.append(f.stem)
        return sorted(set(profiles))


# Keep Demographics class for future use in review tool
@dataclass
class Demographics:
    """User demographic information for range calculation.

    Note: This is kept for the review tool, not used in extraction.
    """
    gender: str | None = None  # "male", "female", "other"
    date_of_birth: str | None = None  # ISO format: "YYYY-MM-DD"
    height_cm: float | None = None
    weight_kg: float | None = None

    @property
    def age(self) -> int | None:
        """Calculate current age from date_of_birth."""
        if not self.date_of_birth:
            return None
        try:
            from datetime import date
            dob = date.fromisoformat(self.date_of_birth)
            today = date.today()
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except ValueError:
            return None


class LabSpecsConfig:
    """Central configuration for lab specifications.

    Loads lab_specs.json once and provides multiple views.
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

        # Pre-compute all views (filter out meta keys starting with _)
        self._standardized_names = sorted(
            key for key in self._specs.keys() if not key.startswith('_')
        )

        # Collect unique units from primary_unit and alternatives
        all_units = set()
        for lab_name, spec in self._specs.items():
            if lab_name.startswith('_'):  # Skip meta keys
                continue
            if not isinstance(spec, dict):  # Skip non-dict entries
                continue
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
            if not lab_name.startswith('_') and isinstance(spec, dict)
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

    def get_primary_unit(self, lab_name: str) -> str | None:
        """Get primary unit for a lab."""
        if lab_name not in self._specs:
            return None
        return self._specs[lab_name].get('primary_unit')

    def get_conversion_factor(self, lab_name: str, from_unit: str) -> float | None:
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

    def get_healthy_range_for_demographics(
        self,
        lab_name: str,
        gender: str | None = None,
        age: int | None = None
    ) -> tuple[float | None, float | None]:
        """Get healthy range for a lab, considering demographics.

        Selection priority:
        1. Gender + age-specific (e.g., "male:0-17", "female:65+")
        2. Gender-specific (e.g., "male", "female")
        3. Default range
        4. (None, None) if no range found

        Args:
            lab_name: Standardized lab name
            gender: "male" or "female" (optional)
            age: Age in years (optional)

        Returns:
            Tuple of (min, max) or (None, None)
        """
        if lab_name not in self._specs:
            return (None, None)

        ranges = self._specs[lab_name].get('ranges', {})
        if not ranges:
            return (None, None)

        # Try gender + age-specific ranges first
        if gender and age is not None:
            for key, value in ranges.items():
                if key.startswith(f"{gender}:"):
                    age_part = key.split(":", 1)[1]
                    if self._age_matches_range(age, age_part):
                        if isinstance(value, list) and len(value) >= 2:
                            return (value[0], value[1])

        # Try gender-specific range
        if gender:
            gender_range = ranges.get(gender)
            if isinstance(gender_range, list) and len(gender_range) >= 2:
                return (gender_range[0], gender_range[1])

        # Fall back to default
        default = ranges.get('default')
        if isinstance(default, list) and len(default) >= 2:
            return (default[0], default[1])

        return (None, None)

    def _age_matches_range(self, age: int, age_spec: str) -> bool:
        """Check if age matches an age specification.

        Supports formats:
        - "0-17" (inclusive range)
        - "65+" (threshold and above)
        - "18-64" (inclusive range)
        """
        if '+' in age_spec:
            threshold = int(age_spec.replace('+', ''))
            return age >= threshold
        if '-' in age_spec:
            parts = age_spec.split('-')
            if len(parts) == 2:
                try:
                    return int(parts[0]) <= age <= int(parts[1])
                except ValueError:
                    return False
        return False

    def get_percentage_variant(self, lab_name: str) -> str | None:
        """Get the (%) variant of a lab name if it exists."""
        if lab_name.endswith("(%)"):
            return None  # Already a percentage variant

        percentage_variant = f"{lab_name} (%)"
        if percentage_variant in self._specs:
            return percentage_variant
        return None

    def get_non_percentage_variant(self, lab_name: str) -> str | None:
        """Get the non-(%) variant of a lab name if it exists.

        For example: "Blood - Neutrophils (%)" -> "Blood - Neutrophils"
        """
        if not lab_name.endswith("(%)"):
            return None  # Not a percentage variant

        # Remove " (%)" suffix
        non_percentage_variant = lab_name[:-4]  # Remove " (%)"
        if non_percentage_variant in self._specs:
            return non_percentage_variant
        return None

    def get_loinc_code(self, lab_name: str) -> str | None:
        """Get LOINC code for a lab test if available.

        LOINC (Logical Observation Identifiers Names and Codes) is a
        universal standard for identifying laboratory observations.

        Args:
            lab_name: Standardized lab name

        Returns:
            LOINC code string (e.g., "1558-6") or None if not configured

        Example:
            >>> lab_specs.get_loinc_code("Blood - Glucose (Fasting)")
            "1558-6"
            >>> lab_specs.get_loinc_code("Blood - Unknown Test")
            None
        """
        if lab_name not in self._specs:
            return None
        return self._specs[lab_name].get('loinc_code')
