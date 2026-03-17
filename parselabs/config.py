"""Configuration management for lab parser."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from parselabs.paths import get_lab_specs_path, get_profiles_dir

logger = logging.getLogger(__name__)

LEGACY_QUALITATIVE_SUFFIX = ", Qualitative"
CANONICAL_QUALITATIVE_SUFFIX = " (Qualitative)"

UNKNOWN_VALUE = "$UNKNOWN$"


@dataclass
class ExtractionConfig:
    """Configuration for extraction pipeline.

    Simplified to require only essential inputs with smart defaults.
    """

    input_path: Path
    output_path: Path
    openrouter_api_key: str
    extract_model_id: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Processing settings
    input_file_regex: str = "*.pdf"
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

    # Runtime settings
    openrouter_api_key: str | None = None
    openrouter_base_url: str | None = None
    extract_model_id: str | None = None
    workers: int | None = None

    # Demographics for personalized healthy ranges
    demographics: Demographics | None = None

    @staticmethod
    def _first_value(*values):
        """Return the first non-empty value from the provided candidates."""

        for value in values:
            if value not in (None, ""):
                return value
        return None

    @staticmethod
    def _resolve_profile_value_path(path_value: str | None, profile_path: Path) -> Path | None:
        """Resolve a profile-managed filesystem path relative to the profile file."""

        if not path_value:
            return None

        candidate = Path(path_value).expanduser()
        if candidate.is_absolute():
            return candidate
        return (profile_path.parent / candidate).resolve()

    @classmethod
    def from_file(cls, profile_path: Path) -> "ProfileConfig":
        """Load profile from YAML or JSON file."""

        profile_path = profile_path.expanduser().resolve()

        # Guard: profile file must exist
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")

        content = profile_path.read_text(encoding="utf-8")

        # Parse based on extension
        if profile_path.suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(content)
        else:
            # Fall back to JSON for all other extensions
            data = json.loads(content)

        # Extract paths
        paths = data.get("paths", {})
        input_path_str = paths.get("input_path") or data.get("input_path")
        output_path_str = paths.get("output_path") or data.get("output_path")
        processing = data.get("processing", {})
        input_file_regex = cls._first_value(
            paths.get("input_file_regex"),
            processing.get("input_file_regex"),
            data.get("input_file_regex"),
        )

        # Extract runtime settings
        openrouter = data.get("openrouter", {})
        models = data.get("models", {})
        openrouter_api_key = cls._first_value(
            openrouter.get("api_key"),
            data.get("openrouter_api_key"),
            data.get("api_key"),
        )
        openrouter_base_url = cls._first_value(
            openrouter.get("base_url"),
            data.get("openrouter_base_url"),
            data.get("base_url"),
        )
        extract_model_id = cls._first_value(
            models.get("extract_model_id"),
            data.get("extract_model_id"),
            data.get("model"),
        )

        # Extract optional overrides
        workers = cls._first_value(processing.get("workers"), data.get("workers"))

        # Extract demographics (for personalized healthy ranges)
        demographics = None
        demo_data = data.get("demographics", {})

        # Build Demographics object if demographic data is present
        if demo_data:
            demographics = Demographics(
                gender=demo_data.get("gender"),
                date_of_birth=demo_data.get("date_of_birth"),
                height_cm=demo_data.get("height_cm"),
                weight_kg=demo_data.get("weight_kg"),
            )

        return cls(
            name=data.get("name", profile_path.stem),
            input_path=cls._resolve_profile_value_path(input_path_str, profile_path),
            output_path=cls._resolve_profile_value_path(output_path_str, profile_path),
            input_file_regex=input_file_regex,
            openrouter_api_key=openrouter_api_key,
            openrouter_base_url=openrouter_base_url,
            extract_model_id=extract_model_id,
            workers=workers,
            demographics=demographics,
        )

    @classmethod
    def get_profiles_dir(cls) -> Path:
        """Return the configured profiles directory."""

        return get_profiles_dir()

    @classmethod
    def find_path(cls, name: str, profiles_dir: Path | None = None) -> Path | None:
        """Find profile file path by name, trying yaml/yml/json extensions."""

        profiles_dir = (profiles_dir or cls.get_profiles_dir()).expanduser()

        # Try each supported extension in priority order
        for ext in (".yaml", ".yml", ".json"):
            p = profiles_dir / f"{name}{ext}"
            if p.exists():
                return p

        # No matching profile file found
        return None

    @classmethod
    def list_profiles(cls, profiles_dir: Path | None = None) -> list[str]:
        """List available profile names."""

        profiles_dir = (profiles_dir or cls.get_profiles_dir()).expanduser()

        # Guard: return empty if profiles directory doesn't exist
        if not profiles_dir.exists():
            return []

        # Collect profile names from all supported formats
        profiles = []
        for ext in ("*.json", "*.yaml", "*.yml"):
            for f in profiles_dir.glob(ext):
                if not f.name.startswith("_"):  # Skip templates
                    profiles.append(f.stem)

        return sorted(set(profiles))

    @classmethod
    def iter_paths(cls, profiles_dir: Path | None = None) -> list[Path]:
        """Return resolved profile file paths in name order."""

        profiles_dir = profiles_dir or cls.get_profiles_dir()
        return [path for name in cls.list_profiles(profiles_dir) if (path := cls.find_path(name, profiles_dir))]


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

        # Guard: no date of birth available
        if not self.date_of_birth:
            return None

        try:
            from datetime import date

            dob = date.fromisoformat(self.date_of_birth)
            today = date.today()
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except ValueError:
            # Invalid date format
            return None


class LabSpecsConfig:
    """Central configuration for lab specifications.

    Loads lab_specs.json once and provides multiple views.
    """

    def __init__(self, config_path: Path | None = None):
        """Load lab specs configuration."""

        # Initialize empty state
        self.config_path = (config_path or get_lab_specs_path()).expanduser().resolve()
        self._specs = {}
        self._standardized_names = []
        self._standardized_units = []
        self._lab_type_map = {}
        self._canonical_name_map = {}
        self._lab_name_aliases = {}

        # Guard: config file must exist
        if not self.config_path.exists():
            logger.warning(f"lab_specs.json not found at {self.config_path}")
            return

        # Load raw specs from JSON
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._specs = json.load(f)

        # Pre-compute all views (filter out meta keys starting with _).
        canonical_names = set()
        for lab_name, spec in self._specs.items():
            if lab_name.startswith("_") or not isinstance(spec, dict):
                continue

            canonical_name = self._build_canonical_lab_name(lab_name, spec)
            self._canonical_name_map[lab_name] = canonical_name
            self._lab_name_aliases[lab_name] = lab_name
            self._lab_name_aliases[canonical_name] = lab_name
            canonical_names.add(canonical_name)

        self._standardized_names = sorted(canonical_names)

        # Collect unique units from primary_unit and alternatives
        all_units = set()
        for lab_name, spec in self._specs.items():
            if lab_name.startswith("_"):  # Skip meta keys
                continue
            if not isinstance(spec, dict):  # Skip non-dict entries
                continue

            primary = spec.get("primary_unit")
            if primary:
                all_units.add(primary)

            for alt in spec.get("alternatives", []):
                unit = alt.get("unit")
                if unit:
                    all_units.add(unit)

        self._standardized_units = sorted(all_units)

        # Build lab_type mapping
        self._lab_type_map = {
            self._canonical_name_map.get(lab_name, lab_name): spec.get("lab_type", "blood")
            for lab_name, spec in self._specs.items()
            if not lab_name.startswith("_") and isinstance(spec, dict)
        }

        logger.info(f"Loaded {len(self._standardized_names)} lab specs, {len(self._standardized_units)} units")

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

    @staticmethod
    def _looks_like_qualitative_boolean_name(lab_name: str, spec: dict) -> bool:
        """Return whether the lab should expose a canonical qualitative suffix."""

        if spec.get("primary_unit") != "boolean":
            return False

        if lab_name.endswith(LEGACY_QUALITATIVE_SUFFIX) or lab_name.endswith(CANONICAL_QUALITATIVE_SUFFIX):
            return True

        qualitative_urine_bases = {
            "Urine Type II - Albumin",
            "Urine Type II - Appearance",
            "Urine Type II - Bilirubin",
            "Urine Type II - Blood",
            "Urine Type II - Color",
            "Urine Type II - Glucose",
            "Urine Type II - Ketones",
            "Urine Type II - Leukocytes",
            "Urine Type II - Nitrites",
            "Urine Type II - Proteins",
            "Urine Type II - Urobilinogen",
        }
        return lab_name in qualitative_urine_bases

    def _build_canonical_lab_name(self, lab_name: str, spec: dict) -> str:
        """Return the canonical exported/display name for one configured lab."""

        if lab_name.endswith(CANONICAL_QUALITATIVE_SUFFIX):
            return lab_name

        if lab_name.endswith(LEGACY_QUALITATIVE_SUFFIX):
            return f"{lab_name.removesuffix(LEGACY_QUALITATIVE_SUFFIX)}{CANONICAL_QUALITATIVE_SUFFIX}"

        if self._looks_like_qualitative_boolean_name(lab_name, spec):
            return f"{lab_name}{CANONICAL_QUALITATIVE_SUFFIX}"

        return lab_name

    def resolve_lab_name(self, lab_name: str) -> str | None:
        """Resolve a canonical or legacy lab name to the configured spec key."""

        if not isinstance(lab_name, str) or not lab_name.strip():
            return None

        if lab_name in self._specs:
            return lab_name

        return self._lab_name_aliases.get(lab_name)

    def get_canonical_lab_name(self, lab_name: str) -> str:
        """Return the canonical exported/display name for a configured lab."""

        resolved_name = self.resolve_lab_name(lab_name)
        if resolved_name is None:
            return lab_name

        return self._canonical_name_map.get(resolved_name, resolved_name)

    def get_lab_type(self, lab_name: str) -> str:
        """Get lab type for a given lab name."""

        return self._lab_type_map.get(self.get_canonical_lab_name(lab_name), "blood")

    def get_primary_unit(self, lab_name: str) -> str | None:
        """Get primary unit for a lab."""

        resolved_name = self.resolve_lab_name(lab_name)

        # Guard: lab not in specs
        if resolved_name is None:
            return None

        return self._specs[resolved_name].get("primary_unit")

    def get_conversion_factor(self, lab_name: str, from_unit: str) -> float | None:
        """Get conversion factor from given unit to primary unit."""

        resolved_name = self.resolve_lab_name(lab_name)

        # Guard: lab not in specs
        if resolved_name is None:
            return None

        spec = self._specs[resolved_name]
        primary_unit = spec.get("primary_unit")

        # Already in primary unit
        if from_unit == primary_unit:
            return 1.0

        # Find conversion factor in alternatives
        for alt in spec.get("alternatives", []):
            if alt.get("unit") == from_unit:
                return alt.get("factor")

        # No conversion available for this unit
        return None

    def get_healthy_range_for_demographics(self, lab_name: str, gender: str | None = None, age: int | None = None) -> tuple[float | None, float | None]:
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

        resolved_name = self.resolve_lab_name(lab_name)

        # Guard: lab not in specs
        if resolved_name is None:
            return (None, None)

        ranges = self._specs[resolved_name].get("ranges", {})

        # Guard: no ranges configured
        if not ranges:
            return (None, None)

        # Try gender + age-specific ranges first
        if gender and age is not None:
            for key, value in ranges.items():
                if key.startswith(f"{gender}:"):
                    age_part = key.split(":", 1)[1]
                    if self._age_matches_range(age, age_part):
                        # Matching gender+age range found
                        if isinstance(value, list) and len(value) >= 2:
                            return (value[0], value[1])

        # Try gender-specific range
        if gender:
            gender_range = ranges.get(gender)
            if isinstance(gender_range, list) and len(gender_range) >= 2:
                return (gender_range[0], gender_range[1])

        # Fall back to default
        default = ranges.get("default")
        if isinstance(default, list) and len(default) >= 2:
            return (default[0], default[1])

        # No applicable range found
        return (None, None)

    def _age_matches_range(self, age: int, age_spec: str) -> bool:
        """Check if age matches an age specification.

        Supports formats:
        - "0-17" (inclusive range)
        - "65+" (threshold and above)
        - "18-64" (inclusive range)
        """

        # Check threshold format (e.g., "65+")
        if "+" in age_spec:
            threshold = int(age_spec.replace("+", ""))
            return age >= threshold

        # Check range format (e.g., "0-17", "18-64")
        if "-" in age_spec:
            parts = age_spec.split("-")
            if len(parts) == 2:
                try:
                    return int(parts[0]) <= age <= int(parts[1])
                except ValueError:
                    # Invalid numeric values in range
                    return False

        # Unrecognized format
        return False

    def get_percentage_variant(self, lab_name: str) -> str | None:
        """Get the (%) variant of a lab name if it exists."""

        lab_name = self.get_canonical_lab_name(lab_name)

        # Guard: already a percentage variant
        if lab_name.endswith("(%)"):
            return None

        # Check if percentage variant exists in specs
        percentage_variant = f"{lab_name} (%)"
        if self.resolve_lab_name(percentage_variant) is not None:
            return self.get_canonical_lab_name(percentage_variant)

        return None

    def get_non_percentage_variant(self, lab_name: str) -> str | None:
        """Get the non-(%) variant of a lab name if it exists.

        For example: "Blood - Neutrophils (%)" -> "Blood - Neutrophils"
        """

        lab_name = self.get_canonical_lab_name(lab_name)

        # Guard: not a percentage variant
        if not lab_name.endswith("(%)"):
            return None

        # Remove " (%)" suffix and check if non-percentage variant exists
        non_percentage_variant = lab_name[:-4]  # Remove " (%)"
        if self.resolve_lab_name(non_percentage_variant) is not None:
            return self.get_canonical_lab_name(non_percentage_variant)

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

        resolved_name = self.resolve_lab_name(lab_name)

        # Guard: lab not in specs
        if resolved_name is None:
            return None

        return self._specs[resolved_name].get("loinc_code")
