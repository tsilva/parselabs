#!/usr/bin/env python3
"""
Schema Validator for lab_specs.json

Validates:
1. JSON structure and syntax
2. Required fields for each lab entry
3. Data types and value ranges
4. LOINC code presence (with known exceptions)
5. LOINC code uniqueness (no duplicates across labs)
6. Relationship configurations
7. Lab name prefixes match lab_type
"""

import json
import sys
from pathlib import Path
from typing import Any


class LabSpecsValidator:
    """Validates lab_specs.json schema and data integrity."""

    # Labs that are allowed to not have LOINC codes (usually experimental/rare tests)
    LOINC_EXCEPTIONS = {
        "Blood - Anti-Dopamine D2 Receptor Antibody (D2R)",  # Rare/experimental test
    }

    VALID_LAB_TYPES = {"blood", "urine", "feces", "saliva", "sample"}
    LAB_NAME_PREFIXES = {
        "blood": ["Blood - "],
        "urine": [
            "Urine - ",
            "Urine Type II - ",
        ],  # Type II is a standard urine test category
        "feces": ["Feces - "],
        "saliva": ["Saliva - "],
        "sample": ["Sample - "],
    }

    def __init__(self, config_path: str = "config/lab_specs.json"):
        self.config_path = Path(config_path)
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.config: dict[str, Any] = {}

    def validate(self) -> bool:
        """Run all validation checks.

        Returns:
            True if validation passes, False otherwise
        """
        if not self._load_config():
            return False

        self._validate_structure()
        self._validate_relationships()
        self._validate_lab_entries()

        return len(self.errors) == 0

    def _load_config(self) -> bool:
        """Load and parse lab_specs.json."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            return True
        except FileNotFoundError:
            self.errors.append(f"Config file not found: {self.config_path}")
            return False
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON syntax: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Failed to load config: {e}")
            return False

    def _validate_structure(self) -> None:
        """Validate top-level structure."""
        if not isinstance(self.config, dict):
            self.errors.append("Config must be a JSON object (dict)")
            return

        # Check for _relationships key
        if "_relationships" not in self.config:
            self.warnings.append("Missing '_relationships' key (optional)")

    def _validate_relationships(self) -> None:
        """Validate _relationships configuration."""
        relationships = self.config.get("_relationships", [])

        if not isinstance(relationships, list):
            self.errors.append("'_relationships' must be an array")
            return

        for i, rel in enumerate(relationships):
            if not isinstance(rel, dict):
                self.errors.append(f"Relationship {i}: Must be an object")
                continue

            # Required fields
            for field in ["name", "formula", "target", "tolerance_percent"]:
                if field not in rel:
                    self.errors.append(f"Relationship '{rel.get('name', i)}': Missing required field '{field}'")

            # Validate types
            if "name" in rel and not isinstance(rel["name"], str):
                self.errors.append(f"Relationship {i}: 'name' must be a string")

            if "formula" in rel and not isinstance(rel["formula"], str):
                self.errors.append(f"Relationship '{rel.get('name', i)}': 'formula' must be a string")

            if "target" in rel and not isinstance(rel["target"], str):
                self.errors.append(f"Relationship '{rel.get('name', i)}': 'target' must be a string")

            if "tolerance_percent" in rel:
                if not isinstance(rel["tolerance_percent"], (int, float)):
                    self.errors.append(f"Relationship '{rel.get('name', i)}': 'tolerance_percent' must be a number")
                elif rel["tolerance_percent"] < 0 or rel["tolerance_percent"] > 100:
                    self.errors.append(f"Relationship '{rel.get('name', i)}': 'tolerance_percent' must be between 0 and 100")

    def _validate_lab_entries(self) -> None:
        """Validate individual lab entries."""
        lab_count = 0
        missing_loinc_count = 0
        loinc_codes: dict[str, list[str]] = {}  # loinc_code -> [lab_names]

        for lab_name, spec in self.config.items():
            # Skip special keys
            if lab_name.startswith("_"):
                continue

            lab_count += 1

            if not isinstance(spec, dict):
                self.errors.append(f"Lab '{lab_name}': Specification must be an object")
                continue

            # Validate required fields
            self._validate_required_fields(lab_name, spec)

            # Validate lab_type and name prefix
            self._validate_lab_type_and_prefix(lab_name, spec)

            # Validate primary_unit
            self._validate_primary_unit(lab_name, spec)

            # Validate alternatives
            self._validate_alternatives(lab_name, spec)

            # Validate ranges
            self._validate_ranges(lab_name, spec)

            # Validate LOINC code
            if not self._validate_loinc_code(lab_name, spec):
                missing_loinc_count += 1
            else:
                # Track LOINC codes for uniqueness check
                loinc_code = spec.get("loinc_code")
                if loinc_code:
                    if loinc_code not in loinc_codes:
                        loinc_codes[loinc_code] = []
                    loinc_codes[loinc_code].append(lab_name)

            # Validate optional biological limits
            self._validate_biological_limits(lab_name, spec)

            # Validate optional temporal constraints
            self._validate_temporal_constraints(lab_name, spec)

            # Validate optional ranges_vary_with_cycle flag
            self._validate_ranges_vary_with_cycle(lab_name, spec)

        # Check for duplicate LOINC codes
        for loinc_code, lab_names in loinc_codes.items():
            if len(lab_names) > 1:
                self.errors.append(f"Duplicate LOINC code '{loinc_code}' used by: {', '.join(sorted(lab_names))}")

        # Summary
        if lab_count == 0:
            self.errors.append("No lab entries found in config")

        if missing_loinc_count > len(self.LOINC_EXCEPTIONS):
            self.warnings.append(f"{missing_loinc_count} labs missing LOINC codes ({len(self.LOINC_EXCEPTIONS)} known exceptions)")

    def _validate_required_fields(self, lab_name: str, spec: dict) -> None:
        """Validate required fields are present."""
        required = ["lab_type", "primary_unit", "loinc_code"]

        for field in required:
            if field not in spec:
                # LOINC code is checked separately with exceptions
                if field == "loinc_code" and lab_name in self.LOINC_EXCEPTIONS:
                    continue
                self.errors.append(f"Lab '{lab_name}': Missing required field '{field}'")

    def _validate_lab_type_and_prefix(self, lab_name: str, spec: dict) -> None:
        """Validate lab_type and name prefix consistency."""
        lab_type = spec.get("lab_type")

        if not lab_type:
            return  # Already flagged in required fields

        if not isinstance(lab_type, str):
            self.errors.append(f"Lab '{lab_name}': 'lab_type' must be a string")
            return

        if lab_type not in self.VALID_LAB_TYPES:
            self.errors.append(f"Lab '{lab_name}': Invalid lab_type '{lab_type}'. Must be one of: {', '.join(sorted(self.VALID_LAB_TYPES))}")
            return

        # Check name prefix matches lab_type
        expected_prefixes = self.LAB_NAME_PREFIXES.get(lab_type, [])
        if expected_prefixes:
            if not any(lab_name.startswith(prefix) for prefix in expected_prefixes):
                prefix_str = "' or '".join(expected_prefixes)
                self.errors.append(f"Lab '{lab_name}': Name must start with '{prefix_str}' for lab_type '{lab_type}'")

    def _validate_primary_unit(self, lab_name: str, spec: dict) -> None:
        """Validate primary_unit field."""
        primary_unit = spec.get("primary_unit")

        if not primary_unit:
            return  # Already flagged in required fields

        if not isinstance(primary_unit, str):
            self.errors.append(f"Lab '{lab_name}': 'primary_unit' must be a string")
            return

        if not primary_unit.strip():
            self.errors.append(f"Lab '{lab_name}': 'primary_unit' cannot be empty")

        # Check consistency with lab name for percentage units
        if primary_unit == "%" and not lab_name.endswith("(%)"):
            self.warnings.append(f"Lab '{lab_name}': Has unit '%' but name doesn't end with '(%)'")

    def _validate_alternatives(self, lab_name: str, spec: dict) -> None:
        """Validate alternatives array."""
        alternatives = spec.get("alternatives", [])

        if not isinstance(alternatives, list):
            self.errors.append(f"Lab '{lab_name}': 'alternatives' must be an array")
            return

        for i, alt in enumerate(alternatives):
            if not isinstance(alt, dict):
                self.errors.append(f"Lab '{lab_name}': Alternative {i} must be an object")
                continue

            # Required fields in alternatives
            if "unit" not in alt:
                self.errors.append(f"Lab '{lab_name}': Alternative {i} missing 'unit'")
            elif not isinstance(alt["unit"], str):
                self.errors.append(f"Lab '{lab_name}': Alternative {i} 'unit' must be a string")

            if "factor" not in alt:
                self.errors.append(f"Lab '{lab_name}': Alternative {i} missing 'factor'")
            elif not isinstance(alt["factor"], (int, float)):
                self.errors.append(f"Lab '{lab_name}': Alternative {i} 'factor' must be a number")
            elif alt["factor"] <= 0:
                self.errors.append(f"Lab '{lab_name}': Alternative {i} 'factor' must be positive")

    def _validate_ranges(self, lab_name: str, spec: dict) -> None:
        """Validate ranges configuration."""
        ranges = spec.get("ranges")

        # Ranges are optional
        if ranges is None:
            return

        if not isinstance(ranges, dict):
            self.errors.append(f"Lab '{lab_name}': 'ranges' must be an object")
            return

        for range_key, range_val in ranges.items():
            if not isinstance(range_val, list):
                self.errors.append(f"Lab '{lab_name}': Range '{range_key}' must be an array")
                continue

            if len(range_val) != 2:
                self.errors.append(f"Lab '{lab_name}': Range '{range_key}' must have exactly 2 values [min, max]")
                continue

            min_val, max_val = range_val

            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                self.errors.append(f"Lab '{lab_name}': Range '{range_key}' values must be numbers")
                continue

            if min_val > max_val:
                self.errors.append(f"Lab '{lab_name}': Range '{range_key}' min ({min_val}) > max ({max_val})")

    def _validate_loinc_code(self, lab_name: str, spec: dict) -> bool:
        """Validate LOINC code presence and format.

        Returns:
            True if LOINC code is present and valid, False otherwise
        """
        loinc_code = spec.get("loinc_code")

        # Allow exceptions
        if lab_name in self.LOINC_EXCEPTIONS:
            if not loinc_code:
                self.warnings.append(f"Lab '{lab_name}': Known exception without LOINC code")
            return True

        if not loinc_code:
            self.errors.append(f"Lab '{lab_name}': Missing LOINC code")
            return False

        if not isinstance(loinc_code, str):
            self.errors.append(f"Lab '{lab_name}': LOINC code must be a string")
            return False

        # Basic format validation (LOINC codes are typically NNNN-N format)
        if not loinc_code.strip():
            self.errors.append(f"Lab '{lab_name}': LOINC code cannot be empty")
            return False

        # Loose format check (allows various LOINC formats)
        if not any(c.isdigit() for c in loinc_code):
            self.warnings.append(f"Lab '{lab_name}': LOINC code '{loinc_code}' has unusual format (no digits)")

        return True

    def _validate_biological_limits(self, lab_name: str, spec: dict) -> None:
        """Validate optional biological_min and biological_max fields."""
        bio_min = spec.get("biological_min")
        bio_max = spec.get("biological_max")

        if bio_min is not None:
            if not isinstance(bio_min, (int, float)):
                self.errors.append(f"Lab '{lab_name}': 'biological_min' must be a number")

        if bio_max is not None:
            if not isinstance(bio_max, (int, float)):
                self.errors.append(f"Lab '{lab_name}': 'biological_max' must be a number")

        # Check min < max if both present
        if bio_min is not None and bio_max is not None:
            if isinstance(bio_min, (int, float)) and isinstance(bio_max, (int, float)):
                if bio_min > bio_max:
                    self.errors.append(f"Lab '{lab_name}': biological_min ({bio_min}) > biological_max ({bio_max})")

    def _validate_temporal_constraints(self, lab_name: str, spec: dict) -> None:
        """Validate optional max_daily_change field."""
        max_daily_change = spec.get("max_daily_change")

        if max_daily_change is not None:
            if not isinstance(max_daily_change, (int, float)):
                self.errors.append(f"Lab '{lab_name}': 'max_daily_change' must be a number")
            elif max_daily_change <= 0:
                self.errors.append(f"Lab '{lab_name}': 'max_daily_change' must be positive")

    def _validate_ranges_vary_with_cycle(self, lab_name: str, spec: dict) -> None:
        """Validate optional ranges_vary_with_cycle field.

        This flag indicates labs where reference ranges legitimately vary
        by menstrual cycle phase (e.g., FSH, LH, Progesterone, Estradiol).
        """
        ranges_vary = spec.get("ranges_vary_with_cycle")

        if ranges_vary is not None:
            if not isinstance(ranges_vary, bool):
                self.errors.append(f"Lab '{lab_name}': 'ranges_vary_with_cycle' must be a boolean")

    def print_report(self) -> None:
        """Print validation report."""
        print("\n=== Lab Specs Schema Validation ===\n")
        print(f"Config: {self.config_path}")

        # Count lab entries
        lab_count = sum(1 for k in self.config.keys() if not k.startswith("_"))
        print(f"Lab entries: {lab_count}")

        # Print errors
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print("\n✅ No errors found")

        # Print warnings
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        print()


def main():
    """Run validation and exit with appropriate code."""
    validator = LabSpecsValidator()

    is_valid = validator.validate()
    validator.print_report()

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
