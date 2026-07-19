"""Compatibility entry point for ``parselabs admin validate-lab-specs``."""

from parselabs.admin.validate_lab_specs_schema import main

if __name__ == "__main__":
    raise SystemExit(main())
