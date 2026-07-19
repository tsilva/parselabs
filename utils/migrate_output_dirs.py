"""Compatibility entry point for ``parselabs admin migrate-output-dirs``."""

from parselabs.admin.migrate_output_dirs import main

if __name__ == "__main__":
    raise SystemExit(main())
