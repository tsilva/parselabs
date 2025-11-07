# Repository Guidelines

## Project Structure & Module Organization
- `main.py` orchestrates ingestion of lab reports, transcription via OpenRouter models, normalization, and CSV/Excel export; treat it as the single entry point for runtime changes.
- `config/lab_names_mappings.json` and `lab_specs.json` define controlled vocabularies used in normalization; edits must keep keys and values in sync with the integrity checks in `test.py`. Lab units are enforced via the `LabUnit` enum in `main.py`.
- `utils/` contains maintenance scripts (e.g., `build_labs_specs.py`, `sort_lab_specs.py`) that regenerate specification assets; keep their outputs version-controlled.
- Data written to `OUTPUT_PATH` (configured via `.env`) is expected to include `all.csv`, which feeds the test suite and downstream analysis.

## Environment & Configuration
- Duplicate `.env.example` into `.env`, set `OPENROUTER_API_KEY`, input/output paths, and model IDs before running any commands.
- Use `uv sync` (preferred) or `pip install -r requirements.txt` to create a Python ≥3.8 environment; the repository ships with an optional local `.venv/`.

## Build, Test, and Development Commands
- `python main.py` — runs the full extraction pipeline against files that match `INPUT_FILE_REGEX`.
- `python test.py` — executes integrity checks and prints a grouped report; rerun after any change that touches mappings, units, or output generation.
- `python utils/build_labs_specs.py` — helper to rebuild lab specification JSON when adding new analytes.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indents, explicit imports, and type hints mirroring existing models (`pydantic` classes and enums in `main.py`).
- Use descriptive, sentence-case logging messages and keep slug fields suffixed with `_slug`, enumerations with `_enum`, mirroring column schema.
- When extending mappings, maintain prefixes (`blood-`, `urine-`, `feces-`) and ensure values adopt the capitalized form expected by the tests.

## Testing Guidelines
- Integrity tests operate on `OUTPUT_PATH/all.csv`; ensure sample data exists or adjust the path in `.env` before running.
- Add new checks by defining `test_*` functions in `test.py` and registering them in `main()`, keeping output messages actionable.

## Commit & Pull Request Guidelines
- Match the existing history: start commit subjects in the imperative mood, include scope (e.g., “Update environment configuration and enhance lab specifications...”).
- Pull requests should summarize behavioral changes, note required config updates, link related issues, and attach key log excerpts or sample CSV diffs when relevant.
