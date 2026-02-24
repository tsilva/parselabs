---
name: code-clarity
description: Code quality and clarity guidelines for Python - enforces flat orchestrator patterns, explicit error handling, and mandatory comments
---

# Code Clarity Skill

## Core Rules

### 1. Flat Orchestrator

Main functions own all flow: max 2 indent levels, early returns as guards, helpers extracted for deeper logic.

```python
def process_single_pdf(pdf_path: Path) -> tuple[Path | None, list[dict]]:
    """Process a single PDF: extract, standardize, save."""
    # Initialize directory structure
    doc_out_dir, csv_path, failed_pages = _setup_paths(pdf_path)

    # Guard: Skip if no pages found
    pages = _convert_to_images(pdf_path)
    if not pages:
        return None, []

    # Extract lab data via vision model
    try:
        data = _extract_via_vision(pdf_path, pages)
    except ExtractionError as e:
        logger.error(f"Extraction failed: {e}")
        return None, failed_pages

    # Guard: No results extracted
    if not data:
        return _handle_empty_results(csv_path), failed_pages

    # Normalize and save
    _apply_standardization(data)
    _save_results(data, csv_path)
    return csv_path, failed_pages
```

**Rules embedded in this example:**
- **Max 2 indent levels** — `try` block is the deepest nesting
- **Early return over else** — guards exit early, remaining code stays flat
- **Extract helpers** — `_extract_via_vision`, `_apply_standardization` keep orchestrator readable
- **Verb-based helper names** — `_setup_paths`, `_convert_to_images`, `_extract_via_vision`
- **Read top-to-bottom** — all decision points visible in main flow

**When to extract a helper:**
- Indentation would reach level 3+
- Block exceeds 5-7 contiguous lines
- Logic is independently testable or reusable

### 2. Errors Propagate Up

Helpers raise exceptions. Orchestrators catch and decide.

```python
# BAD - Helper hides failure
def _fetch_data():
    try:
        return api.call()
    except Exception:
        return None  # Caller can't distinguish failure from empty

# GOOD - Helper raises, orchestrator decides
def _fetch_data():
    return api.call()  # Let it raise

def process():
    try:
        data = _fetch_data()
    except APIError as e:
        logger.error(f"Fetch failed: {e}")
        return None
    return transform(data)
```

### 3. Comment Every Branch

Every guard clause, if/elif/else branch, and logical block gets a comment explaining intent.

```python
# Build and validate configuration
config, errors = build_config(args)

# Guard: Bail if validation failed
if errors:
    return None, errors

# Check error type for appropriate response
if "401" in error_msg:
    # Authentication failure - credentials invalid
    return False, "Auth failed"
elif "timeout" in error_msg.lower():
    # Server didn't respond in time
    return False, "Server timeout"
else:
    # Unknown error - fail safe
    return False, "Unknown error"
```

**Mandatory check** — verify a comment exists before every:
- `return` / `continue` / `break` guard
- `if` / `elif` / `else` branch
- Logical block (group of related statements)
