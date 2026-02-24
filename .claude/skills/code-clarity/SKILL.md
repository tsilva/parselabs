---
name: code-clarity
description: Code quality and clarity guidelines for Python - enforces flat orchestrator patterns, explicit error handling, and mandatory comments
---

# Code Clarity Skill

## Core Rules

### 1. Max 2 Indentation Levels in Orchestrators

Main orchestrator functions must not exceed 2 levels of indentation. Extract helpers when deeper.

```python
# GOOD - Flat structure with early returns
def process():
    if not valid:           # Level 1
        return None
    
    result = fetch()        # Level 1
    if not result:          # Level 1
        return None
    
    return transform(result)

# BAD - Deep nesting requires extraction
def process():
    if condition1:          # Level 1
        if condition2:       # Level 2
            for item in items: # Level 3 ← EXTRACT THIS
                process(item)
```

### 2. Helpers Must Not Silently Catch Errors

Reusable helpers let exceptions propagate. Orchestrator handles all error flow.

```python
# BAD - Silent failure hidden in helper
def helper():
    try:
        return api.call()
    except Exception:
        return None  # Silent!

def main():
    result = helper()  # Did it fail? Unknown!
    if result:         # Defensive check required
        process(result)

# CORRECT - Explicit error handling in orchestrator
def helper():
    return api.call()  # Let it raise

def main():
    try:
        result = helper()
        process(result)
    except APIError as e:
        logger.error(f"API call failed: {e}")
        return None
```

### 3. All Comments Required

**Every code block, branch, and early exit must have an explanatory comment.**

#### Block Comments
Comments explain PURPOSE and INTENT, not just restate code.

```python
# Build and validate configuration from user args
config, errors = build_config(args)

# Guard: Return errors if validation failed
if errors:
    return None, errors
```

#### Branch Comments
All `if/elif/else` branches must have comments explaining the condition.

```python
# Guard: Skip processing if data is missing
if not data:
    return None
# Check for authentication errors specifically
elif "401" in error_msg:
    return False, "Auth failed"
# Handle timeout scenarios
elif "timeout" in error_msg.lower():
    return False, "Server timeout"
# Any other error - fail safe
else:
    return False, "Unknown error"
```

**TRIGGER: Always check these for missing comments:**
- Error classification chains (multiple elif branches)
- State machine logic (different states)
- Strategy selection (choosing between algorithms)
- Protocol handlers (HTTP status codes, API response codes)

#### Early Exit Comments (Guard Clauses)
**ALL early exits MUST have comments.** No exceptions.

```python
# CORRECT - Guard clause with explanation
# Guard: Return errors if validation failed
if errors:
    return None, errors

# Skip non-directory entries
if not pdf_dir.is_dir():
    continue

# Process remaining PDFs if cache incomplete
if pdfs_to_process:
    _process()
```

**MANDATORY CHECK:** Verify each has a comment:
- `if condition:` → `return` / `continue` / `break`
- Any guard causing early exit from function or loop

### 4. Extract Helpers for Complex Logic

Break logic into well-named helpers when:
- Hit indentation level 3 or deeper
- More than 5-7 lines of contiguous logic
- Logic can be reused or tested independently

**Naming:** Use verbs describing intent.
```python
# GOOD
def _extract_via_vision()
def _apply_standardization()
def _try_load_cached_extraction()

# BAD
def _convert_pdf()        # Implementation detail
def _load_cache()         # Too vague
def _process_data()       # Too generic
```

**Document tuple returns in docstring:**
```python
def _extract_via_vision(pdf_path: Path) -> tuple[list[Result], datetime | None]:
    """Extract lab results from PDF using vision model.
    
    Returns:
        tuple: (list of extracted results, document date or None)
    """
```

### 5. Orchestrator Pattern

Main function controls all flow decisions. Error handling visible at each step. Read top-to-bottom without jumping into helpers.

```python
def process_single_pdf(pdf_path: Path) -> tuple[Path | None, list[dict]]:
    """Process a single PDF: extract, standardize, save."""
    # Initialize directory structure
    doc_out_dir, csv_path, failed_pages = _setup_paths(pdf_path)

    # Attempt text extraction, fall back to vision if needed
    try:
        result = _try_text_extraction(pdf_path)
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        result = None

    # Choose processing path based on extraction success
    if result:
        data = _process_text_results(result)
    else:
        try:
            data = _extract_via_vision(pdf_path)
        except Exception as e:
            logger.error(f"Vision extraction failed: {e}")
            return None, failed_pages

    # Guard: Check if extraction yielded results
    if not data:
        return _handle_empty_results(csv_path), failed_pages

    # Post-processing
    _apply_standardization(data)
    _save_results(data, csv_path)

    return csv_path, failed_pages
```

**Principles:**
- Read top-to-bottom without jumping into helpers
- All decision points visible in main flow
- Error handling explicit at each step
- Guard clauses for early returns
