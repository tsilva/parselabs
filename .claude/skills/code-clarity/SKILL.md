# Code Clarity Skill

## Core Rules

1. **Max 2 indentation levels** in orchestrators. Extract helpers when deeper.

2. **Helpers MUST NOT silently catch errors.** Let exceptions propagate to orchestrator.

3. **Every code block needs inline comments** explaining purpose/intent.

4. **All if/elif/else branches must have comments** explaining the condition and why that path is taken:
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

# Any other error - fail safe and assume unavailable
else:
    return False, "Unknown error"
```

5. **All early exits (continue, break, return) must have comments** explaining the guard condition:
```python
# Skip non-directory entries
if not pdf_dir.is_dir():
    continue

# Ignore hidden directories
if pdf_dir.name.startswith("."):
    continue

# Only process directories matching the input pattern
if pdf_dir.name not in matching_stems:
    continue
```

6. **Extract helpers** for logic >5 lines or complex conditionals. Name by intent: `_extract_via_vision()`, not `_convert_pdf()`.

7. **Use guard clauses** for early returns:
```python
# Skip invalid data entries
if not data:
    return None
```

## Orchestrator Pattern
- Main function controls all flow decisions
- Error handling visible at each step
- Read top-to-bottom without jumping into helpers

## Anti-Patterns
| Issue | Solution |
|-------|----------|
| Indentation >2 levels | Extract helper |
| Silent try/except | Remove, propagate exceptions |
| Missing comments | Add inline explanation |
| Missing branch comments | Comment every if/elif/else path |
| Early exits without comments | Add comment explaining guard condition |
| Long functions >30 lines | Extract logical sections |

## Example
```python
def main():
    # Setup paths for processing
    paths = _setup_paths()
    
    # Explicit error handling visible here
    try:
        result = _helper()  # Helper raises, doesn't catch
    except Exception as e:
        logger.error(f"Failed: {e}")
        return None
    
    # Guard clause for early exit on invalid state
    if not result:
        return None
    
    # Continue processing...
    return _process(result)
```
