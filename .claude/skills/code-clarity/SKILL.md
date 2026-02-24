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

**TRIGGER: Always check these patterns for missing comments:**
- Error classification chains (multiple elif branches checking error codes/messages)
- State machine logic (if/elif/else handling different states)
- Strategy selection (choosing between algorithms, formats, approaches)
- Protocol handlers (HTTP status codes, API response codes, error types)

5. **ALL early exit conditions MUST have a comment** directly above them explaining the guard condition. No exceptions.

```python
# CORRECT - Comment explains WHY we exit
# Guard: Return errors if validation failed (config will be None)
if errors:
    return None, errors

# CORRECT - Comment explains WHAT we're filtering
# Skip non-directory entries (files, symlinks, etc.)
if not pdf_dir.is_dir():
    continue

# CORRECT - Comment explains the threshold
# Process remaining PDFs if cache doesn't have all files
if pdfs_to_process:
    _process()

# INCORRECT - Missing comment
if errors:
    return None, errors

# INCORRECT - Missing comment  
if not pdf_dir.is_dir():
    continue
```

**MANDATORY CHECK: Scan for these patterns and verify each has a comment:**
- `if condition:` followed by `return` → Must have comment above `if`
- `if condition:` followed by `continue` → Must have comment above `if`  
- `if condition:` followed by `break` → Must have comment above `if`
- Any guard clause that causes early exit from function or loop

**NO EXCEPTIONS:** Even "obvious" conditions like `if errors:` or `if not data:` MUST have comments.

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
| **Guard clause without comment** | **Add comment above EVERY early exit** |
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
