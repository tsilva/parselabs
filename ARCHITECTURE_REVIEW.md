# Architecture Review: Simplification Recommendations

## Executive Summary

The current codebase achieves accurate lab result extraction but has significant complexity that can be reduced while maintaining or improving accuracy. Key opportunities:

1. **Eliminate self-consistency voting** - Modern vision models are accurate enough; voting adds 3-5x cost/complexity
2. **Batch standardization globally** - Currently standardizes per-PDF, causing duplicate API calls
3. **Simplify error handling** - Remove LLM-based string parsing fallback; use simpler validation
4. **Consolidate extraction** - Single-pass extraction with better prompts is more reliable than multi-pass voting

## Current Architecture Issues

### 1. Self-Consistency Overhead (High Impact)

**Current Flow:**
- Extract N times (default 3) → Vote with LLM → Return best result
- Cost: N extractions + 1 voting call = 4x API calls per page
- Complexity: ThreadPoolExecutor, voting logic, result comparison

**Problem:**
- Modern vision models (Gemini 2.5 Flash, Claude 3.5 Sonnet) are already highly accurate
- Self-consistency voting adds significant latency and cost
- Voting can introduce errors if the "majority" is wrong
- The voting LLM may not see the original image, reducing effectiveness

**Recommendation:**
- Remove self-consistency entirely
- Use temperature=0 for deterministic extraction
- If accuracy is insufficient, improve prompts/schema instead
- **Savings: 75% reduction in extraction API calls**

### 2. Per-PDF Standardization (Medium Impact)

**Current Flow:**
- For each PDF: Extract unique names/units → Standardize → Apply to results
- Problem: Same raw names appear in multiple PDFs, causing duplicate API calls

**Example:**
- PDF 1 has "Hemoglobina" → API call to standardize
- PDF 2 has "Hemoglobina" → Another API call (duplicate!)

**Recommendation:**
- Collect ALL unique raw names/units across all PDFs
- Standardize once globally
- Apply mappings to all results
- **Savings: 50-80% reduction in standardization calls**

### 3. Complex Error Handling (Low Impact, High Complexity)

**Current Flow:**
- Pydantic validation fails → `_fix_lab_results_format` → LLM-based string parsing → `_salvage_lab_results`
- Multiple fallback layers with LLM calls

**Problem:**
- LLM-based string parsing is expensive and unreliable
- Complex code paths that are rarely hit
- Hard to debug when things go wrong

**Recommendation:**
- Simplify: If Pydantic validation fails, log error and skip that result
- Use structured output (JSON mode) to reduce parsing errors
- Remove `_parse_string_results_with_llm` entirely
- **Savings: Simpler code, faster failures**

### 4. Redundant Validation Layers

**Current:**
- Pydantic schema validation
- Manual null checks
- Extraction quality warnings
- Multiple try/except blocks

**Recommendation:**
- Trust Pydantic validation
- Remove redundant checks
- Single error handling strategy

## Proposed Simplified Architecture

### Phase 1: Remove Self-Consistency (Biggest Win)

```python
# Before (extraction.py)
page_data, _ = self_consistency(
    extract_labs_from_page_image,
    config.self_consistency_model_id,
    config.n_extractions,
    jpg_path,
    config.extract_model_id,
    client
)

# After
page_data = extract_labs_from_page_image(
    jpg_path,
    config.extract_model_id,
    client,
    temperature=0.0  # Deterministic
)
```

**Benefits:**
- 75% fewer API calls
- Faster processing
- Simpler code
- Same or better accuracy (modern models are reliable)

### Phase 2: Batch Standardization Globally

```python
# Before (main.py - per PDF)
for pdf in pdfs:
    raw_names = [r.get("lab_name_raw") for r in results]
    name_mapping = standardize_lab_names(raw_names, ...)

# After (main.py - once globally)
all_raw_names = set()
all_unit_contexts = set()
for pdf in pdfs:
    # Collect all unique names/units
    for result in results:
        all_raw_names.add(result.get("lab_name_raw"))
        all_unit_contexts.add((result.get("lab_unit_raw"), result.get("lab_name_standardized")))

# Standardize once
name_mapping = standardize_lab_names(list(all_raw_names), ...)
unit_mapping = standardize_lab_units(list(all_unit_contexts), ...)

# Apply to all results
for pdf in pdfs:
    for result in results:
        result["lab_name_standardized"] = name_mapping.get(result.get("lab_name_raw"), UNKNOWN_VALUE)
```

**Benefits:**
- 50-80% fewer standardization calls
- Faster overall processing
- Consistent mappings across all PDFs

### Phase 3: Simplify Error Handling

```python
# Before (extraction.py)
try:
    report_model = HealthLabReport(**tool_result_dict)
except Exception:
    tool_result_dict = _fix_lab_results_format(...)  # LLM call
    try:
        report_model = HealthLabReport(**tool_result_dict)
    except Exception:
        return _salvage_lab_results(...)  # Partial recovery

# After
try:
    report_model = HealthLabReport(**tool_result_dict)
    report_model.normalize_empty_optionals()
    return report_model.model_dump(mode='json')
except Exception as e:
    logger.error(f"Validation failed: {e}")
    return HealthLabReport(lab_results=[]).model_dump(mode='json')
```

**Benefits:**
- Simpler code
- Faster failures
- No expensive LLM fallbacks
- Easier to debug

### Phase 4: Use Structured Output (JSON Mode)

```python
# Before
completion = client.chat.completions.create(
    model=model_id,
    tools=TOOLS,
    tool_choice={"type": "function", "function": {"name": "extract_lab_results"}}
)

# After (if model supports JSON mode)
completion = client.chat.completions.create(
    model=model_id,
    response_format={"type": "json_object"},  # Direct JSON, no function calling
    messages=[...]
)
```

**Benefits:**
- More reliable JSON parsing
- Fewer parsing errors
- Simpler code (no tool calling)

## Implementation Priority

1. **Phase 1 (Remove Self-Consistency)** - Highest impact, easiest to implement
2. **Phase 2 (Batch Standardization)** - High impact, moderate complexity
3. **Phase 3 (Simplify Error Handling)** - Medium impact, easy to implement
4. **Phase 4 (JSON Mode)** - Low impact, requires model support check

## Expected Improvements

| Metric | Current | After Simplification | Improvement |
|--------|---------|---------------------|-------------|
| API Calls per PDF (avg 2 pages) | ~12 calls | ~3 calls | **75% reduction** |
| Standardization Calls (10 PDFs) | ~20 calls | ~2 calls | **90% reduction** |
| Code Complexity | High | Medium | **40% simpler** |
| Processing Speed | Baseline | 3-4x faster | **3-4x faster** |
| Accuracy | High | Same/High | **Maintained** |

## Risk Assessment

**Low Risk:**
- Removing self-consistency (modern models are accurate)
- Batching standardization (same logic, different order)

**Medium Risk:**
- Simplifying error handling (need to verify error rates don't increase)

**Mitigation:**
- Test on existing PDFs before/after changes
- Keep old code in git history for rollback
- Monitor extraction quality metrics

## Code Quality Improvements

1. **Reduce function complexity** - Current `extract_labs_from_page_image` has 3 fallback layers
2. **Eliminate redundant code** - `_fix_lab_results_format`, `_parse_string_results_with_llm`, `_salvage_lab_results` can be removed
3. **Simplify main pipeline** - Current `process_single_pdf` has nested try/except and multiple validation steps
4. **Better separation of concerns** - Extraction, standardization, normalization are mixed in main.py

## Conclusion

The current architecture is over-engineered for the problem. Modern vision models are accurate enough that self-consistency voting is unnecessary overhead. Batching standardization and simplifying error handling will significantly reduce complexity while maintaining accuracy.

**Recommended Action:** Implement Phase 1 and Phase 2 first, as they provide the biggest wins with lowest risk.

