# Pipeline Improvement Suggestions

This document outlines potential simplifications to reduce complexity while maintaining the same extraction quality.

## Current Pipeline Complexity

The current pipeline has 8 major stages with multiple intermediate files per page:
- **Per-page artifacts**: `.jpg`, `.txt`, `.json` (3 files × pages)
- **Models**: 3 separate model configurations (transcribe, extract, self-consistency)
- **Config files**: 3 JSON files (names mapping, units mapping, lab specs)
- **Processing steps**: Image preprocessing → Transcription → Extraction → Normalization → Mapping → Deduplication

## Major Simplification Opportunities

### 1. Eliminate Transcription Step ⭐ (Biggest Win)

**Current Architecture**:
```
Image → Vision Model (Transcription) → LLM (Extraction)
      → .txt file                   → .json file
```

**Simplified Architecture**:
```
Image → Vision Model (Direct Extraction) → .json file
```

**Implementation**:
- Use vision model with function calling directly on images
- Modern vision models (Gemini Flash, GPT-4o, Claude 3.5) support structured output from images
- Pass Pydantic schema directly to vision model

**Benefits**:
- ✅ 50% fewer API calls per page
- ✅ Removes ~100 lines of code (`transcription_from_page_image()` and related logic)
- ✅ Eliminates all `.txt` files from output
- ✅ No transcription errors propagating to extraction phase
- ✅ Removes `TRANSCRIBE_MODEL_ID` and `N_TRANSCRIPTIONS` configuration
- ✅ Faster processing (one API call instead of two)

**Trade-offs**:
- ⚠️ Slightly less debuggable (can't inspect intermediate text)
- ⚠️ May require better prompt engineering for vision model

**Code Impact**: Remove functions
- `transcription_from_page_image()`
- One call to `self_consistency()` per page
- Text file I/O logic

---

### 2. Multi-Page Batching

**Current**: Each page processed independently in separate API calls
**Proposed**: Batch 2-5 pages per request

**Implementation**:
```python
# Instead of:
for page in pages:
    result = extract_from_page(page)

# Do:
for page_batch in chunk(pages, size=3):
    results = extract_from_pages(page_batch)  # Multi-image request
```

**Benefits**:
- ✅ 2-3x fewer API calls for multi-page reports
- ✅ Model sees full document context (better date resolution, cross-page references)
- ✅ Eliminates per-page JSON files (one JSON per PDF)
- ✅ Automatic date resolution (no complex fallback logic needed)
- ✅ Better handling of "continued on next page" scenarios

**Trade-offs**:
- ⚠️ Larger token usage per request
- ⚠️ Harder to cache individual pages (need cache invalidation strategy)
- ⚠️ If one page fails, entire batch may need retry

**Recommended Strategy**:
- Use batch size of 3 pages (empirically good balance)
- Fallback to single-page processing if batch fails
- Keep page-level caching for JPG images, batch-level for JSON

---

### 3. LLM-Driven Normalization

**Current Post-Processing Pipeline**:
```
Extract raw values → Slugify → Map to enum → Convert units
```

**Simplified LLM-Driven**:
```
Extract normalized values directly (provide lab_specs.json as context)
```

**Implementation**:
- Include `lab_specs.json` in system prompt or as tool definition
- Instruct model to output standardized names/units directly
- Model returns "Blood - Hemoglobin A1c" instead of "HbA1c" or "Hemoglobin A1C"

**Benefits**:
- ✅ Eliminates `lab_names_mappings.json` (468 entries to maintain)
- ✅ Eliminates `lab_units_mappings.json` (50+ entries)
- ✅ Removes `slugify()` function and all slugification logic
- ✅ Removes `map_lab_name_enum()` and `map_lab_unit_enum()` functions
- ✅ Removes error logging for unmapped values
- ✅ Simpler dataframe processing (~200 lines removed from `main()`)
- ✅ Automatic handling of new lab name variations (no manual mapping updates)

**Trade-offs**:
- ⚠️ Larger prompts (include entire lab specs)
- ⚠️ Model may make normalization errors
- ⚠️ Less explicit control over mappings
- ⚠️ Harder to debug why specific mapping was chosen

**Hybrid Approach** (Recommended):
- Use LLM for initial normalization
- Keep `lab_specs.json` as source of truth
- Add validation step: check if extracted names exist in `lab_specs.json`
- Log unmapped values for manual review/spec updates

---

### 4. Reduce Intermediate File Output

**Current File Structure** (per 3-page PDF):
```
output/document_name/
  ├── document_name.pdf (copy)
  ├── document_name.001.jpg
  ├── document_name.001.txt
  ├── document_name.001.json
  ├── document_name.002.jpg
  ├── document_name.002.txt
  ├── document_name.002.json
  ├── document_name.003.jpg
  ├── document_name.003.txt
  ├── document_name.003.json
  └── document_name.csv
```
**Total**: 13 files per 3-page document

**Simplified Structure**:
```
output/document_name/
  ├── document_name.001.jpg
  ├── document_name.002.jpg
  ├── document_name.003.jpg
  ├── document_name.json (all pages)
  └── document_name.csv
```
**Total**: 5 files per 3-page document (62% reduction)

**Benefits**:
- ✅ Less disk I/O (faster processing)
- ✅ Cleaner output directories (easier to navigate)
- ✅ Reduced filesystem operations (better for network drives)
- ✅ Simpler existence checks

**Trade-offs**:
- ⚠️ Harder to debug individual pages
- ⚠️ No incremental caching for extraction (all-or-nothing)
- ⚠️ Reprocessing one page requires reprocessing all pages

**Recommendation**: Keep `.jpg` files for debugging, eliminate `.txt` and per-page `.json`

---

### 5. Simplify Self-Consistency Pattern

**Current Implementation**:
- Run operation N times (N_TRANSCRIPTIONS, N_EXTRACTIONS)
- If outputs differ, call separate voting model (SELF_CONSISTENCY_MODEL_ID)
- Complex fallback logic (confidence-based selection)

**Simplified Options**:

#### Option A: Single-Call with Structured Output
```python
# Remove self-consistency entirely
result = extract_labs_from_image(image, model_id, temperature=0)
```

**Benefits**:
- ✅ Removes entire `self_consistency()` function (~80 lines)
- ✅ Removes voting logic and fallback mechanisms
- ✅ N times faster (N=1 instead of N=3 or N=5)
- ✅ Lower API costs
- ✅ Simpler configuration (remove N_TRANSCRIPTIONS, N_EXTRACTIONS, SELF_CONSISTENCY_MODEL_ID)

**When to use**: If N=1 is already working well (which is the default)

#### Option B: Model's Native Confidence Scores
```python
# Single call, use model's built-in confidence
result = extract_labs_from_image(image, model_id)
# Filter results where confidence < threshold
filtered = [r for r in result.lab_results if r.confidence > 0.7]
```

**Benefits**: Uses model's own uncertainty estimation, no external voting

#### Option C: Single-Model Voting
```python
# If keeping N>1, vote within the same model
results = [extract_labs() for _ in range(N)]
best = model.vote(results)  # Same model votes on its own outputs
```

**Benefits**: Removes separate SELF_CONSISTENCY_MODEL configuration

**Recommendation**:
- Start with Option A (single call, temperature=0)
- If accuracy drops, implement Option C (single-model voting)
- Default N=1, allow N>1 via config for critical use cases

---

### 6. Consolidate Model Configuration

**Current**: 3 separate model IDs
```env
TRANSCRIBE_MODEL_ID=google/gemini-2.5-flash
EXTRACT_MODEL_ID=google/gemini-2.5-flash
SELF_CONSISTENCY_MODEL_ID=google/gemini-2.5-flash
```

**Simplified**: 1 model ID
```env
MODEL_ID=google/gemini-2.5-flash
```

**Benefits**:
- ✅ Simpler configuration
- ✅ Easier to switch models (one change instead of three)
- ✅ Lower latency (connection pooling to single endpoint)

**When different models make sense**:
- Fast model for transcription, powerful model for extraction
- In practice, using same model is common

---

### 7. Simplify Configuration Files

**Current**: 3 JSON config files
- `lab_names_mappings.json` (468 entries)
- `lab_units_mappings.json` (50+ entries)
- `lab_specs.json` (primary units, conversions, ranges)

**Simplified**: 1 unified config file
```json
{
  "Blood - Hemoglobin A1c": {
    "aliases": ["HbA1c", "Hemoglobin A1C", "A1c", "Glycated Hemoglobin"],
    "primary_unit": "%",
    "alternative_units": {
      "mmol/mol": {"factor": 10.93}
    },
    "healthy_range": {"min": 4.0, "max": 5.6}
  }
}
```

**Benefits**:
- ✅ Single source of truth
- ✅ Easier to maintain (all info for one lab in one place)
- ✅ No need to keep 3 files in sync
- ✅ Simpler loading logic

**Implementation**:
- Merge all 3 files into `lab_specs.json`
- Add `aliases` array to each lab spec
- Matching logic: check if extracted name is in aliases, map to canonical name

---

## Recommended Simplified Architecture

### Minimal Pipeline (Aggressive Simplification)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PDF → Multi-page image batches (3 pages/batch)          │
│ 2. Vision model → Direct structured extraction              │
│    - Input: Images + lab_specs.json context                │
│    - Output: Normalized, structured data                    │
│ 3. Per-PDF JSON → Merged CSV → Excel + Plots               │
└─────────────────────────────────────────────────────────────┘
```

**Removed**:
- Transcription step (no `.txt` files)
- Per-page JSON files
- Slugification logic
- Mapping files (merged into specs)
- Self-consistency voting
- Unit conversion post-processing (done by model)
- 3 model configs → 1 model config

**Code reduction**: ~40-50% fewer lines in `main.py`

**Config reduction**:
```env
MODEL_ID=google/gemini-2.5-flash
INPUT_PATH=./data/pdfs
OUTPUT_PATH=./output
LAB_SPECS_PATH=config/lab_specs.json
MAX_WORKERS=4
BATCH_SIZE=3  # Pages per request
```

---

### Hybrid Pipeline (Conservative Simplification)

**Recommended for maintaining debuggability while simplifying:**

| Component | Keep/Change | Rationale |
|-----------|-------------|-----------|
| Page-by-page processing | ✅ Keep | Good for caching, debugging |
| Transcription step | ❌ Remove | Redundant with modern vision models |
| Direct image→structured | ✅ Add | Reduces API calls by 50% |
| Config files | ✅ Keep separate | More maintainable than LLM-driven normalization |
| Per-page JSONs | ❌ Remove | Save only per-PDF results |
| Slugification & mapping | ✅ Keep | Explicit control, easier debugging |
| Self-consistency | ⚠️ Simplify | Default N=1, same model if N>1 |
| Multi-page batching | ⚠️ Optional | Consider for >2 page docs |

**Code reduction**: ~30% fewer lines
**Complexity reduction**: Moderate
**Risk**: Low (maintains most existing logic)

---

## Implementation Priority

### Phase 1: Low-Risk, High-Impact (Do First)
1. **Eliminate transcription step** - Direct image→extraction
   - Effort: Medium (rewrite extraction function)
   - Risk: Low (vision models support this well)
   - Impact: 50% fewer API calls, simpler code

2. **Consolidate model configuration** - Use single MODEL_ID
   - Effort: Low (config change + minor refactoring)
   - Risk: None
   - Impact: Simpler configuration

3. **Remove per-page JSON files** - Save only per-PDF JSON
   - Effort: Low (file I/O changes)
   - Risk: Low (debugging slightly harder)
   - Impact: Cleaner output directories

### Phase 2: Medium-Risk, High-Impact (Do Second)
4. **Simplify self-consistency** - Default N=1, remove voting
   - Effort: Medium (remove function, update callers)
   - Risk: Medium (may affect accuracy)
   - Impact: Faster processing, simpler code

5. **Multi-page batching** - Process 3 pages per request
   - Effort: High (significant refactoring)
   - Risk: Medium (caching strategy changes)
   - Impact: 2-3x fewer API calls

### Phase 3: Higher-Risk, Research Needed (Evaluate)
6. **LLM-driven normalization** - Model outputs standardized values
   - Effort: High (prompt engineering + validation)
   - Risk: High (model errors in normalization)
   - Impact: Removes mapping maintenance burden

7. **Unified config file** - Merge 3 JSONs into 1
   - Effort: Medium (file restructuring + migration)
   - Risk: Low (mostly organizational)
   - Impact: Easier maintenance

---

## Estimated Impact Summary

| Improvement | LOC Reduction | API Calls | Complexity | Risk |
|-------------|---------------|-----------|------------|------|
| Remove transcription | -150 lines | -50% | -30% | Low |
| Multi-page batching | +50 lines | -60% | +10% | Medium |
| LLM normalization | -250 lines | 0% | -40% | High |
| Simplify self-consistency | -100 lines | -66% | -20% | Medium |
| Reduce files | -50 lines | 0% | -10% | Low |
| **Total (all)** | **-500 lines** | **-75%** | **-50%** | **Mixed** |
| **Hybrid (conservative)** | **-200 lines** | **-50%** | **-30%** | **Low** |

---

## Migration Strategy

### For Existing Data
- Keep old pipeline in `main.py`
- Create `main_v2.py` with simplified pipeline
- Run both in parallel on sample PDFs
- Compare outputs for accuracy
- Switch when confident

### Backward Compatibility
- Maintain ability to read old intermediate files (`.txt`, per-page `.json`)
- Add migration script to convert old output structure to new
- Keep old config files for 1-2 versions

---

## Questions to Consider

1. **How often do you debug transcription vs extraction separately?**
   - If rarely: Remove transcription step
   - If often: Keep separate steps

2. **What's your current accuracy with N=1?**
   - If >95%: Remove self-consistency
   - If <90%: Keep but simplify

3. **How important is exact control over name/unit mappings?**
   - Critical for medical: Keep explicit mappings
   - Flexible: Try LLM-driven normalization

4. **What's your typical PDF page count?**
   - Mostly 1-2 pages: Keep page-by-page
   - Often 5+ pages: Implement batching

5. **How much disk space vs debugging visibility?**
   - Tight on space: Minimize intermediate files
   - Need debugging: Keep `.txt` and per-page `.json`

---

## Next Steps

1. Review improvements and prioritize based on your use case
2. Implement Phase 1 improvements (low-risk, high-impact)
3. Test on sample PDFs and measure:
   - Accuracy (compare with current pipeline)
   - Speed (processing time reduction)
   - API costs (token usage reduction)
4. Iterate based on results
5. Document findings and update this file
