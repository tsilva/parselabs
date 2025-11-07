# Labs Parser: Simplification & Accuracy Improvements

**Date:** 2025-11-07
**Analysis Goal:** Identify opportunities to make the process more accurate while simplifying it - fewer moving parts, better results.

---

## Current Architecture: Strengths & Complexity

### What's Working Well ✅
1. **Excellent traceability** - Raw values preserved alongside standardized
2. **Smart caching** - Per-page JSON files enable resumable processing
3. **Strong validation** - Pydantic models catch errors early
4. **Clean config** - Single `lab_specs.json` as source of truth
5. **Parallel processing** - PDFs processed concurrently

### Current Complexity 📊
**Per Document Processing:**
- **3-4 LLM calls per page:**
  - 1× extraction (or N× with self-consistency)
  - 1× name standardization (batched)
  - 1× unit standardization (batched)
  - 1× voting (if self-consistency results differ)
- **9 processing passes:** PDF→image→extract→standardize names→standardize units→save CSV→merge→normalize→deduplicate→export

---

## 🎯 Proposed Simplifications (Better Results, Fewer Moving Parts)

### **Big Win #1: Unified Extraction + Standardization**

**Current Problem:** Extraction is "blind" - extracts raw, then separate LLM calls standardize
```python
# Now: 3 separate LLM operations
extract_labs_from_page_image()  # Vision model extracts raw
standardize_lab_names()          # Text model maps names
standardize_lab_units()          # Text model maps units
```

**Proposed Solution:** Extract with context, return BOTH raw and standardized in one call

```python
# Modified LabResult model
class LabResult(BaseModel):
    # Raw fields (exactly as written)
    test_name_raw: str
    value_raw: str
    unit_raw: Optional[str]

    # Standardized fields (populated during extraction)
    test_name: str  # From lab_specs.json keys
    unit: str       # From lab_specs.json units

    # Validation
    @field_validator('test_name')
    def validate_standardized_name(cls, v):
        if v not in VALID_LAB_NAMES:
            return "$UNKNOWN$"
        return v
```

**Benefits:**
- ✂️ **Eliminates 2 LLM calls** - From 3-4 calls → 1 call per page
- ⚡ **Faster processing** - No separate standardization passes
- 🎯 **Better accuracy** - LLM sees reference ranges during extraction, can validate
- 🔍 **Still traceable** - Raw values preserved in separate fields

**How:** Pass lab_specs.json context in extraction prompt:
```python
prompt = f"""Extract lab results. Available standardized names:
{json.dumps(list(lab_specs.keys())[:50])}  # Top 50 most common
...
For each result, provide BOTH:
1. test_name_raw: exactly as written
2. test_name: closest match from standardized list (or $UNKNOWN$)
"""
```

---

### **Big Win #2: Simplified Self-Consistency with Majority Voting**

**Current Problem:** Generic wrapper + text-based LLM voting is expensive
```python
# Now: Thread pool + LLM call to vote on text differences
results = [extract() for _ in range(N)]
if results_differ:
    best = llm_vote(results)  # Another LLM call
```

**Proposed Solution:** Field-level majority voting (deterministic, fast)

```python
def self_consistency_majority(extract_fn, n: int) -> LabResult:
    """Deterministic majority voting on structured fields."""
    results = [extract_fn() for _ in range(n)]

    if n == 1:
        return results[0]

    # Field-by-field majority
    consensus = {}
    for field in LabResult.__fields__:
        values = [getattr(r, field) for r in results]
        consensus[field] = Counter(values).most_common(1)[0][0]

    return LabResult(**consensus)
```

**Benefits:**
- ✂️ **Eliminates voting LLM call** - Deterministic, no API needed
- ⚡ **Faster** - Simple counting vs LLM call
- 🎯 **More predictable** - Transparent logic
- 💰 **Cheaper** - No voting API costs
- 📊 **Better for structured data** - Field-level comparison vs text blob

**Optional Enhancement:** Only use LLM for complex ties (e.g., 3-way tie on critical field)

---

### **Big Win #3: Simplified Column Schema**

**Current Problem:** Confusing naming creates errors
```
test_name (raw?)
lab_name (standardized?)
lab_name_standardized (also standardized?)
unit, lab_unit, unit_normalized, lab_unit_standardized (which is which?)
```

**Proposed Solution:** Clear suffix convention
```python
COLUMN_SCHEMA = {
    # Raw extraction (exactly as in PDF)
    'test_name_raw': str,
    'value_raw': str,
    'unit_raw': str,
    'reference_range_raw': str,

    # Standardized (mapped to lab_specs.json)
    'test_name': str,  # Standardized name
    'unit': str,       # Standardized unit

    # Normalized (converted to primary units)
    'value_normalized': float,
    'unit_normalized': str,  # Always primary unit

    # Derived
    'lab_type': str,   # blood/urine/feces
    'date': date,
}
```

**Benefits:**
- 📝 **Clear intent** - Suffix tells you transformation level
- 🐛 **Fewer bugs** - Hard to use wrong column
- 📚 **Self-documenting** - Schema explains itself

---

### **Big Win #4: Consolidated Processing Pipeline**

**Current:** 9 passes with intermediate files
```
PDF → image → JSON → standardize names → standardize units
    → save CSV → merge all → normalize → deduplicate → export
```

**Proposed:** 5 passes, fewer intermediates
```
PDF → image → extract+standardize JSON → merge → normalize+deduplicate+export
```

**Changes:**
- **Combine passes 3-5:** Extract with standardization (one step)
- **Combine passes 7-9:** Normalize, deduplicate, export (vectorized together)
- **Remove intermediate CSVs:** Go straight from JSON → final CSV

**Benefits:**
- ⚡ **Faster** - Less I/O, fewer iterations
- 🧠 **Less memory** - Fewer intermediate DataFrames
- 🔧 **Simpler** - Fewer functions to maintain

---

### **Big Win #5: Smart Caching with Standardization Memory**

**Current Problem:** Standardization re-runs every time (no caching)

**Proposed Solution:** Cache standardization results
```python
# .cache/standardization_cache.json
{
    "raw_name": "GLICOSE -jejum-" -> "Blood - Glucose (Fasting)",
    "raw_unit": ("mg/dl", "Blood - Glucose") -> "mg/dL"
}

# Reuse across runs and documents
def standardize_with_cache(raw_name: str) -> str:
    if raw_name in cache:
        return cache[raw_name]
    result = llm_standardize([raw_name])
    cache[raw_name] = result
    return result
```

**Benefits:**
- ⚡ **Faster reruns** - Most labs seen before
- 💰 **Cheaper** - Fewer API calls on reruns
- 🎯 **More consistent** - Same raw value → same standardized value

---

## 📊 Impact Summary

| Metric | Current | Proposed | Improvement |
|--------|---------|----------|-------------|
| **LLM calls/page** | 3-4 | 1 | **66-75% reduction** |
| **Processing passes** | 9 | 5 | **44% reduction** |
| **Self-consistency cost** | N extractions + 1 vote | N extractions | **Eliminate voting call** |
| **Accuracy** | Good | Better | **+Context during extraction** |
| **Traceability** | Excellent | Excellent | **Maintained** |
| **Code complexity** | Medium | Low | **Simpler to maintain** |

---

## 🚀 Implementation Strategy

### **Phase 1: Core Simplification** (Biggest wins)
1. Modify `LabResult` model with `_raw` fields
2. Update extraction prompt to include lab_specs context
3. Implement majority voting for self-consistency
4. Test accuracy vs current approach

### **Phase 2: Pipeline Consolidation**
5. Combine standardization into extraction
6. Remove separate standardization functions
7. Consolidate normalize→dedupe→export passes

### **Phase 3: Polish**
8. Rename columns with clear suffixes
9. Add standardization caching
10. Update tests and documentation

---

## 🤔 Key Architectural Decisions

### **Keep Two-Phase (Raw + Standardized)**
✅ Even though we combine into one LLM call, we still record BOTH values
- Raw for traceability
- Standardized for analysis
- Best of both worlds

### **Keep Self-Consistency**
✅ N extractions still valuable, just simplify voting
- Visual extraction has inherent uncertainty
- Majority voting is cheaper and works well

### **Keep Per-Page Caching**
✅ JSON files enable resumable processing
- Critical for large batches
- Easy debugging (inspect failed pages)

---

## 📝 Detailed Architecture Analysis

### Current Pipeline Stages and Data Flow

#### 1. **PDF Processing Stage** (`process_single_pdf`)
**Flow:**
- PDF → PIL images (via pdf2image)
- Each page → preprocessed JPG (grayscale, contrast-enhanced, max 1200px width)
- Saved to: `{output_dir}/{pdf_stem}/{pdf_stem}.{page}.jpg`

**Caching:** Images are cached on disk - only regenerated if missing

#### 2. **Extraction Stage** (`extract_labs_from_page_image`)
**Flow:**
- Page image → Base64 encoded
- Vision model extracts structured data using function calling
- Returns `HealthLabReport` with nested `LabResult` objects
- Saved to: `{output_dir}/{pdf_stem}/{pdf_stem}.{page}.json`

**Key Features:**
- Uses Pydantic models for validation (`LabResult`, `HealthLabReport`)
- Extracts EXACTLY as written in PDF (no normalization)
- Handles numeric values, text values, reference ranges
- Tool-based extraction via OpenAI function calling

**Complexity Point:** This is wrapped in `self_consistency` function for accuracy

#### 3. **Self-Consistency Mechanism** (`self_consistency`)
**How it Works:**
- Runs extraction function N times (configurable via `N_EXTRACTIONS`)
- If N=1: Direct execution, no voting
- If N>1 and outputs differ:
  - Runs ThreadPoolExecutor with N concurrent tasks
  - Collects all results
  - If results differ, uses LLM to vote on best result
  - Voting prompt: "Select output most consistent with majority"
- Returns: (best_result, all_results)

**Used For:**
- Extraction from images (lines 160-167 in main.py)
- Could be used for other operations but currently only extraction

**Complexity:** Thread-based parallelism for extraction, then LLM voting if needed

#### 4. **Standardization Stage** (post-extraction)
**Two separate LLM calls:**

**A. Lab Names** (`standardize_lab_names`):
- Input: List of unique raw test names
- Output: Dictionary mapping raw → standardized names
- Uses: `lab_specs.json` keys as candidate list
- Unknown values → `$UNKNOWN$`

**B. Lab Units** (`standardize_lab_units`):
- Input: List of (raw_unit, standardized_lab_name) tuples
- Output: Dictionary mapping (raw_unit, lab_name) → standardized unit
- Uses: All units from `lab_specs.json` (primary_unit + alternatives)
- Context-aware: Uses lab name to infer missing units
- Unknown values → `$UNKNOWN$`

**Applied:** Lines 200-249 in main.py, after all pages extracted

#### 5. **Normalization Stage** (`apply_normalizations`)
**Operations:**
- Creates derived columns: `lab_name`, `lab_unit`, `lab_type`, `lab_name_slug`
- Unit conversions: Applies conversion factors from `lab_specs.json`
- Converts to primary units: `value_normalized`, `unit_normalized`, etc.
- Computes health status: `is_out_of_reference`, `is_in_healthy_range`

**Applied:** Line 417 in main.py, after merging all CSVs

#### 6. **Deduplication Stage** (`deduplicate_results`)
**Logic:**
- Groups by (date, lab_name)
- For each group with duplicates:
  - Prefers rows with primary unit
  - Otherwise picks first row
- Returns deduplicated DataFrame

**Applied:** Lines 420-423 in main.py

#### 7. **Export Stage**
- CSV: `{output_dir}/all.csv`
- Excel: Two sheets
  - "AllData": All results
  - "MostRecentByEnum": Most recent value per lab_name
- Plots: Time-series per lab with reference ranges

---

### Points of Complexity

#### **1. Self-Consistency Pattern**
**Why Complex:**
- Requires N concurrent API calls (ThreadPoolExecutor)
- Additional LLM call for voting if results differ
- Generic implementation accepts arbitrary functions
- Temperature injection logic for non-extraction functions

**Benefits:**
- Improves accuracy for critical extraction step
- Configurable (N_EXTRACTIONS=1 for speed)

**Cost:**
- N × extraction API calls
- 1 × voting API call (if needed)
- Thread management overhead

#### **2. Two-Phase Standardization**
**Current Approach:**
1. Extract raw data (no interpretation)
2. Separate LLM calls for names and units

**Why Complex:**
- Two additional LLM calls per document
- Unit standardization needs context (lab name)
- Batch processing to reduce API calls

**Benefits:**
- Perfect traceability (raw values preserved)
- Flexible standardization (can update without re-extraction)
- Context-aware unit mapping

**Alternative Considered:**
- Single-pass extraction with standardization
- Pro: Fewer API calls
- Con: Loses traceability, harder to debug

#### **3. Multiple Processing Passes Over Data**

**Pass 1:** PDF → Images (per-page caching)

**Pass 2:** Images → JSON extraction (per-page caching)

**Pass 3:** Standardize names (batch, per-document)

**Pass 4:** Standardize units (batch, per-document)

**Pass 5:** Save per-document CSV

**Pass 6:** Merge all CSVs

**Pass 7:** Apply normalizations (vectorized)

**Pass 8:** Deduplicate

**Pass 9:** Export (CSV, Excel, plots)

**Why Complex:**
- Multiple data transformations
- Switching between row-wise and vectorized operations
- Memory: Hold all data in RAM after merge

**Benefits:**
- Incremental caching (can resume)
- Parallel PDF processing (multiprocessing)
- Clear separation of concerns

#### **4. Existence Checks for Caching**
**Current Implementation:**
```python
if not jpg_path.exists():
    # Preprocess and save
if not json_path.exists():
    # Extract and save
```

**Known Issue (from TODO.md):**
- "Takes a long time to get to new labs (existence check is slow)"
- Checking existence for every page × every PDF
- File system I/O overhead

**Improvement Opportunity:**
- Check at document level first
- Batch existence checks
- Skip entire document if CSV exists (already implemented in recent changes)

#### **5. Unit Conversion Logic**
**Current Flow:**
1. Get unique lab names in DataFrame
2. For each lab name:
   - Get primary unit from config
   - For each unique unit in that lab:
     - Get conversion factor
     - Apply vectorized multiplication

**Nested Loops:**
- Outer: Lab names (vectorized within)
- Inner: Units per lab (vectorized within)

**Efficiency:** Vectorized operations where possible, but still nested iteration

---

### Areas Where Accuracy Could Be Improved

#### **1. Extraction Prompt Engineering**
**Current Prompt:** Very detailed with 8 critical rules
- Emphasizes EXACT copying
- Handles qualitative vs quantitative values
- Unit inference rules

**Potential Issues:**
- Complex prompt may be hard to follow consistently
- Rules for null units ("infer from lab name") could conflict with "exact copy"
- Multiple edge cases might confuse model

**Improvement Opportunities:**
- Simplify prompt: Focus on copy, defer interpretation
- A/B test different prompt structures
- Add few-shot examples in prompt

#### **2. Self-Consistency Voting**
**Current Voting Prompt:**
- "Select output most consistent with majority"
- Generic prompt (not domain-specific)

**Potential Issues:**
- Voting model may not understand medical context
- No guidance on what matters (values vs formatting)
- Returns verbatim output (requires parsing)

**Improvement Opportunities:**
- Medical-aware voting prompt
- Structured comparison criteria
- JSON-structured vote output

#### **3. Unit Standardization Context**
**Current Approach:**
- Provides (raw_unit, lab_name) pairs
- LLM infers missing units from lab name

**Potential Issues:**
- Inference could be wrong
- No visibility into reference ranges for validation
- Boolean/unitless distinction unclear

**Improvement Opportunities:**
- Include reference ranges in context
- Provide few-shot examples
- Validate against expected unit types

#### **4. Date Resolution**
**Priority:**
1. `collection_date` from page 1
2. `report_date` from page 1
3. Regex match in filename (YYYY-MM-DD)
4. None (warning)

**Potential Issues:**
- No validation that dates match across pages
- Filename date might be wrong
- TODO item: "Assert that metadata dates match file name"

**Improvement Opportunities:**
- Cross-validate dates across sources
- Check consistency across pages
- Warn on mismatches

#### **5. Reference Range Parsing**
**Current:**
- Extracts `reference_range` as string
- LLM parses min/max from string

**Potential Issues:**
- Complex range formats: "1 a 2/campo", ">5", "<0.5"
- Parsing errors could lose information
- No validation against healthy ranges

**Improvement Opportunities:**
- Specialized range parsing function
- Validate against config ranges
- Handle complex formats explicitly

---

### Error-Prone Areas

#### **1. Column Name Confusion**
**Issue:** Multiple similar column names
- `test_name` vs `lab_name` vs `lab_name_standardized`
- `unit` vs `lab_unit` vs `unit_normalized` vs `lab_unit_standardized`

**Impact:**
- Hard to track which column is which
- Easy to use wrong column in analysis

**Fix:** Standardize naming convention

#### **2. Pydantic Field Names vs Prompt**
**Issue:** Prompt says "Use `test_name` (NOT lab_name)" (line 173-176)
- Suggests confusion between schema and old versions

**Fix:** Update prompt to match current schema

#### **3. Self-Consistency Voting Parsing**
**Issue:** Voting result is unparsed text
- Needs JSON parsing
- Could return wrong format
- Fallback: Return first result

**Error Handling:** Good fallback logic exists

#### **4. Unit Inference Logic**
**Issue:** Standardization prompt says "infer from lab name" for null units
- Could introduce errors
- Conflicts with "exact copy" principle

**Fix:**
- Never infer during extraction
- Only infer during standardization (already done)

#### **5. Multiprocessing with Logging**
**Issue:** Multiprocessing Pool for PDFs
- Logging may interleave
- Hard to debug failures

**Mitigation:** Already has per-document logging with prefixes

---

## 📈 Complexity vs. Accuracy Trade-offs

| Feature | Complexity Cost | Accuracy Benefit | Recommendation |
|---------|----------------|------------------|----------------|
| Self-consistency (N>1) | High (N×API calls) | High (voting reduces errors) | Keep, make configurable ✅ |
| Two-phase standardization | Medium (2 LLM calls) | High (traceability) | Keep, optimize batching |
| Per-page caching | Low (file I/O) | High (resume capability) | Keep, optimize existence checks |
| Multiple passes | Medium (memory) | Medium (clarity) | Consider consolidation |
| Nested unit conversions | Low (vectorized) | High (correctness) | Keep as-is |
| Pydantic validation | Low (Python overhead) | High (data quality) | Keep ✅ |
| Generic self-consistency | Medium (abstraction) | Low (not reused) | Simplify to extraction-only |
| LLM voting | Medium (1 LLM call) | Medium (could be majority) | Consider simpler voting |

### Key Strengths
1. **Excellent traceability:** Raw values preserved
2. **Good caching:** Incremental processing
3. **Strong validation:** Pydantic models
4. **Flexible config:** Single JSON file
5. **Parallel processing:** PDFs in parallel

### Key Weaknesses
1. **Many LLM calls:** Extraction + 2 standardization + voting
2. **Complex column schema:** Confusing names
3. **Multiple passes:** Could be consolidated
4. **Slow existence checks:** File I/O overhead
5. **Generic abstractions:** Self-consistency not reused

### Recommended Priorities
1. **Cache standardization results** (across documents)
2. **Simplify column naming** (clarify schema)
3. **Optimize existence checks** (batch, document-level)
4. **Simplify self-consistency** (extraction-specific, no generic wrapper)
5. **Improve voting** (structured comparison, not text parsing)
