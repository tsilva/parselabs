# Eliminating lab_names_mappings.json

## The Core Problem

The mapping file is essentially a **brittle lookup table** that:
- Requires manual updates for every lab name variant
- Fails silently on unmapped values
- Doesn't leverage the AI that's already doing extraction

## Recommended Approach: Direct Standardized Extraction

### Phase 1: Eliminate the Mapping Step Entirely

1. **Modify the extraction prompt** to output standardized lab names directly
   - Load all existing `lab_name_enum` values from `lab_specs.json` at runtime
   - Pass this list to the extraction LLM as valid options
   - Update the Pydantic schema to use `Literal` type with all valid enums (or make it accept any string with validation logic)

2. **Two-tier extraction strategy**:
   - **Primary**: LLM tries to match to existing standardized names from `lab_specs.json`
   - **Fallback**: If truly novel lab, LLM generates a new standard name following conventions:
     - Prefix with lab type ("Blood - ", "Urine - ", etc.)
     - Use clear, standardized medical terminology
     - Apply unit suffix rules (e.g., "(%)" for percentage-based labs)

3. **Add a normalization verification step**:
   ```python
   def verify_standardization(raw_text: str, standardized_name: str,
                              available_enums: list[str]) -> dict:
       """Ask LLM to confirm the standardization is appropriate"""
       # Returns: {confirmed: bool, suggested_alternative: str, confidence: float}
   ```

### Phase 2: Self-Learning Cache (Optional Enhancement)

4. **Implement a learned mapping cache**:
   - SQLite database tracking: `(raw_lab_name, raw_unit) -> (standardized_enum, confidence, last_used, frequency)`
   - Populated automatically as documents are processed
   - Used for fast-path lookups on subsequent runs
   - Periodically exported to JSON for version control/review

5. **Add a review mechanism**:
   - Flag low-confidence mappings for manual review
   - Generate a report of new standardized names for approval
   - Update `lab_specs.json` for newly confirmed labs

## Alternative: Semantic Similarity Fallback

If you want determinism without always hitting the LLM:

1. Pre-compute embeddings for all standardized lab names in `lab_specs.json`
2. For extracted raw names, compute embedding and find nearest neighbor
3. If similarity > threshold (e.g., 0.85), use that match
4. Otherwise, call LLM for normalization
5. Cache all decisions

## Benefits of This Approach

- **No manual mapping maintenance** - system learns from lab_specs.json
- **More reliable** - AI understands medical terminology context, not just string matching
- **Handles variants naturally** - "HbA1c", "Hemoglobin A1c", "Glycated Hemoglobin" all map correctly
- **Self-documenting** - new labs follow same conventions automatically
- **Graceful degradation** - can review and correct standardizations post-processing

## Migration Path

1. Keep existing mapping file initially
2. Run new approach in parallel, comparing outputs
3. Build confidence via validation on historical data
4. Switch over when accuracy >= current system
5. Archive mapping file as reference

## Implementation Notes

### Current Flow (with mappings)
```
PDF → Transcription → Extraction (raw names) → Mapping lookup → Standardized enums
                                                     ↓
                                              lab_names_mappings.json
```

### Proposed Flow (direct standardization)
```
PDF → Transcription → Extraction with context → Standardized enums
                            ↑
                      lab_specs.json (valid options)
```

### Key Changes Required

1. **In extraction prompt**:
   - Include: "Use these standardized lab names: [list from lab_specs.json]"
   - Include: "If lab not in list, create new name following pattern: '{LabType} - {StandardName}'"

2. **In LabResult Pydantic model**:
   - Change `lab_name: str` to validate against known enums + convention rules
   - Add `lab_name_confidence: float` field
   - Add `is_novel_lab: bool` field

3. **Post-extraction validation**:
   - Check if extracted name follows conventions
   - Flag novel labs for review
   - Log all name mappings to cache

4. **Reporting**:
   - Generate summary of new labs discovered per run
   - Track mapping confidence over time
   - Alert on low-confidence standardizations
