Please extract ALL lab test results from this medical lab report text.

CRITICAL: For EACH lab test you find, you MUST extract:
1. lab_name_raw - The test name EXACTLY as shown (required)
2. value_raw - The result value EXACTLY as shown (ALWAYS PUT THE RESULT HERE - whether numeric or text)
3. lab_unit_raw - The unit EXACTLY as shown (extract what you see, can be null if no unit)
4. reference_range - The reference range text (if visible)
5. reference_min_raw and reference_max_raw - Parse the numeric bounds from the reference range
{std_reminder}
Extract test names, values, units, and reference ranges EXACTLY as they appear.
Pay special attention to preserving the exact formatting and symbols.

CRITICAL: Extract EVERY lab test you see, including:
- Numeric results → Put in value_raw (examples: "5.2", "14.8", "0.75")
- Text-based qualitative results → Put in value_raw (examples: "NEGATIVO", "POSITIVO", "NORMAL", "AMARELA", "NAO CONTEM", "AUSENTE", "PRESENTE")
- Range results → Put in value_raw (examples: "1 a 2", "1-5 / campo", "0-3 / campo")

IMPORTANT: The value_raw field should contain the ACTUAL TEST RESULT, whether it's a number or text.

Also set page_has_lab_data:
- true if this document contains lab test results
- false if this is a cover page, instructions, or administrative content with no lab tests

--- DOCUMENT TEXT ---
{text}
--- END OF DOCUMENT ---
