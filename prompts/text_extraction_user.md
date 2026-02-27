Please extract ALL lab test results from this medical lab report text.

CRITICAL: For EACH lab test you find, you MUST extract:
1. raw_lab_name - The test name EXACTLY as shown (required)
2. raw_value - The result value EXACTLY as shown (ALWAYS PUT THE RESULT HERE - whether numeric or text)
3. raw_lab_unit - The unit EXACTLY as shown (extract what you see, can be null if no unit)
4. raw_reference_range - The reference range text (if visible)
5. raw_reference_min and raw_reference_max - Parse the numeric bounds from the reference range

Extract test names, values, units, and reference ranges EXACTLY as they appear.
Pay special attention to preserving the exact formatting and symbols.

CRITICAL: Extract EVERY lab test you see, including:
- Numeric results → Put in raw_value (examples: "5.2", "14.8", "0.75")
- Text-based qualitative results → Put in raw_value (examples: "NEGATIVO", "POSITIVO", "NORMAL", "AMARELA", "NAO CONTEM", "AUSENTE", "PRESENTE")
- Range results → Put in raw_value (examples: "1 a 2", "1-5 / campo", "0-3 / campo")

IMPORTANT: The raw_value field should contain the ACTUAL TEST RESULT, whether it's a number or text.

Also set page_has_lab_data:
- true if this document contains lab test results
- false if this is a cover page, instructions, or administrative content with no lab tests

--- DOCUMENT TEXT ---
{text}
--- END OF DOCUMENT ---
