Please extract ALL lab test results from this medical lab report text.

CRITICAL: For EACH lab test you find, you MUST extract:
1. raw_lab_name - The test name EXACTLY as shown (required)
2. raw_value - The result value EXACTLY as shown (ALWAYS PUT THE RESULT HERE - whether numeric or text)
3. raw_lab_unit - The unit EXACTLY as shown (extract what you see, can be null if no unit)
4. raw_reference_range - The reference range text (if visible)
5. raw_reference_min and raw_reference_max - Parse the numeric bounds from the reference range
6. bbox_left, bbox_top, bbox_right, bbox_bottom - For text-only extraction, set all four to null because no page coordinates are available

Extract test names, values, units, and reference ranges EXACTLY as they appear.
Pay special attention to preserving the exact formatting and symbols.

CRITICAL: Extract EVERY lab test you see, including:
- Numeric results → Put in raw_value (examples: "5.2", "14.8", "0.75")
- Text-based qualitative results → Put in raw_value (examples: "NEGATIVO", "POSITIVO", "NORMAL", "AMARELA", "NAO CONTEM", "AUSENTE", "PRESENTE")
- Range results → Put in raw_value (examples: "1 a 2", "1-5 / campo", "0-3 / campo")

IMPORTANT: The raw_value field should contain the ACTUAL TEST RESULT, whether it's a number or text.

REFERENCE RANGE RULES:
- raw_reference_range should copy the full visible range text exactly
- raw_reference_min and raw_reference_max must contain only plain numbers
- "< 40" -> raw_reference_min=null, raw_reference_max=40
- "> 150" -> raw_reference_min=150, raw_reference_max=null
- "150 - 400" -> raw_reference_min=150, raw_reference_max=400
- "0.2 a 1.0" -> raw_reference_min=0.2, raw_reference_max=1.0
- If you cannot reliably parse numeric bounds, leave both as null

FIELD SEPARATION RULES:
- raw_lab_name must contain only the test name
- raw_value must contain only the result
- raw_lab_unit must contain only the unit
- bbox_left/bbox_top/bbox_right/bbox_bottom must all be null in this text-only mode
- Never embed field labels or multiple fields inside one field

COMMON COMPLEX SCENARIOS:

A) Qualitative and quantitative results on one line:
- Example: "Anticorpo Anti-HBs: POSITIVO - 864 UI/L"
- Extract TWO results:
  1. raw_lab_name="Anticorpo Anti-HBs (qualitative)", raw_value="POSITIVO", raw_lab_unit=null
  2. raw_lab_name="Anticorpo Anti-HBs (quantitative)", raw_value="864", raw_lab_unit="UI/L"

B) Numeric result followed by "=" and interpretation:
- Example: "ANTICORPO ANTI SCL 70  9= NR  < 19 U"
- raw_value must be "9"
- raw_comments should capture the interpretation text when possible
- The numeric value before "=" is the result

C) White blood cell differentials with absolute count and percentage:
- Example: "Neutrofilos 3,3 10^9/L 62,9 % 35.0 - 85.0"
- Extract TWO results:
  1. raw_value="3.3", raw_lab_unit="10^9/L"
  2. raw_value="62.9", raw_lab_unit="%", raw_reference_min=35.0, raw_reference_max=85.0
- Do not merge these into one result

D) Text-only qualitative rows:
- Example: "Urine Color: AMARELA"
- raw_value should be "AMARELA"
- raw_lab_unit should be null if no unit is shown

E) Range-like qualitative values:
- Example: "1 a 2", "1-5 / campo", "0-3 / campo"
- Put the exact text in raw_value

PAGE CLASSIFICATION:
- Set page_has_lab_data=true if this text contains any lab test results
- Set page_has_lab_data=false if this is administrative or non-lab content

Also set page_has_lab_data:
- true if this document contains lab test results
- false if this is a cover page, instructions, or administrative content with no lab tests

--- DOCUMENT TEXT ---
{text}
--- END OF DOCUMENT ---
