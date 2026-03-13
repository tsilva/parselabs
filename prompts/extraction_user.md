Please extract ALL lab test results from this medical lab report image.

CRITICAL: For EACH lab test you find, you MUST extract:
1. raw_lab_name - The test name EXACTLY as shown (required)
2. raw_section_name - The nearest visible section/header governing that row (copy EXACTLY as shown, or null if none is visible)
3. raw_value - The result value EXACTLY as shown (ALWAYS PUT THE RESULT HERE - whether numeric or text)
4. raw_lab_unit - The unit EXACTLY as shown (extract what you see, can be null if no unit)
5. raw_reference_range - The reference range text (if visible)
6. raw_reference_min and raw_reference_max - Parse the numeric bounds from the reference range
7. bbox_left, bbox_top, bbox_right, bbox_bottom - The result bounding box in normalized 0-1000 page coordinates

Extract test names, values, units, and reference ranges EXACTLY as they appear.
Pay special attention to preserving the exact formatting and symbols.

IMPORTANT FOR SECTION NAMES:
- Use the most specific visible section/header for the row
- If the page has nested headers, choose the innermost visible one that governs the row
- If no governing section/header is visible, set raw_section_name to null
- Prefer keeping the section/header in raw_section_name instead of repeating it inside raw_lab_name

IMPORTANT FOR BOUNDING BOXES:
- Use the full page image as the coordinate space
- 0 means the top/left edge, 1000 means the bottom/right edge
- The box should cover the visible region for that extracted result
- If you are not confident about the location, set all four bbox fields to null

CRITICAL: Extract EVERY lab test you see, including:
- Numeric results → Put in raw_value (examples: "5.2", "14.8", "0.75")
- Text-based qualitative results → Put in raw_value (examples: "NEGATIVO", "POSITIVO", "NORMAL", "AMARELA", "NAO CONTEM", "AUSENTE", "PRESENTE")
- Range results → Put in raw_value (examples: "1 a 2", "1-5 / campo", "0-3 / campo")

IMPORTANT: The raw_value field should contain the ACTUAL TEST RESULT, whether it's a number or text.
Do NOT put test results in the raw_comments field - that's only for additional notes.
Do NOT skip or omit text-based results - they are just as important as numeric results.

Also set page_has_lab_data:
- true if this page contains lab test results
- false if this is a cover page, instructions, or administrative content with no lab tests

BEFORE OUTPUTTING EACH RESULT, VERIFY:
✓ raw_lab_name contains ONLY the test name (no values, units, or ranges)
✓ raw_value contains ONLY the result (no test names or units)
✓ raw_lab_unit contains ONLY the unit (no values)
✓ No field contains text like "raw_value:", "raw_lab_unit:", etc.
✓ Bounding boxes use the normalized 0-1000 page coordinate system or are all null
