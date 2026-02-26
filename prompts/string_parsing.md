You are parsing lab test result strings into structured format.

For each string below, extract:
- lab_name_raw: The name of the lab test
- value_raw: Numeric value (null if text-only result like "Negative")
- lab_unit_raw: Unit of measurement (null if none)
- reference_range: Reference range text (null if none)
- reference_min_raw: Min reference value (null if not available)
- reference_max_raw: Max reference value (null if not available)
- comments: Any qualitative results or notes
- source_text: The original string

Input strings:
{string_results_json}

Return a JSON array of parsed lab results matching the LabResult schema.
Each result must have at minimum: lab_name_raw, value_raw, lab_unit_raw, and source_text fields.
