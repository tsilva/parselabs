# labs-parser

Scripts to extract data from health lab reports (portuguese)

# TODO

- Fix invalid lab spec mappings
- Fix invalid lab spec ranges
- Change lab names to match selfdecode's
- Dont use LLM for pairing lab specs when there is an exact match (remove dots)
- Validate inconsistencies (check explanations)
- Check for json pages with [] that dont belong to pages with n/a
- Search for two entries with the same name in a row and delete the second one
- Confirm that dates in files match file name

# Validados manualmente

- TSH
- T4 Livre
- Anti-Tireoglobulina
- Anti-TPO
