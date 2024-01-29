# labs-parser

Scripts to extract data from health lab reports (portuguese)

# TODO

- Dont use LLM for pairing lab specs when there is an exact match (remove dots)
- Validate JSON against TXT (eg: 1998-03-03 - analises.001.txt broken)
- Validate inconsistencies (check explanations)
- TSH plot
- Manual validation of final file
- Validate lab spec units
- Fix invalid mappings
- Fix invalid units
- Check for json pages with [] that dont belong to pages with n/a
- Are there missing TSH values? are they valid and with same unit?
- Plot T3/T4/etc
- Regen json files