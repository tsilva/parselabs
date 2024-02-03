# labs-parser

Scripts to extract data from health lab reports (portuguese)

# TODO

- Fix invalid lab spec mappings
- Fix invalid lab spec ranges
- Change lab names to match selfdecode's
- Dont use LLM for pairing lab specs when there is an exact match (remove dots)
- Validate JSON against TXT (eg: 1998-03-03 - analises.001.txt broken)
- Validate inconsistencies (check explanations)
- Check for json pages with [] that dont belong to pages with n/a
