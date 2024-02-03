# labs-parser

Scripts to extract data from health lab reports (portuguese)

# TODO

- Fix invalid mappings
# @tsilva TODO: support mixed plots
- Ask GPT for anomalies and plot those
- Change lab names to match selfdecode's
- Ask GPT about last values
- Dont use LLM for pairing lab specs when there is an exact match (remove dots)
- Validate JSON against TXT (eg: 1998-03-03 - analises.001.txt broken)
- Validate inconsistencies (check explanations)
- Manual validation of final file
- Check for json pages with [] that dont belong to pages with n/a
- Mix with manual history