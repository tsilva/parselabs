# GOAL

Given a set a directory of pdf documents containing lab results (eg: blood tests, urine tests, etc), extract all with 100% accuracy and standardize all to same unit, and generate a single CSV containing all entries. The purpose is to be able to create a clean dataset where we can perform data analysis. For performing this process, accuracy is more important than cost, and cost is more important than speed, but all are important.

# REQUIREMENTS

- The system accepts PDF files containing laboratory test reports as input.
- The system should skip failures and continue extracting remaining documents. Running pipeline again extracts missing documents.
- Extractions are performed page by page to increase accuracy and reduce effort of retrying and resuming.
- The system uses LLMs to perform structured extraction, this can be from text if PDF allows it, otherwise with vision capbilities.
- Raw values are preserved in extraction for audit purposes.
- Document date should be the sample collection date found in the document, otherwise the document issue date, otherwise the first date found. The sooner in the document that date is found the more likely it is the real date. Once identified all pages are assumed as having same date.
- Lab names should be mapped to a canonical vocabulary of standardized test names to allow cross-referencing them.
- Each standardized lab name is prefixed with its lab type: "Blood - ", "Urine - ", or "Feces - ".
- Standardized lab names with percentage units must have standardized names ending in "(%)" to distinguish them from their non-fraction counterparts.
- Duplicate results for the same date, lab_name should be flagged, this should never happen, if it happens its most likely an extraction issue.
- The target extraction folder should not have duplicate documents, use file hash to determine this, if extracting documents from raw folder and a document with this hash is found in output folder skip and log warning.
- The output folder should contain a folder per extracted document with the document's name followed by its file hash, folder should contain original pdf plus the extracted data.
- The system validates extracted data for biological plausibility: negative values (where biologically impossible), values exceeding known biological limits, and percentage values outside valid bounds.
- The system validates inter-lab relationships where one result can be calculated from others (e.g., LDL cholesterol from the Friedewald formula) and flags mismatches beyond a configured tolerance.
- The system validates component-total constraints (e.g., a component value should not exceed its parent total).
- The system detects implausible rates of change between consecutive results for the same lab over time, using per-lab maximum daily change thresholds.
- The system detects format artifacts from extraction errors, such as value-reference concatenation (e.g., "52.6=1946") or excessive decimal places suggesting concatenated numbers.
- The system validates reference ranges for consistency (e.g., min should not exceed max) and flags values that deviate extremely far from the reference range.
- Each flagged result includes one or more reason codes describing why it was flagged.
- The system provides an interactive web-based viewer for browsing, plotting, and reviewing extracted results.
- The viewer displays a data table of all results that can be filtered by: lab name, review status (needs review, abnormal, unhealthy, unreviewed), and a latest-only toggle.
- The viewer shows summary statistics: total results, unique tests, date range, items needing review, abnormal count, unhealthy count, and reviewed count.
- The viewer plots time-series charts for selected lab results, with visual bands for both the PDF-reported reference range and the configured healthy range.
- The viewer supports multi-lab selection with stacked subplots.
- The viewer displays the source page image alongside each result so the user can visually verify the extraction.
- The viewer provides a side-by-side comparison of raw extracted values versus their standardized counterparts.
- The viewer supports an Accept/Reject/Skip review workflow where review decisions are persisted across sessions.
- The viewer supports keyboard shortcuts for review actions (accept, reject, skip) and row navigation.
- Healthy reference ranges in the viewer are adjusted based on user demographics (age, gender) when provided.
- The primary output is a merged dataset combining results from all processed documents, available in both CSV and Excel formats.
- The output schema includes 17 columns: date, source file, page number, standardized lab name/value/unit, reference min/max from PDF, raw lab name/value/unit, confidence, review flags (needed/reason/confidence), lab type, and result index.
- Per-document intermediate outputs are preserved: preprocessed page images, per-page extracted data, and per-document CSV files.
- Excel output includes formatting: frozen header row, optimized column widths, and internal-only columns hidden from view.
-  Users configure the system through profiles that specify: a name, input path (where PDFs live), output path (where results go), an optional file pattern filter, optional worker count override, and optional demographics (gender, date of birth, height, weight).
- AI model selection and API credentials are configured separately from profiles via environment variables.
- The lab specifications configuration defines: the canonical vocabulary of lab names, each lab's type and primary unit, alternative units with conversion factors, healthy reference ranges (with optional age/gender variants), biological plausibility limits, maximum daily change rates, inter-lab relationships, and LOINC codes.
- Every lab in the specifications must have a LOINC code for interoperability.
- The CLI supports: running all profiles or a specific one, listing available profiles, and overriding model, worker count, file pattern, and environment file.
- A validation/test suite verifies both configuration integrity (schema, LOINC codes, naming conventions, conversion factors, range consistency) and data integrity (no missing dates, no duplicates, prefix consistency, unit consistency, outlier detection).
- The system logs processing details and reports any pages that failed extraction, with reasons.
