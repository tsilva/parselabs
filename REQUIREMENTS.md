# Requirements

This document describes the functional requirements for Labs Parser — a tool that extracts laboratory test results from medical documents and converts them into structured, standardized data for review and analysis.

**Design principle:** Extraction is objective (what's on the page). Analysis is subjective (health status, custom ranges) and belongs in a separate tool.

---

R1. The system accepts PDF files containing laboratory test reports as input.

R2. The system handles both text-based PDFs and scanned/image-based PDFs (falling back to vision-based extraction when text extraction fails).

R3. Each page of a multi-page document is processed independently.

R4. The system supports batch processing of multiple PDFs in a single run.

R5. The system processes pages in parallel using a configurable number of workers.

R6. Already-processed documents are cached and skipped on subsequent runs unless the source file has changed.

R7. The system extracts structured lab results from each page, capturing: test name, numeric value, unit, reference range (min/max), and date.

R8. The system uses AI vision models to extract data directly from page images when text extraction is insufficient.

R9. The system supports a self-consistency mode where multiple independent extractions of the same page are compared to improve accuracy.

R10. Raw extracted values (name, value, unit) are preserved alongside their standardized counterparts for audit purposes.

R11. Collection or report dates are extracted from the document content, with fallback to date patterns in the filename.

R12. Limit values (e.g., "<0.05", ">738") are recognized and flagged as below-limit or above-limit rather than discarded.

R13. Extracted lab names are mapped to a canonical vocabulary of standardized test names (300+ labs across blood, urine, and feces categories).

R14. Unit notation is normalized to a consistent format (e.g., case, symbols, spacing) without changing the unit type.

R15. Values are converted from their extracted unit to a canonical primary unit using defined conversion factors.

R16. Each standardized lab name is prefixed with its lab type: "Blood - ", "Urine - ", or "Feces - ".

R17. Labs with percentage units must have standardized names ending in "(%)".

R18. Duplicate results for the same (date, lab_name) pair are deduplicated.

R19. The system validates extracted data for biological plausibility: negative values (where biologically impossible), values exceeding known biological limits, and percentage values outside valid bounds.

R20. The system validates inter-lab relationships where one result can be calculated from others (e.g., LDL cholesterol from the Friedewald formula) and flags mismatches beyond a configured tolerance.

R21. The system validates component-total constraints (e.g., a component value should not exceed its parent total).

R22. The system detects implausible rates of change between consecutive results for the same lab over time, using per-lab maximum daily change thresholds.

R23. The system detects format artifacts from extraction errors, such as value-reference concatenation (e.g., "52.6=1946") or excessive decimal places suggesting concatenated numbers.

R24. The system validates reference ranges for consistency (e.g., min should not exceed max) and flags values that deviate extremely far from the reference range.

R25. Each flagged result includes one or more reason codes describing why it was flagged, plus an adjusted confidence score reflecting the severity.

R26. The system provides an interactive web-based viewer for browsing, plotting, and reviewing extracted results.

R27. The viewer displays a data table of all results that can be filtered by: lab name, review status (needs review, abnormal, unhealthy, unreviewed), and a latest-only toggle.

R28. The viewer shows summary statistics: total results, unique tests, date range, items needing review, abnormal count, unhealthy count, and reviewed count.

R29. The viewer plots time-series charts for selected lab results, with visual bands for both the PDF-reported reference range and the configured healthy range.

R30. The viewer supports multi-lab selection with stacked subplots.

R31. The viewer displays the source page image alongside each result so the user can visually verify the extraction.

R32. The viewer provides a side-by-side comparison of raw extracted values versus their standardized counterparts.

R33. The viewer supports an Accept/Reject/Skip review workflow where review decisions are persisted across sessions.

R34. The viewer supports keyboard shortcuts for review actions (accept, reject, skip) and row navigation.

R35. The viewer supports exporting the currently filtered results as a CSV file.

R36. The viewer supports switching between multiple user profiles without restarting.

R37. Healthy reference ranges in the viewer are adjusted based on user demographics (age, gender) when provided.

R38. The primary output is a merged dataset combining results from all processed documents, available in both CSV and Excel formats.

R39. The output schema includes 17 columns: date, source file, page number, standardized lab name/value/unit, reference min/max from PDF, raw lab name/value/unit, confidence, review flags (needed/reason/confidence), lab type, and result index.

R40. Per-document intermediate outputs are preserved: preprocessed page images, per-page extracted data, and per-document CSV files.

R41. A snapshot of the lab specifications used during extraction is saved alongside the output for reproducibility.

R42. Excel output includes formatting: frozen header row, optimized column widths, and internal-only columns hidden from view.

R43. Users configure the system through profiles that specify: a name, input path (where PDFs live), output path (where results go), an optional file pattern filter, optional worker count override, and optional demographics (gender, date of birth, height, weight).

R44. AI model selection and API credentials are configured separately from profiles via environment variables.

R45. The lab specifications configuration defines: the canonical vocabulary of lab names, each lab's type and primary unit, alternative units with conversion factors, healthy reference ranges (with optional age/gender variants), biological plausibility limits, maximum daily change rates, inter-lab relationships, and LOINC codes.

R46. Every lab in the specifications must have a LOINC code for interoperability.

R47. The CLI supports: running all profiles or a specific one, listing available profiles, and overriding model, worker count, file pattern, and environment file.

R48. A validation/test suite verifies both configuration integrity (schema, LOINC codes, naming conventions, conversion factors, range consistency) and data integrity (no missing dates, no duplicates, prefix consistency, unit consistency, outlier detection).

R49. When extraction produces no results for a document, the user is prompted interactively to decide whether to reprocess it.

R50. The system logs processing details and reports any pages that failed extraction, with reasons.
