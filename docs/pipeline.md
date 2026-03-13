# Current Pipeline

Based on the current code, not prior documentation:

## Subsystem Layout

- `parselabs/runtime.py` is now the shared bootstrap surface for profile resolution, runtime context creation, OpenAI client caching, and runtime path access.
- `parselabs/store.py` is now the shared filesystem/state surface for hashed document discovery, page JSON persistence, review actions, and legacy merged-review CSV fallback.
- `parselabs/dataset.py` is now the shared data surface for review/export row builders, schema metadata, normalization/standardization/validation exports, and integrity-report helpers.
- `parselabs/review.py` is now the shared UI helper surface for bbox scaling, source-page overlays, reference formatting, and results-explorer dataframe loading.
- Legacy public modules such as `parselabs.profiles` and `parselabs.documents` remain import-compatible as thin shims over the new subsystem modules.

1. Profile/runtime setup.
- `parselabs/cli.py` dispatches the top-level `parselabs` command into `extract`, `review`, and `admin` flows.
- Extraction-mode flags are still parsed in `parselabs/pipeline.py`.
- It resolves the selected profile through `RuntimeContext.from_profile(...)` in `parselabs/runtime.py`.
- For normal extraction runs, it requires input path, output path, API settings, and logging.
- It copies the active `lab_specs.json` into the output directory so the run remains reproducible.

2. API validation before any file work.
- `run_for_profile()` calls `validate_api_access(...)` before PDF discovery.
- That sends a minimal chat completion using the configured extraction model.
- If the key is invalid, the model is unavailable, the endpoint is wrong, or the server times out, the run stops immediately.
- This is a hard preflight check, not a best-effort warning.

3. PDF discovery.
- PDF enumeration is delegated to `discover_pdf_files(...)` in `parselabs/store.py`.
- It lists only top-level files in the input directory.
- It filters them against `input_file_regex`, case-insensitively.
- Missing directories, permission problems, and other OS errors are converted into pipeline errors.

4. Exact-content deduplication for the current run.
- `run_pipeline_for_pdf_files()` calls `_prepare_pdf_run(...)`.
- That hashes every discovered PDF with SHA-256 and keeps the first 8 hex chars.
- If two input files have identical content, only the first one is processed.
- Later duplicates are logged as skipped duplicates.
- There is no persisted run manifest cache in the current pipeline.

5. Output path derivation.
- Each unique PDF is assigned a hashed output directory: `{stem}_{hash}`.
- The per-document CSV path is derived from that directory as `{stem}.csv`.
- Hashes prevent collisions when different source files share the same stem.
- This path logic is shared with `parselabs/store.py`.

6. Parallel document processing.
- Unique PDFs are processed through `_process_pdfs_in_parallel(...)`.
- Worker count is `min(config.max_workers, len(pdfs_to_process))`.
- A single-worker run stays in-process; multi-worker runs use `multiprocessing.Pool`.
- Logging is initialized inside each worker.
- Each worker calls `process_single_pdf(...)`.

7. Per-document setup inside a worker.
- `process_single_pdf(...)` creates the hashed document directory if needed.
- It copies the original source PDF into that directory.
- That copied PDF becomes the local artifact used for page conversion and later review.
- The worker keeps track of page-level extraction failures separately from total document failure.

8. PDF conversion and text extraction bootstrap.
- The PDF is converted to PIL images with `pdf2image.convert_from_path(...)`.
- In parallel with image-based work, the runtime runs `pdftotext -layout` once for the document.
- The extracted text is split into page-sized chunks using form-feed boundaries.
- If `pdftotext` fails, the page text list becomes empty strings and extraction can still continue via vision.

9. Per-page cached artifact setup.
- For each page, the runtime builds canonical image paths like `{stem}.{page:03d}.jpg` and `{stem}.{page:03d}.fallback.jpg`.
- It generates the primary and fallback page image variants only if they do not already exist.
- Primary and fallback images are cached on disk in the document folder.
- This means page images are reused on reruns even when extraction is retried.

10. Page JSON reuse rules.
- For each page, `_extract_or_load_page_data(...)` checks whether `{stem}.{page:03d}.json` already exists.
- It reads the payload through `read_page_payload(...)`.
- Existing page JSON is reused only if `is_page_payload_reusable(...)` returns true.
- Right now that means the payload must be valid JSON and must not be marked `_extraction_failed`.
- Failed or unreadable page JSON is intentionally re-extracted on the next run.

11. Deterministic per-page routing.
- The page first tries text extraction if the page text has at least 80 non-whitespace characters.
- That uses `extract_labs_from_text(...)`.
- If the text result is weak, failed, or empty on a likely lab page, the runtime falls back to vision.
- Vision first uses the primary page image with `extract_labs_from_page_image(...)`.
- If that still looks unusable, the runtime retries once with the fallback image.
- The runtime does not compare candidates; it follows a fixed fallback chain.

12. Extraction payload persistence.
- After extraction, the runtime adds `source_file` programmatically because the model cannot know it.
- Each extracted row can now also carry `raw_section_name`, copied directly from the governing section/header visible on the page.
- If extraction failed and returned no lab rows, `_ensure_extraction_failure_placeholder(...)` inserts a synthetic placeholder row.
- Every useful outcome is saved as page JSON: extracted rows, explicit no-lab pages, and explicit extraction failures.
- Completely empty non-final outcomes are not cached, so they will be retried later.

13. Page-level metadata enrichment.
- Extracted result rows get `result_index`, `page_number`, and `source_file`.
- `result_index` tracks the row position within the page JSON’s `lab_results` list.
- `page_number` is 1-based.
- These fields let the review UI write decisions back to the exact JSON row that produced the review row.

14. Per-document review CSV rebuild from page JSON.
- After page extraction finishes, `process_single_pdf(...)` always calls `rebuild_document_csv(...)`.
- That means the document CSV is derived from current page JSON, not treated as stored truth.
- `build_document_review_dataframe(...)` loads one review row per extracted JSON result.
- Empty documents still get a stable empty review schema.

15. Review-row loading and flattening.
- `load_document_review_rows(...)` scans all page JSON files in a processed document directory.
- It flattens each page’s `lab_results` list into rows.
- It carries through raw fields, bounding boxes, `review_status`, and `review_completed_at`.
- If a page payload has `_extraction_failed=True`, every row from that page is marked review-needed and gets `EXTRACTION_FAILED` appended to `review_reason`.
- The document date is taken from the first usable page payload, with fallback to a `YYYY-MM-DD` token in the filename stem.

16. Shared row preparation pipeline.
- Both review-mode rows and export-mode rows now go through one shared function: `prepare_rows(...)` in `parselabs/rows.py`.
- The caller selects `mode="review"` or `mode="export"`.
- Optional review-status filtering can happen first.
- Then the same standardization, normalization, ambiguity-flagging, and validation stack is applied, with mode-specific behavior after that.

17. Cached standardization.
- `apply_cached_standardization(...)` standardizes raw lab names using `standardize_lab_names(...)`.
- Name standardization is now section-aware: it prefers a cache key built from `(raw_lab_name, raw_section_name)` when a section/header was extracted for the row.
- Legacy bare-name cache keys remain as a fallback only for rows that have no extracted section name.
- It standardizes units using `standardize_lab_units(...)` only after the lab name is known.
- Unknown name mappings become `$UNKNOWN$`.
- Blank raw units are only inferred from the lab spec primary unit for a narrow safe set: `boolean`, `pH`, and `unitless`.
- After unit mapping, the code remaps percentage-vs-absolute sibling analytes using the standardized unit as the tie-breaker.

18. Deterministic normalization.
- `apply_normalizations(...)` runs before review/export divergence.
- This is the objective cleanup layer: numeric parsing, comparison-operator handling, unit conversion, reference normalization, and related mechanical transforms.
- It is intentionally not a subjective correction layer.
- The public export aliases like `lab_name`, `value`, `lab_unit`, `reference_min`, and `reference_max` are added later, after normalization.

19. Review ambiguity flags.
- `_flag_review_ambiguities(...)` adds reviewer-facing flags without changing extracted meaning.
- It marks unknown lab mappings with `UNKNOWN_LAB_MAPPING`.
- It marks unknown or missing publishable units with `UNKNOWN_UNIT_MAPPING`.
- It marks unresolved percentage-vs-absolute cases with `AMBIGUOUS_PERCENTAGE_VARIANT`.
- It marks suspicious normalized reference ranges with `SUSPICIOUS_REFERENCE_RANGE`.

20. Review-mode behavior.
- In `mode="review"`, every row remains visible.
- The code flags duplicates with `flag_duplicate_entries(...)`, but does not deduplicate them away.
- It adds export-style alias columns so the UI and CSV have a stable shape.
- It then runs `ValueValidator.validate(...)`.
- The result becomes the per-document review CSV and the basis for the reviewer UI.

21. Export-mode behavior.
- In `mode="export"`, unresolved rows are filtered out first by `_filter_exportable_rows(...)`.
- Rows with unknown standardized labs or missing/non-publishable primary units are removed.
- Duplicate-looking rows are still flagged first.
- If lab specs exist, `deduplicate_results(...)` then removes publish-time duplicates.
- Export aliases are added.
- Validation runs after deduplication.

22. Value validation.
- `ValueValidator` is shared by both review mode and export mode.
- It checks biological plausibility, inter-lab relationships, component-total constraints, temporal consistency, format artifacts, and reference-range consistency.
- It appends reason codes into `review_reason` and sets `review_needed=True` where needed.
- It also records validation stats such as total flagged rows and counts by reason.

23. Corpus-level result returned by the extraction pipeline.
- `run_pipeline_for_pdf_files(...)` returns a `PipelineRunResult`.
- Its `final_df` is not the merged review CSV snapshot.
- It is the accepted-row export dataframe built by `_build_final_export_from_document_dirs(...)` with `allow_pending=True`.
- That helper rebuilds document CSVs from JSON again, loads only rows with `review_status == accepted`, applies the shared export path, and returns the final export-shaped dataframe.

24. What `all.csv` and `all.xlsx` contain on the first export pass.
- `run_for_profile()` does not write `pipeline_result.final_df` to disk.
- Instead, it reloads the per-document CSVs and merges them with `_build_merged_review_dataframe_from_csv_paths(...)`.
- It writes that merged review-state dataframe to `output/all.csv` and `output/all.xlsx`.
- So the first export pass still writes the merged review-state dataframe, not accepted-only reviewed truth.

25. End-of-run standardization auto-refresh.
- After the first export pass, the pipeline scans merged review rows for uncached standardization names and unit pairs.
- By default, it runs one in-process cache refresh pass using the same OpenRouter credentials and extraction model as the active profile.
- The shared refresh helper updates raw-name mappings first, using the extracted `raw_section_name` as part of the name-cache key when available, then rescans unit pairs using the newly resolved standardized names before calling the unit standardizer.
- This means a row that was `$UNKNOWN$` on the first pass can still contribute a unit mapping in the same automatic refresh cycle.
- If the refresh adds any cache entries, the pipeline rebuilds per-document CSVs and merged outputs from persisted page JSON only.
- It does not re-extract PDFs and does not repeat extraction API calls.
- If `--no-auto-standardize` is passed, the refresh is skipped and the pipeline logs a final manual-fallback summary instead.
- If refresh calls fail or some mappings remain unresolved, the run keeps the best-effort outputs and ends with a warning summary instead of failing.

26. Reviewed-JSON rebuild mode.
- If `--rebuild-from-json` is passed, extraction is skipped entirely.
- `_run_reviewed_json_rebuild(...)` rebuilds every per-document CSV from page JSON, writes merged `all.csv` and `all.xlsx`, and then runs the same optional post-export auto-refresh step.
- This path regenerates review outputs from existing reviewed JSON state while keeping the auto-refresh behavior consistent with normal extraction runs.

27. Strict reviewed-truth helpers still exist separately.
- `build_final_output_dataframe_from_reviewed_json(...)` and related helpers build accepted-row export data from reviewed JSON.
- In strict mode, `ensure_document_fixture_ready(...)` blocks promotion if any rows are still pending or any missing-row markers remain unresolved.
- `--allow-pending` affects those reviewed-truth helpers.
- It does not change what the normal merged `all.csv` export contains.

28. Canonical source of truth.
- The canonical persisted intermediate is the per-page JSON in each hashed document directory.
- Per-document CSVs are derived review snapshots.
- Top-level `all.csv` and `all.xlsx` are merged review-state outputs, potentially rebuilt once more after automatic standardization refresh.
- Accepted-only final export data is computed through a separate export path and returned by helpers, but not written by the normal extraction run.
