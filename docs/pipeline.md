# Current Pipeline

This document describes the current extraction, review, and export pipeline.

## Subsystem Layout

- `parselabs/runtime.py` is the shared bootstrap surface for profile resolution, runtime context creation, OpenAI client caching, and runtime path access.
- `parselabs/store.py` is the shared filesystem/state surface for hashed document discovery, page JSON persistence, and review actions.
- `parselabs/rows.py` is the shared review/export row-building surface; `parselabs/dataset.py` owns integrity reports.
- `parselabs/review.py` owns shared UI presentation helpers, while `parselabs/review_data.py` owns results-explorer dataframe loading.
- `frontend/` owns the light Svelte Results and Review screens. `parselabs/ui.py` launches the loopback-only FastAPI app in `parselabs/dashboard.py`, serving the compiled `static/dashboard/` assets and one selected profile's review API.
- The dashboard loads rows through `load_results_dataframe(..., include_document_key=True)`, preserving each hashed document directory in API identities so reports with the same filename remain distinct. Source pages and bounding boxes use the shared review/store helpers; decisions use `ReviewTarget` and `apply_review_action_for_target(...)` for that exact document. Review uses the original extracted value and reference; Results uses standardized values with raw fallbacks for unmapped units. A session token protects writes, stale status preconditions prevent overwrites, and source access is contained within the profile's output root. Refresh reloads page JSON; each decision reads and updates only its source page without rebuilding the entire corpus or derived CSV/XLSX exports. Other callers retain the existing row schema by default.
- `parselabs/admin/` owns installable maintenance-command implementations; `utils/*.py` contains compatibility entry points only.

1. Profile/runtime setup.
- `parselabs/cli.py` dispatches the top-level `parselabs` command into `extract`, `review`, and `admin` flows.
- Extraction-mode flags are parsed in `parselabs/pipeline.py`.
- It resolves the selected profile through `RuntimeContext.from_profile(...)` in `parselabs/runtime.py`.
- For normal extraction runs, it requires input path, output path, API settings, and logging.
- Console logging defaults to a quiet normal mode: only user-facing phase notices, summarized action-needed warnings, errors, progress bars, and the final summary are shown.
- Detailed INFO diagnostics are still written to `output/logs/info.log`, and errors are also written to `output/logs/error.log`; `--verbose`, `--debug`, and `--quiet` adjust only console verbosity.
- It copies the active `lab_specs.json` into the output directory so the run remains reproducible.

2. API validation before any file work.
- `run_for_profile()` calls `validate_api_access(...)` before PDF discovery.
- That sends a minimal chat completion using the configured extraction model.
- The preflight allows 120 seconds for AgentBridge model startup and queueing. SDK retries are disabled for this check because a timed-out request may still occupy a Codex worker.
- Validation and vision extraction explicitly request high reasoning effort for `codex/gpt-6.1-sol`.
- If the key is invalid, the model is unavailable, the endpoint is wrong, or the server times out, the run stops immediately.
- This is a hard preflight check, not a best-effort warning.

3. PDF discovery.
- PDF enumeration is delegated to `discover_pdf_files(...)` in `parselabs/store.py`.
- It lists only top-level files in the input directory.
- It filters them against `input_file_regex`, case-insensitively.
- Missing directories, permission problems, and other OS errors are converted into pipeline errors.

4. Exact-content deduplication and reusable-output scan for the current run.
- `run_pipeline_for_pdf_files()` calls `_prepare_pdf_run(...)`.
- That shows a `Scanning PDFs` progress bar in normal console output, hashes every discovered PDF with SHA-256, and keeps the first 8 hex chars.
- If two input files have identical content, only the first one is processed.
- Later duplicates are logged as skipped duplicates.
- Each unique PDF is then checked for a reusable per-document CSV plus reusable page JSON for every known page.
- Fully reusable documents are carried forward as cached CSV paths and do not enter the worker pool.
- There is no persisted run manifest cache in the current pipeline.

5. Output path derivation.
- Each unique PDF is assigned a hashed output directory: `{stem}_{hash}`.
- The per-document CSV path is derived from that directory as `{stem}.csv`.
- Hashes prevent collisions when different source files share the same stem.
- This path logic is shared with `parselabs/store.py`.

6. Parallel document processing.
- Only unique PDFs that were not fully reusable in the preflight scan are processed through `_process_pdfs_in_parallel(...)`.
- Worker count is `min(config.max_workers, len(pdfs_to_process))`.
- A single-worker run stays in-process; multi-worker runs use `multiprocessing.Pool`.
- Logging is initialized inside each worker with the same console verbosity mode as the parent run.
- The `Processing PDFs` progress bar is created after the scan and its total is only the documents that require extraction/rebuild work.
- The progress bar shows a compact `active:` postfix with the document names currently assigned to workers.
- Each worker calls `process_single_pdf(...)`.

7. Per-document setup inside a worker.
- `process_single_pdf(...)` creates the hashed document directory if needed.
- It copies only the original PDF contents into that directory, without propagating restrictive filesystem flags or unrelated metadata.
- That copied PDF becomes the local artifact used for page conversion and later review.
- The worker keeps track of page-level extraction failures separately from total document failure.
- Expected vision API failures are converted to `ExtractionAPIError` and reported as document failures with the document name and failure reason; unexpected internal runtime errors propagate instead of being folded into failed-page output.

8. PDF conversion bootstrap.
- The PDF is converted to PIL images with `pdf2image.convert_from_path(...)`.
- Those page images are the only extraction input now.
- This keeps every extraction path tied to a source image so rows can carry bounding boxes.
- There is no `pdftotext` routing or text-only extraction pass in the current pipeline.

9. Per-page cached artifact setup.
- For each page, the runtime builds canonical image paths like `{stem}.{page:03d}.jpg` and `{stem}.{page:03d}.fallback.jpg`.
- It generates the primary and fallback page image variants only if they do not already exist.
- Both variants now add a uniform white page border before resize/contrast adjustments so edge-clipped scan text gets margin for vision extraction.
- Primary and fallback images are cached on disk in the document folder.
- This means page images are reused on reruns even when extraction is retried.

10. Page JSON reuse rules.
- For each page, `_extract_or_load_page_data(...)` checks whether `{stem}.{page:03d}.json` already exists.
- It reads the payload through `read_page_payload(...)`.
- Existing page JSON is reused only if `is_page_payload_reusable(...)` returns true.
- Right now that means the payload must be valid JSON and must not be marked `_extraction_failed`.
- Failed or unreadable page JSON is intentionally re-extracted on the next run.

11. Deterministic per-page routing.
- Every page first uses the primary page image with `extract_labs_from_page_image(...)`.
- Any extracted payload that contains lab rows but lacks a complete four-value bbox on even one row is treated as malformed output.
- Any extracted payload whose bbox geometry looks like a narrow table-value column instead of a horizontal row region is also treated as malformed output.
- If that result is weak, failed, or empty on a likely lab page, the runtime retries once with the fallback image.
- The runtime does not compare candidates; it follows a fixed vision-only fallback chain.

12. Extraction payload persistence.
- After extraction, the runtime adds `source_file` programmatically because the model cannot know it.
- Validated fresh extraction excludes explicitly labelled normal/control PT/APTT measurements, which describe control specimens rather than the patient. Patient clotting times and qualitative `Normal` results remain. The vision prompt also excludes these controls and preserves an inline result unit when it conflicts with a shared heading, recording the conflict for review instead of inferring a correction.
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
- Before standardization, the row builder now backfills missing `raw_section_name` values for clearly urine-context pages using deterministic page-level label heuristics.
- This protects ambiguous bare labels such as `Glicose`, `LEUCOCITOS`, and `Eritrocitos` from being misread as blood analytes when the model omitted the visible urine section header.

16. Shared row preparation pipeline.
- Both review-mode rows and export-mode rows now go through one shared function: `prepare_rows(...)` in `parselabs/rows.py`.
- The caller selects `mode="review"` or `mode="export"`.
- Optional review-status filtering can happen first.
- Then the same standardization, normalization, ambiguity-flagging, and validation stack is applied, with mode-specific behavior after that.

17. Cached standardization.
- `apply_cached_standardization(...)` standardizes raw lab names using `standardize_lab_names(...)`.
- Name standardization is now section-aware: it prefers a cache key built from `(raw_lab_name, raw_section_name)` when a section/header was extracted for the row.
- Older caches that still store contextual mappings as `section - raw_lab_name` are now read transparently alongside the newer `raw_lab_name|section` format.
- Contextual lookup now also uses a folded compatibility pass that collapses stylized spaced-letter headers and accent/OCR variants before giving up on the cache.
- Leading dagger/double-dagger footnote markers are ignored for name/section lookup only. Raw names and section text remain literal, and conflicting folded aliases remain unresolved.
- Bare-name fallback is still guarded: it is only reused when the cache has no conflicting contextual mappings for that raw label.
- Unit-cache keys are normalized on both the stored and queried sides for multiline whitespace, accented `leucócitos`/`células` count labels and equivalent micro symbols. Conflicting normalized aliases remain unresolved; distinct unit scales and symbols stay distinct, and raw source text is unchanged.
- Explicit conversion-unit matching also folds those count-label accents, so remapping nRBC ratios from the absolute-count assay to the separate `/100 WBC` assay does not lose a supported denominator.
- When the same raw assay appears on one page as both a numeric result and a qualitative interpretation, the row builder now probes explicit `(quantitative)` / `(qualitative)` cache keys first before falling back to the undecorated raw name.
- A generic adjacent `Resultado`/`Result` numeric row can inherit the preceding qualitative assay's context for cache lookup while preserving its objective raw label in audit columns.
- It standardizes units using `standardize_lab_units(...)` only after the lab name is known.
- Unknown name mappings become `$UNKNOWN$`.
- Blank raw units are only inferred from the lab spec primary unit for a narrow safe set: `boolean`, `pH`, `unitless`, `mm/h`, and `titer`. Explicit dilution results on titer-backed assays bypass stale index/boolean unit-cache mappings.
- For percentage-vs-absolute sibling analytes, explicit non-percent raw units now override stale `%` cache entries so absolute differential counts do not stay pinned to percentage variants.
- A percentage-shaped value and reference interval can resolve a missing unit when they decisively match the percentage sibling. An explicit printed concentration unit is preserved even when its value or interval looks like a percentage; the discrepancy is flagged for source review.
- For percentage-vs-absolute sibling analytes, blank raw units can also be inferred from the report's own extracted reference ranges when those ranges clearly match one sibling variant over the other.
- After unit mapping, the code remaps percentage-vs-absolute sibling analytes using the standardized unit as the tie-breaker.

18. Deterministic normalization.
- `apply_normalizations(...)` runs before review/export divergence.
- This is the objective cleanup layer: numeric parsing, comparison-operator handling, unit conversion, reference normalization, and related mechanical transforms.
- Portuguese non-reactive qualitative results such as `Não reativo` are normalized to boolean false for boolean-backed assays.
- `Normal` and uncertain interpretations such as `DUVIDOSO` do not establish absence. Unclassified qualitative results keep a null numeric value, the complete `value_text`, and the `text` scale. Four-digit urine specific gravity notation such as `1012` normalizes to `1.012`; the source notation remains unchanged.
- Reported text and intervals are preserved in `value_text`, alongside any derived numeric/boolean `value`. Textual sediment observations retain their reported wording without an invented cell count. Quantitative assays with only a negative label no longer become zero concentrations; configured qualitative ANA/CRP/MPO/PR3 siblings carry boolean interpretations separately. Urine-strip grades such as `Contém(+--)` preserve their full grading text.
- Numeric annotations and approximations such as `9= NR` and `Ca. 125` also retain their complete source text alongside the parsed number. A bare `U` remains `U`, without inventing international calibration or a volume denominator. Legacy CH50 `U C'H 50` remains unconverted because no validated conversion to modern `U/mL` is configured.
- Antibody assay `U/mL`, `U/L` and `kU/L` units are preserved without assuming international calibration. Conversely, explicit `IU/mL` cannot become a dimensionless index. Unverified unit aliases for SSA, anti-tTG IgA, rheumatoid factor and thyroid antibodies have been removed.
- Explicit MCHC `%` and reticulocyte hemoglobin `pg/L` source scales remain literal; stale unit caches cannot silently replace them with `g/dL` or `pg`.
- Platelet distribution width reported as `10(GSD)` keeps that index scale; cached mappings cannot relabel it as a volume in `fL`.
- Immature granulocyte percentage and absolute-count observations use separate specifications (LOINC `38518-7` and `51584-1`), preserving both printed companions.
- Nucleated red-cell results per `100 WBC` use a separate ratio specification (LOINC `19048-8`), so they survive deduplication alongside an absolute concentration of the same cells.
- Qualitative sediment wording such as `Raras` retains an explicitly printed `/µL` unit and a null numeric result. Inline `/campo` result notation can establish a field-based scale, while the raw unit column remains available for source audit.
- Numeric urine concentrations/counts with explicit units retain their quantitative values, including zero and one-sided reference intervals. Explicit `Leu/µL`, `Erit/µL` and named-cell `Cells/µL` strip estimates use quantitative test-strip definitions; sediment observations retain their microscopy definitions. Unit-free strip grades can still use qualitative variants. HLA-B27 presence without a printed method has a method-neutral specification (LOINC `4821-5`); genotyping is retained only when supported by the source.
- When units convert, normalized displays use the converted number with its mapped unit, retaining approximation/interpretation markers. The complete original wording remains in `value_text` and source details.
- Urobilinogen Ehrlich units convert at factor one to mg/dL, supported by [LOINC's mapping guidance for 19161-9](https://loinc.org/19161-9/); the raw source unit remains available.
- Activity-derived urine cell estimates use separate test-strip definitions for [leukocytes 20408-1](https://loinc.org/20408-1) and [erythrocytes 20409-9](https://loinc.org/20409-9), rather than asserting sediment microscopy. Printed approximate counts retain their qualifier, and an absent/rare textual result never becomes an invented numeric count.
- Color, appearance, ABO group and Rh type are nominal results with primary unit/scale `text`, no numeric `value`, and no inferred normality code. This overrides legacy boolean unit-cache entries while retaining the canonical qualitative assay names for color and appearance. Blood-group definitions follow [ABO 883-9](https://loinc.org/883-9) and [Rh 10331-7](https://loinc.org/10331-7).
- Unqualified glucose has its own [2345-7 definition](https://loinc.org/2345-7), separate from explicitly fasting glucose. A legacy fasting mapping is downgraded unless a non-negated fasting condition appears in the raw name, section or comments. Source review must preserve a separately printed fasting condition in the raw comments rather than add a condition absent from the page.
- Separately printed urine hemoglobin uses [725-2](https://loinc.org/725-2), preserving its identity alongside a separately printed blood observation. Macroscopic urine deposit abundance and unspecified sediment cells are retained as text with explicit unmapped-LOINC exceptions; a crystal or cell-subtype-specific code must not be substituted without source evidence.
- Second-hour sedimentation uses [18184-2](https://loinc.org/18184-2) and the `mm/(2.h)` scale. An explicitly printed `mm` distance on the second-hour assay retains its value, overriding stale hourly-unit caches. An explicitly printed `mm/h` rate converts by a factor of two, including its reference bounds; the exact source unit remains in `raw_unit`.
- Stool chymotrypsin activity retains `U/g` ([25375-7](https://loinc.org/25375-7)); calprotectin mass per stool mass retains `mg/kg` ([38445-3](https://loinc.org/38445-3)). Intrinsic-factor antibodies without a printed isotype use a generic definition ([11564-2](https://loinc.org/11564-2)) and preserve relative units without assuming calibration equivalence.
- Separately printed thyroid-antibody presence interpretations have dedicated qualitative specifications ([thyroglobulin 15210-8](https://loinc.org/15210-8), [TPO 32042-4](https://loinc.org/32042-4)); their quantitative companions keep their original calibrated units.
- ANA and Widal dilution results use a `titer` denominator value while retaining the printed `1:N`/`1/N` notation and comparison in `value_text`. Standalone dilution reference comparisons are parsed; interpretive threshold prose is kept in the raw range without creating a normal interval. The titer definitions follow [ANA 5048-4](https://loinc.org/5048-4), [Widal O 42959-7](https://loinc.org/42959-7/), and [Widal H 42960-5](https://loinc.org/42960-5). Separate presence assays use [ANA 42254-3](https://loinc.org/42254-3), [CRP 11039-5](https://loinc.org/11039-5), [MPO 29643-4](https://loinc.org/29643-4), and [PR3 63310-7](https://loinc.org/63310-7). A qualitative ANA result preserves the original pattern text; its reference dilution remains raw context and never becomes a boolean reference bound or a measured titer.
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
- The code flags duplicates with `flag_duplicate_entries(...)`, but only when same-date same-lab rows still disagree after normalization; equivalent dual-unit repeats stay quiet.
- It adds export-style alias columns so the UI and CSV have a stable shape.
- It then runs `ValueValidator.validate(...)`.
- The result becomes the per-document review CSV and the basis for the reviewer UI.

Reticulocyte counts printed in millions per microliter (`10E6/uL`) convert to `10⁹/L` with a factor of 1000; a percentage reference interval is not an absolute-count interval.

21. Export-mode behavior.
- In `mode="export"`, unresolved rows are filtered out first by `_filter_exportable_rows(...)`.
- Rows with unknown standardized labs or missing/non-publishable primary units are removed.
- Duplicate-looking rows are still flagged first.
- If lab specs exist, `deduplicate_results(...)` then removes publish-time duplicates.
- Deduplication and date/lab integrity keys include `value_text` so separate qualitative findings from the same assay are retained; identical findings still collapse and numeric dual-unit handling is unchanged.
- Explicit specimen numbers (for example, `1ª Amostra` or `Sample 2`) also distinguish separate source specimens during deduplication, even when their results match.
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
- Instead, it rebuilds every processed document from canonical page JSON with `_rebuild_review_outputs_from_processed_documents(...)`.
- It writes that corpus result's merged review-state dataframe to `output/all.csv` and `output/all.xlsx`.
- So the first export pass still writes the merged review-state dataframe, not accepted-only reviewed truth.
- After rebuilding and exporting all available review outputs, a normal extraction run raises `PipelineError` if any PDF failed completely or any page/document failure record remains, so incomplete runs cannot exit successfully while retaining their diagnostic artifacts.

25. End-of-run standardization auto-refresh.
- After the first export pass, the pipeline scans merged review rows for uncached standardization names and unit pairs.
- By default, it runs one in-process cache refresh pass using the same OpenRouter credentials and extraction model resolved for the active runtime.
- The shared refresh helper preserves deterministic first-seen order and sends bounded batches instead of one corpus-sized request. Name and unit calls both default to at most 50 items; the admin command can override these limits with `--name-batch-size` and `--unit-batch-size`.
- Raw-name batches run first, using the extracted `raw_section_name` as part of the name-cache key when available. Every valid batch result is merged into the working cache and persisted immediately, so later name failures cannot discard earlier progress and newly resolved names are available when the refresh rescans dependent unit pairs.
- Unit mappings use the same batched, eager-persistence flow. Every returned name or unit must correspond to an item in its current batch and use a configured candidate value. Hallucinated inputs, malformed items, out-of-candidate values, and `$UNKNOWN$` results are never cached.
- Standardization uses a dedicated OpenAI client configuration with SDK retries disabled; extraction client retry behavior is unchanged. A retryable API failure can consume at most one application-level retry per item. Timed-out batches retry as two smaller deterministic sub-batches, while omitted or malformed response items retry only the incomplete subset. Retry splitting is not recursive.
- A batch that remains unsuccessful is recorded in a concise aggregate error, but independent later batches continue. Update and unresolved counts therefore cover the complete best-effort refresh pass, including partial successes.
- For `codex/gpt-6.1-sol`, name and unit standardization calls explicitly use high reasoning effort. Other `codex/...` models use medium effort. Other providers do not receive that provider-specific option.
- Bundled standardization caches are immutable defaults; automatic and admin refreshes persist user overrides under `~/.config/parselabs/cache/`.
- This means a row that was `$UNKNOWN$` on the first pass can still contribute a unit mapping in the same automatic refresh cycle.
- If the refresh adds any cache entries, the pipeline rebuilds per-document CSVs and merged outputs from persisted page JSON only.
- It does not re-extract PDFs and does not repeat extraction API calls.
- If `--no-auto-standardize` is passed, the refresh is skipped and the pipeline logs a final manual-fallback summary instead.
- If refresh calls return structured API/cache errors or leave mappings unresolved, the run keeps the best-effort outputs and ends with a warning summary instead of failing.
- Unexpected exceptions inside the refresh orchestration propagate so implementation bugs are visible during the run.

26. Reviewed-JSON rebuild mode.
- If `--rebuild-from-json` is passed, extraction is skipped entirely.
- `_run_reviewed_json_rebuild(...)` rebuilds every per-document CSV from page JSON, then writes either the in-progress merged review ledger or the accepted-only final export depending on review completeness.
- With `--allow-pending`, it writes the merged review-state dataframe so reviewers can refresh the live queue without requiring a fully reviewed corpus.
- Without `--allow-pending`, it writes the accepted-only deduplicated final export dataframe, so fixture-ready reviewed corpora publish clean `all.csv` and `all.xlsx` outputs.
- This path regenerates review outputs from existing reviewed JSON state while keeping the auto-refresh behavior consistent with normal extraction runs.

27. Strict reviewed-truth helpers still exist separately.
- `build_final_output_dataframe_from_reviewed_json(...)` and related helpers build accepted-row export data from reviewed JSON.
- In strict mode, `ensure_document_fixture_ready(...)` blocks promotion if any rows are still pending or any missing-row markers remain unresolved.
- `--allow-pending` affects those reviewed-truth helpers.
- It only changes whether reviewed rebuild mode publishes the review ledger or the accepted final export.
- Golden-corpus promotion additionally requires a separate completed source/export audit bound to PDF/page SHA-256 hashes and the canonical export fingerprint. Pending rows, unresolved omissions, stale source evidence, or changed normalized output block promotion.
- Private golden artifacts live outside Git under `~/.local/share/parselabs/approved/<profile>/<case-id>/`. Promotion freezes reviewed page JSON alongside the source PDF and expected CSV; changed existing baselines require explicit replacement, and unrelated cases remain frozen.
- `RUN_LOCAL_GOLDENS=1 uv run pytest -m golden_local` checks frozen reviewed JSON through current normalization/export without API calls. The separate `RUN_APPROVED_DOCS=1` suite reruns vision extraction through the configured API. Ordinary CI skips both private modes.
- `make regression-quick` runs fresh AgentBridge extraction for an explicitly curated local subset. `make regression-quick-offline` replays the same subset without API calls. The private sibling `quick-suite.json` pins case identities and PDF hashes, records layout rationale/tags and required profiles, and enforces document/page budgets. The full corpus remains available independently. Fresh runs verify source attestations before model calls and fail incomplete extractions. Both private modes reject CI execution before discovering patient artifacts.
- Fresh runs match frozen review decisions by page and normalized observation content before comparing accepted exports. Missing/extra observations and substantive date/value/unit/reference/limit differences fail; reordered rows, equivalent raw numeric formatting and qualitative sentence-ending punctuation are allowed. Ambiguous decisions fail rather than transfer by position. Private outputs, logs and a completed comparison's `latest-result.json` are retained outside Git under the corpus's sibling `regression-runs/`; approved baselines remain unchanged. Quick fresh extraction uses two workers and remains subject to model latency.

28. Canonical source of truth.
- The canonical persisted intermediate is the per-page JSON in each hashed document directory.
- Per-document CSVs are derived review snapshots.
- Normal extraction-mode `all.csv` and `all.xlsx` are merged review-state outputs, potentially rebuilt once more after automatic standardization refresh.
- Strict reviewed rebuild mode writes the accepted-only final export instead of the review ledger.

### Fidelity checks and cached reconstruction

Discovery is recursive; clearly labelled prescriptions, orders and leaflets are not lab-result candidates. A printed patient name is checked against the configured profile name before page results are published. Continuation pages without names do not establish identity. Rebuilds preserve each page’s collection/report date instead of imposing one PDF-wide date; a document fallback fills only undated rows. Percentage and absolute-count sibling rows with identical reference bounds on the same source page are flagged rather than assigned invented replacement ranges. One-sided intervals are also checked. When the absolute sibling has no interval, a percentage interval that matches the configured absolute range substantially better is flagged for source review, without transferring or deleting bounds automatically. Explicit reticulocyte count units survive stale unit mappings, and numeric conversion columns support decimal factors even when all source values are integers.

Date extraction keeps specimen collection, laboratory registration/entry and report issue dates separate. Registration includes `DT. ENTRADA` and older analytical-study `Data` headers beside `Hora Inscrição`, or episode-header `Data` beside tube/episode identifiers; a later FINAL EM or web-page print date does not replace that event date. A continuation page lacking an event date inherits an explicit collection/registration date from the document before using its report issue date. Explicit different page event dates remain distinct. Each result may also carry an explicit `collection_date`; flattening gives it precedence over the page date for that specimen. This survives extraction-model serialization and cached-JSON rebuilds. An undated result keeps the existing page/document fallback, but an unverified date cannot support golden-corpus validation.

Reported numeric intervals remain in `value_text` with an empty numeric `value`; no midpoint is invented. Explicit sediment per-volume units override cached per-field mappings. Source rows with conflicting per-field and per-volume labels require review.

### Assay identity and explicit scale

DGP IgA and IgG use distinct deamidated-gliadin definitions ([58709-7](https://loinc.org/58709-7), [58710-5](https://loinc.org/58710-5)). Method-specific HbA2 observations stay separate for [HPLC](https://loinc.org/42245-1) and [electrophoresis](https://loinc.org/4552-6). Quantitative urinary hemoglobin uses a [concentration definition](https://loinc.org/726-0), while presence-only strip observations retain their qualitative definition. High-sensitivity CRP has a [distinct method definition](https://loinc.org/30522-7). Stool cultures use the [stool specimen definition](https://loinc.org/625-4), and Gram-stain descriptions retain nominal text. Qualitative urine leukocytes use [5799-2](https://loinc.org/5799-2).

Explicit Anti-HBs IU/mL converts to mIU/mL with factor 1000. Colony counts labelled `milhares UFC/mL` convert with factor 1000. Estradiol pmol/L converts to pg/mL with factor 0.272382, calculated from the [NIST molecular weight](https://webbook.nist.gov/cgi/cbook.cgi?ID=C50282). A printed urine glucose result of `Normal` remains text without an invented zero concentration. Exact units can come from an explicitly referenced attached table; obscured assay identities and contradictory source units require source review rather than inferred repairs.
