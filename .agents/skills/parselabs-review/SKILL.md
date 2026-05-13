---
name: parselabs-review
description: Use when reviewing Parselabs extracted lab rows for a profile. By default, review and accept/reject only unconfirmed rows through the Browser Use review UI or helper artifacts. Also supports explicit full re-audit of accepted/rejected rows, rejected-row root-cause investigation, source-evidence comparison against page/bbox crops, patching extracted page JSON when values or review decisions are proven wrong, and rebuilding profile outputs.
---

# Parselabs Review

Use this skill for Parselabs profile review work where source-page evidence matters. The default task is conservative: review only unconfirmed rows and persist accept/reject decisions. Escalate to full re-audit or JSON patching only when the user asks for it or when rejected-row investigation requires it.

## Modes

### Default: Unconfirmed Review

Use when the user asks to review a profile, review pending rows, approve/reject labs, or clear the review queue.

1. Resolve `~/.config/parselabs/profiles/<profile>.yaml` and its `paths.output_path`.
2. Run `scripts/audit_profile.py --profile <profile> --status pending`.
3. Launch `parselabs review --profile <profile> --tab review`.
4. Open `http://localhost:7863` with the official Browser Use plugin.
5. For each unconfirmed row, compare the visible row/crop to the stored extraction.
6. Accept when the extraction is source-faithful; reject when it is wrong, duplicated, non-lab text, or ambiguous.
7. Rebuild outputs from reviewed JSON if review decisions were persisted outside an already-syncing UI flow.

### Full Re-Audit

Use only when the user explicitly asks to re-review confirmed rows, review everything, or verify previous accept/reject decisions.

1. Audit all statuses with `scripts/audit_profile.py --profile <profile> --status all`.
2. Use Browser Use, OCR helpers, and page images to sample or exhaustively inspect accepted and rejected rows as requested.
3. Flip review decisions only when the source evidence clearly contradicts the current status.
4. Rebuild and report before/after counts.

### Patch Extracted Data

Use when the user asks to patch wrong values, fix extracted data, investigate rejected rows, or repair source JSON.

1. Gather evidence from Browser Use, source page images, bbox crops, and/or `scripts/ocr_review_rows.py`.
2. Patch the canonical page JSON in the processed document directory, not `all.csv`.
3. Patch only fields proven wrong by the actual document:
   - `raw_lab_name`
   - `raw_section_name`
   - `raw_value`
   - `raw_lab_unit`
   - `raw_reference_range`, `raw_reference_min`, `raw_reference_max`
   - `raw_comments`
   - `bbox_left`, `bbox_top`, `bbox_right`, `bbox_bottom`
   - `review_status`, `review_completed_at`
4. Store bbox coordinates on the extraction 0-1000 normalized page scale.
5. Rebuild with `uv run parselabs --profile <profile> --rebuild-from-json --allow-pending --no-auto-standardize`.
6. Verify the rebuilt `all.csv`.

## Decision Standard

Accept only when the visible document supports the stored row:

- lab name and section
- value
- unit
- reference range/min/max
- comments
- bbox points to the same physical row

Reject when any material field is wrong, the bbox points to the wrong row, the row is duplicate translated text, collection metadata, narrative/comment-only text without a result, or the evidence is ambiguous.

## Required Discipline

- Review source fidelity, not health meaning.
- Use Browser Use for Gradio/browser review.
- Do not patch from inference alone. Patch only from visible source evidence.
- Preserve unrelated worktree changes; read `git status --short` before repo edits.
- Reading or writing profile outputs under Google Drive usually needs escalated filesystem access.
- Data-only patches require rebuild verification; repo code changes also require focused tests.

## Helper Scripts

- `scripts/audit_profile.py --profile <profile> --status pending`: show pending rows for default review.
- `scripts/audit_profile.py --profile <profile> --status rejected`: show rejected row clusters.
- `scripts/audit_profile.py --profile <profile> --status all`: show all rows for full re-audit planning.
- `scripts/ocr_review_rows.py --profile <profile> --status rejected`: OCR full pages and bbox crops for rejected rows.
- Add `--source-file` and `--page-number` to narrow OCR output.

Helper output is evidence-gathering, not final authority. The source page image and bbox crop are the source of truth.

## Root-Cause Checks

- Many rejected rows on one page with matching crops usually means prior review decisions were wrong; patch statuses only.
- Shifted crops usually mean stale bad bboxes; patch bboxes only if the visible row is clear.
- Missing bboxes can be patched from OCR/page evidence or left rejected for re-extraction.
- `$UNKNOWN$` rows require checking `config/lab_specs.json` and `config/cache/name_standardization.json` before deciding whether they are real labs, duplicates, or metadata.
- Wrong standardization from section context should be fixed by patching `raw_section_name` only when the page shows the correct governing section.

## Reporting

End with:

- profile reviewed and mode used
- rows accepted/rejected/patched
- final accepted/rejected/pending counts
- remaining rejected or pending rows and why
- rebuild command and verification/tests run
