---
name: parselabs-review-mcp
description: Use when reviewing parsed lab results for a profile through the Parselabs review MCP server. This skill is for systematically working through all unreviewed rows in a profile, inspecting the full page image and deterministic bbox crop, and persisting accept or reject for every row until no pending rows remain.
---

# Parselabs Review MCP

Use this skill when the user wants a full review pass for a profile through the MCP workflow instead of the Gradio reviewer.

## Preconditions

- The current session must have access to the Parselabs review MCP server.
- The MCP server should come from this project workspace and expose `next_pending_row` and `decide_row`.
- The target profile must already have processed output.

If the MCP server is unavailable, say so briefly and stop. Do not silently fall back to Playwright, OCR, or a different review path unless the user explicitly asks for that fallback.

## Goal

Clear the entire unreviewed queue for one profile by repeating:

1. Fetch the next pending row.
2. Inspect the returned full page image and bbox crop.
3. Decide `accept` or `reject`.
4. Persist the decision immediately.

Continue until the MCP returns `done: true`.

## Required Workflow

For a full profile review:

1. Call `next_pending_row` with the requested profile.
2. Read the structured payload first:
   - `row_id`
   - `stored_result`
   - `page_number`
   - `result_index`
   - `artifact_error`
3. Inspect both returned images:
   - Full page image for context
   - Bbox clip for the exact row text
4. Compare the visible document text to the stored extraction fields.
5. Call `decide_row` immediately with the same `row_id`.
6. Repeat until `done: true`.

Do not batch decisions in memory. Persist each row before moving to the next one.

## Decision Standard

Review the extraction, not the health meaning.

Primary fields to verify:

- `raw_lab_name`
- `raw_value`
- `raw_lab_unit`
- `raw_reference_range`
- `raw_reference_min`
- `raw_reference_max`
- `raw_comments`
- bbox points to the correct row

Accept when the stored raw extraction is clearly consistent with the visible source row.

Reject when any of these is true:

- The bbox points to the wrong row
- The row text does not match the stored lab name
- The value is wrong
- The unit is wrong
- The reference text is materially wrong
- The extraction merged multiple rows incorrectly
- The extraction split one row incorrectly
- The crop or page is too ambiguous to confirm safely

When evidence is ambiguous, reject instead of accept.

## Practical Heuristics

- Trust the crop first for row-local text, then use the full page to disambiguate nearby rows.
- If the crop is slightly tight or loose, use the full page image to confirm whether the bbox still targets the intended row.
- For qualitative rows, verify the literal label as printed, not an inferred numeric meaning.
- For rows with blank units or blank ranges, accept only if the document also omits them or the stored extraction otherwise clearly matches.
- Ignore downstream normalization concerns during this pass. This is a source-faithfulness review.

## Loop Discipline

Default target is all unreviewed rows in the profile, not only `review_needed` rows, unless the user explicitly narrows scope.

Keep a running count while you work:

- accepted
- rejected
- reviewed total

At the end, report:

- profile reviewed
- accepted count
- rejected count
- whether the queue was fully cleared
- any recurring failure patterns worth fixing in extraction

## Example Tool Pattern

Use this exact shape conceptually:

```text
next_pending_row(profile="cristina")
decide_row(profile="cristina", row_id="...", decision="accept")
next_pending_row(profile="cristina")
decide_row(profile="cristina", row_id="...", decision="reject")
...
next_pending_row(profile="cristina") -> done: true
```

## What Not To Do

- Do not switch to the Gradio reviewer when this skill is active unless the user asks.
- Do not re-run extraction.
- Do not edit JSON files directly if `decide_row` can persist the decision.
- Do not stop early because the queue is large. Keep going until `done: true` or a real blocker occurs.
