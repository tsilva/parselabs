from __future__ import annotations

from pathlib import Path

import pandas as pd

import review_documents
from parselabs.review_sync import ProcessedDocument


def _make_review_df(statuses: list[str], pages: list[int], rows: list[int]) -> pd.DataFrame:
    """Build a minimal review dataframe for reviewer-state tests."""

    records: list[dict] = []

    # Build one synthetic review row per requested status.
    for idx, status in enumerate(statuses):
        records.append(
            {
                "page_number": pages[idx],
                "result_index": rows[idx],
                "raw_lab_name": f"Raw Lab {idx}",
                "raw_value": str(90 + idx),
                "raw_lab_unit": "mg/dL",
                "lab_name": f"Blood - Lab {idx}",
                "value": 90 + idx,
                "lab_unit": "mg/dL",
                "review_status": status,
                "review_reason": "",
                "raw_comments": "",
                "raw_reference_range": "70 - 100",
                "reference_min": 70,
                "reference_max": 100,
            }
        )

    return pd.DataFrame(records)


def test_build_queue_state_hides_reviewed_rows_and_sorts_pending_first():
    review_df = _make_review_df(
        statuses=["accepted", "", "", "rejected"],
        pages=[1, 2, 1, 1],
        rows=[0, 0, 1, 2],
    )

    hidden_queue = review_documents._build_queue_state(review_df, show_reviewed=False)
    visible_queue = review_documents._build_queue_state(review_df, show_reviewed=True)

    assert hidden_queue["actual_index"].tolist() == [2, 1]
    assert hidden_queue["status_code"].tolist() == ["P", "P"]
    assert visible_queue["actual_index"].tolist() == [2, 1, 0, 3]
    assert visible_queue["status_code"].tolist() == ["P", "P", "A", "R"]


def test_resolve_current_index_falls_back_to_first_visible_pending_row():
    review_df = _make_review_df(
        statuses=["accepted", "", ""],
        pages=[1, 1, 2],
        rows=[0, 1, 0],
    )

    pending_only_queue = review_documents._build_queue_state(review_df, show_reviewed=False)
    all_rows_queue = review_documents._build_queue_state(review_df, show_reviewed=True)

    assert review_documents._resolve_current_index(pending_only_queue, requested_index=0, prefer_first_visible=False) == 1
    assert review_documents._resolve_current_index(all_rows_queue, requested_index=0, prefer_first_visible=False) == 0


def test_choose_next_pending_index_prefers_same_page_before_other_pages():
    review_df = _make_review_df(
        statuses=["accepted", "", ""],
        pages=[1, 2, 1],
        rows=[0, 0, 1],
    )

    assert review_documents._choose_next_pending_index(review_df, current_index=0) == 2


def test_build_dropdown_choices_prioritizes_incomplete_documents(monkeypatch):
    alpha = ProcessedDocument(
        doc_dir=Path("/tmp/alpha_deadbeef"),
        stem="alpha",
        pdf_path=Path("/tmp/alpha_deadbeef/alpha.pdf"),
        csv_path=Path("/tmp/alpha_deadbeef/alpha.csv"),
    )
    beta = ProcessedDocument(
        doc_dir=Path("/tmp/beta_deadbeef"),
        stem="beta",
        pdf_path=Path("/tmp/beta_deadbeef/beta.pdf"),
        csv_path=Path("/tmp/beta_deadbeef/beta.csv"),
    )
    gamma = ProcessedDocument(
        doc_dir=Path("/tmp/gamma_deadbeef"),
        stem="gamma",
        pdf_path=Path("/tmp/gamma_deadbeef/gamma.pdf"),
        csv_path=Path("/tmp/gamma_deadbeef/gamma.csv"),
    )

    review_frames = {
        alpha.doc_dir.name: _make_review_df(statuses=["", ""], pages=[1, 1], rows=[0, 1]),
        beta.doc_dir.name: _make_review_df(statuses=["accepted"], pages=[1], rows=[0]),
        gamma.doc_dir.name: _make_review_df(statuses=["accepted", ""], pages=[1, 2], rows=[0, 0]),
    }

    # Stub review-frame loading so dropdown sorting can be tested without touching disk.
    monkeypatch.setattr(review_documents, "_get_review_frame", lambda document: review_frames[document.doc_dir.name])

    choices = review_documents._build_dropdown_choices([beta, gamma, alpha], filter_mode="All")

    assert [doc_id for _, doc_id in choices] == [
        alpha.doc_dir.name,
        gamma.doc_dir.name,
        beta.doc_dir.name,
    ]
