from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from parselabs import document_reviewer as review_documents
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


def test_build_page_image_value_adds_overlay_for_current_row(tmp_path):
    doc_dir = tmp_path / "glucose_deadbeef"
    doc_dir.mkdir(parents=True)
    Image.new("RGB", (200, 100), "white").save(doc_dir / "glucose.001.jpg")

    document = ProcessedDocument(
        doc_dir=doc_dir,
        stem="glucose",
        pdf_path=doc_dir / "glucose.pdf",
        csv_path=doc_dir / "glucose.csv",
    )
    row = pd.Series(
        {
            "page_number": 1,
            "bbox_left": 100,
            "bbox_top": 200,
            "bbox_right": 600,
            "bbox_bottom": 800,
        }
    )

    image_value = review_documents._build_page_image_value(document, row)

    assert image_value is not None
    image_path, annotations = image_value
    assert image_path.endswith("glucose.001.jpg")
    assert annotations == [((20, 20, 120, 80), review_documents.SOURCE_BBOX_LABEL)]


def test_build_inspector_html_shows_raw_and_mapped_reference_ranges():
    review_df = pd.DataFrame(
        [
            {
                "page_number": 1,
                "result_index": 0,
                "raw_lab_name": "Glucose",
                "raw_value": "5.0",
                "raw_lab_unit": "mmol/L",
                "lab_name": "Blood - Glucose",
                "value": 90.0,
                "lab_unit": "mg/dL",
                "review_status": "",
                "review_reason": "",
                "raw_comments": "",
                "raw_reference_range": "3.9 - 5.5 mmol/L",
                "reference_min": 70.2,
                "reference_max": 99.0,
            }
        ]
    )

    html_output = review_documents._build_inspector_html(document=object(), review_df=review_df, current_index=0, show_reviewed=True)

    assert "3.9 - 5.5 mmol/L" in html_output
    assert "70.2 - 99.0 mg/dL" in html_output
    assert html_output.index("Mapped</div>") < html_output.index("Raw</div>")


def test_reference_formatters_render_one_sided_ranges_as_inequalities():
    min_only_row = pd.Series(
        {
            "reference_min": 4.5,
            "reference_max": None,
            "lab_unit": "10¹²/L",
        }
    )
    max_only_row = pd.Series(
        {
            "reference_min": None,
            "reference_max": 5.9,
            "lab_unit": "10¹²/L",
        }
    )

    assert review_documents._format_reference_text(min_only_row) == ">4.5"
    assert review_documents._format_reference_text(max_only_row) == "<5.9"
    assert review_documents._format_mapped_reference_text(min_only_row) == ">4.5 10¹²/L"
    assert review_documents._format_mapped_reference_text(max_only_row) == "<5.9 10¹²/L"


def test_build_queue_display_shows_reviewed_status_as_icons():
    review_df = _make_review_df(
        statuses=["accepted", "", "rejected"],
        pages=[1, 1, 1],
        rows=[0, 1, 2],
    )

    queue_state = review_documents._build_queue_state(review_df, show_reviewed=True)
    display_df = review_documents._build_queue_display(queue_state, current_index=1)

    assert display_df["St"].tolist() == ["", "✅", "❌"]


def test_get_review_frame_uses_row_pipeline_even_without_csv(monkeypatch, tmp_path):
    doc_dir = tmp_path / "glucose_deadbeef"
    doc_dir.mkdir(parents=True)

    document = ProcessedDocument(
        doc_dir=doc_dir,
        stem="glucose",
        pdf_path=doc_dir / "glucose.pdf",
        csv_path=doc_dir / "glucose.csv",
    )
    expected_df = pd.DataFrame([{"page_number": 1, "result_index": 0, "review_status": ""}])

    monkeypatch.setattr(review_documents, "get_lab_specs", lambda: object())
    monkeypatch.setattr(review_documents.RowPipeline, "build_review_rows", lambda doc_dir, lab_specs: expected_df)

    review_df = review_documents._get_review_frame(document)

    assert review_df.equals(expected_df.fillna(""))
