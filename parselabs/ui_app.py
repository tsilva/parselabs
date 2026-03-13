"""Combined Gradio app for the results explorer and document reviewer."""

from __future__ import annotations

import logging

import gradio as gr

from parselabs import document_reviewer, results_view
from parselabs.runtime import RuntimeContext

logger = logging.getLogger(__name__)

RESULTS_TAB_ID = "results-explorer"
REVIEW_TAB_ID = "review-queue"


def build_app(context: RuntimeContext, default_tab: str) -> gr.Blocks:
    """Build the combined Gradio app with a selectable default tab."""

    selected_tab = _normalize_default_tab(default_tab)
    results_view.apply_runtime_context(context)
    document_reviewer.apply_runtime_context(context)
    explorer_app = results_view.create_app()
    reviewer_app = document_reviewer.build_app()

    with gr.Blocks(title="Parselabs") as demo:
        with gr.Tabs(selected=selected_tab):
            with gr.Tab("Results Explorer", id=RESULTS_TAB_ID):
                explorer_app.render()

            with gr.Tab("Review Queue", id=REVIEW_TAB_ID):
                reviewer_app.render()

    return demo


def launch_app(context: RuntimeContext, default_tab: str) -> None:
    """Launch the combined Gradio app for one selected profile."""

    demo = build_app(context, default_tab)
    allowed_paths = RuntimeContext.list_output_roots()
    css = "\n".join([document_reviewer.CUSTOM_CSS, results_view.CUSTOM_CSS])
    head = "\n".join([document_reviewer.KEYBOARD_SHORTCUTS_JS, results_view.KEYBOARD_JS])
    server_port = 7862 if _normalize_default_tab(default_tab) == RESULTS_TAB_ID else 7863

    logger.info(
        "Starting Parselabs app on http://localhost:%s with default tab %s",
        server_port,
        selected_tab_label(default_tab),
    )
    demo.launch(
        server_name="127.0.0.1",
        server_port=server_port,
        show_error=True,
        inbrowser=False,
        allowed_paths=allowed_paths,
        css=css,
        head=head,
    )


def selected_tab_label(default_tab: str) -> str:
    """Return a human-readable label for the selected default tab."""

    return "Results Explorer" if _normalize_default_tab(default_tab) == RESULTS_TAB_ID else "Review Queue"


def _normalize_default_tab(default_tab: str) -> str:
    """Normalize user-facing default-tab values to concrete tab ids."""

    normalized = str(default_tab).strip().lower()
    if normalized in {"review", "reviewer", REVIEW_TAB_ID}:
        return REVIEW_TAB_ID
    return RESULTS_TAB_ID
