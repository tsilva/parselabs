"""Unified Gradio review workspace."""

from __future__ import annotations

import logging

import gradio as gr

from parselabs import document_reviewer, results_view
from parselabs.runtime import RuntimeContext

logger = logging.getLogger(__name__)

RESULTS_TAB_ID = "results-explorer"
REVIEW_TAB_ID = "review-queue"

def build_app(context: RuntimeContext, default_tab: str) -> gr.Blocks:
    """Build the single review workspace."""

    _ = default_tab
    return results_view.create_app(context)


def launch_app(context: RuntimeContext, default_tab: str) -> None:
    """Launch the unified review workspace for one selected profile."""

    demo = build_app(context, default_tab)
    allowed_paths = RuntimeContext.list_output_roots()
    css = "\n".join([document_reviewer.CUSTOM_CSS, results_view.CUSTOM_CSS])
    head = results_view.KEYBOARD_JS
    server_port = 7862 if _normalize_default_tab(default_tab) == RESULTS_TAB_ID else 7863

    logger.info(
        "Starting Parselabs review workspace on http://localhost:%s with launch mode %s",
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
    """Return a human-readable launch mode label."""

    return "Results Explorer" if _normalize_default_tab(default_tab) == RESULTS_TAB_ID else "Review Queue"


def _normalize_default_tab(default_tab: str) -> str:
    """Normalize legacy launch-mode values."""

    normalized = str(default_tab).strip().lower()
    if normalized in {"review", "reviewer", REVIEW_TAB_ID}:
        return REVIEW_TAB_ID
    return RESULTS_TAB_ID
