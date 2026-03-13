"""Combined Gradio app for the results explorer and document reviewer."""

from __future__ import annotations

import logging

import gradio as gr

import review_documents
import viewer
from parselabs.config import ProfileConfig
from parselabs.runtime import RuntimeContext

logger = logging.getLogger(__name__)

RESULTS_TAB_ID = "results-explorer"
REVIEW_TAB_ID = "review-queue"


def build_app(context: RuntimeContext, default_tab: str) -> gr.Blocks:
    """Build the combined Gradio app with a selectable default tab."""

    selected_tab = _normalize_default_tab(default_tab)
    viewer.apply_runtime_context(context)
    review_documents.apply_runtime_context(context)
    explorer_app = viewer.create_app()
    reviewer_app = review_documents.build_app()

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
    allowed_paths = _build_allowed_paths()
    css = "\n".join([review_documents.CUSTOM_CSS, viewer.CUSTOM_CSS])
    head = "\n".join([review_documents.KEYBOARD_SHORTCUTS_JS, viewer.KEYBOARD_JS])
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


def _build_allowed_paths() -> list[str]:
    """Return filesystem roots the combined app may serve."""

    allowed_paths: set[str] = set()

    for profile_name in ProfileConfig.list_profiles():
        profile_path = ProfileConfig.find_path(profile_name)
        if not profile_path:
            continue

        profile = ProfileConfig.from_file(profile_path)
        if not profile.output_path:
            continue

        allowed_paths.add(str(profile.output_path))
        if profile.output_path.parent != profile.output_path:
            allowed_paths.add(str(profile.output_path.parent))

    return sorted(allowed_paths)
