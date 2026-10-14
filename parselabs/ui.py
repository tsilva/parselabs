"""Launch the local Svelte results and review workspace."""

from __future__ import annotations

import logging

import uvicorn

from parselabs.dashboard import create_dashboard_app
from parselabs.runtime import RuntimeContext

logger = logging.getLogger(__name__)

RESULTS_TAB_ID = "results-explorer"
REVIEW_TAB_ID = "review-queue"


def build_app(context: RuntimeContext, default_tab: str):
    """Build the single review workspace."""

    return create_dashboard_app(context, default_tab)


def launch_app(context: RuntimeContext, default_tab: str) -> None:
    """Launch the unified review workspace for one selected profile."""

    app = build_app(context, default_tab)
    server_port = 7862 if _normalize_default_tab(default_tab) == RESULTS_TAB_ID else 7863

    logger.info(
        "Starting Parselabs review workspace on http://localhost:%s with launch mode %s",
        server_port,
        selected_tab_label(default_tab),
    )
    uvicorn.run(app, host="127.0.0.1", port=server_port)


def selected_tab_label(default_tab: str) -> str:
    """Return a human-readable launch mode label."""

    return "Results Explorer" if _normalize_default_tab(default_tab) == RESULTS_TAB_ID else "Review Queue"


def _normalize_default_tab(default_tab: str) -> str:
    """Normalize launch-mode values."""

    normalized = str(default_tab).strip().lower()
    if normalized in {"review", REVIEW_TAB_ID}:
        return REVIEW_TAB_ID
    return RESULTS_TAB_ID
