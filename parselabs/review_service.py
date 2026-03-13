"""High-level review actions backed by canonical page JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from parselabs.review_sync import save_missing_row_marker, save_review_status

ReviewAction = Literal["accept", "reject", "clear", "missing_row"]


class ReviewService:
    """Apply reviewer actions without exposing storage details to callers."""

    _STATUS_BY_ACTION = {
        "accept": "accepted",
        "reject": "rejected",
        "clear": None,
    }

    @classmethod
    def apply_action(
        cls,
        doc_dir: Path,
        page_number: int,
        result_index: int,
        action: ReviewAction,
    ) -> tuple[bool, str]:
        """Persist a supported reviewer action for one extracted row."""

        # Missing-row markers are stored on the page payload, not on the row itself.
        if action == "missing_row":
            return save_missing_row_marker(doc_dir, page_number, result_index)

        status = cls._STATUS_BY_ACTION[action]
        return save_review_status(doc_dir, page_number, result_index, status)
