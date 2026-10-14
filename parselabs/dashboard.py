"""Local Svelte dashboard API backed by the existing review store."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from parselabs.paths import get_static_dir
from parselabs.review import get_bbox_coordinates, normalize_review_status, scale_bbox_to_pixels
from parselabs.review_data import load_results_dataframe
from parselabs.review_state import ReviewTarget, apply_review_action_for_target
from parselabs.runtime import RuntimeContext
from parselabs.store import HASHED_DOCUMENT_DIR_RE, get_page_image_path, get_page_json_path, parse_page_number, read_page_payload


class Decision(BaseModel):
    """A review command with an optimistic concurrency precondition."""

    action: Literal["accept", "reject", "clear", "missing_row"]
    expected_status: Literal["accepted", "rejected", ""]


def _row_id(entry: dict) -> str:
    """Keep API identities stable without exposing filesystem paths."""

    identity = [entry.get("document_key"), entry.get("source_file"), entry.get("page_number"), entry.get("result_index")]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]


def _serialize_rows(frame: pd.DataFrame) -> list[dict]:
    """Convert pandas dates and missing values to JSON-safe review records."""

    # Pandas' JSON encoder handles timestamps, numpy types, and NaN consistently.
    records = json.loads(frame.to_json(orient="records", date_format="iso"))
    for entry in records:
        entry["id"] = _row_id(entry)
        entry["review_status"] = entry.get("review_status") or ""
    return records


class DashboardStore:
    """Cache loaded rows and serialize review writes for one selected profile."""

    def __init__(self, context: RuntimeContext):
        self.context = context
        self.output = Path(context.output_path).resolve()
        self.lock = threading.RLock()
        self.rows: dict[str, dict] = {}
        self.loaded = False

    def refresh(self) -> None:
        """Reload authoritative page JSON through the shared review loader."""

        frame = load_results_dataframe(self.output, self.context.lab_specs, self.context.demographics, include_document_key=True)
        self.rows = {entry["id"]: entry for entry in _serialize_rows(frame)}
        self.loaded = True

    def get_row(self, row_id: str) -> dict:
        """Resolve a server-issued identity, never a caller-supplied path."""

        # Load on first access so empty profiles still launch successfully.
        if not self.loaded:
            self.refresh()

        entry = self.rows.get(row_id)

        # Unknown identities cannot select another document or filesystem path.
        if entry is None:
            raise HTTPException(404, "Result no longer exists. Refresh the dashboard.")
        return entry

    def document_dir(self, entry: dict) -> Path:
        """Resolve and contain page access inside the selected profile's output."""

        key = entry.get("document_key")

        # Rows without a real document cannot serve sources or accept decisions.
        if not isinstance(key, str) or Path(key).name != key or HASHED_DOCUMENT_DIR_RE.fullmatch(key) is None:
            raise HTTPException(404, "Source document identity is unavailable.")

        directory = (self.output / key).resolve()

        # Reject symlinks escaping the selected profile's output root.
        if not directory.is_relative_to(self.output):
            raise HTTPException(403, "Source is outside the selected profile.")

        # A page JSON symlink must not allow review writes outside the profile.
        if not get_page_json_path(directory, int(entry["page_number"])).resolve().is_relative_to(self.output):
            raise HTTPException(403, "Source is outside the selected profile.")
        return directory

    def source(self, row_id: str, page: int | None) -> dict:
        """Return available document pages and a normalized source highlight."""

        entry = self.get_row(row_id)
        directory = self.document_dir(entry)
        pages = sorted({number for path in directory.glob("*.json") if (number := parse_page_number(path)) is not None})
        selected_page = page if page is not None else int(entry["page_number"])

        # Only known extracted pages are navigable.
        if selected_page not in pages:
            raise HTTPException(404, "Source page is unavailable.")

        image_path = get_page_image_path(directory, selected_page)
        bbox = get_bbox_coordinates(entry) if selected_page == int(entry["page_number"]) else None
        highlight = None

        # Report missing source images without hiding review metadata.
        if image_path is None:
            return {"pages": pages, "page": selected_page, "image_url": None, "bbox": None}

        image_path = image_path.resolve()

        # Image symlinks must follow the same profile containment rule.
        if not image_path.is_relative_to(self.output):
            raise HTTPException(403, "Source is outside the selected profile.")

        with Image.open(image_path) as image:
            width, height = image.size

        # Convert all supported bbox formats using the established review helper.
        if bbox is not None:
            pixels = scale_bbox_to_pixels(bbox, (width, height))

            # Invalid or empty boxes should not draw a misleading highlight.
            if pixels is not None:
                left, top, right, bottom = pixels
                highlight = {"left": left / width * 100, "top": top / height * 100, "width": (right - left) / width * 100, "height": (bottom - top) / height * 100}

        return {"pages": pages, "page": selected_page, "image_url": f"/api/results/{row_id}/image?page={selected_page}", "bbox": highlight}

    def persisted_result(self, entry: dict) -> dict:
        """Read only the selected page when checking a review precondition."""

        directory = self.document_dir(entry)
        payload = read_page_payload(get_page_json_path(directory, int(entry["page_number"])))
        results = payload.get("lab_results") if payload is not None else None
        index = int(entry["result_index"])

        # A removed row cannot be replaced by an in-memory decision.
        if not isinstance(results, list) or index < 0 or index >= len(results) or not isinstance(results[index], dict):
            raise HTTPException(404, "Source result is unavailable. Refresh the dashboard.")
        return results[index]


def create_dashboard_app(context: RuntimeContext, default_tab: str = "results") -> FastAPI:
    """Serve the compiled Svelte application and profile-scoped review API."""

    # Keep the existing Python services authoritative for normalization and writes.
    store = DashboardStore(context)
    token = secrets.token_urlsafe(32)
    app = FastAPI(title="Parselabs", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @app.middleware("http")
    async def protect_local_data(request: Request, call_next):
        """Prevent cross-origin decisions and caching of private report data."""

        # Every write requires the same-origin bootstrap token, not just cookies.
        if request.method not in {"GET", "HEAD", "OPTIONS"} and request.headers.get("x-parselabs-token") != token:
            return Response("Invalid dashboard session.", status_code=403)

        response = await call_next(request)

        # Source images and lab results must not persist in shared browser caches.
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/api/bootstrap")
    def bootstrap():
        """Provide one selected profile and its initial screen."""

        return {"tab": "review" if default_tab in {"review", "review-queue"} else "results", "token": token}

    @app.get("/api/results")
    def results():
        """Reload page JSON so external decisions appear on refresh."""

        with store.lock:
            store.refresh()
            return {"rows": list(store.rows.values())}

    @app.get("/api/results/{row_id}/source")
    def source(row_id: str, page: int | None = None):
        """Expose source metadata for a stable row identity."""

        with store.lock:
            return store.source(row_id, page)

    @app.get("/api/results/{row_id}/image")
    def image(row_id: str, page: int | None = None):
        """Serve only a verified image belonging to the selected profile."""

        with store.lock:
            descriptor = store.source(row_id, page)

            # Missing images are a visible source-unavailable state in the UI.
            if descriptor["image_url"] is None:
                raise HTTPException(404, "Source image is unavailable.")
            directory = store.document_dir(store.get_row(row_id))
            return FileResponse(get_page_image_path(directory, descriptor["page"]), media_type="image/jpeg")

    @app.post("/api/results/{row_id}/decision")
    def decision(row_id: str, command: Decision):
        """Persist a review decision and return the updated authoritative row."""

        with store.lock:
            entry = store.get_row(row_id)
            persisted = store.persisted_result(entry)

            # Avoid overwriting decisions made in another browser or review tool.
            if normalize_review_status(persisted.get("review_status")) != command.expected_status:
                raise HTTPException(409, "This result changed elsewhere. Refresh before reviewing it.")

            target = ReviewTarget(store.document_dir(entry), int(entry["page_number"]), int(entry["result_index"]))
            success, error = apply_review_action_for_target(target, command.action)

            # Failed writes remain visible and must not advance the review queue.
            if not success:
                raise HTTPException(400, error)

            # Review changes only status metadata; avoid rebuilding the whole corpus.
            persisted = store.persisted_result(entry)
            updated = {**entry, "review_status": normalize_review_status(persisted.get("review_status")), "review_completed_at": persisted.get("review_completed_at")}
            store.rows[row_id] = updated
            return updated

    # Bundle the frontend with the existing resource system for installed wheels.
    static_dir = get_static_dir()
    app.mount("/brand", StaticFiles(directory=static_dir / "brand"), name="brand")
    app.mount("/", StaticFiles(directory=static_dir / "dashboard", html=True), name="dashboard")
    return app
