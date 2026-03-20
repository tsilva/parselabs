"""FastMCP server for deterministic row-by-row lab review."""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import CallToolResult, TextContent

from parselabs.review_artifacts_backend import (
    apply_review_decision,
    format_json_text,
    get_next_review_payload,
)


def build_server() -> FastMCP:
    """Build the stdio MCP server for review artifacts."""

    server = FastMCP(
        name="Parselabs Review MCP",
        instructions=(
            "Use this server to review one pending parsed lab row at a time. "
            "Call next_pending_row to get the next unresolved row together with the full page image "
            "and a deterministic bounding-box clip. After inspecting the images, call decide_row "
            "with accept or reject for the returned row_id."
        ),
    )

    @server.tool(
        name="next_pending_row",
        description="Return the next pending parsed lab row with the full page image and bbox clip.",
    )
    def next_pending_row(
        profile: str,
        artifacts_dir: str | None = None,
        review_needed_only: bool = False,
    ) -> CallToolResult:
        """Return the next pending row plus visual artifacts."""

        payload = get_next_review_payload(
            profile,
            artifacts_dir=Path(artifacts_dir) if artifacts_dir else None,
            review_needed_only=review_needed_only,
        )

        content = [
            TextContent(
                type="text",
                text=format_json_text(payload),
            )
        ]

        # Include the full page image when it exists so the model can use page context.
        if payload.get("page_image_path"):
            content.append(Image(path=payload["page_image_path"]).to_image_content())

        # Include the deterministic bbox crop when it exists so the model can inspect the row directly.
        if payload.get("bbox_clip_path"):
            content.append(Image(path=payload["bbox_clip_path"]).to_image_content())

        return CallToolResult(
            content=content,
            structuredContent=payload,
            isError=False,
        )

    @server.tool(
        name="decide_row",
        description="Persist an accept, reject, or clear decision for one previously returned row_id.",
    )
    def decide_row(
        profile: str,
        row_id: str,
        decision: str,
    ) -> CallToolResult:
        """Persist one row review decision."""

        success, payload = apply_review_decision(profile, row_id, decision)

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=format_json_text(payload),
                )
            ],
            structuredContent=payload,
            isError=not success,
        )

    return server


def main() -> None:
    """Run the review MCP server over stdio."""

    build_server().run("stdio")


if __name__ == "__main__":
    main()
