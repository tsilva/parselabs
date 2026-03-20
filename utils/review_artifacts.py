"""Deterministic review-artifact CLI for row-by-row human or model auditing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from parselabs.review_artifacts_backend import (
    apply_review_decision,
    format_json_text,
    get_next_review_payload,
)
from parselabs.runtime import add_profile_arguments, list_non_template_profiles, resolve_profile_name


def main(argv: list[str] | None = None) -> int:
    """Run the deterministic review-artifact utility."""

    args = _parse_args(argv)

    # List configured profiles before any profile-backed work starts.
    if args.list_profiles:
        _print_profiles()
        return 0

    # Resolve one explicit profile for the requested review action.
    profile_name = resolve_profile_name(args.profile)

    # Route the request to the chosen subcommand.
    if args.command == "next":
        return _run_next(profile_name, args)

    # Route persisted decisions through the existing review store.
    if args.command == "decide":
        return _run_decide(profile_name, args)

    raise RuntimeError(f"Unsupported command: {args.command}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse review-artifact CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Fetch deterministic review artifacts and persist row decisions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    next_parser = subparsers.add_parser(
        "next",
        help="Return the next pending row plus page-image and bbox-crop artifacts.",
    )
    add_profile_arguments(next_parser, profile_help="Profile name")
    next_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory where deterministic bbox crops will be written.",
    )
    next_parser.add_argument(
        "--review-needed-only",
        action="store_true",
        help="Return only pending rows that were flagged with review_needed.",
    )

    decide_parser = subparsers.add_parser(
        "decide",
        help="Persist one accept/reject decision for a previously returned row_id.",
    )
    add_profile_arguments(decide_parser, profile_help="Profile name")
    decide_parser.add_argument(
        "--row-id",
        required=True,
        help="Opaque row identifier returned by the next subcommand.",
    )
    decide_parser.add_argument(
        "--decision",
        choices=["accept", "reject", "clear"],
        required=True,
        help="Review action to persist for the resolved row.",
    )

    return parser.parse_args(sys.argv[1:] if argv is None else argv)


def _print_profiles() -> None:
    """Print every selectable profile on its own line."""

    for profile_name in list_non_template_profiles():
        print(profile_name)


def _run_next(profile_name: str, args: argparse.Namespace) -> int:
    """Return the next pending review row and write its deterministic artifacts."""

    payload = get_next_review_payload(
        profile_name,
        artifacts_dir=args.artifacts_dir,
        review_needed_only=bool(args.review_needed_only),
    )
    _print_json(payload)
    return 0


def _run_decide(profile_name: str, args: argparse.Namespace) -> int:
    """Persist one decision for the requested row identifier."""

    success, payload = apply_review_decision(profile_name, args.row_id, args.decision)
    _print_json(payload)
    return 0 if success else 1


def _print_json(payload: dict) -> None:
    """Print one JSON payload with stable human-readable formatting."""

    print(format_json_text(payload))


if __name__ == "__main__":
    raise SystemExit(main())
