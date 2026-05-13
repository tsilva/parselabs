#!/usr/bin/env python3
"""Audit Parselabs profile review status and rejected row clusters."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


BBOX_COLUMNS = ["bbox_left", "bbox_top", "bbox_right", "bbox_bottom"]


def _load_profile(profile: str) -> dict[str, Any]:
    """Load one YAML or JSON Parselabs profile."""

    profile_dir = Path.home() / ".config" / "parselabs" / "profiles"
    yaml_path = profile_dir / f"{profile}.yaml"
    json_path = profile_dir / f"{profile}.json"

    # Prefer YAML because project profiles are normally YAML files.
    if yaml_path.exists():
        import yaml

        return yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    # Support JSON profiles for parity with Parselabs config loading.
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))

    raise FileNotFoundError(f"Profile not found: {profile}")


def _resolve_output_path(args: argparse.Namespace) -> Path:
    """Resolve the output path from either an explicit path or a profile name."""

    # Explicit output path is useful when auditing copied profile outputs.
    if args.output_path:
        return Path(args.output_path).expanduser()

    # Guard: callers must provide one source of output path information.
    if not args.profile:
        raise SystemExit("Provide --profile or --output-path.")

    profile = _load_profile(args.profile)
    return Path(profile["paths"]["output_path"]).expanduser()


def _normalized_status(series: pd.Series) -> pd.Series:
    """Return normalized review status text."""

    return series.fillna("").astype(str).str.strip().str.lower()


def _print_counts(df: pd.DataFrame) -> None:
    """Print review-status totals for one dataframe."""

    status = _normalized_status(df.get("review_status", pd.Series([""] * len(df))))
    print(
        {
            "rows": len(df),
            "accepted": int((status == "accepted").sum()),
            "rejected": int((status == "rejected").sum()),
            "pending": int((status == "").sum()),
        }
    )


def _print_reason_counts(rows: pd.DataFrame) -> None:
    """Print semicolon-delimited review reason counts."""

    reason_counts: Counter[str] = Counter()
    reasons = rows.get("review_reason", pd.Series(dtype=str)).fillna("").astype(str)

    for reason in reasons:
        parts = [part.strip() for part in reason.split(";") if part.strip()]
        reason_counts.update(parts or ["<blank>"])

    for reason, count in reason_counts.most_common():
        print(f"{reason}: {count}")


def _print_bbox_completeness(rows: pd.DataFrame) -> None:
    """Print bbox completeness for selected rows."""

    # Guard: older exports may not have bbox columns.
    if rows.empty or not set(BBOX_COLUMNS).issubset(rows.columns):
        print({"complete": 0, "missing_or_partial": 0})
        return

    complete = rows[BBOX_COLUMNS].notna().all(axis=1)
    print({"complete": int(complete.sum()), "missing_or_partial": int((~complete).sum())})


def _print_row_details(rows: pd.DataFrame) -> None:
    """Print compact selected-row details for manual triage."""

    detail_columns = [
        "source_file",
        "page_number",
        "result_index",
        "date",
        "raw_lab_name",
        "raw_section_name",
        "raw_value",
        "raw_lab_unit",
        "lab_name",
        "value",
        "lab_unit",
        "review_reason",
    ]
    available_columns = [column for column in detail_columns if column in rows.columns]

    for idx, row in rows[available_columns].iterrows():
        print(
            f"#{idx} {row.get('source_file')} p{int(row.get('page_number'))} "
            f"r{int(row.get('result_index'))}: {row.get('raw_lab_name')!r} "
            f"sec={row.get('raw_section_name')!r} raw={row.get('raw_value')!r} "
            f"{row.get('raw_lab_unit')!r} => {row.get('lab_name')!r} "
            f"{row.get('value')!r} {row.get('lab_unit')!r} "
            f"reason={row.get('review_reason')!r}"
        )


def main() -> None:
    """Run the profile audit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Parselabs profile name, e.g. tiago")
    parser.add_argument("--output-path", help="Explicit profile output directory")
    parser.add_argument("--status", default="rejected", help="Status to detail: rejected, accepted, pending, or all")
    args = parser.parse_args()

    output_path = _resolve_output_path(args)
    csv_path = output_path / "all.csv"

    # Guard: the profile must have a merged output to audit.
    if not csv_path.exists():
        raise SystemExit(f"Missing merged CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    status = _normalized_status(df.get("review_status", pd.Series([""] * len(df))))

    print(f"output_path={output_path}")
    _print_counts(df)

    if args.status == "all":
        selected = df.copy()
    elif args.status == "pending":
        selected = df[status == ""].copy()
    else:
        selected = df[status == args.status].copy()

    print(f"\nSelected rows: status={args.status} count={len(selected)}")

    print("\nBy source/page:")
    if selected.empty:
        print("<none>")
    else:
        for (source, page), group in selected.groupby(["source_file", "page_number"], dropna=False):
            print(f"{source} p{int(page)}: {len(group)}")

    print("\nReview reason counts:")
    _print_reason_counts(selected)

    print("\nBBox completeness:")
    _print_bbox_completeness(selected)

    print("\nRow details:")
    _print_row_details(selected)


if __name__ == "__main__":
    main()
