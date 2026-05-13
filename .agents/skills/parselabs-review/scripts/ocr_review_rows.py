#!/usr/bin/env python3
"""OCR Parselabs review rows and their bbox crops for source comparison."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from parselabs.review import scale_bbox_to_pixels


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


def _doc_dir_for(output_path: Path, source_file: str) -> Path:
    """Return the processed document directory for one source file."""

    candidates = sorted(output_path.glob(f"{Path(source_file).stem}_*"))

    # Guard: missing processed document directories cannot be OCRed.
    if not candidates:
        raise FileNotFoundError(f"No processed document directory for {source_file}")

    return candidates[0]


def _page_image_path(output_path: Path, source_file: str, page_number: int) -> Path:
    """Return the cached page image path for one review row."""

    doc_dir = _doc_dir_for(output_path, source_file)
    stem = doc_dir.name.rsplit("_", 1)[0]
    return doc_dir / f"{stem}.{page_number:03d}.jpg"


def _run_tesseract_tsv(image_path: Path, lang: str) -> list[dict[str, int | str]]:
    """Return OCR line boxes for one page image."""

    with tempfile.NamedTemporaryFile(suffix=".tsv") as tmp:
        subprocess.run(
            ["tesseract", str(image_path), tmp.name[:-4], "-l", lang, "--psm", "6", "tsv"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        rows = list(csv.DictReader(Path(tmp.name).read_text(errors="replace").splitlines(), delimiter="\t"))

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        text = (row.get("text") or "").strip()

        # Skip empty OCR tokens.
        if not text:
            continue

        key = (row.get("block_num") or "", row.get("par_num") or "", row.get("line_num") or "")
        groups.setdefault(key, []).append(row)

    lines = []
    for words in groups.values():
        left = min(int(word["left"]) for word in words)
        top = min(int(word["top"]) for word in words)
        right = max(int(word["left"]) + int(word["width"]) for word in words)
        bottom = max(int(word["top"]) + int(word["height"]) for word in words)
        text = " ".join((word.get("text") or "").strip() for word in words if (word.get("text") or "").strip())
        lines.append({"text": text, "left": left, "top": top, "right": right, "bottom": bottom})

    return sorted(lines, key=lambda line: (int(line["top"]), int(line["left"])))


def _row_bbox(row: pd.Series) -> tuple[float, float, float, float] | None:
    """Return the normalized bbox for one review row."""

    # Guard: missing bbox coordinates cannot produce a crop.
    if row[BBOX_COLUMNS].isna().any():
        return None

    return tuple(float(row[column]) for column in BBOX_COLUMNS)


def _crop_ocr(image_path: Path, bbox: tuple[float, float, float, float] | None, lang: str) -> str:
    """Return OCR text from one bbox crop."""

    # Guard: rows without a bbox can only be inspected on the full page.
    if bbox is None:
        return ""

    image = Image.open(image_path)
    pixel_bbox = scale_bbox_to_pixels(bbox, image.size)

    # Guard: malformed bboxes cannot produce a crop.
    if pixel_bbox is None:
        return ""

    left, top, right, bottom = pixel_bbox
    crop_box = (
        max(0, left - 20),
        max(0, top - 8),
        min(image.width, right + 20),
        min(image.height, bottom + 8),
    )

    with tempfile.NamedTemporaryFile(suffix=".png") as image_tmp, tempfile.NamedTemporaryFile(suffix=".txt") as text_tmp:
        image.crop(crop_box).save(image_tmp.name)
        subprocess.run(
            ["tesseract", image_tmp.name, text_tmp.name[:-4], "-l", lang, "--psm", "6"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return Path(text_tmp.name).read_text(errors="replace").strip().replace("\n", " | ")


def _select_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Return the rows requested for OCR inspection."""

    status = _normalized_status(df.get("review_status", pd.Series([""] * len(df))))

    if args.status == "all":
        selected = df.copy()
    elif args.status == "pending":
        selected = df[status == ""].copy()
    else:
        selected = df[status == args.status].copy()

    if args.source_file:
        selected = selected[selected["source_file"].astype(str) == args.source_file]

    if args.page_number is not None:
        selected = selected[selected["page_number"].astype(int) == args.page_number]

    return selected


def main() -> None:
    """Run OCR inspection for selected review rows."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Parselabs profile name, e.g. tiago")
    parser.add_argument("--output-path", help="Explicit profile output directory")
    parser.add_argument("--status", default="rejected", help="Status to inspect: rejected, accepted, pending, or all")
    parser.add_argument("--source-file", help="Restrict OCR to one source_file value")
    parser.add_argument("--page-number", type=int, help="Restrict OCR to one page number")
    parser.add_argument("--lang", default="eng", help="Tesseract language, default: eng")
    args = parser.parse_args()

    output_path = _resolve_output_path(args)
    df = pd.read_csv(output_path / "all.csv")
    selected = _select_rows(df, args)

    # Guard: empty selections should fail loudly enough to avoid false confidence.
    if selected.empty:
        raise SystemExit("No rows matched the requested filters.")

    for (source_file, page_number), group in selected.groupby(["source_file", "page_number"], dropna=False):
        image_path = _page_image_path(output_path, str(source_file), int(page_number))
        print(f"\n===== {source_file} page {int(page_number)} =====")
        print(f"image={image_path}")

        print("-- OCR lines --")
        for idx, line in enumerate(_run_tesseract_tsv(image_path, args.lang)):
            print(
                f"L{idx:02d} y={int(line['top']):04d}-{int(line['bottom']):04d} "
                f"x={int(line['left']):04d}-{int(line['right']):04d} {line['text']}"
            )

        print("-- row crops --")
        for idx, row in group.iterrows():
            print(
                f"#{idx} r{int(row['result_index'])} target={row['raw_lab_name']!r} "
                f"raw={row['raw_value']!r} crop={_crop_ocr(image_path, _row_bbox(row), args.lang)!r}"
            )


if __name__ == "__main__":
    main()
