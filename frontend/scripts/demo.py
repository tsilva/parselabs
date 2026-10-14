"""Run an isolated dashboard with synthetic reports for UI development."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import uvicorn
from PIL import Image, ImageDraw, ImageFont

from parselabs import standardization
from parselabs.config import LabSpecsConfig, ProfileConfig
from parselabs.dashboard import create_dashboard_app
from parselabs.runtime import RuntimeContext

LABS = [
    ("Ferritin", "48", "ng/mL", 30, 400, "2276-4"),
    ("Vitamin B12", "420", "pg/mL", 200, 900, "2132-9"),
    ("TSH", "2.1", "mIU/L", 0.4, 4, "3016-3"),
    ("Platelets", "246", "10³/µL", 150, 400, "777-3"),
    ("Glucose", "92", "mg/dL", 70, 100, "2345-7"),
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Use a local system font, with a portable Pillow fallback."""

    for path in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        # Select an installed font without downloading assets or reading profiles.
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _write_document(output: Path, stem: str, date: str, values: list[tuple], accepted: bool) -> None:
    """Create a synthetic source bitmap, PDF, and matching extraction JSON."""

    directory = output / f"{stem}_deadbeef"
    directory.mkdir()
    image = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 70), "Synthetic Laboratory", fill="#10233f", font=_font(30))
    draw.text((70, 125), "LABORATORY RESULTS", fill="#10233f", font=_font(26))
    draw.text((70, 180), f"Report date: {date}     ·     Demo data only", fill="#536986", font=_font(18))
    draw.line((70, 230, 930, 230), fill="#bac8d7", width=2)
    draw.text((80, 260), "Test name", fill="#10233f", font=_font(18))
    draw.text((485, 260), "Result", fill="#10233f", font=_font(18))
    draw.text((635, 260), "Unit", fill="#10233f", font=_font(18))
    draw.text((790, 260), "Reference", fill="#10233f", font=_font(18))
    results = []

    for index, (name, value, unit, low, high, _) in enumerate(values):
        y = 315 + index * 65
        draw.text((80, y), name, fill="#10233f", font=_font(19))
        draw.text((485, y), value, fill="#10233f", font=_font(19))
        draw.text((635, y), unit, fill="#10233f", font=_font(18))
        draw.text((790, y), f"{low} – {high}", fill="#10233f", font=_font(18))
        draw.line((70, y + 48, 930, y + 48), fill="#dce5ee")
        result = {
            "raw_lab_name": name, "raw_value": value, "raw_lab_unit": unit,
            "raw_reference_min": low, "raw_reference_max": high,
            "raw_reference_range": f"{low} – {high}",
            "bbox_left": 70, "bbox_top": y / 1400 * 1000,
            "bbox_right": 930, "bbox_bottom": (y + 42) / 1400 * 1000,
        }

        # Historical measurements are already reviewed; the latest report is pending.
        if accepted:
            result["review_status"] = "accepted"
        results.append(result)

    draw.text((70, 1280), "Synthetic fixture · no patient information", fill="#536986", font=_font(17))
    image.save(directory / f"{stem}.001.jpg", quality=95)
    image.save(directory / f"{stem}.pdf", "PDF")
    (directory / f"{stem}.001.json").write_text(json.dumps({"collection_date": date, "lab_results": results}), encoding="utf-8")
    (directory / f"{stem}.002.json").write_text(json.dumps({"collection_date": date, "lab_results": []}), encoding="utf-8")
    # A second synthetic page makes page navigation testable.
    image.save(directory / f"{stem}.002.jpg", quality=95)


def build_demo_context() -> RuntimeContext:
    """Keep every fixture and cache outside the repository and real profiles."""

    output = Path(tempfile.mkdtemp(prefix="parselabs-dashboard-demo-"))
    cache = output / "cache"
    cache.mkdir()
    names = {name.lower(): f"Blood - {name}" for name, *_ in LABS}
    units = {f"{unit.lower()}|blood - {name.lower()}": unit for name, _, unit, *_ in LABS}
    (cache / "name_standardization.json").write_text(json.dumps(names))
    (cache / "unit_standardization.json").write_text(json.dumps(units))
    standardization.CACHE_DIR = cache
    standardization.DEFAULT_CACHE_DIR = cache
    specs = {f"Blood - {name}": {"primary_unit": unit, "lab_type": "blood", "loinc_code": loinc, "ranges": {"default": [low, high]}} for name, _, unit, low, high, loinc in LABS}
    spec_path = output / "lab_specs.json"
    spec_path.write_text(json.dumps(specs))
    _write_document(output, "demo-report", "2026-10-08", LABS, False)

    for date, value in [("2026-07-16", "96"), ("2026-05-12", "90"), ("2026-03-10", "94"), ("2026-01-14", "89")]:
        _write_document(output, f"demo-{date}", date, [("Glucose", value, "mg/dL", 70, 100, "2345-7")], True)

    return RuntimeContext(
        profile_name="demo", profile=ProfileConfig(name="Demo", output_path=output),
        extraction_config=None, input_path=None, output_path=output, demographics=None,
        lab_specs=LabSpecsConfig(config_path=spec_path), logger=logging.getLogger(__name__),
    )


def main() -> None:
    """Start a loopback-only API and compiled dashboard for synthetic review."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    context = build_demo_context()
    print(f"Synthetic demo files: {context.output_path}", flush=True)
    uvicorn.run(create_dashboard_app(context), host="127.0.0.1", port=args.port)


# Only run the demo server when explicitly invoked.
if __name__ == "__main__":
    main()
