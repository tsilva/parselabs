"""Shared utility functions for the labs parser."""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)


_PRIMARY_IMAGE_MAX_WIDTH = 1800
_FALLBACK_IMAGE_MAX_WIDTH = 1800
_PAGE_BORDER_PADDING_PX = 64


def _resize_image(image: Image.Image, max_width: int) -> Image.Image:
    """Resize an image while preserving aspect ratio."""

    if image.width <= max_width:
        return image

    ratio = max_width / image.width
    new_height = int(image.height * ratio)
    return image.resize((max_width, new_height), Image.Resampling.LANCZOS)


def _add_page_padding(image: Image.Image, padding_px: int = _PAGE_BORDER_PADDING_PX) -> Image.Image:
    """Add a uniform white border around a page image before extraction."""

    if padding_px <= 0:
        return image

    fill = 255 if image.mode == "L" else (255, 255, 255)
    return ImageOps.expand(image, border=padding_px, fill=fill)


def create_page_image_variants(image: Image.Image) -> dict[str, Image.Image]:
    """Create image variants for extraction.

    The primary variant preserves color and more detail for table structure,
    while the fallback variant exaggerates contrast in grayscale for difficult OCR.
    """

    base_image = ImageOps.exif_transpose(image)
    padded_rgb = _add_page_padding(base_image.convert("RGB"))
    padded_gray = _add_page_padding(base_image.convert("L"))

    primary = _resize_image(padded_rgb, _PRIMARY_IMAGE_MAX_WIDTH)
    primary = ImageOps.autocontrast(primary, cutoff=1)
    primary = ImageEnhance.Sharpness(primary).enhance(1.15)

    fallback = _resize_image(padded_gray, _FALLBACK_IMAGE_MAX_WIDTH)
    fallback = ImageOps.autocontrast(fallback, cutoff=1)
    fallback = ImageEnhance.Contrast(fallback).enhance(2.2)
    fallback = fallback.filter(ImageFilter.SHARPEN)

    return {
        "primary": primary,
        "fallback": fallback,
    }


def preprocess_page_image(image: Image.Image) -> Image.Image:
    """Backward-compatible grayscale preprocessing wrapper."""

    return create_page_image_variants(image)["fallback"]


def slugify(value: Any) -> str:
    """Create a normalized slug for mapping/debugging purposes."""

    # Skip missing values
    if pd.isna(value):
        return ""

    # Normalize unicode characters and replace special symbols
    value = str(value).strip().lower().replace("µ", "micro").replace("μ", "micro").replace("%", "percent")
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")

    # Strip punctuation, collapse whitespace, remove hyphens
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_]+", "-", value).strip("-")
    value = value.replace("-", "")

    return value


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from text."""

    # Only process if text starts with a code fence
    if text.startswith("```"):
        lines = text.split("\n")

        # Strip opening fence line
        if lines[0].startswith("```"):
            lines = lines[1:]

        # Strip closing fence line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


def parse_llm_json_response(text: str, fallback: Any = None) -> Any:
    """Parse JSON from LLM response, handling markdown fences."""

    text = strip_markdown_fences(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Return fallback when JSON parsing fails
        return fallback


def ensure_columns(df: pd.DataFrame, columns: list[str], default: Any = None) -> pd.DataFrame:
    """Ensure DataFrame has specified columns, adding them with default value if missing."""

    for col in columns:
        # Add missing column with default value
        if col not in df.columns:
            df[col] = default

    return df


def setup_logging(log_dir: Path, clear_logs: bool = False) -> logging.Logger:
    """Configure file and console logging, optionally clearing existing logs."""

    # Set up log directory and file paths
    log_dir.mkdir(exist_ok=True)
    info_log_path = log_dir / "info.log"
    error_log_path = log_dir / "error.log"

    # Clear existing log files if requested
    if clear_logs:
        for log_file in (info_log_path, error_log_path):
            if log_file.exists():
                log_file.write_text("", encoding="utf-8")

    # Configure root logger so all modules inherit the same level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers from root logger
    # Note: Don't close() handlers in multiprocessing context - they share file descriptors
    # with the parent process, and closing them can corrupt the parent's file handles
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

        # Only close if we're in the main process (clear_logs=True indicates main process)
        if clear_logs:
            handler.close()

    # Formatters
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # File handlers
    info_handler = logging.FileHandler(info_log_path, encoding="utf-8")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(file_formatter)

    error_handler = logging.FileHandler(error_log_path, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Register all handlers with the root logger
    root_logger.addHandler(info_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    # Return a logger for the calling module
    logger = logging.getLogger(__name__)
    return logger
