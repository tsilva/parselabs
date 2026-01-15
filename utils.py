"""Shared utility functions for the labs parser."""

import json
import unicodedata
import re
import shutil
import logging
from pathlib import Path
from typing import Any
from PIL import Image, ImageEnhance
import pandas as pd


def preprocess_page_image(image: Image.Image) -> Image.Image:
    """Convert image to grayscale, resize, and enhance contrast."""
    gray_image = image.convert('L')
    MAX_WIDTH = 1200
    if gray_image.width > MAX_WIDTH:
        ratio = MAX_WIDTH / gray_image.width
        new_height = int(gray_image.height * ratio)
        gray_image = gray_image.resize((MAX_WIDTH, new_height), Image.Resampling.LANCZOS)
    return ImageEnhance.Contrast(gray_image).enhance(2.0)


def slugify(value: Any) -> str:
    """Create a normalized slug for mapping/debugging purposes."""
    if pd.isna(value):
        return ""
    value = str(value).strip().lower().replace('µ', 'micro').replace('μ', 'micro').replace('%', 'percent')
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_]+", "-", value).strip('-')
    value = value.replace("-", "")
    return value


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from text."""
    if text.startswith("```"):
        lines = text.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = '\n'.join(lines).strip()
    return text


def parse_llm_json_response(text: str, fallback: Any = None) -> Any:
    """Parse JSON from LLM response, handling markdown fences."""
    text = strip_markdown_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


def clear_directory(dir_path: Path) -> None:
    """Remove all contents of a directory without deleting the directory itself."""
    if dir_path.exists():
        for item in dir_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            else:
                shutil.rmtree(item)


def ensure_columns(df: pd.DataFrame, columns: list[str], default: Any = None) -> pd.DataFrame:
    """Ensure DataFrame has specified columns, adding them with default value if missing."""
    for col in columns:
        if col not in df.columns:
            df[col] = default
    return df


def setup_logging(log_dir: Path, clear_logs: bool = False) -> logging.Logger:
    """Configure file and console logging, optionally clearing existing logs."""
    log_dir.mkdir(exist_ok=True)
    info_log_path = log_dir / "info.log"
    error_log_path = log_dir / "error.log"

    if clear_logs:
        # Clear logs in the specified directory
        for log_file in (info_log_path, error_log_path):
            if log_file.exists():
                log_file.write_text("", encoding="utf-8")

        # Also clear project root logs/ directory if it exists (legacy location)
        project_root_logs = Path("logs")
        if project_root_logs.exists() and project_root_logs.is_dir():
            for log_file in (project_root_logs / "info.log", project_root_logs / "error.log"):
                if log_file.exists():
                    log_file.write_text("", encoding="utf-8")

    # Configure root logger so all modules inherit the same level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers from root logger
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    # Formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')

    # File handlers
    info_handler = logging.FileHandler(info_log_path, encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(file_formatter)

    error_handler = logging.FileHandler(error_log_path, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    root_logger.addHandler(info_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    # Return a logger for the calling module
    logger = logging.getLogger(__name__)
    return logger
