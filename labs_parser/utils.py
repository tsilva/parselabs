"""Shared utility functions for the labs parser."""

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


def load_dotenv_with_env() -> str | None:
    """Load .env.{name} file based on --env flag (default: "local").

    Parses --env from sys.argv before full argument parsing to allow
    loading the correct environment before module-level initialization.

    Returns:
        The environment name (defaults to "local").
    """
    # Extract --env value from sys.argv (default: "local")
    env_name = "local"
    for i, arg in enumerate(sys.argv):
        # Match --env <value> form
        if arg == "--env" and i + 1 < len(sys.argv):
            env_name = sys.argv[i + 1]
            break

        # Match --env=<value> form
        if arg.startswith("--env="):
            env_name = arg.split("=", 1)[1]
            break

    # Load .env.{name} file
    env_file = Path(f".env.{env_name}")
    if env_file.exists():
        load_dotenv(env_file, override=True)
        logger.info(f"Loaded environment: .env.{env_name}")
    else:
        # Warn when env file is not found
        logger.warning(f".env.{env_name} not found")

    return env_name


def preprocess_page_image(image: Image.Image) -> Image.Image:
    """Convert image to grayscale, resize, and enhance contrast."""

    # Convert to grayscale
    gray_image = image.convert("L")

    # Downscale if wider than max width
    MAX_WIDTH = 1200
    if gray_image.width > MAX_WIDTH:
        ratio = MAX_WIDTH / gray_image.width
        new_height = int(gray_image.height * ratio)
        gray_image = gray_image.resize((MAX_WIDTH, new_height), Image.Resampling.LANCZOS)

    # Enhance contrast for better OCR readability
    return ImageEnhance.Contrast(gray_image).enhance(2.0)


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
