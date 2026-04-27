from __future__ import annotations

import logging
import sys

import pytest

from parselabs import pipeline
from parselabs.utils import log_user_info, setup_logging


@pytest.fixture
def restore_root_logger():
    """Restore pytest's logging handlers after setup_logging rewires the root logger."""

    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    yield

    # Remove handlers installed by setup_logging during the test.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    # Restore pytest's original logging configuration.
    root_logger.setLevel(original_level)
    for handler in original_handlers:
        root_logger.addHandler(handler)


def _flush_root_handlers() -> None:
    """Flush active root handlers so file assertions can read current log content."""

    for handler in logging.getLogger().handlers:
        handler.flush()


def test_setup_logging_normal_console_filters_detail_info(tmp_path, capsys, restore_root_logger):
    setup_logging(tmp_path / "logs", console_mode="normal")
    test_logger = logging.getLogger("tests.quiet_logging")

    test_logger.info("detail info")
    log_user_info(test_logger, "phase info")
    test_logger.warning("warning info")
    test_logger.error("error info")
    _flush_root_handlers()

    captured = capsys.readouterr()
    info_log = (tmp_path / "logs" / "info.log").read_text(encoding="utf-8")

    assert "detail info" not in captured.err
    assert "phase info" in captured.err
    assert "warning info" in captured.err
    assert "error info" in captured.err
    assert "detail info" in info_log


def test_setup_logging_verbose_console_includes_detail_info(tmp_path, capsys, restore_root_logger):
    setup_logging(tmp_path / "logs", console_mode="verbose")
    test_logger = logging.getLogger("tests.quiet_logging")

    test_logger.info("detail info")

    assert "detail info" in capsys.readouterr().err


def test_setup_logging_quiet_console_shows_only_errors(tmp_path, capsys, restore_root_logger):
    setup_logging(tmp_path / "logs", console_mode="quiet")
    test_logger = logging.getLogger("tests.quiet_logging")

    test_logger.warning("warning info")
    test_logger.error("error info")

    captured = capsys.readouterr()

    assert "warning info" not in captured.err
    assert "error info" in captured.err


def test_parse_args_supports_console_logging_modes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["parselabs", "--profile", "tsilva", "--verbose"])

    args = pipeline.parse_args()

    assert args.profile == "tsilva"
    assert args.console_mode == "verbose"


def test_parse_args_rejects_conflicting_console_logging_modes(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["parselabs", "--quiet", "--debug"])

    with pytest.raises(SystemExit) as exc_info:
        pipeline.parse_args()

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "not allowed with argument" in captured.err
