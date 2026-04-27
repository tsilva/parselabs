from __future__ import annotations

import logging
import sys

import pytest

from parselabs import pipeline, standardization
from parselabs.standardization_refresh import StandardizationRefreshResult
from parselabs.utils import log_user_info, log_user_warning, setup_logging


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


def test_setup_logging_normal_console_filters_detail_info_and_warnings(tmp_path, capsys, restore_root_logger):
    setup_logging(tmp_path / "logs", console_mode="normal")
    test_logger = logging.getLogger("tests.quiet_logging")

    test_logger.info("detail info")
    log_user_info(test_logger, "phase info")
    log_user_warning(test_logger, "summary warning")
    test_logger.warning("warning info")
    test_logger.error("error info")
    _flush_root_handlers()

    captured = capsys.readouterr()
    info_log = (tmp_path / "logs" / "info.log").read_text(encoding="utf-8")

    assert "detail info" not in captured.err
    assert "phase info" in captured.err
    assert "summary warning" in captured.err
    assert "warning info" not in captured.err
    assert "error info" in captured.err
    assert "INFO:" not in captured.err
    assert "detail info" in info_log
    assert "warning info" in info_log


def test_setup_logging_verbose_console_includes_detail_info(tmp_path, capsys, restore_root_logger):
    setup_logging(tmp_path / "logs", console_mode="verbose")
    test_logger = logging.getLogger("tests.quiet_logging")

    test_logger.info("detail info")

    assert "detail info" in capsys.readouterr().err


def test_standardization_cache_misses_stay_out_of_normal_console(tmp_path, capsys, monkeypatch, restore_root_logger):
    setup_logging(tmp_path / "logs", console_mode="normal")
    monkeypatch.setattr(standardization, "CACHE_DIR", tmp_path / "cache")

    standardization.standardize_lab_units([("", "Urine Type II - Glucose")])
    _flush_root_handlers()

    captured = capsys.readouterr()
    info_log = (tmp_path / "logs" / "info.log").read_text(encoding="utf-8")

    assert "[unit_standardization]" not in captured.err
    assert "[unit_standardization]" in info_log


def test_standardization_unresolved_summary_is_user_visible(tmp_path, capsys, restore_root_logger):
    setup_logging(tmp_path / "logs", console_mode="normal")
    result = StandardizationRefreshResult(
        uncached_names=(),
        uncached_unit_pairs=(("", "Urine Type II - Glucose"),),
        name_updates=0,
        unit_updates=0,
        unresolved_names=(),
        unresolved_unit_pairs=(("", "Urine Type II - Glucose"),),
    )

    pipeline._log_standardization_refresh_summary(result, auto_standardize=True, profile_name="tiago")

    assert "Remaining uncached mappings after auto-refresh" in capsys.readouterr().err


def test_setup_profile_environment_logs_profile_after_configured_logging(tmp_path, capsys, monkeypatch, restore_root_logger):
    setup_logging(tmp_path / "logs", console_mode="normal")
    calls = []

    class FakeContext:
        logger = logging.getLogger("tests.profile")
        extraction_config = type(
            "FakeExtractionConfig",
            (),
            {
                "input_path": tmp_path / "input",
                "output_path": tmp_path / "output",
                "extract_model_id": "test-model",
            },
        )()
        lab_specs = object()

        def copy_lab_specs_to_output(self):
            return None

    def fake_from_profile(*args, **kwargs):
        calls.append(kwargs["console_mode"])
        return FakeContext()

    monkeypatch.setattr(pipeline.RuntimeContext, "from_profile", fake_from_profile)

    pipeline._setup_profile_environment(type("Args", (), {"console_mode": "normal"})(), "tiago")

    captured = capsys.readouterr()

    assert calls == ["normal"]
    assert "Processing profile: tiago" in captured.err
    assert "INFO:" not in captured.err


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
