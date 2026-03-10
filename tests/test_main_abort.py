from argparse import Namespace

import pytest

import main
from parselabs.exceptions import ConfigurationError


def test_main_aborts_after_first_failed_profile(monkeypatch):
    attempts: list[str] = []

    monkeypatch.setattr(
        main,
        "parse_args",
        lambda: Namespace(profile=None, list_profiles=False, model=None, workers=None, pattern=None),
    )
    monkeypatch.setattr(main.ProfileConfig, "list_profiles", lambda: ["alpha", "beta"])

    def fake_run_for_profile(args, profile_name):
        attempts.append(profile_name)
        raise ConfigurationError("bad key")

    monkeypatch.setattr(main, "run_for_profile", fake_run_for_profile)

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 1
    assert attempts == ["alpha"]
