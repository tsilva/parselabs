from pathlib import Path

from parselabs.paths import get_profiles_dir


def test_get_profiles_dir_uses_profiles_subdirectory(monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/tmp/test-home")))

    assert get_profiles_dir() == Path("/tmp/test-home/.config/parselabs/profiles")
