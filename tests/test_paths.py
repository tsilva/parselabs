from pathlib import Path

from parselabs.paths import get_env_file, get_profiles_dir


def test_get_profiles_dir_uses_profiles_subdirectory(monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/tmp/test-home")))

    assert get_profiles_dir() == Path("/tmp/test-home/.config/parselabs/profiles")


def test_get_env_file_uses_user_config_directory(monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/tmp/test-home")))

    assert get_env_file() == Path("/tmp/test-home/.config/parselabs/.env")
