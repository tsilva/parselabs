from pathlib import Path

from parselabs import paths


def test_writable_cache_is_user_config_state(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert paths.get_cache_dir() == tmp_path / ".config" / "parselabs" / "cache"


def test_bundled_resources_resolve_in_source_checkout():
    resource_root = paths.get_bundled_resources_dir()

    assert (resource_root / "config" / "lab_specs.json").is_file()
    assert (resource_root / "prompts" / "extraction_system.md").is_file()
    assert (resource_root / "static" / "viewer.css").is_file()
    assert paths.get_source_lab_specs_path() == resource_root / "config" / "lab_specs.json"
