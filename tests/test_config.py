from pathlib import Path

from parselabs.config import ProfileConfig, load_config_env


def test_profile_config_loads_env_runtime_settings_and_resolves_relative_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("EXTRACT_MODEL_ID", "google/gemini-test")

    profile_path = tmp_path / "demo.yaml"
    profile_path.write_text(
        """
name: "Demo"
paths:
  input_path: "./input"
  output_path: "./output"
processing:
  input_file_regex: "2024-*.pdf"
  workers: 3
demographics:
  gender: "female"
  date_of_birth: "1990-01-15"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    profile = ProfileConfig.from_file(profile_path)

    assert profile.name == "Demo"
    assert profile.input_path == (tmp_path / "input").resolve()
    assert profile.output_path == (tmp_path / "output").resolve()
    assert profile.input_file_regex == "2024-*.pdf"
    assert profile.workers == 3
    assert profile.openrouter_api_key == "env-key"
    assert profile.openrouter_base_url == "https://env.example/v1"
    assert profile.extract_model_id == "google/gemini-test"
    assert profile.demographics is not None
    assert profile.demographics.gender == "female"
    assert profile.demographics.date_of_birth == "1990-01-15"


def test_profile_config_ignores_profile_runtime_keys_without_env_sources(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    (home_dir / ".config" / "parselabs").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home_dir))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("EXTRACT_MODEL_ID", raising=False)
    load_config_env.cache_clear()

    profile_path = tmp_path / "flat.json"
    profile_path.write_text(
        """
{
  "name": "Flat",
  "input_path": "/tmp/input",
  "output_path": "/tmp/output",
  "openrouter_api_key": "flat-key",
  "openrouter_base_url": "https://openrouter.example/v1",
  "extract_model_id": "openai/gpt-test",
  "workers": 8
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    profile = ProfileConfig.from_file(profile_path)

    assert profile.openrouter_api_key is None
    assert profile.openrouter_base_url is None
    assert profile.extract_model_id is None
    assert profile.workers == 8
    assert profile.input_path == Path("/tmp/input")
    assert profile.output_path == Path("/tmp/output")


def test_profile_config_prefers_shared_config_env_over_profile_keys(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    env_dir = home_dir / ".config" / "parselabs"
    env_dir.mkdir(parents=True)
    (env_dir / ".env").write_text(
        (
            'OPENROUTER_API_KEY="shared-key"\n'
            'OPENROUTER_BASE_URL="https://shared.example/v1"\n'
            'EXTRACT_MODEL_ID="google/gemini-shared"\n'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home_dir))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("EXTRACT_MODEL_ID", raising=False)
    load_config_env.cache_clear()

    profile_path = tmp_path / "env-first.yaml"
    profile_path.write_text(
        """
name: "Env First"
openrouter:
  api_key: "stale-profile-key"
  base_url: "https://stale.example/v1"
models:
  extract_model_id: "google/gemini-stale"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    profile = ProfileConfig.from_file(profile_path)

    assert profile.openrouter_api_key == "shared-key"
    assert profile.openrouter_base_url == "https://shared.example/v1"
    assert profile.extract_model_id == "google/gemini-shared"
    load_config_env.cache_clear()


def test_profile_config_prefers_process_env_over_shared_config_env(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    env_dir = home_dir / ".config" / "parselabs"
    env_dir.mkdir(parents=True)
    (env_dir / ".env").write_text(
        (
            "OPENROUTER_API_KEY=shared-key\n"
            "OPENROUTER_BASE_URL=https://shared.example/v1\n"
            "EXTRACT_MODEL_ID=google/gemini-shared\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home_dir))
    monkeypatch.setenv("OPENROUTER_API_KEY", "shell-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://shell.example/v1")
    monkeypatch.setenv("EXTRACT_MODEL_ID", "google/gemini-shell")
    load_config_env.cache_clear()

    profile_path = tmp_path / "shell-first.yaml"
    profile_path.write_text('name: "Shell First"\n', encoding="utf-8")

    profile = ProfileConfig.from_file(profile_path)

    assert profile.openrouter_api_key == "shell-key"
    assert profile.openrouter_base_url == "https://shell.example/v1"
    assert profile.extract_model_id == "google/gemini-shell"
    load_config_env.cache_clear()
