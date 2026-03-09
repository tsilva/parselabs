from pathlib import Path

from parselabs.config import ProfileConfig


def test_profile_config_loads_runtime_settings_and_resolves_relative_paths(tmp_path):
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
openrouter:
  api_key: "test-key"
  base_url: "https://example.invalid/v1"
models:
  extract_model_id: "google/gemini-test"
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
    assert profile.openrouter_api_key == "test-key"
    assert profile.openrouter_base_url == "https://example.invalid/v1"
    assert profile.extract_model_id == "google/gemini-test"
    assert profile.demographics is not None
    assert profile.demographics.gender == "female"
    assert profile.demographics.date_of_birth == "1990-01-15"


def test_profile_config_supports_flat_runtime_keys(tmp_path):
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

    assert profile.openrouter_api_key == "flat-key"
    assert profile.openrouter_base_url == "https://openrouter.example/v1"
    assert profile.extract_model_id == "openai/gpt-test"
    assert profile.workers == 8
    assert profile.input_path == Path("/tmp/input")
    assert profile.output_path == Path("/tmp/output")
