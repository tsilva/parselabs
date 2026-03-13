from __future__ import annotations

import argparse

from parselabs import config as config_module
from parselabs.profiles import add_profile_arguments, load_ui_context
from parselabs.ui import selected_tab_label


def test_add_profile_arguments_parses_shared_flags():
    parser = add_profile_arguments(
        argparse.ArgumentParser(),
        profile_help="Profile name",
    )

    args = parser.parse_args(["--profile", "tsilva", "--list-profiles"])

    assert args.profile == "tsilva"
    assert args.list_profiles is True


def test_load_ui_context_defaults_to_first_available_profile(monkeypatch, tmp_path):
    profiles_dir = tmp_path / "profiles"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    profiles_dir.mkdir()
    input_dir.mkdir()
    output_dir.mkdir()

    (profiles_dir / "_template.yaml").write_text("name: Template\n", encoding="utf-8")
    (profiles_dir / "alpha.yaml").write_text(
        "\n".join(
            [
                'name: "Alpha"',
                "paths:",
                "  input_path: ../input",
                "  output_path: ../output",
                "openrouter:",
                '  api_key: "test-key"',
                "models:",
                '  extract_model_id: "test-model"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(config_module, "get_profiles_dir", lambda: profiles_dir)

    context = load_ui_context(None)

    assert context.profile_name == "alpha"
    assert context.output_path == output_dir.resolve()


def test_selected_tab_label_handles_review_aliases():
    assert selected_tab_label("results") == "Results Explorer"
    assert selected_tab_label("reviewer") == "Review Queue"
