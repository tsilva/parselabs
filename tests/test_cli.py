from __future__ import annotations

import pytest

from parselabs import cli


def test_main_defaults_to_legacy_extract_mode(monkeypatch):
    calls: list[tuple[list[str], str]] = []

    def fake_run_extract(argv, *, program_name="parselabs"):
        calls.append((list(argv), program_name))

    monkeypatch.setattr(cli, "_run_extract", fake_run_extract)

    cli.main(["--profile", "tsilva"])

    assert calls == [(["--profile", "tsilva"], "parselabs")]


def test_main_routes_extract_subcommand(monkeypatch):
    calls: list[tuple[list[str], str]] = []

    def fake_run_extract(argv, *, program_name="parselabs"):
        calls.append((list(argv), program_name))

    monkeypatch.setattr(cli, "_run_extract", fake_run_extract)

    cli.main(["extract", "--profile", "tsilva"])

    assert calls == [(["--profile", "tsilva"], "parselabs extract")]


def test_main_routes_review_subcommand(monkeypatch):
    calls: list[tuple[list[str], str, str]] = []

    def fake_run_review(argv, *, program_name, default_tab="results"):
        calls.append((list(argv), program_name, default_tab))

    monkeypatch.setattr(cli, "_run_review", fake_run_review)

    cli.main(["review", "--profile", "tsilva", "--tab", "review"])

    assert calls == [(["--profile", "tsilva", "--tab", "review"], "parselabs review", "results")]


def test_main_routes_admin_subcommand(monkeypatch):
    calls: list[list[str]] = []

    monkeypatch.setattr(cli, "_run_admin", lambda argv: calls.append(list(argv)))

    cli.main(["admin", "validate-lab-specs"])

    assert calls == [["validate-lab-specs"]]


def test_parse_review_args_supports_tab_selection():
    args = cli._parse_review_args(
        ["--profile", "tsilva", "--tab", "review"],
        program_name="parselabs review",
        default_tab="results",
    )

    assert args.profile == "tsilva"
    assert args.tab == "review"


def test_main_help_surfaces_subcommands(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 0
    assert "parselabs review" in captured.out
    assert "parselabs admin" in captured.out
