from types import SimpleNamespace

from parselabs import admin_commands


def test_admin_dispatch_passes_arguments_without_mutating_sys_argv(monkeypatch):
    received: list[list[str]] = []
    fake_module = SimpleNamespace(main=lambda argv: received.append(argv) or 0)
    monkeypatch.setattr(admin_commands.importlib, "import_module", lambda name: fake_module)
    original_argv = list(admin_commands.sys.argv)

    result = admin_commands.run_admin_command("validate-lab-specs", ["--example"])

    assert result == 0
    assert received == [["--example"]]
    assert admin_commands.sys.argv == original_argv
