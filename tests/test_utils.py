from pathlib import Path

from parselabs.utils import load_dotenv_with_env


def test_load_dotenv_with_env_reads_config_dir_files(monkeypatch, tmp_path):
    home_dir = tmp_path / "home"
    config_dir = home_dir / ".config" / "parselabs"
    config_dir.mkdir(parents=True)

    (config_dir / ".env").write_text("OPENROUTER_API_KEY=base-key\nEXTRACT_MODEL_ID=base-model\n", encoding="utf-8")
    (config_dir / ".env.local").write_text("EXTRACT_MODEL_ID=local-model\nMAX_WORKERS=7\n", encoding="utf-8")

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home_dir))
    monkeypatch.setattr("sys.argv", ["parselabs"])
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("EXTRACT_MODEL_ID", raising=False)
    monkeypatch.delenv("MAX_WORKERS", raising=False)
    monkeypatch.chdir(tmp_path)

    env_name = load_dotenv_with_env()

    assert env_name == "local"
    assert Path.home() == home_dir
    assert __import__("os").getenv("OPENROUTER_API_KEY") == "base-key"
    assert __import__("os").getenv("EXTRACT_MODEL_ID") == "local-model"
    assert __import__("os").getenv("MAX_WORKERS") == "7"
