"""Exercise the portable profile example and optional lab filename convention."""

from pathlib import Path

from parselabs.config import ProfileConfig
from parselabs.store import discover_pdf_files


def test_profile_template_loads_from_a_private_runtime_copy(tmp_path):
    template = Path(__file__).parents[1] / "profiles/template.yaml.example"
    copied = tmp_path / "example.yaml"
    copied.write_bytes(template.read_bytes())
    profile = ProfileConfig.from_file(copied)

    assert profile.input_file_regex == "*.pdf"
    assert profile.input_path.is_absolute()
    assert profile.output_path.is_absolute()
    assert profile.demographics is None
    assert profile.workers is None


def test_shared_folder_glob_selects_nested_labs_and_excludes_orders(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    names = [
        "Analises.PDF", "2025-01-02 - analises - painel.pdf",
        "2025-01-02 - ultrasound.pdf", "2025-01-02 - receita - analises.pdf",
        "2025-01-02 - pedido exame - analises.pdf", "analises.txt",
    ]
    for name in names:
        (nested / name).touch()

    assert {p.name for p in discover_pdf_files(tmp_path, "*analises*.pdf")} == set(names[:2])
