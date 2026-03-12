from parselabs import standardization


def test_standardize_lab_units_normalizes_missing_unit_tokens(monkeypatch):
    monkeypatch.setattr(
        standardization,
        "load_cache",
        lambda name: {
            "null|urine type ii - ph": "pH",
            "null|urine type ii - specific gravity": "unitless",
        },
    )

    result = standardization.standardize_lab_units(
        [
            ("", "Urine Type II - pH"),
            ("nan", "Urine Type II - Specific Gravity"),
        ]
    )

    assert result[("", "Urine Type II - pH")] == "pH"
    assert result[("nan", "Urine Type II - Specific Gravity")] == "unitless"


def test_load_cache_ignores_unknown_entries_for_standardization(tmp_path, monkeypatch):
    monkeypatch.setattr(standardization, "CACHE_DIR", tmp_path)
    (tmp_path / "unit_standardization.json").write_text(
        '{"null|urine type ii - ph":"$UNKNOWN$","null|urine type ii - specific gravity":"unitless"}',
        encoding="utf-8",
    )

    cache = standardization.load_cache("unit_standardization")

    assert cache == {"null|urine type ii - specific gravity": "unitless"}


def test_load_cache_ignores_unit_entries_with_unknown_lab_name(tmp_path, monkeypatch):
    monkeypatch.setattr(standardization, "CACHE_DIR", tmp_path)
    (tmp_path / "unit_standardization.json").write_text(
        '{"mg/dl|$unknown$":"mg/dL","null|urine type ii - ph":"pH"}',
        encoding="utf-8",
    )

    cache = standardization.load_cache("unit_standardization")

    assert cache == {"null|urine type ii - ph": "pH"}
