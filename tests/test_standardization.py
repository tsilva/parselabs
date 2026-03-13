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


def test_standardize_lab_names_prefers_section_aware_keys(monkeypatch):
    monkeypatch.setattr(
        standardization,
        "load_cache",
        lambda name: {
            "glicose": "Blood - Glucose (Fasting)",
            "glicose|elementos anormais": "Urine Type II - Glucose",
        },
    )

    result = standardization.standardize_lab_names(
        [
            ("Glicose", "Elementos anormais"),
            ("Glicose", None),
        ]
    )

    assert result[("Glicose", "Elementos anormais")] == "Urine Type II - Glucose"
    assert result[("Glicose", None)] == "Blood - Glucose (Fasting)"


def test_standardize_lab_names_does_not_fallback_to_legacy_key_when_section_is_present(monkeypatch):
    monkeypatch.setattr(
        standardization,
        "load_cache",
        lambda name: {
            "leucocitos": "Blood - Leukocytes",
        },
    )

    result = standardization.standardize_lab_names(
        [
            ("LEUCOCITOS", "Sedimento urinário"),
        ]
    )

    assert result[("LEUCOCITOS", "Sedimento urinário")] == "$UNKNOWN$"


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
