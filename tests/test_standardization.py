import pytest

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


def test_standardize_lab_names_reads_legacy_section_prefix_keys(monkeypatch):
    monkeypatch.setattr(
        standardization,
        "load_cache",
        lambda name: {
            "bioquímica - 25 hidroxivitamina d (25(oh)vit. d)": "Blood - 25-OH Vitamin D",
        },
    )

    result = standardization.standardize_lab_names(
        [
            ("25 Hidroxivitamina D (25(OH)Vit. D)", "Bioquímica"),
        ]
    )

    assert result[("25 Hidroxivitamina D (25(OH)Vit. D)", "Bioquímica")] == "Blood - 25-OH Vitamin D"


def test_standardize_lab_names_uses_safe_bare_fallback_when_contextual_values_agree(monkeypatch):
    monkeypatch.setattr(
        standardization,
        "load_cache",
        lambda name: {
            "ferritina": "Blood - Ferritin",
            "bioquimica - ferritina": "Blood - Ferritin",
        },
    )

    result = standardization.standardize_lab_names(
        [
            ("Ferritina", "Metabolismo do Ferro"),
        ]
    )

    assert result[("Ferritina", "Metabolismo do Ferro")] == "Blood - Ferritin"


def test_standardize_lab_names_blocks_bare_fallback_when_contextual_values_conflict(monkeypatch):
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
            ("Glicose", "Bioquímica"),
        ]
    )

    assert result[("Glicose", "Bioquímica")] == "$UNKNOWN$"


def test_standardize_lab_names_matches_folded_contextual_keys(monkeypatch):
    monkeypatch.setattr(
        standardization,
        "load_cache",
        lambda name: {
            "resistencia osmotica dos eritrocitos (apos incubacao) - hemolise inicial": "Blood - Osmotic Resistance Initial (After Incubation)",
            "imunologia - atc anti-transglutaminase (iga)": "Blood - Anti-Tissue Transglutaminase Antibody IgA (Anti-tTG IgA)",
        },
    )

    result = standardization.standardize_lab_names(
        [
            ("Hemolise Inicial", "Resistencia Osmotica dos Eritrocitos (após incubaçao)"),
            ("ATC ANTI-TRANSGLUTAMINASE (IgA)", "I M U N O L O G I A"),
        ]
    )

    assert result[("Hemolise Inicial", "Resistencia Osmotica dos Eritrocitos (após incubaçao)")] == "Blood - Osmotic Resistance Initial (After Incubation)"
    assert result[("ATC ANTI-TRANSGLUTAMINASE (IgA)", "I M U N O L O G I A")] == "Blood - Anti-Tissue Transglutaminase Antibody IgA (Anti-tTG IgA)"


def test_standardize_lab_names_does_not_fallback_to_legacy_key_when_section_is_present(monkeypatch):
    monkeypatch.setattr(
        standardization,
        "load_cache",
        lambda name: {
            "leucocitos": "Blood - Leukocytes",
            "leucocitos|exame microscopico do sedimento": "Urine Type II - Sediment - Leukocytes",
        },
    )

    result = standardization.standardize_lab_names(
        [
            ("LEUCOCITOS", "Sedimento urinário"),
        ]
    )

    assert result[("LEUCOCITOS", "Sedimento urinário")] == "$UNKNOWN$"


def test_standardize_lab_units_normalizes_compact_exponent_notation(monkeypatch):
    monkeypatch.setattr(
        standardization,
        "load_cache",
        lambda name: {
            "x10^9/l|blood - leukocytes": "10⁹/L",
            "x10^12/l|blood - erythrocytes": "10¹²/L",
        },
    )

    result = standardization.standardize_lab_units(
        [
            ("x109/L", "Blood - Leukocytes"),
            ("x1012/L", "Blood - Erythrocytes"),
        ]
    )

    assert result[("x109/L", "Blood - Leukocytes")] == "10⁹/L"
    assert result[("x1012/L", "Blood - Erythrocytes")] == "10¹²/L"


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


@pytest.mark.parametrize("raw_unit,cached_unit,standard_unit", [
    ("μg/g\nfezes", "µg/g fezes", "µg/g"),
    ("/100\nLeucocitos", "/100 Leucócitos", "/100 WBC"),
    ("  mg/dL\t", "mg/dl", "mg/dL"),
])
def test_unit_lookup_normalizes_wrapping_accents_and_micro_symbols(monkeypatch, raw_unit, cached_unit, standard_unit):
    name = "Feces - Synthetic"
    monkeypatch.setattr(standardization, "load_cache", lambda _: {f"{cached_unit}|{name.lower()}": standard_unit})
    assert standardization.standardize_lab_units([(raw_unit, name)])[(raw_unit, name)] == standard_unit


def test_unit_lookup_refuses_conflicting_folded_aliases_and_preserves_scale(monkeypatch):
    name = "Blood - Synthetic"
    monkeypatch.setattr(standardization, "load_cache", lambda _: {
        f"µg/g\nfezes|{name.lower()}": "µg/g",
        f"μg/g fezes|{name.lower()}": "mg/g",
        f"10⁹/l|{name.lower()}": "10⁹/L",
        f"10¹²/l|{name.lower()}": "10¹²/L",
        f"å|{name.lower()}": "Å",
        f"a|{name.lower()}": "A",
    })
    result = standardization.standardize_lab_units([("µg/g fezes", name), ("10⁹/L", name), ("10¹²/L", name)])
    assert result[("µg/g fezes", name)] == "$UNKNOWN$"
    assert result[("10⁹/L", name)] == "10⁹/L"
    assert result[("10¹²/L", name)] == "10¹²/L"
    symbols = standardization.standardize_lab_units([("Å", name), ("A", name)])
    assert symbols[("Å", name)] == "Å"
    assert symbols[("A", name)] == "A"


def test_name_lookup_preserves_context_while_ignoring_leading_footnote_daggers(monkeypatch):
    monkeypatch.setattr(standardization, "load_cache", lambda _: {
        "percentagem|reticulócitos": "Blood - Reticulocyte Count (%)",
        "percentagem|other panel": "Blood - Other (%)",
    })
    context = ("† Percentagem", "† RETICULÓCITOS")
    assert standardization.standardize_lab_names([context])[context] == "Blood - Reticulocyte Count (%)"


def test_name_lookup_refuses_ambiguous_footnote_folded_aliases(monkeypatch):
    monkeypatch.setattr(standardization, "load_cache", lambda _: {"assay|panel": "Blood - A", "† assay|panel": "Blood - B"})
    context = ("‡ Assay", "Panel")
    assert standardization.standardize_lab_names([context])[context] == "$UNKNOWN$"
