import json

import pandas as pd

from parselabs.config import LabSpecsConfig, UNKNOWN_VALUE
from parselabs import standardization_refresh as refresh


def _make_lab_specs(tmp_path):
    config_path = tmp_path / "lab_specs.json"
    config_path.write_text(
        json.dumps(
            {
                "Blood - Glucose": {
                    "primary_unit": "mg/dL",
                    "lab_type": "blood",
                    "loinc_code": "2345-7",
                },
                "Urine Type II - pH": {
                    "primary_unit": "pH",
                    "lab_type": "urine",
                    "loinc_code": "5803-2",
                },
                "Urine Type II - Glucose": {
                    "primary_unit": "boolean",
                    "lab_type": "urine",
                    "loinc_code": "5792-7",
                },
            }
        ),
        encoding="utf-8",
    )
    return LabSpecsConfig(config_path=config_path)


def _install_cache_store(monkeypatch, *, name_cache=None, unit_cache=None):
    cache_store = {
        "name_standardization": dict(name_cache or {}),
        "unit_standardization": dict(unit_cache or {}),
    }

    monkeypatch.setattr(refresh, "load_cache", lambda name: dict(cache_store[name]))
    monkeypatch.setattr(refresh, "save_cache", lambda name, cache: cache_store.__setitem__(name, dict(cache)))
    return cache_store


def test_refresh_standardization_caches_noop_when_all_cached(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _install_cache_store(
        monkeypatch,
        name_cache={"glucose": "Blood - Glucose"},
        unit_cache={"mg/dl|blood - glucose": "mg/dL"},
    )
    dataframe = pd.DataFrame(
        [
            {
                "raw_lab_name": "Glucose",
                "raw_unit": "mg/dL",
                "lab_name": "Blood - Glucose",
            }
        ]
    )

    result = refresh.refresh_standardization_caches_from_dataframe(
        dataframe,
        lab_specs,
        model_id="test-model",
        dry_run=True,
    )

    assert result.attempted is False
    assert result.rebuild_required is False
    assert result.unresolved_names == ()
    assert result.unresolved_unit_pairs == ()


def test_refresh_standardization_caches_updates_name_and_dependent_unit_in_one_pass(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    cache_store = _install_cache_store(monkeypatch)
    dataframe = pd.DataFrame(
        [
            {
                "raw_lab_name": "pH",
                "raw_unit": "",
                "lab_name": UNKNOWN_VALUE,
            }
        ]
    )

    monkeypatch.setattr(
        refresh,
        "_standardize_names_with_llm",
        lambda uncached_names, standardized_names, client, model_id: {("pH", None): "Urine Type II - pH"},
    )

    def fake_standardize_units(uncached_pairs, standardized_units, client, model_id, lab_specs):
        assert uncached_pairs == [("", "Urine Type II - pH")]
        return {"null|urine type ii - ph": "pH"}

    monkeypatch.setattr(refresh, "_standardize_units_with_llm", fake_standardize_units)

    result = refresh.refresh_standardization_caches_from_dataframe(
        dataframe,
        lab_specs,
        model_id="test-model",
        client=object(),
    )

    assert result.uncached_names == (("pH", None),)
    assert result.uncached_unit_pairs == (("", "Urine Type II - pH"),)
    assert result.name_updates == 1
    assert result.unit_updates == 1
    assert result.unresolved_names == ()
    assert result.unresolved_unit_pairs == ()
    assert cache_store["name_standardization"]["ph"] == "Urine Type II - pH"
    assert cache_store["unit_standardization"]["null|urine type ii - ph"] == "pH"


def test_refresh_standardization_caches_handles_unit_only_misses(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    cache_store = _install_cache_store(
        monkeypatch,
        name_cache={"glucose": "Blood - Glucose"},
    )
    dataframe = pd.DataFrame(
        [
            {
                "raw_lab_name": "Glucose",
                "raw_unit": "mg/dl",
                "lab_name": "Blood - Glucose",
            }
        ]
    )

    monkeypatch.setattr(refresh, "_standardize_units_with_llm", lambda *args, **kwargs: {"mg/dl|blood - glucose": "mg/dL"})

    result = refresh.refresh_standardization_caches_from_dataframe(
        dataframe,
        lab_specs,
        model_id="test-model",
        client=object(),
    )

    assert result.uncached_names == ()
    assert result.uncached_unit_pairs == (("mg/dl", "Blood - Glucose"),)
    assert result.unit_updates == 1
    assert result.unresolved_unit_pairs == ()
    assert cache_store["unit_standardization"]["mg/dl|blood - glucose"] == "mg/dL"


def test_refresh_standardization_caches_reports_partial_unresolved_results(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _install_cache_store(monkeypatch)
    dataframe = pd.DataFrame(
        [
            {
                "raw_lab_name": "pH",
                "raw_unit": "",
                "lab_name": UNKNOWN_VALUE,
            }
        ]
    )

    monkeypatch.setattr(
        refresh,
        "_standardize_names_with_llm",
        lambda uncached_names, standardized_names, client, model_id: {("pH", None): "Urine Type II - pH"},
    )
    monkeypatch.setattr(refresh, "_standardize_units_with_llm", lambda *args, **kwargs: {})

    result = refresh.refresh_standardization_caches_from_dataframe(
        dataframe,
        lab_specs,
        model_id="test-model",
        client=object(),
    )

    assert result.name_updates == 1
    assert result.unit_updates == 0
    assert result.unresolved_names == ()
    assert result.unresolved_unit_pairs == (("", "Urine Type II - pH"),)


def test_refresh_standardization_caches_persists_contextual_name_keys(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    cache_store = _install_cache_store(
        monkeypatch,
        name_cache={"glicose": "Blood - Glucose"},
    )
    dataframe = pd.DataFrame(
        [
            {
                "raw_lab_name": "Glicose",
                "raw_section_name": "Elementos anormais",
                "raw_unit": "",
                "lab_name": UNKNOWN_VALUE,
            }
        ]
    )

    monkeypatch.setattr(
        refresh,
        "_standardize_names_with_llm",
        lambda uncached_names, standardized_names, client, model_id: {
            ("Glicose", "Elementos anormais"): "Urine Type II - Glucose"
        },
    )
    monkeypatch.setattr(
        refresh,
        "_standardize_units_with_llm",
        lambda *args, **kwargs: {"null|urine type ii - glucose": "boolean"},
    )

    result = refresh.refresh_standardization_caches_from_dataframe(
        dataframe,
        lab_specs,
        model_id="test-model",
        client=object(),
    )

    assert result.uncached_names == (("Glicose", "Elementos anormais"),)
    assert result.name_updates == 1
    assert result.unresolved_names == ()
    assert cache_store["name_standardization"]["glicose"] == "Blood - Glucose"
    assert cache_store["name_standardization"]["glicose|elementos anormais"] == "Urine Type II - Glucose"
