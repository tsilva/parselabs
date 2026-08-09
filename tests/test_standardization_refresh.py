import json
import logging
from types import SimpleNamespace

import pandas as pd

from parselabs import standardization_refresh as refresh
from parselabs.config import UNKNOWN_VALUE, LabSpecsConfig
from parselabs.runtime import get_openai_client_for_credentials


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


def _make_fake_client():
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))]
                )
            )
        )
    )


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
            ("Glicose", "Elementos anormais"): "Urine Type II - Glucose (Qualitative)"
        },
    )
    monkeypatch.setattr(
        refresh,
        "_standardize_units_with_llm",
        lambda *args, **kwargs: {"null|urine type ii - glucose (qualitative)": "boolean"},
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
    assert cache_store["name_standardization"]["glicose|elementos anormais"] == "Urine Type II - Glucose (Qualitative)"


def test_standardize_names_with_llm_ignores_malformed_items(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    monkeypatch.setattr(refresh, "load_prompt_template", lambda _: "prompt")
    monkeypatch.setattr(
        refresh,
        "parse_llm_json_response",
        lambda *args, **kwargs: [
            {"raw_lab_name": "Glucose", "raw_section_name": None, "standardized_name": "Blood - Glucose"},
            {"raw_lab_name": "Invented input", "raw_section_name": None, "standardized_name": "Blood - Glucose"},
            {"raw_lab_name": "Glucose", "raw_section_name": None, "standardized_name": "Invented candidate"},
            {"raw_lab_name": 1, "standardized_name": "Blood - Glucose"},
            {"raw_lab_name": "Glucose", "standardized_name": 123},
            {"raw_lab_name": "Glucose", "raw_section_name": [], "standardized_name": "Blood - Glucose"},
        ],
    )

    result = refresh._standardize_names_with_llm(
        [("Glucose", None)],
        lab_specs.standardized_names,
        _make_fake_client(),
        "test-model",
    )

    assert result == {("Glucose", None): "Blood - Glucose"}


def test_standardize_units_with_llm_ignores_malformed_items(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    monkeypatch.setattr(refresh, "load_prompt_template", lambda _: "prompt")
    monkeypatch.setattr(
        refresh,
        "parse_llm_json_response",
        lambda *args, **kwargs: [
            {"raw_unit": "mg/dL", "lab_name": "Blood - Glucose", "standardized_unit": "mg/dL"},
            {"raw_unit": "invented", "lab_name": "Blood - Glucose", "standardized_unit": "mg/dL"},
            {"raw_unit": "mg/dL", "lab_name": "Blood - Glucose", "standardized_unit": "invented"},
            {"raw_unit": None, "lab_name": "Blood - Glucose", "standardized_unit": "mg/dL"},
            {"raw_unit": "mg/dL", "lab_name": 1, "standardized_unit": "mg/dL"},
            {"raw_unit": "mg/dL", "lab_name": "Blood - Glucose", "standardized_unit": 5},
        ],
    )

    result = refresh._standardize_units_with_llm(
        [("mg/dL", "Blood - Glucose")],
        lab_specs.standardized_units,
        _make_fake_client(),
        "test-model",
        lab_specs,
    )

    assert result == {"mg/dl|blood - glucose": "mg/dL"}


def test_render_prompt_template_preserves_literal_braces():
    template = 'Use "{unknown}" and keep examples like {test} plus JSON {"a": 1}.'

    rendered = refresh._render_prompt_template(template, unknown="$UNKNOWN$")

    assert rendered == 'Use "$UNKNOWN$" and keep examples like {test} plus JSON {"a": 1}.'


def test_render_prompt_template_replaces_known_placeholders_only():
    template = "Candidates: {candidates}\nCount: {num_candidates}\nContext: {primary_units_context}"

    rendered = refresh._render_prompt_template(
        template,
        candidates='["Blood - Glucose"]',
        num_candidates=1,
        primary_units_context="",
    )

    assert rendered == 'Candidates: ["Blood - Glucose"]\nCount: 1\nContext: '


def test_prune_unknown_cache_entries_removes_unknown_values():
    pruned_cache, removed_count = refresh._prune_unknown_cache_entries(
        {
            "a": "Blood - Glucose",
            "b": "$UNKNOWN$",
            "c": "mg/dL",
        }
    )

    assert pruned_cache == {"a": "Blood - Glucose", "c": "mg/dL"}
    assert removed_count == 1


def test_prune_unknown_cache_entries_removes_unit_entries_without_resolved_lab_name():
    pruned_cache, removed_count = refresh._prune_unknown_cache_entries(
        {
            "mg/dl|blood - glucose": "mg/dL",
            "mg/dl|$unknown$": "mg/dL",
            "null|": "unitless",
        }
    )

    assert pruned_cache == {"mg/dl|blood - glucose": "mg/dL"}
    assert removed_count == 2


def _make_uncached_name_dataframe(count):
    return pd.DataFrame([{"raw_lab_name": f"Synthetic name {index}", "lab_name": UNKNOWN_VALUE} for index in range(count)])


def _install_successful_name_standardizer(monkeypatch, calls):
    def fake_standardize_names(batch, standardized_names, client, model_id):
        calls.append(list(batch))
        return {item: "Blood - Glucose" for item in batch}

    monkeypatch.setattr(refresh, "_standardize_names_with_llm", fake_standardize_names)


def test_refresh_batches_795_names_at_default_size_in_first_seen_order(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _install_cache_store(monkeypatch)
    calls = []
    _install_successful_name_standardizer(monkeypatch, calls)

    result = refresh.refresh_standardization_caches_from_dataframe(
        _make_uncached_name_dataframe(795),
        lab_specs,
        model_id="test-model",
        client=object(),
    )

    assert len(calls) == 16
    assert max(map(len, calls)) == refresh.DEFAULT_NAME_STANDARDIZATION_BATCH_SIZE == 50
    assert [item for batch in calls for item in batch] == [(f"Synthetic name {index}", None) for index in range(795)]
    assert result.name_updates == 795
    assert result.unresolved_names == ()


def test_refresh_batches_435_names_at_default_size(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _install_cache_store(monkeypatch)
    calls = []
    _install_successful_name_standardizer(monkeypatch, calls)

    result = refresh.refresh_standardization_caches_from_dataframe(
        _make_uncached_name_dataframe(435),
        lab_specs,
        model_id="test-model",
        client=object(),
    )

    assert len(calls) == 9
    assert max(map(len, calls)) == refresh.DEFAULT_NAME_STANDARDIZATION_BATCH_SIZE
    assert result.name_updates == 435
    assert result.unresolved_names == ()


def test_refresh_accepts_configured_name_batch_size(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _install_cache_store(monkeypatch)
    calls = []
    _install_successful_name_standardizer(monkeypatch, calls)

    refresh.refresh_standardization_caches_from_dataframe(
        _make_uncached_name_dataframe(9),
        lab_specs,
        model_id="test-model",
        client=object(),
        name_batch_size=4,
    )

    assert [len(batch) for batch in calls] == [4, 4, 1]


def test_refresh_batches_unit_mappings(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _install_cache_store(monkeypatch, name_cache={"glucose": "Blood - Glucose"})
    calls = []
    dataframe = pd.DataFrame(
        [
            {
                "raw_lab_name": "Glucose",
                "raw_unit": f"synthetic-unit-{index}",
                "lab_name": "Blood - Glucose",
            }
            for index in range(121)
        ]
    )

    def fake_standardize_units(batch, standardized_units, client, model_id, received_lab_specs):
        calls.append(list(batch))
        return {refresh._build_unit_cache_key(*item): "mg/dL" for item in batch}

    monkeypatch.setattr(refresh, "_standardize_units_with_llm", fake_standardize_units)

    result = refresh.refresh_standardization_caches_from_dataframe(
        dataframe,
        lab_specs,
        model_id="test-model",
        client=object(),
    )

    assert len(calls) == 3
    assert max(map(len, calls)) == refresh.DEFAULT_UNIT_STANDARDIZATION_BATCH_SIZE == 50
    assert result.unit_updates == 121
    assert result.unresolved_unit_pairs == ()


def test_successful_batches_persist_before_timeout_and_later_batches_continue(tmp_path, monkeypatch, caplog):
    lab_specs = _make_lab_specs(tmp_path)
    cache_store = _install_cache_store(monkeypatch)
    calls = []
    failed_names = {"Synthetic name 2", "Synthetic name 3"}

    def fake_standardize_names(batch, standardized_names, client, model_id):
        calls.append(list(batch))
        if any(raw_name in failed_names for raw_name, _ in batch):
            assert cache_store["name_standardization"]["synthetic name 0"] == "Blood - Glucose"
            assert cache_store["name_standardization"]["synthetic name 1"] == "Blood - Glucose"
            raise TimeoutError("synthetic timeout")
        return {item: "Blood - Glucose" for item in batch}

    monkeypatch.setattr(refresh, "_standardize_names_with_llm", fake_standardize_names)

    with caplog.at_level(logging.INFO, logger="parselabs.standardization_refresh"):
        result = refresh.refresh_standardization_caches_from_dataframe(
            _make_uncached_name_dataframe(5),
            lab_specs,
            model_id="test-model",
            client=object(),
            name_batch_size=2,
        )

    assert [len(batch) for batch in calls] == [2, 2, 1, 1, 1]
    assert calls[-1] == [("Synthetic name 4", None)]
    assert result.name_updates == 3
    assert result.unresolved_names == (("Synthetic name 2", None), ("Synthetic name 3", None))
    assert result.name_error is not None
    assert "1 name batch(es) incomplete" in result.name_error
    assert "Name batch 2/3 timed out; retrying as 2 smaller batches" in caplog.text


def test_timeout_retry_splits_once_and_is_bounded(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _install_cache_store(monkeypatch)
    call_sizes = []

    def always_timeout(batch, standardized_names, client, model_id):
        call_sizes.append(len(batch))
        raise TimeoutError("synthetic timeout")

    monkeypatch.setattr(refresh, "_standardize_names_with_llm", always_timeout)

    result = refresh.refresh_standardization_caches_from_dataframe(
        _make_uncached_name_dataframe(4),
        lab_specs,
        model_id="test-model",
        client=object(),
        name_batch_size=4,
    )

    assert call_sizes == [4, 2, 2]
    assert result.name_updates == 0
    assert len(result.unresolved_names) == 4
    assert result.name_error is not None


def test_missing_and_invalid_response_items_retry_only_incomplete_subset(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    cache_store = _install_cache_store(monkeypatch)
    calls = []

    def partially_malformed(batch, standardized_names, client, model_id):
        calls.append(list(batch))
        if len(calls) == 1:
            return {
                batch[0]: "Blood - Glucose",
                batch[1]: "Hallucinated candidate",
                ("Hallucinated input", None): "Blood - Glucose",
            }
        return {item: "Blood - Glucose" for item in batch}

    monkeypatch.setattr(refresh, "_standardize_names_with_llm", partially_malformed)

    result = refresh.refresh_standardization_caches_from_dataframe(
        _make_uncached_name_dataframe(3),
        lab_specs,
        model_id="test-model",
        client=object(),
        name_batch_size=3,
    )

    assert calls == [
        [("Synthetic name 0", None), ("Synthetic name 1", None), ("Synthetic name 2", None)],
        [("Synthetic name 1", None), ("Synthetic name 2", None)],
    ]
    assert result.name_updates == 3
    assert result.unresolved_names == ()
    assert result.name_error is None
    assert "hallucinated input" not in cache_store["name_standardization"]


def test_unknown_response_is_complete_but_remains_uncached(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    cache_store = _install_cache_store(monkeypatch)
    calls = []

    def return_unknown(batch, standardized_names, client, model_id):
        calls.append(list(batch))
        return {item: UNKNOWN_VALUE for item in batch}

    monkeypatch.setattr(refresh, "_standardize_names_with_llm", return_unknown)

    result = refresh.refresh_standardization_caches_from_dataframe(
        _make_uncached_name_dataframe(1),
        lab_specs,
        model_id="test-model",
        client=object(),
    )

    assert len(calls) == 1
    assert cache_store["name_standardization"] == {}
    assert result.name_updates == 0
    assert result.unresolved_names == (("Synthetic name 0", None),)
    assert result.name_error is None


def test_codex_standardization_requests_use_medium_reasoning(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    requests = []
    client = _make_fake_client()
    client.chat.completions.create = lambda **kwargs: (
        requests.append(kwargs) or SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))])
    )
    monkeypatch.setattr(refresh, "load_prompt_template", lambda _: "prompt")

    refresh._standardize_names_with_llm([("Glucose", None)], lab_specs.standardized_names, client, "codex/standardizer")
    refresh._standardize_units_with_llm(
        [("mg/dL", "Blood - Glucose")],
        lab_specs.standardized_units,
        client,
        "codex/standardizer",
        lab_specs,
    )

    assert len(requests) == 2
    assert all(request["reasoning_effort"] == "medium" for request in requests)


def test_non_codex_standardization_requests_omit_reasoning_effort(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    requests = []
    client = _make_fake_client()
    client.chat.completions.create = lambda **kwargs: (
        requests.append(kwargs) or SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))])
    )
    monkeypatch.setattr(refresh, "load_prompt_template", lambda _: "prompt")

    refresh._standardize_names_with_llm([("Glucose", None)], lab_specs.standardized_names, client, "openai/gpt-5")
    refresh._standardize_units_with_llm(
        [("mg/dL", "Blood - Glucose")],
        lab_specs.standardized_units,
        client,
        "openai/gpt-5",
        lab_specs,
    )

    assert len(requests) == 2
    assert all("reasoning_effort" not in request for request in requests)


def test_standardization_disables_sdk_retries_without_mutating_shared_client(tmp_path, monkeypatch):
    lab_specs = _make_lab_specs(tmp_path)
    _install_cache_store(monkeypatch)
    shared_client = get_openai_client_for_credentials("https://example.invalid/v1", "synthetic-shared-client-key")
    observed_retry_limits = []

    def capture_retry_limit(batch, standardized_names, client, model_id):
        observed_retry_limits.append(client.max_retries)
        return {item: "Blood - Glucose" for item in batch}

    monkeypatch.setattr(refresh, "_standardize_names_with_llm", capture_retry_limit)

    result = refresh.refresh_standardization_caches_from_dataframe(
        _make_uncached_name_dataframe(1),
        lab_specs,
        model_id="test-model",
        client=shared_client,
    )
    standardization_client = refresh._build_client(
        base_url="https://example.invalid/v1",
        api_key="synthetic-standardization-key",
    )

    assert result.name_updates == 1
    assert observed_retry_limits == [0]
    assert standardization_client.max_retries == 0
    assert shared_client.max_retries != 0
    assert standardization_client is not shared_client
