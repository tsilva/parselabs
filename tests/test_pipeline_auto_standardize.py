import sys
from argparse import Namespace
from types import SimpleNamespace

import pandas as pd
import pytest

from parselabs import pipeline as main
from parselabs.config import ExtractionConfig
from parselabs.standardization_refresh import StandardizationRefreshResult


def _build_config(tmp_path):
    return ExtractionConfig(
        input_path=tmp_path / "input",
        output_path=tmp_path / "output",
        openrouter_api_key="test-key",
        extract_model_id="test-model",
        max_workers=1,
    )


def _write_all_csv(output_path, *, value="Glucose"):
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"raw_lab_name": value, "raw_unit": "mg/dL", "lab_name": "Blood - Glucose"}]).to_csv(
        output_path / "all.csv",
        index=False,
    )


def test_parse_args_supports_no_auto_standardize(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["parselabs", "--profile", "tsilva", "--no-auto-standardize"])

    args = main.parse_args()

    assert args.profile == "tsilva"
    assert args.auto_standardize is False


def test_maybe_auto_standardize_outputs_rebuilds_when_refresh_changes(tmp_path, monkeypatch):
    output_path = tmp_path / "output"
    rebuilt_csv = output_path / "doc.csv"
    rebuilt_df = pd.DataFrame([{"raw_lab_name": "pH", "raw_unit": "", "lab_name": "Urine Type II - pH"}])
    _write_all_csv(output_path)
    calls = []

    monkeypatch.setattr(
        main,
        "refresh_standardization_caches_from_dataframe",
        lambda *args, **kwargs: StandardizationRefreshResult(
            uncached_names=(("pH", None),),
            uncached_unit_pairs=(("", "Urine Type II - pH"),),
            name_updates=1,
            unit_updates=1,
            unresolved_names=(),
            unresolved_unit_pairs=(),
        ),
    )
    monkeypatch.setattr(
        main,
        "_rebuild_review_outputs_from_processed_documents",
        lambda *args, **kwargs: SimpleNamespace(merged_review_df=rebuilt_df, csv_paths=[rebuilt_csv]),
    )
    monkeypatch.setattr(
        main,
        "_export_final_results",
        lambda final_df, hidden_cols, widths, output_path: calls.append(final_df.copy()),
    )

    csv_paths = main._maybe_auto_standardize_outputs(
        output_path=output_path,
        lab_specs=SimpleNamespace(),
        hidden_cols=[],
        widths={},
        model_id="test-model",
        base_url="https://example.com",
        api_key="test-key",
        auto_standardize=True,
        profile_name="tsilva",
        allow_pending=True,
    )

    assert csv_paths == [rebuilt_csv]
    assert len(calls) == 1
    assert calls[0].to_dict("records") == rebuilt_df.to_dict("records")


def test_maybe_auto_standardize_outputs_exports_final_rows_only_in_strict_mode(tmp_path, monkeypatch):
    output_path = tmp_path / "output"
    rebuilt_csv = output_path / "doc.csv"
    reviewed_corpus = SimpleNamespace(
        merged_review_df=pd.DataFrame([{"kind": "review"}]),
        final_df=pd.DataFrame([{"kind": "final"}]),
        csv_paths=[rebuilt_csv],
    )
    exported_frames = []
    _write_all_csv(output_path)

    monkeypatch.setattr(
        main,
        "refresh_standardization_caches_from_dataframe",
        lambda *args, **kwargs: StandardizationRefreshResult(
            uncached_names=(("pH", None),),
            uncached_unit_pairs=(),
            name_updates=1,
            unit_updates=0,
            unresolved_names=(),
            unresolved_unit_pairs=(),
        ),
    )
    monkeypatch.setattr(
        main,
        "_rebuild_review_outputs_from_processed_documents",
        lambda *args, **kwargs: reviewed_corpus,
    )
    monkeypatch.setattr(
        main,
        "_export_final_results",
        lambda final_df, hidden_cols, widths, output_path: exported_frames.append(final_df.copy()),
    )

    csv_paths = main._maybe_auto_standardize_outputs(
        output_path=output_path,
        lab_specs=SimpleNamespace(),
        hidden_cols=[],
        widths={},
        model_id="test-model",
        base_url="https://example.com",
        api_key="test-key",
        auto_standardize=True,
        profile_name="tsilva",
        allow_pending=False,
    )

    assert csv_paths == [rebuilt_csv]
    assert len(exported_frames) == 1
    assert exported_frames[0].to_dict("records") == [{"kind": "final"}]


def test_maybe_auto_standardize_outputs_skips_rebuild_when_disabled(tmp_path, monkeypatch):
    output_path = tmp_path / "output"
    _write_all_csv(output_path)
    refresh_calls = []

    monkeypatch.setattr(
        main,
        "refresh_standardization_caches_from_dataframe",
        lambda *args, **kwargs: refresh_calls.append(kwargs["dry_run"])
        or StandardizationRefreshResult(
            uncached_names=(("Glucose", None),),
            uncached_unit_pairs=(),
            name_updates=0,
            unit_updates=0,
            unresolved_names=(("Glucose", None),),
            unresolved_unit_pairs=(),
        ),
    )
    monkeypatch.setattr(main, "_rebuild_review_outputs_from_processed_documents", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rebuild not expected")))

    csv_paths = main._maybe_auto_standardize_outputs(
        output_path=output_path,
        lab_specs=SimpleNamespace(),
        hidden_cols=[],
        widths={},
        model_id="test-model",
        base_url="https://example.com",
        api_key="test-key",
        auto_standardize=False,
        profile_name="tsilva",
        allow_pending=True,
    )

    assert refresh_calls == [True]
    assert csv_paths == []


def test_maybe_auto_standardize_outputs_keeps_outputs_when_refresh_returns_structured_error(tmp_path, monkeypatch):
    output_path = tmp_path / "output"
    _write_all_csv(output_path)

    monkeypatch.setattr(
        main,
        "refresh_standardization_caches_from_dataframe",
        lambda *args, **kwargs: StandardizationRefreshResult(
            uncached_names=(("Glucose", None),),
            uncached_unit_pairs=(),
            name_updates=0,
            unit_updates=0,
            unresolved_names=(("Glucose", None),),
            unresolved_unit_pairs=(),
            name_error="boom",
        ),
    )
    monkeypatch.setattr(main, "_rebuild_review_outputs_from_processed_documents", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rebuild not expected")))

    csv_paths = main._maybe_auto_standardize_outputs(
        output_path=output_path,
        lab_specs=SimpleNamespace(),
        hidden_cols=[],
        widths={},
        model_id="test-model",
        base_url="https://example.com",
        api_key="test-key",
        auto_standardize=True,
        profile_name="tsilva",
        allow_pending=True,
    )

    assert csv_paths == []


def test_maybe_auto_standardize_outputs_propagates_unexpected_refresh_errors(tmp_path, monkeypatch):
    output_path = tmp_path / "output"
    _write_all_csv(output_path)

    monkeypatch.setattr(main, "refresh_standardization_caches_from_dataframe", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(main, "_rebuild_review_outputs_from_processed_documents", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rebuild not expected")))

    with pytest.raises(RuntimeError, match="boom"):
        main._maybe_auto_standardize_outputs(
            output_path=output_path,
            lab_specs=SimpleNamespace(),
            hidden_cols=[],
            widths={},
            model_id="test-model",
            base_url="https://example.com",
            api_key="test-key",
            auto_standardize=True,
            profile_name="tsilva",
            allow_pending=True,
        )


def test_run_for_profile_calls_auto_standardize_by_default(tmp_path, monkeypatch):
    config = _build_config(tmp_path)
    config.input_path.mkdir(parents=True, exist_ok=True)
    config.output_path.mkdir(parents=True, exist_ok=True)
    pdf_path = config.input_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    auto_calls = []

    monkeypatch.setattr(main, "_setup_profile_environment", lambda args, profile_name: (config, SimpleNamespace()))
    monkeypatch.setattr(main, "get_openai_client", lambda config: object())
    monkeypatch.setattr(main, "validate_api_access", lambda client, model_id: (True, "ok"))
    monkeypatch.setattr(main, "discover_pdf_files", lambda input_path, pattern: [pdf_path])
    monkeypatch.setattr(
        main,
        "run_pipeline_for_pdf_files",
        lambda pdf_files, config, lab_specs: SimpleNamespace(
            merged_review_df=pd.DataFrame([{"raw_lab_name": "Glucose"}]),
            failed_pages=[],
            csv_paths=[],
        ),
    )
    monkeypatch.setattr(main, "get_column_lists", lambda schema: ([], [], {}, {}))
    monkeypatch.setattr(main, "_export_final_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main,
        "_maybe_auto_standardize_outputs",
        lambda **kwargs: auto_calls.append(kwargs["auto_standardize"]) or [],
    )
    monkeypatch.setattr(main, "_report_extraction_failures", lambda *args, **kwargs: None)

    main.run_for_profile(Namespace(auto_standardize=True), "tsilva")

    assert auto_calls == [True]


def test_run_reviewed_json_rebuild_uses_auto_standardize_setting(tmp_path, monkeypatch):
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    auto_calls = []

    monkeypatch.setattr(main, "_setup_rebuild_environment", lambda profile_name: (SimpleNamespace(output_path=output_path, openrouter_base_url=None, openrouter_api_key=None, extract_model_id="model"), SimpleNamespace()))
    monkeypatch.setattr(main, "get_column_lists", lambda schema: ([], [], {}, {}))
    monkeypatch.setattr(
        main,
        "_rebuild_review_outputs_from_processed_documents",
        lambda *args, **kwargs: SimpleNamespace(merged_review_df=pd.DataFrame(), csv_paths=[]),
    )
    monkeypatch.setattr(main, "_export_final_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main,
        "_maybe_auto_standardize_outputs",
        lambda **kwargs: auto_calls.append(kwargs["auto_standardize"]) or [],
    )
    monkeypatch.setattr(main, "_report_extraction_failures", lambda *args, **kwargs: None)

    main._run_reviewed_json_rebuild(Namespace(auto_standardize=False, model=None), "tsilva", allow_pending=True)

    assert auto_calls == [False]


def test_run_reviewed_json_rebuild_exports_final_rows_only_when_fixture_ready(tmp_path, monkeypatch):
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    exported_frames = []
    reviewed_corpus = SimpleNamespace(
        merged_review_df=pd.DataFrame([{"kind": "review"}]),
        final_df=pd.DataFrame([{"kind": "final"}]),
        csv_paths=[],
    )

    monkeypatch.setattr(main, "_setup_rebuild_environment", lambda profile_name: (SimpleNamespace(output_path=output_path, openrouter_base_url=None, openrouter_api_key=None, extract_model_id="model"), SimpleNamespace()))
    monkeypatch.setattr(main, "get_column_lists", lambda schema: ([], [], {}, {}))
    monkeypatch.setattr(main, "_rebuild_review_outputs_from_processed_documents", lambda *args, **kwargs: reviewed_corpus)
    monkeypatch.setattr(main, "_export_final_results", lambda df, *args, **kwargs: exported_frames.append(df.copy()))
    monkeypatch.setattr(main, "_maybe_auto_standardize_outputs", lambda **kwargs: [])
    monkeypatch.setattr(main, "_report_extraction_failures", lambda *args, **kwargs: None)

    main._run_reviewed_json_rebuild(Namespace(auto_standardize=False, model=None), "tsilva", allow_pending=True)
    main._run_reviewed_json_rebuild(Namespace(auto_standardize=False, model=None), "tsilva", allow_pending=False)

    assert exported_frames[0].to_dict("records") == [{"kind": "review"}]
    assert exported_frames[1].to_dict("records") == [{"kind": "final"}]
