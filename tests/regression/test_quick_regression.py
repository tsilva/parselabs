"""Privacy boundaries and curated coverage using synthetic local fixtures only."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from parselabs.regression import build_fresh_case_diff, discover_approved_cases, get_required_regression_profile, private_regression_enabled, replay_review_decisions, select_regression_cases


@pytest.fixture
def quick_corpus(tmp_path, monkeypatch):
    root = tmp_path / "approved"
    entries = []
    for profile, layout in [("person-a", "scan"), ("person-b", "digital"), ("person-c", "hospital")]:
        directory = root / profile / "synthetic_12345678"
        (directory / "reviewed").mkdir(parents=True)
        (directory / "document.pdf").write_bytes(profile.encode())
        (directory / "expected.csv").write_text("date,lab_name\n")
        (directory / "reviewed" / "synthetic.001.json").write_text('{"lab_results": []}')
        (directory / "case.json").write_text(json.dumps({
            "stem": "synthetic", "file_hash": "12345678", "original_filename": "synthetic.pdf", "profile": profile,
        }))
        entries.append({
            "profile": profile, "case_id": directory.name, "pdf_sha256": hashlib.sha256(profile.encode()).hexdigest(),
            "tags": [layout], "rationale": "Synthetic layout coverage.",
        })
    manifest = {"version": 1, "required_profiles": [entry["profile"] for entry in entries], "max_documents": 3, "max_pages": 3, "cases": entries}
    path = tmp_path / "quick-suite.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setenv("PARSELABS_APPROVED_FIXTURES_DIR", str(root))
    return root, path, manifest


def test_quick_suite_selects_pinned_cases_in_curated_order(quick_corpus):
    root, _, manifest = quick_corpus
    cases = discover_approved_cases(root)
    manifest["cases"].reverse()
    quick_corpus[1].write_text(json.dumps(manifest))
    selected = select_regression_cases(cases, "quick")
    assert [case.profile for case in selected] == ["person-c", "person-b", "person-a"]


@pytest.mark.parametrize("problem,message", [
    ("missing", "missing or duplicate"), ("duplicate", "missing or duplicate"),
    ("profiles", "required profiles"), ("pages", "page budget"), ("hash", "PDF changed"),
])
def test_quick_suite_rejects_stale_or_incomplete_coverage(quick_corpus, problem, message):
    root, path, manifest = quick_corpus
    if problem == "missing":
        manifest["cases"][0]["case_id"] = "missing"
    elif problem == "duplicate":
        manifest["cases"].append(manifest["cases"][0])
    elif problem == "profiles":
        manifest["cases"].pop()
    elif problem == "pages":
        manifest["max_pages"] = 2
    else:
        manifest["cases"][0]["pdf_sha256"] = "stale"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        select_regression_cases(discover_approved_cases(root), "quick")


def test_quick_manifest_symlink_inside_git_is_rejected(quick_corpus, tmp_path):
    root, path, _ = quick_corpus
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    target = repository / "private.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="outside every Git"):
        select_regression_cases(discover_approved_cases(root), "quick")


def test_full_suite_does_not_require_a_quick_manifest(quick_corpus):
    root, path, _ = quick_corpus
    path.unlink()
    cases = discover_approved_cases(root)
    assert select_regression_cases(cases, "full") == cases


@pytest.mark.parametrize("flag", ["RUN_LOCAL_GOLDENS", "RUN_APPROVED_DOCS"])
def test_ci_refuses_private_suite_before_accessing_any_corpus(tmp_path, flag):
    env = dict(os.environ, CI="true", PARSELABS_APPROVED_FIXTURES_DIR=str(tmp_path / "missing-private-corpus"))
    env.pop("RUN_LOCAL_GOLDENS", None)
    env.pop("RUN_APPROVED_DOCS", None)
    env[flag] = "1"
    marker = "golden_local" if flag == "RUN_LOCAL_GOLDENS" else "approved_docs"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/regression/test_approved_docs.py", "-q", "-m", marker, "--golden-suite=quick"],
        cwd=Path(__file__).parents[2], env=env, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "Private medical regression tests cannot run in CI" in result.stdout
    assert "FileNotFoundError" not in result.stdout


def test_ordinary_tests_do_not_enable_private_access_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv("RUN_LOCAL_GOLDENS", raising=False)
    assert not private_regression_enabled("RUN_LOCAL_GOLDENS")


def test_fresh_regression_allows_gateway_without_authentication(monkeypatch):
    monkeypatch.setattr("parselabs.regression.load_profile_config", lambda name: SimpleNamespace(extract_model_id="synthetic", openrouter_api_key=""))
    assert get_required_regression_profile("synthetic").extract_model_id == "synthetic"


def test_fresh_review_replay_follows_observations_after_reordering(tmp_path):
    reviewed = tmp_path / "reviewed"
    fresh = tmp_path / "fresh"
    reviewed.mkdir()
    fresh.mkdir()
    first = {"raw_lab_name": "Assay A", "raw_value": "2.00", "raw_lab_unit": "mg/L", "review_status": "accepted"}
    second = {"raw_lab_name": "Assay B", "raw_value": "3", "raw_lab_unit": "mg/L", "review_status": "rejected"}
    (reviewed / "synthetic.001.json").write_text(json.dumps({"lab_results": [first, second]}))
    actual = [{**second, "review_status": None}, {**first, "raw_value": "2", "review_status": None}]
    target = fresh / "synthetic.001.json"
    target.write_text(json.dumps({"lab_results": actual}))
    replay_review_decisions(reviewed, fresh)
    assert [row["review_status"] for row in json.loads(target.read_text())["lab_results"]] == ["rejected", "accepted"]


def test_fresh_review_replay_refuses_added_rows_without_partial_writes(tmp_path):
    reviewed = tmp_path / "reviewed"
    fresh = tmp_path / "fresh"
    reviewed.mkdir()
    fresh.mkdir()
    row = {"raw_lab_name": "Assay", "raw_value": "2", "raw_lab_unit": "mg/L", "review_status": "accepted"}
    for number in [1, 2]:
        name = f"synthetic.{number:03}.json"
        (reviewed / name).write_text(json.dumps({"lab_results": [row]}))
        actual = [{**row, "review_status": None}] * number
        (fresh / name).write_text(json.dumps({"lab_results": actual}))
    before = {p.name: p.read_bytes() for p in fresh.iterdir()}
    with pytest.raises(ValueError, match="Fresh source rows differ"):
        replay_review_decisions(reviewed, fresh)
    assert {p.name: p.read_bytes() for p in fresh.iterdir()} == before


def test_fresh_comparison_ignores_position_but_detects_measurement_and_omission_changes():
    expected = pd.DataFrame([{"date": "2024-01-01", "lab_name": "Blood - A", "value": 2, "lab_unit": "mg/L", "page_number": 1, "result_index": 0, "raw_value": "2.00"}])
    actual = expected.assign(result_index=5, raw_value="2")
    assert not build_fresh_case_diff(expected, actual, "synthetic")
    assert build_fresh_case_diff(expected, actual.assign(value=3), "synthetic")
    assert build_fresh_case_diff(expected, actual.iloc[:0], "synthetic")


def test_fresh_comparison_ignores_qualitative_sentence_punctuation_but_not_result_changes():
    expected = pd.DataFrame([{"date": "2024-01-01", "lab_name": "Blood - A", "value_text": "NEGATIVE", "lab_unit": "unitless", "page_number": 1}])
    assert not build_fresh_case_diff(expected, expected.assign(value_text="NEGATIVE."), "synthetic")
    assert build_fresh_case_diff(expected, expected.assign(value_text="POSITIVE"), "synthetic")
