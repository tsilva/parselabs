from utils.update_standardization_caches import _prune_unknown_cache_entries, _render_prompt_template


def test_render_prompt_template_preserves_literal_braces():
    template = 'Use "{unknown}" and keep examples like {test} plus JSON {"a": 1}.'

    rendered = _render_prompt_template(template, unknown="$UNKNOWN$")

    assert rendered == 'Use "$UNKNOWN$" and keep examples like {test} plus JSON {"a": 1}.'


def test_render_prompt_template_replaces_known_placeholders_only():
    template = "Candidates: {candidates}\nCount: {num_candidates}\nContext: {primary_units_context}"

    rendered = _render_prompt_template(
        template,
        candidates='["Blood - Glucose"]',
        num_candidates=1,
        primary_units_context="",
    )

    assert rendered == 'Candidates: ["Blood - Glucose"]\nCount: 1\nContext: '


def test_prune_unknown_cache_entries_removes_unknown_values():
    pruned_cache, removed_count = _prune_unknown_cache_entries(
        {
            "a": "Blood - Glucose",
            "b": "$UNKNOWN$",
            "c": "mg/dL",
        }
    )

    assert pruned_cache == {"a": "Blood - Glucose", "c": "mg/dL"}
    assert removed_count == 1


def test_prune_unknown_cache_entries_removes_unit_entries_without_resolved_lab_name():
    pruned_cache, removed_count = _prune_unknown_cache_entries(
        {
            "mg/dl|blood - glucose": "mg/dL",
            "mg/dl|$unknown$": "mg/dL",
            "null|": "unitless",
        }
    )

    assert pruned_cache == {"mg/dl|blood - glucose": "mg/dL"}
    assert removed_count == 2
