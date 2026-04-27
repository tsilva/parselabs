"""Shared prompt template loading helpers."""

from __future__ import annotations

from parselabs.paths import get_prompts_dir

_PROMPTS_DIR = get_prompts_dir()


def load_prompt_template(name: str) -> str:
    """Load one prompt template from the shared prompts directory."""

    return (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
