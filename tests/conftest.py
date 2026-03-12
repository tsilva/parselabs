from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pdf2image  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pdf2image"] = types.SimpleNamespace(convert_from_path=lambda *args, **kwargs: [])

try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    class _OpenAI:
        pass

    class _APIError(Exception):
        pass

    sys.modules["openai"] = types.SimpleNamespace(OpenAI=_OpenAI, APIError=_APIError)
