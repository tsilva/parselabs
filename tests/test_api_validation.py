from types import SimpleNamespace

from main import _classify_api_check_error, validate_api_access


class _FakeCompletions:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._error is not None:
            raise self._error
        return self._result


class _FakeClient:
    def __init__(self, result=None, error=None):
        self.chat = SimpleNamespace(completions=_FakeCompletions(result=result, error=error))


def test_validate_api_access_runs_minimal_completion():
    result = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])
    client = _FakeClient(result=result)

    is_valid, message = validate_api_access(client, "google/gemini-test", timeout=7)

    assert is_valid is True
    assert message == "API key and model validation passed"
    assert client.chat.completions.last_kwargs == {
        "model": "google/gemini-test",
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0,
        "max_tokens": 5,
        "timeout": 7,
    }


def test_validate_api_access_surfaces_authentication_failures():
    client = _FakeClient(error=RuntimeError("401 Unauthorized"))

    is_valid, message = validate_api_access(client, "google/gemini-test")

    assert is_valid is False
    assert "Authentication failed" in message


def test_classify_api_check_error_reports_missing_model():
    is_valid, message = _classify_api_check_error("404 model not found", timeout=10)

    assert is_valid is False
    assert "Model or endpoint not found" in message
