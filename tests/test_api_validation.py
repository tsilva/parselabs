from types import SimpleNamespace

import httpx
from openai import OpenAI

from parselabs.pipeline import _classify_api_check_error, validate_api_access


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

    def with_options(self, **kwargs):
        return self


def test_validate_api_access_runs_minimal_completion():
    result = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])
    client = _FakeClient(result=result)

    is_valid, message = validate_api_access(client, "google/gemini-test", timeout=7)

    assert is_valid is True
    assert message == "API key and model validation passed"
    assert client.chat.completions.last_kwargs == {
        "model": "openrouter/google/gemini-test",
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0,
        "max_tokens": 16,
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


def test_codex_validation_allows_slow_gateway_response():
    """A successful 15-second Codex request must survive the preflight deadline."""
    import json

    def respond(request):
        if request.extensions["timeout"]["read"] < 15:
            raise httpx.ReadTimeout("Codex response takes 15 seconds", request=request)
        payload = json.loads(request.content)
        assert payload["model"] == "codex/gpt-6.1-sol"
        assert payload["reasoning_effort"] == "high"
        return httpx.Response(200, json={
            "id": "synthetic-validation", "object": "chat.completion", "created": 0,
            "model": payload["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
        })

    with OpenAI(api_key="synthetic", base_url="http://agentbridge.test/api/v1", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        is_valid, message = validate_api_access(client, "codex/gpt-6.1-sol")
    assert is_valid, message


def test_validation_timeout_does_not_enqueue_retries():
    calls = []

    def respond(request):
        calls.append(request)
        raise httpx.ReadTimeout("Request timed out.", request=request)

    with OpenAI(api_key="synthetic", base_url="http://agentbridge.test/api/v1", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        is_valid, message = validate_api_access(client, "codex/gpt-6.1-sol", timeout=7)
    assert not is_valid
    assert len(calls) == 1
    assert "7s" in message
