import pytest

from src.infra.azure import AzureClient


def test_build_messages_splits_generic_prompt_roles():
    client = AzureClient(endpoint="https://example.test", api_key="key", model="m")
    prompt = "System:\nRules\n\nUser:\nConvert to ATL formula: x\n\nAssistant:\n"

    messages = client._build_messages(prompt)

    assert messages == [
        {"role": "system", "content": "Rules"},
        {"role": "user", "content": "Convert to ATL formula: x"},
    ]


def test_build_messages_keeps_plain_prompt_as_user_message():
    client = AzureClient(endpoint="https://example.test", api_key="key", model="m")

    assert client._build_messages("plain") == [{"role": "user", "content": "plain"}]


class _FakeResponse:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        import json as _json

        return _json.loads(self.text)


def test_is_content_filter_error_detects_markers():
    from src.infra.azure import _is_content_filter_error

    assert _is_content_filter_error("...'code': 'content_filter'...")
    assert _is_content_filter_error("ResponsibleAIPolicyViolation")
    assert _is_content_filter_error("triggering the content management policy")
    assert not _is_content_filter_error("HTTP 500: internal error")
    assert not _is_content_filter_error("")


def test_generate_raises_content_filter_error_without_retrying(monkeypatch):
    from src.infra.azure import ContentFilterError

    client = AzureClient(endpoint="https://example.test", api_key="key", model="m")

    calls = {"count": 0}
    filtered_body = (
        "{\"detail\":\"Error code: 400 - {'error': {'message': 'The response was "
        "filtered due to the prompt triggering Azure OpenAI's content management "
        "policy.', 'code': 'content_filter', 'status': 400, 'innererror': "
        "{'code': 'ResponsibleAIPolicyViolation'}}}\"}"
    )

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        return _FakeResponse(500, filtered_body)

    monkeypatch.setattr(client.session, "post", fake_post)

    with pytest.raises(ContentFilterError):
        client.generate("a prompt")

    # Deterministic rejection must not be retried.
    assert calls["count"] == 1


def test_generate_retries_transient_errors(monkeypatch):
    client = AzureClient(endpoint="https://example.test", api_key="key", model="m")

    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        return _FakeResponse(503, "service unavailable")

    monkeypatch.setattr(client.session, "post", fake_post)
    monkeypatch.setattr("src.infra.azure.time.sleep", lambda *a, **k: None)

    with pytest.raises(RuntimeError):
        client.generate("a prompt")

    # Transient server errors are retried up to three attempts.
    assert calls["count"] == 3
