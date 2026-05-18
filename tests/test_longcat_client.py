from __future__ import annotations

import pytest

from src.nodes import longcat_client


def _clear_longcat_env(monkeypatch):
    for key in (
        "WF_B_AI_ENABLED",
        "LONGCAT_API_KEY",
        "LONGCAT_APP_KEY",
        "LONGCAT_BASE_URL",
        "LONGCAT_MODEL",
        "LONGCAT_TIMEOUT_SECONDS",
        "LONGCAT_MAX_TOKENS",
        "LONGCAT_TEMPERATURE",
    ):
        monkeypatch.delenv(key, raising=False)


def test_longcat_config_is_default_off(monkeypatch):
    _clear_longcat_env(monkeypatch)

    assert longcat_client.is_b_ai_enabled() is False
    assert longcat_client.load_longcat_config() is None


def test_longcat_config_requires_key_when_enabled(monkeypatch):
    _clear_longcat_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")

    assert longcat_client.is_b_ai_enabled() is True
    assert longcat_client.load_longcat_config() is None


def test_chat_completion_uses_openai_compatible_endpoint(monkeypatch):
    _clear_longcat_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "true")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")
    calls = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "model": "LongCat-Flash-Chat",
                "choices": [
                    {
                        "message": {"content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 7},
            }

    def fake_post(url, headers, json, timeout):
        calls.update(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return FakeResponse()

    monkeypatch.setattr(longcat_client.requests, "post", fake_post)

    result = longcat_client.chat_completion([{"role": "user", "content": "hi"}])

    assert calls["url"] == "https://api.longcat.chat/openai/v1/chat/completions"
    assert calls["headers"]["Authorization"] == "Bearer test-key"
    assert calls["json"]["model"] == "LongCat-Flash-Chat"
    assert calls["json"]["messages"] == [{"role": "user", "content": "hi"}]
    assert result["content"] == "hello"
    assert result["usage"] == {"total_tokens": 7}


def test_chat_completion_reports_http_errors_without_key(monkeypatch):
    _clear_longcat_env(monkeypatch)
    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")

    class FakeResponse:
        status_code = 429
        text = "rate limited"

        def json(self):
            return {"error": {"message": "rate limited"}}

    monkeypatch.setattr(
        longcat_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(longcat_client.LongCatAPIError) as exc:
        longcat_client.chat_completion([{"role": "user", "content": "hi"}])

    message = longcat_client.sanitize_longcat_error(exc.value)
    assert "429" in message
    assert "test-key" not in message
