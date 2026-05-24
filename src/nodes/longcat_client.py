"""Optional LongCat OpenAI-compatible chat client."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import requests


DEFAULT_LONGCAT_BASE_URL = "https://api.longcat.chat/openai"
DEFAULT_LONGCAT_MODEL = "LongCat-Flash-Chat"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_TOKENS = 700
DEFAULT_TEMPERATURE = 0.2

TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}


class LongCatConfigurationError(RuntimeError):
    """Raised when LongCat is enabled but not configured."""


class LongCatAPIError(RuntimeError):
    """Raised when the LongCat API returns an unusable response."""


@dataclass(frozen=True)
class LongCatConfig:
    api_key: str
    base_url: str = DEFAULT_LONGCAT_BASE_URL
    model: str = DEFAULT_LONGCAT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE


def openai_chat_completions_url(base_url: str) -> str:
    """Build the OpenAI-format Chat Completions URL from a configured base URL."""

    cleaned = base_url.strip().rstrip("/")
    if cleaned.endswith("/v1/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    if cleaned == "https://api.longcat.chat":
        return f"{cleaned}/openai/v1/chat/completions"
    return f"{cleaned}/v1/chat/completions"


def _env_mapping(env: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return os.environ if env is None else env


def is_b_ai_enabled(env: Mapping[str, str] | None = None) -> bool:
    value = _env_mapping(env).get("WF_B_AI_ENABLED", "")
    return value.strip().lower() in TRUTHY_VALUES


def _read_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw_value = env.get(key)
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _read_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw_value = env.get(key)
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def load_longcat_config(env: Mapping[str, str] | None = None) -> LongCatConfig | None:
    """Load LongCat config only when the B AI switch is explicitly enabled."""

    env = _env_mapping(env)
    if not is_b_ai_enabled(env):
        return None

    api_key = (env.get("LONGCAT_API_KEY") or env.get("LONGCAT_APP_KEY") or "").strip()
    if not api_key:
        return None

    base_url = (env.get("LONGCAT_BASE_URL") or DEFAULT_LONGCAT_BASE_URL).strip().rstrip("/")
    model = (env.get("LONGCAT_MODEL") or DEFAULT_LONGCAT_MODEL).strip()

    return LongCatConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=_read_float(env, "LONGCAT_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        max_tokens=_read_int(env, "LONGCAT_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        temperature=_read_float(env, "LONGCAT_TEMPERATURE", DEFAULT_TEMPERATURE),
    )


def sanitize_longcat_error(error: BaseException, env: Mapping[str, str] | None = None) -> str:
    """Return an error string with known API keys redacted."""

    text = str(error)
    env = _env_mapping(env)
    for key_name in ("LONGCAT_API_KEY", "LONGCAT_APP_KEY"):
        key_value = env.get(key_name)
        if key_value:
            text = text.replace(key_value, "<redacted>")
    return text


def chat_completion(
    messages: Sequence[dict[str, str]],
    *,
    config: LongCatConfig | None = None,
) -> dict[str, Any]:
    """Call LongCat's OpenAI-compatible chat completion endpoint."""

    config = config or load_longcat_config()
    if config is None:
        raise LongCatConfigurationError(
            "LongCat B AI is disabled or LONGCAT_API_KEY/LONGCAT_APP_KEY is missing"
        )

    response = requests.post(
        openai_chat_completions_url(config.base_url),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "messages": list(messages),
            "stream": False,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        },
        timeout=config.timeout_seconds,
    )

    if response.status_code >= 400:
        try:
            error_payload = response.json().get("error", {})
            message = error_payload.get("message") or response.text
        except ValueError:
            message = response.text
        raise LongCatAPIError(f"LongCat API returned HTTP {response.status_code}: {message}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise LongCatAPIError("LongCat API returned non-JSON response") from exc

    choices = payload.get("choices") or []
    if not choices:
        raise LongCatAPIError("LongCat API returned no choices")

    first_choice = choices[0] or {}
    message = first_choice.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LongCatAPIError("LongCat API returned empty message content")

    return {
        "content": content.strip(),
        "model": payload.get("model") or config.model,
        "usage": payload.get("usage") or {},
        "finish_reason": first_choice.get("finish_reason"),
    }
