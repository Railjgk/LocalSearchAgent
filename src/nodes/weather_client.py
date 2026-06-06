"""B-stage weather context adapter built on C's WeatherForecaster node."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Mapping

from .b_utils import weather_city_from_constraints
from .weather_forecaster import WeatherForecaster


DEFAULT_CITY_ADCODE = "310000"
DEFAULT_TIMEOUT_SECONDS = 8.0

RAINY_KEYWORDS = ("rain", "shower", "storm", "thunder", "雨", "阵雨", "雷", "暴雨")
SNOW_KEYWORDS = ("snow", "雪")
WINDY_KEYWORDS = ("wind", "gale", "风")


def _env_mapping(env: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _read_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw_value = env.get(key)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _gaode_api_key(env: Mapping[str, str]) -> str:
    return (env.get("GAODE_WEATHER_API_KEY") or env.get("GAODE_API_KEY") or "").strip()


def _city_from_constraints(constraints: dict[str, Any] | None) -> str:
    return weather_city_from_constraints(constraints, default=DEFAULT_CITY_ADCODE)


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def classify_weather(weather: str | None, temperature: Any = None, windpower: Any = None) -> dict[str, Any]:
    """Convert C weather output into planner-friendly tags and risks."""

    weather_text = str(weather or "").strip()
    temp = _as_float(temperature)
    wind_text = str(windpower or "").strip()

    tags: list[str] = []
    risks: list[str] = []
    prefer_indoor = False

    if _contains_any(weather_text, RAINY_KEYWORDS):
        tags.append("rainy")
        risks.append("rain")
        prefer_indoor = True
    if _contains_any(weather_text, SNOW_KEYWORDS):
        tags.append("snowy")
        risks.append("snow")
        prefer_indoor = True
    if _contains_any(weather_text, WINDY_KEYWORDS) or any(token in wind_text for token in ("5", "6", "7", "8", "9")):
        tags.append("windy")
        risks.append("wind")

    if temp is not None:
        if temp >= 32:
            tags.append("hot")
            risks.append("heat")
            prefer_indoor = True
        elif temp <= 5:
            tags.append("cold")
            risks.append("cold")
            prefer_indoor = True
        elif 12 <= temp <= 28 and not risks:
            tags.append("comfortable")

    if not tags:
        tags.append("unknown")

    return {
        "condition_tags": sorted(set(tags)),
        "risk_tags": sorted(set(risks)),
        "prefer_indoor": prefer_indoor,
        "outdoor_caution": bool(risks),
    }


def _fallback_context(city: str, source: str, **extra: Any) -> dict[str, Any]:
    return {
        "available": False,
        "source": source,
        "city": city,
        "adcode": city,
        "condition_tags": ["unknown"],
        "risk_tags": [],
        "prefer_indoor": False,
        "outdoor_caution": False,
        **extra,
    }


def _normalize_weather_result(result: dict[str, Any], *, city: str) -> dict[str, Any]:
    weather = result.get("weather")
    temperature = result.get("temperature")
    windpower = result.get("windpower")
    classification = classify_weather(weather, temperature, windpower)
    feasible = bool(result.get("feasible"))

    return {
        "available": feasible,
        "source": "weather_forecaster",
        "city": result.get("city") or city,
        "adcode": result.get("adcode") or city,
        "weather": weather,
        "temperature": _as_float(temperature),
        "winddirection": result.get("winddirection"),
        "windpower": windpower,
        "humidity": _as_float(result.get("humidity")),
        "reporttime": result.get("reporttime"),
        **classification,
        "raw_weather": result,
    }


@lru_cache(maxsize=64)
def _fetch_weather_cached(city: str, api_key: str) -> dict[str, Any]:
    result = WeatherForecaster(api_key=api_key).get_weather(city, extensions="base")
    if not result.get("feasible"):
        return _fallback_context(
            city,
            "weather_forecaster",
            reason=result.get("reason") or "weather_unavailable",
            raw_weather=result,
        )
    return _normalize_weather_result(result, city=city)


def get_weather_context(
    constraints: dict[str, Any] | None = None,
    *,
    existing_context: dict[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return live weather context through C's WeatherForecaster, with safe fallback."""

    if isinstance(existing_context, dict) and existing_context:
        return dict(existing_context)

    env = _env_mapping(env)
    enabled = _read_bool(env, "WF_WEATHER_ENABLED", True)
    city = _city_from_constraints(constraints)
    if not enabled:
        return _fallback_context(city, "disabled")

    api_key = _gaode_api_key(env)
    if not api_key:
        return _fallback_context(city, "missing_api_key")

    try:
        return _fetch_weather_cached(city, api_key)
    except Exception as exc:
        return _fallback_context(
            city,
            "weather_forecaster",
            error_type=type(exc).__name__,
            error=str(exc)[:200].replace(api_key, "<redacted>"),
        )


def weather_context_is_active(weather_context: dict[str, Any] | None) -> bool:
    return bool(weather_context and weather_context.get("available"))
