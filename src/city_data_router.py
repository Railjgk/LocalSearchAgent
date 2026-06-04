"""Resolve a request city to the local mock POI data directory."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, Iterator, MutableMapping


REPO_ROOT = Path(__file__).resolve().parents[1]

CITY_MOCK_DATA_DIRS = {
    "上海": REPO_ROOT / "experiments" / "mock_data" / "gaode_supply_shanghai_v2_20260527_full",
    "北京": REPO_ROOT / "experiments" / "mock_data" / "gaode_supply_beijing_v1",
    "青岛": REPO_ROOT / "experiments" / "mock_data" / "gaode_supply_qingdao_v1",
}

CITY_ALIASES = {
    "上海市": "上海",
    "shanghai": "上海",
    "北京市": "北京",
    "beijing": "北京",
    "青岛市": "青岛",
    "qingdao": "青岛",
}


def normalize_city(value: object | None) -> str | None:
    """Normalize a user/front-end city value into a supported city name."""

    text = str(value or "").strip()
    if not text:
        return None
    return CITY_ALIASES.get(text) or CITY_ALIASES.get(text.lower()) or (text if text in CITY_MOCK_DATA_DIRS else None)


def infer_city_from_text(text: object | None) -> str | None:
    """Infer supported cities from free text."""

    raw = str(text or "")
    raw_lower = raw.lower()
    for token, city in {**{city: city for city in CITY_MOCK_DATA_DIRS}, **CITY_ALIASES}.items():
        if token and token in raw:
            return city
        if token.isascii() and token in raw_lower:
            return city
    return None


def resolve_request_city(*, explicit_city: object | None = None, user_input: object | None = None) -> str | None:
    """Resolve city with explicit request value taking precedence over text."""

    return normalize_city(explicit_city) or infer_city_from_text(user_input)


def mock_data_dir_for_city(city: str | None) -> Path | None:
    """Return the configured mock data directory for a supported city."""

    normalized = normalize_city(city)
    if not normalized:
        return None
    path = CITY_MOCK_DATA_DIRS.get(normalized)
    return path if path and path.exists() else None


def apply_route_info(target: MutableMapping[str, Any], route_info: dict[str, str | None]) -> MutableMapping[str, Any]:
    """Attach resolved city routing metadata to a planning state/result."""

    city = route_info.get("city")
    mock_data_dir = route_info.get("mock_data_dir")
    if city:
        target["requested_city"] = city
        constraints = dict(target.get("constraints") or {})
        constraints["city"] = city
        hints = dict(constraints.get("route_pattern_hints") or target.get("route_pattern_hints") or {})
        if hints:
            hints["city"] = city
            constraints["route_pattern_hints"] = hints
            target["route_pattern_hints"] = hints
        target["constraints"] = constraints
    if mock_data_dir:
        target["mock_data_dir"] = mock_data_dir
    return target


@contextmanager
def routed_mock_data_dir(*, explicit_city: object | None = None, user_input: object | None = None) -> Iterator[dict[str, str | None]]:
    """Temporarily route B/C mock data reads for one request.

    Existing explicit `WF_MOCK_DATA_DIR` settings are restored after the request.
    If the city is unsupported or the data directory does not exist, the
    environment is left unchanged and the default mock data directory is used.
    """

    city = resolve_request_city(explicit_city=explicit_city, user_input=user_input)
    data_dir = mock_data_dir_for_city(city)
    previous = os.environ.get("WF_MOCK_DATA_DIR")
    if data_dir is not None:
        os.environ["WF_MOCK_DATA_DIR"] = str(data_dir)
    try:
        yield {
            "city": city,
            "mock_data_dir": str(data_dir) if data_dir is not None else None,
        }
    finally:
        if previous is None:
            os.environ.pop("WF_MOCK_DATA_DIR", None)
        else:
            os.environ["WF_MOCK_DATA_DIR"] = previous
