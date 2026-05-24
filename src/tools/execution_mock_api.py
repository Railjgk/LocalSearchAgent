"""C-stage execution mock API backed by B supply fixtures."""

from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "experiments" / "mock_data"
STATE_DIR = DATA_DIR / "c_execution"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return copy.deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return copy.deepcopy(default)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_fixture(name: str, default: Any) -> Any:
    return _load_json(DATA_DIR / name, default)


def _load_state(name: str, default: Any) -> Any:
    return _load_json(STATE_DIR / name, default)


def _write_state(name: str, data: Any) -> None:
    _write_json(STATE_DIR / name, data)


def reset_execution_state() -> None:
    """Reset mutable C execution state files for tests and demos."""

    defaults = {
        "availability_state.json": {"slots": {}},
        "reservation_state.json": {"reservations": []},
        "coupon_state.json": {"deals": {}, "purchases": []},
        "order_state.json": {"orders": []},
        "route_state.json": {"checked_routes": []},
        "execution_state.json": {"executions": []},
    }
    for name, payload in defaults.items():
        _write_state(name, payload)


def _index(items: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    return {item[key]: item for item in items if item.get(key)}


def _fixtures() -> Dict[str, Any]:
    activities = _load_fixture("activities.json", [])
    restaurants = _load_fixture("restaurants.json", [])
    merchants = _load_fixture("merchants.json", [])
    products = _load_fixture("products.json", [])
    deals = _load_fixture("deals.json", [])
    routes = _load_fixture("routes.json", [])

    pois = {}
    pois.update(_index(activities, "poi_id"))
    pois.update(_index(restaurants, "poi_id"))

    return {
        "activities": activities,
        "restaurants": restaurants,
        "merchants": _index(merchants, "merchant_id"),
        "products": _index(products, "product_id"),
        "deals": _index(deals, "deal_id"),
        "routes": routes,
        "pois": pois,
    }


def _response(
    *,
    success: bool,
    status: str,
    failure_reason: str | None = None,
    alternatives: List[Dict[str, Any]] | None = None,
    verified_fields: List[str] | None = None,
    estimated_fields: List[str] | None = None,
    raw_api_results: Dict[str, Any] | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    payload = {
        "success": success,
        "status": status,
        "failure_reason": failure_reason,
        "alternatives": alternatives or [],
        "verified_fields": verified_fields or [],
        "estimated_fields": estimated_fields or [],
        "raw_api_results": raw_api_results or {},
    }
    payload.update(extra)
    return payload


def _party_size(payload: Dict[str, Any]) -> int:
    value = payload.get("party_size", payload.get("quantity", payload.get("people", 1)))
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def _merchant_serves_poi(merchant: Dict[str, Any], poi_id: str | None) -> bool:
    if not merchant or not poi_id:
        return False
    if merchant.get("poi_id") == poi_id:
        return True
    poi_ids = merchant.get("poi_ids") or []
    return str(poi_id) in {str(item) for item in poi_ids}


def _deal_matches_refs(
    deal: Dict[str, Any],
    *,
    poi_id: str | None,
    merchant_id: str | None,
    product_id: str | None,
) -> bool:
    if not deal:
        return False
    if deal.get("product_id") != product_id:
        return False
    if deal.get("poi_id") not in (None, "", poi_id):
        return False
    if deal.get("merchant_id") not in (None, "", merchant_id):
        return False
    return True


def _resolve_refs(payload: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
    data = _fixtures()
    products = data["products"]
    pois = data["pois"]
    merchants = data["merchants"]
    deals = data["deals"]

    product_id = payload.get("product_id")
    product = products.get(product_id) if product_id else None

    poi_id = payload.get("poi_id") or (product or {}).get("poi_id")
    poi = pois.get(poi_id)
    if not poi:
        return None, _response(
            success=False,
            status="unavailable",
            failure_reason="unknown_poi",
            raw_api_results={"poi_id": poi_id},
        )

    merchant_id = payload.get("merchant_id") or poi.get("merchant_id")
    merchant = merchants.get(merchant_id)
    if not _merchant_serves_poi(merchant, poi_id):
        return None, _response(
            success=False,
            status="unavailable",
            failure_reason="unknown_poi",
            raw_api_results={"merchant_id": merchant_id, "poi_id": poi_id},
        )

    product_ids = poi.get("product_ids") or []
    product_id = product_id or poi.get("default_product_id") or next(iter(product_ids), None)
    product = products.get(product_id)
    if not product:
        return None, _response(
            success=False,
            status="unavailable",
            failure_reason="unknown_product",
            raw_api_results={"product_id": product_id},
        )

    if (
        product.get("poi_id") != poi_id
        or product.get("merchant_id") != merchant_id
        or product.get("available") is False
    ):
        return None, _response(
            success=False,
            status="unavailable",
            failure_reason="product_unavailable",
            raw_api_results={
                "poi_id": poi_id,
                "merchant_id": merchant_id,
                "product_id": product_id,
            },
        )

    deal = None
    deal_id = payload.get("deal_id")
    if deal_id:
        deal = deals.get(deal_id)
        if not deal:
            return None, _response(
                success=False,
                status="unavailable",
                failure_reason="unknown_deal",
                raw_api_results={"deal_id": deal_id},
            )
        if not _deal_matches_refs(
            deal,
            poi_id=poi_id,
            merchant_id=merchant_id,
            product_id=product_id,
        ):
            return None, _response(
                success=False,
                status="unavailable",
                failure_reason="unknown_deal",
                raw_api_results={
                    "deal_id": deal_id,
                    "product_id": product_id,
                    "merchant_id": merchant_id,
                },
            )

    refs = {
        "poi": poi,
        "poi_id": poi_id,
        "merchant": merchant,
        "merchant_id": merchant_id,
        "product": product,
        "product_id": product_id,
        "deal": deal,
        "deal_id": deal_id,
    }
    return refs, {}


def _slot_from_poi_overlay(overlay: Dict[str, Any], time_slot: str) -> Dict[str, Any] | None:
    slots = overlay.get("available_slots") or []
    if not isinstance(slots, list):
        return None

    match = next(
        (
            slot
            for slot in slots
            if isinstance(slot, dict) and slot.get("time") == time_slot
        ),
        None,
    )
    if not match:
        return None

    remaining = match.get(
        "remaining",
        match.get("inventory_left", overlay.get("inventory_left", 0)),
    )
    return {
        "remaining": remaining,
        "requires_reservation": bool(
            overlay.get("reservation_required", overlay.get("requires_reservation", False))
        ),
        "queue_time_min": overlay.get("queue_time_min", match.get("queue_time_min", 0)),
    }


def _poi_id_for_product(product_id: str) -> str | None:
    if not product_id:
        return None
    return (_fixtures()["products"].get(product_id) or {}).get("poi_id")


def _slot_snapshot(
    product_id: str,
    time_slot: str,
    poi_id: str | None = None,
) -> Dict[str, Any] | None:
    base = _load_fixture("availability.json", {})
    product_overlay = base.get(product_id, {})
    slot = None
    if isinstance(product_overlay, dict):
        slot = copy.deepcopy(product_overlay.get(time_slot))

    poi_id = poi_id or _poi_id_for_product(product_id)
    poi_overlay = base.get(poi_id, {}) if poi_id else {}
    if slot is None and isinstance(poi_overlay, dict):
        slot = _slot_from_poi_overlay(poi_overlay, time_slot)

    state = _load_state("availability_state.json", {"slots": {}})
    override = state.get("slots", {}).get(product_id, {}).get(time_slot)
    if override is not None:
        slot = copy.deepcopy(override)
    return slot


def _normalize_route_mode(mode: Any) -> str:
    value = str(mode or "drive").strip().lower()
    if value in {"driving", "car", "auto"}:
        return "drive"
    if value in {"walking", "walk"}:
        return "walk"
    return value or "drive"


def _route_records(raw_routes: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_routes, list):
        return [route for route in raw_routes if isinstance(route, dict)]
    if not isinstance(raw_routes, dict):
        return []

    records: List[Dict[str, Any]] = []
    overrides = raw_routes.get("overrides")
    if isinstance(overrides, list):
        records.extend(route for route in overrides if isinstance(route, dict))

    overlays = raw_routes.get("gaode_seed_v1_overlays")
    if isinstance(overlays, dict):
        for poi_id, route in overlays.items():
            if isinstance(route, dict):
                copied = dict(route)
                copied.setdefault("to_id", poi_id)
                records.append(copied)
    return records


def _route_endpoint(route: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = route.get(key)
        if value not in (None, ""):
            return value
    return None


def _set_slot_remaining(product_id: str, time_slot: str, remaining: int) -> None:
    state = _load_state("availability_state.json", {"slots": {}})
    state.setdefault("slots", {}).setdefault(product_id, {})
    slot = _slot_snapshot(product_id, time_slot) or {}
    slot["remaining"] = max(0, remaining)
    state["slots"][product_id][time_slot] = slot
    _write_state("availability_state.json", state)


def _available_alternatives(
    product_id: str,
    party_size: int,
    merchant: Dict[str, Any],
    poi_id: str | None = None,
) -> List[Dict[str, Any]]:
    poi_id = poi_id or _poi_id_for_product(product_id)
    base = _load_fixture("availability.json", {})
    product_overlay = base.get(product_id, {})
    product_times = set(product_overlay.keys()) if isinstance(product_overlay, dict) else set()
    poi_overlay = base.get(poi_id, {}) if poi_id else {}
    poi_times = {
        slot.get("time")
        for slot in (poi_overlay.get("available_slots") or [])
        if isinstance(slot, dict) and slot.get("time")
    }
    merchant_times = set(merchant.get("open_slots") or [])
    times = sorted(product_times | poi_times | merchant_times)
    alternatives = []
    for candidate_time in times:
        open_slots = merchant.get("open_slots") or times
        if candidate_time not in open_slots:
            continue
        slot = _slot_snapshot(product_id, candidate_time, poi_id)
        if not slot:
            continue
        remaining = int(slot.get("remaining", 0))
        if remaining >= party_size:
            alternatives.append(
                {
                    "time": candidate_time,
                    "remaining": remaining,
                    "queue_time_min": slot.get("queue_time_min", 0),
                }
            )
    return alternatives


def availability_check(**payload: Any) -> Dict[str, Any]:
    """Implementation of `/availability/check`."""

    refs, error = _resolve_refs(payload)
    if error:
        return error

    time_slot = payload.get("time")
    party_size = _party_size(payload)
    product = refs["product"]
    merchant = refs["merchant"]
    product_id = refs["product_id"]
    poi_id = refs["poi_id"]

    max_party_size = int(product.get("max_party_size") or 99)
    if party_size > max_party_size:
        return _response(
            success=False,
            status="unavailable",
            failure_reason="party_size_exceeded",
            alternatives=_available_alternatives(product_id, max_party_size, merchant, poi_id),
            verified_fields=["product_id", "party_size", "max_party_size"],
            raw_api_results={"max_party_size": max_party_size},
        )

    alternatives = _available_alternatives(product_id, party_size, merchant, poi_id)
    open_slots = merchant.get("open_slots") or [item["time"] for item in alternatives]
    if time_slot not in open_slots:
        return _response(
            success=False,
            status="unavailable",
            failure_reason="merchant_closed",
            alternatives=alternatives,
            verified_fields=["merchant_id", "time"],
            raw_api_results={"open_slots": merchant.get("open_slots", [])},
        )

    slot = _slot_snapshot(product_id, time_slot, poi_id)
    if not slot:
        return _response(
            success=False,
            status="unavailable",
            failure_reason="merchant_closed",
            alternatives=alternatives,
            verified_fields=["product_id", "time"],
        )

    remaining = int(slot.get("remaining", 0))
    queue_time_min = int(slot.get("queue_time_min", 0))

    if remaining < party_size:
        failure_reason = (
            "inventory_empty"
            if remaining <= 0 and not [item for item in alternatives if item["time"] != time_slot]
            else "slot_full"
        )
        return _response(
            success=False,
            status="unavailable",
            failure_reason=failure_reason,
            alternatives=[item for item in alternatives if item["time"] != time_slot],
            remaining=remaining,
            requires_reservation=bool(slot.get("requires_reservation")),
            queue_time_min=queue_time_min,
            verified_fields=["poi_id", "merchant_id", "product_id", "time", "remaining"],
            estimated_fields=["queue_time_min"],
            raw_api_results={"slot": slot},
        )

    return _response(
        success=True,
        status="available",
        failure_reason=None,
        alternatives=alternatives,
        available=True,
        remaining=remaining,
        requires_reservation=bool(slot.get("requires_reservation")),
        queue_time_min=queue_time_min,
        verified_fields=["poi_id", "merchant_id", "product_id", "time", "remaining"],
        estimated_fields=["queue_time_min"],
        raw_api_results={"slot": slot},
    )


def route_check(**payload: Any) -> Dict[str, Any]:
    """Implementation of `/route/check` using offline routes first."""

    from_id = payload.get("from_id") or payload.get("origin")
    to_id = payload.get("to_id") or payload.get("destination")
    mode = _normalize_route_mode(payload.get("mode", "drive"))
    routes = _route_records(_load_fixture("routes.json", []))

    match = next(
        (
            route
            for route in routes
            if _route_endpoint(route, "from_id", "from", "origin") == from_id
            and _route_endpoint(route, "to_id", "to", "destination") == to_id
            and _normalize_route_mode(route.get("mode", "drive")) == mode
        ),
        None,
    )
    if not match:
        match = next(
            (
                route
                for route in routes
                if _route_endpoint(route, "from_id", "from", "origin") == from_id
                and _route_endpoint(route, "to_id", "to", "destination") == to_id
            ),
            None,
        )

    if not match:
        result = _response(
            success=False,
            status="unavailable",
            failure_reason="route_not_found",
            feasible=False,
            distance_km=None,
            duration_min=None,
            mode=mode,
            traffic_status=None,
            verified_fields=["from_id", "to_id", "mode"],
        )
    else:
        feasible = match.get("feasible", True)
        distance_km = match.get("distance_km")
        duration_min = match.get("duration_min", match.get("travel_time_min"))
        result = _response(
            success=bool(feasible),
            status="available" if feasible else "unavailable",
            failure_reason=None if feasible else "route_not_found",
            feasible=bool(feasible),
            distance_km=distance_km,
            duration_min=duration_min,
            mode=_normalize_route_mode(match.get("mode", mode)),
            traffic_status=match.get("traffic_status", "unknown"),
            verified_fields=["from_id", "to_id", "mode", "distance_km", "duration_min"],
            raw_api_results={"route": match},
        )

    state = _load_state("route_state.json", {"checked_routes": []})
    state.setdefault("checked_routes", []).append(
        {
            "from_id": from_id,
            "to_id": to_id,
            "mode": mode,
            "success": result["success"],
            "checked_at": _now_iso(),
        }
    )
    _write_state("route_state.json", state)
    return result


def reservation_check(**payload: Any) -> Dict[str, Any]:
    """Implementation of `/reservation/check`."""

    availability = availability_check(**payload)
    if not availability["success"]:
        return _response(
            success=False,
            status="unavailable",
            failure_reason=availability.get("failure_reason"),
            alternatives=availability.get("alternatives", []),
            reservable=False,
            available_slots=availability.get("alternatives", []),
            verified_fields=availability.get("verified_fields", []),
            estimated_fields=availability.get("estimated_fields", []),
            raw_api_results={"availability": availability},
        )

    return _response(
        success=True,
        status="reservable",
        reservable=True,
        available_slots=availability.get("alternatives", []),
        failure_reason=None,
        alternatives=availability.get("alternatives", []),
        verified_fields=["poi_id", "merchant_id", "product_id", "time"],
        estimated_fields=availability.get("estimated_fields", []),
        raw_api_results={"availability": availability},
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _id(prefix: str, existing_count: int) -> str:
    return f"{prefix}_mock_{existing_count + 1:03d}_{uuid.uuid4().hex[:6]}"


def reservation_create(**payload: Any) -> Dict[str, Any]:
    """Implementation of `/reservation/create`."""

    check = reservation_check(**payload)
    if not check["success"]:
        return _response(
            success=False,
            status="failed",
            failure_reason=check.get("failure_reason"),
            alternatives=check.get("alternatives", []),
            reservation_id=None,
            raw_api_results={"reservation_check": check},
        )

    refs, error = _resolve_refs(payload)
    if error:
        return error

    time_slot = payload.get("time")
    party_size = _party_size(payload)
    product_id = refs["product_id"]
    slot = _slot_snapshot(product_id, time_slot, refs["poi_id"]) or {}
    remaining = int(slot.get("remaining", 0))

    state = _load_state("reservation_state.json", {"reservations": []})
    reservations = state.setdefault("reservations", [])
    reservation_id = _id("rsv", len(reservations))
    expires_at = (
        datetime.now(timezone.utc).replace(microsecond=0) + timedelta(minutes=20)
    ).isoformat()

    reservation = {
        "reservation_id": reservation_id,
        "status": "reserved",
        "poi_id": refs["poi_id"],
        "merchant_id": refs["merchant_id"],
        "product_id": product_id,
        "time": time_slot,
        "party_size": party_size,
        "expires_at": expires_at,
        "created_at": _now_iso(),
    }
    reservations.append(reservation)
    _write_state("reservation_state.json", state)
    _set_slot_remaining(product_id, time_slot, remaining - party_size)

    return _response(
        success=True,
        status="reserved",
        failure_reason=None,
        reservation_id=reservation_id,
        time=time_slot,
        expires_at=expires_at,
        verified_fields=["reservation_id", "time", "party_size"],
        raw_api_results={"reservation": reservation},
    )


def _deal_remaining(deal: Dict[str, Any]) -> int:
    state = _load_state("coupon_state.json", {"deals": {}, "purchases": []})
    override = state.get("deals", {}).get(deal["deal_id"], {})
    return int(
        override.get(
            "remaining",
            deal.get("remaining", deal.get("stock_limit_per_slot", 0)),
        )
    )


def _set_deal_remaining(deal_id: str, remaining: int) -> None:
    state = _load_state("coupon_state.json", {"deals": {}, "purchases": []})
    state.setdefault("deals", {}).setdefault(deal_id, {})
    state["deals"][deal_id]["remaining"] = max(0, remaining)
    _write_state("coupon_state.json", state)


def coupon_check(**payload: Any) -> Dict[str, Any]:
    """Implementation of `/coupon/check`."""

    if payload.get("deal_id") in (None, ""):
        return _response(
            success=True,
            status="skipped_no_deal",
            failure_reason=None,
            verified_fields=["deal_id"],
            raw_api_results={"deal_id": payload.get("deal_id")},
        )

    refs, error = _resolve_refs(payload)
    if error:
        return error

    deal = refs["deal"]
    time_slot = payload.get("time")
    valid_slots = deal.get("valid_slots") or deal.get("valid_time") or []
    if time_slot not in valid_slots:
        return _response(
            success=False,
            status="not_valid_for_slot",
            failure_reason="not_valid_for_slot",
            alternatives=[{"time": slot} for slot in valid_slots],
            verified_fields=["deal_id", "time", "valid_slots"],
            raw_api_results={"deal": deal},
        )

    remaining = _deal_remaining(deal)
    if remaining <= 0:
        return _response(
            success=False,
            status="unavailable",
            failure_reason="inventory_empty",
            alternatives=[],
            verified_fields=["deal_id", "remaining"],
            raw_api_results={"deal": deal},
        )

    return _response(
        success=True,
        status="coupon_available",
        failure_reason=None,
        amount=deal.get("amount", deal.get("sale_price", deal.get("price"))),
        remaining=remaining,
        refund_policy=deal.get("refund_policy"),
        verified_fields=["deal_id", "product_id", "valid_slots", "remaining"],
        raw_api_results={"deal": deal},
    )


def coupon_buy(**payload: Any) -> Dict[str, Any]:
    """Implementation of `/coupon/buy`."""

    check = coupon_check(**payload)
    if check.get("status") == "skipped_no_deal":
        return _response(
            success=True,
            status="skipped_no_deal",
            failure_reason=None,
            order_id=None,
            redeem_code=None,
            amount=0,
            payment_required=False,
            refund_policy=None,
            raw_api_results={"coupon_check": check},
        )
    if not check["success"]:
        return _response(
            success=False,
            status=check.get("status", "failed"),
            failure_reason=check.get("failure_reason"),
            alternatives=check.get("alternatives", []),
            order_id=None,
            redeem_code=None,
            amount=0,
            payment_required=False,
            raw_api_results={"coupon_check": check},
        )

    refs, error = _resolve_refs(payload)
    if error:
        return error

    deal = refs["deal"]
    amount = deal.get("amount", deal.get("sale_price", deal.get("price")))
    state = _load_state("coupon_state.json", {"deals": {}, "purchases": []})
    purchases = state.setdefault("purchases", [])
    order_id = _id("cpn_ord", len(purchases))
    redeem_code = f"MOCK{uuid.uuid4().hex[:8].upper()}"
    purchase = {
        "order_id": order_id,
        "deal_id": refs["deal_id"],
        "product_id": refs["product_id"],
        "merchant_id": refs["merchant_id"],
        "redeem_code": redeem_code,
        "amount": amount,
        "payment_required": False,
        "refund_policy": deal.get("refund_policy"),
        "created_at": _now_iso(),
    }
    purchases.append(purchase)
    state.setdefault("deals", {}).setdefault(refs["deal_id"], {})
    state["deals"][refs["deal_id"]]["remaining"] = _deal_remaining(deal) - 1
    _write_state("coupon_state.json", state)

    return _response(
        success=True,
        status="coupon_bought",
        failure_reason=None,
        order_id=order_id,
        redeem_code=redeem_code,
        amount=amount,
        payment_required=False,
        refund_policy=deal.get("refund_policy"),
        verified_fields=["deal_id", "order_id", "redeem_code", "amount"],
        raw_api_results={"purchase": purchase},
    )


def _order_amount(
    product: Dict[str, Any],
    party_size: int,
    coupon_order: Dict[str, Any] | None = None,
) -> float:
    if coupon_order and coupon_order.get("amount"):
        return float(coupon_order["amount"])
    price = float(product.get("price", 0))
    if product.get("price_type") == "package":
        return price
    return price * party_size


def order_create(**payload: Any) -> Dict[str, Any]:
    """Implementation of `/order/create`."""

    refs, error = _resolve_refs(payload)
    if error:
        return error

    party_size = _party_size(payload)
    time_slot = payload.get("time")
    consume_inventory = payload.get("consume_inventory", True)
    if consume_inventory:
        availability = availability_check(**payload)
        if not availability["success"]:
            return _response(
                success=False,
                status="failed",
                failure_reason=availability.get("failure_reason"),
                alternatives=availability.get("alternatives", []),
                completion_status="failed",
                raw_api_results={"availability": availability},
            )
        slot = _slot_snapshot(refs["product_id"], time_slot) or {}
        _set_slot_remaining(
            refs["product_id"],
            time_slot,
            int(slot.get("remaining", 0)) - party_size,
        )

    coupon_order = payload.get("coupon_order")
    amount = _order_amount(refs["product"], party_size, coupon_order)
    state = _load_state("order_state.json", {"orders": []})
    orders = state.setdefault("orders", [])
    order_id = _id("ord", len(orders))
    order_type = refs["product"].get("product_type", "product_order")
    order = {
        "order_id": order_id,
        "order_type": order_type,
        "poi_id": refs["poi_id"],
        "merchant_id": refs["merchant_id"],
        "product_id": refs["product_id"],
        "deal_id": refs["deal_id"],
        "reservation_id": payload.get("reservation_id"),
        "coupon_order_id": (coupon_order or {}).get("order_id"),
        "time": time_slot,
        "party_size": party_size,
        "amount": amount,
        "completion_status": "completed",
        "created_at": _now_iso(),
    }
    orders.append(order)
    _write_state("order_state.json", state)

    return _response(
        success=True,
        status="created",
        failure_reason=None,
        order_id=order_id,
        order_type=order_type,
        amount=amount,
        completion_status="completed",
        verified_fields=["order_id", "order_type", "amount", "completion_status"],
        raw_api_results={"order": order},
    )


def _step_id(action_type: str, index: int) -> str:
    if action_type == "order_activity_ticket":
        return f"step_activity_{index}"
    if action_type == "reserve_restaurant":
        return f"step_restaurant_{index}"
    return f"step_{index}"


def _action_payload(action: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(action)
    if "party_size" not in payload:
        payload["party_size"] = _party_size(action)
    return payload


def _retry_on_slot_full(
    payload: Dict[str, Any],
    check_result: Dict[str, Any],
    retry_history: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if check_result.get("failure_reason") != "slot_full":
        return payload, check_result

    alternatives = check_result.get("alternatives", [])
    retry_entry = {
        "reason": "slot_full",
        "original_time": payload.get("time"),
        "alternatives": alternatives,
        "retried": False,
    }
    if alternatives:
        retry_payload = dict(payload)
        retry_payload["time"] = alternatives[0]["time"]
        retry_result = availability_check(**retry_payload)
        retry_entry.update(
            {
                "retried": True,
                "retry_time": retry_payload["time"],
                "retry_success": retry_result.get("success", False),
            }
        )
        retry_history.append(retry_entry)
        return retry_payload, retry_result

    retry_history.append(retry_entry)
    return payload, check_result


def execution_commit(
    *,
    plan_id: str | None = None,
    user_id: str | None = None,
    action_hints: List[Dict[str, Any]] | None = None,
    execution_contract: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Implementation of `/execution/commit`."""

    action_hints = action_hints or []
    retry_history: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []
    execution_id = f"exec_mock_{uuid.uuid4().hex[:8]}"

    if execution_contract and execution_contract.get("ready") is False:
        result = {
            "success": False,
            "execution_id": execution_id,
            "overall_status": "rejected",
            "steps": [],
            "failed_step": "execution_contract",
            "failure_reason": "schema_mismatch",
            "retry_history": [],
            "raw_api_results": {"execution_contract": execution_contract},
        }
        _record_execution(plan_id, user_id, result)
        return result

    previous_poi_id = None
    for index, action in enumerate(action_hints, start=1):
        action_type = action.get("action_type")
        payload = _action_payload(action)
        refs, ref_error = _resolve_refs(payload)
        step = {
            "step_id": _step_id(action_type or "unknown", index),
            "action_type": action_type,
            "status": "pending",
            "poi_id": payload.get("poi_id"),
            "merchant_id": payload.get("merchant_id"),
            "product_id": payload.get("product_id"),
            "deal_id": payload.get("deal_id"),
            "time": payload.get("time"),
        }

        if ref_error:
            step.update(
                {
                    "status": "failed",
                    "success": False,
                    "failure_reason": ref_error.get("failure_reason"),
                    "raw_api_results": {"refs": ref_error},
                }
            )
            steps.append(step)
            result = _commit_result(execution_id, "failed", steps, step, retry_history)
            _record_execution(plan_id, user_id, result)
            return result

        payload.setdefault("poi_id", refs["poi_id"])
        payload.setdefault("merchant_id", refs["merchant_id"])
        payload.setdefault("product_id", refs["product_id"])
        step.update(
            {
                "poi_id": refs["poi_id"],
                "merchant_id": refs["merchant_id"],
                "product_id": refs["product_id"],
            }
        )

        if previous_poi_id and previous_poi_id != refs["poi_id"]:
            route = route_check(
                from_id=previous_poi_id,
                to_id=refs["poi_id"],
                mode=payload.get("mode", "drive"),
            )
            step["route_check"] = route
            if not route["success"]:
                step.update(
                    {
                        "status": "failed",
                        "success": False,
                        "failure_reason": route.get("failure_reason"),
                    }
                )
                steps.append(step)
                result = _commit_result(execution_id, "failed", steps, step, retry_history)
                _record_execution(plan_id, user_id, result)
                return result

        availability = availability_check(**payload)
        payload, availability = _retry_on_slot_full(payload, availability, retry_history)
        step["availability_check"] = availability
        step["time"] = payload.get("time")

        if not availability["success"]:
            step.update(
                {
                    "status": "failed",
                    "success": False,
                    "failure_reason": availability.get("failure_reason"),
                    "alternatives": availability.get("alternatives", []),
                }
            )
            steps.append(step)
            result = _commit_result(execution_id, "failed", steps, step, retry_history)
            _record_execution(plan_id, user_id, result)
            return result

        reservation = None
        requires_reservation = payload.get(
            "requires_reservation",
            refs["product"].get("requires_reservation", False),
        )
        if requires_reservation or action_type == "reserve_restaurant":
            reservation = reservation_create(**payload)
            step["reservation_check"] = reservation
            if not reservation["success"]:
                step.update(
                    {
                        "status": "failed",
                        "success": False,
                        "failure_reason": reservation.get("failure_reason"),
                        "alternatives": reservation.get("alternatives", []),
                    }
                )
                steps.append(step)
                result = _commit_result(execution_id, "failed", steps, step, retry_history)
                _record_execution(plan_id, user_id, result)
                return result
            step["reservation_id"] = reservation.get("reservation_id")

        coupon = coupon_check(**payload)
        step["coupon_check"] = coupon
        if not coupon["success"] and coupon.get("status") != "skipped_no_deal":
            step.update(
                {
                    "status": "failed",
                    "success": False,
                    "failure_reason": coupon.get("failure_reason"),
                    "alternatives": coupon.get("alternatives", []),
                }
            )
            steps.append(step)
            result = _commit_result(execution_id, "failed", steps, step, retry_history)
            _record_execution(plan_id, user_id, result)
            return result

        if action_type == "order_activity_ticket":
            coupon_order = coupon_buy(**payload)
            step["coupon_order"] = coupon_order
            if not coupon_order["success"]:
                step.update(
                    {
                        "status": "failed",
                        "success": False,
                        "failure_reason": coupon_order.get("failure_reason"),
                    }
                )
                steps.append(step)
                result = _commit_result(execution_id, "failed", steps, step, retry_history)
                _record_execution(plan_id, user_id, result)
                return result

            order = order_create(
                **payload,
                reservation_id=(reservation or {}).get("reservation_id"),
                coupon_order=coupon_order,
                consume_inventory=False,
            )
            step["order_create"] = order
            if not order["success"]:
                step.update(
                    {
                        "status": "failed",
                        "success": False,
                        "failure_reason": order.get("failure_reason"),
                    }
                )
                steps.append(step)
                result = _commit_result(execution_id, "failed", steps, step, retry_history)
                _record_execution(plan_id, user_id, result)
                return result

            step.update(
                {
                    "status": "ordered",
                    "success": True,
                    "order_id": order.get("order_id"),
                    "amount": order.get("amount"),
                    "completion_status": order.get("completion_status"),
                }
            )
        elif action_type == "reserve_restaurant":
            step.update(
                {
                    "status": "reserved",
                    "success": True,
                    "completion_status": "completed",
                }
            )
        else:
            step.update(
                {
                    "status": "failed",
                    "success": False,
                    "failure_reason": "unsupported_action_type",
                }
            )
            steps.append(step)
            result = _commit_result(execution_id, "failed", steps, step, retry_history)
            _record_execution(plan_id, user_id, result)
            return result

        steps.append(step)
        previous_poi_id = refs["poi_id"]

    result = {
        "success": True,
        "execution_id": execution_id,
        "overall_status": "completed",
        "steps": steps,
        "failed_step": None,
        "failure_reason": None,
        "retry_history": retry_history,
    }
    _record_execution(plan_id, user_id, result)
    return result


def _commit_result(
    execution_id: str,
    overall_status: str,
    steps: List[Dict[str, Any]],
    failed_step: Dict[str, Any],
    retry_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "success": False,
        "execution_id": execution_id,
        "overall_status": overall_status,
        "steps": steps,
        "failed_step": failed_step.get("step_id"),
        "failure_reason": failed_step.get("failure_reason"),
        "retry_history": retry_history,
    }


def _record_execution(
    plan_id: str | None,
    user_id: str | None,
    result: Dict[str, Any],
) -> None:
    state = _load_state("execution_state.json", {"executions": []})
    record = {
        "plan_id": plan_id,
        "user_id": user_id,
        "created_at": _now_iso(),
        **result,
    }
    state.setdefault("executions", []).append(record)
    _write_state("execution_state.json", state)


ENDPOINTS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "/availability/check": availability_check,
    "/route/check": route_check,
    "/reservation/check": reservation_check,
    "/reservation/create": reservation_create,
    "/coupon/check": coupon_check,
    "/coupon/buy": coupon_buy,
    "/order/create": order_create,
    "/execution/commit": execution_commit,
}


def call_execution_api(path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Small in-process dispatcher mirroring the desired C API paths."""

    if path not in ENDPOINTS:
        return _response(
            success=False,
            status="not_found",
            failure_reason="unknown_endpoint",
            raw_api_results={"path": path},
        )
    return ENDPOINTS[path](**(payload or {}))
