import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.execution_mock_api import (
    availability_check,
    call_execution_api,
    execution_commit,
    reset_execution_state,
    route_check,
)


def _execution_slot_alignment_actions(deal_id="deal_act_001_ticket"):
    return [
        {
            "action_type": "order_activity_ticket",
            "poi_id": "act_001",
            "merchant_id": "m_act_001",
            "product_id": "prod_act_001_ticket",
            "deal_id": deal_id,
            "time": "14:00",
            "quantity": 3,
            "requires_reservation": True,
        },
        {
            "action_type": "reserve_restaurant",
            "poi_id": "res_001",
            "merchant_id": "m_res_001",
            "product_id": "prod_res_001_light_set",
            "deal_id": None,
            "time": "17:30",
            "people": 3,
            "requires_reservation": True,
        },
    ]


def _micro_vacation_actions():
    return [
        {
            "action_type": "order_activity_ticket",
            "poi_id": "act_micro_vacation_spa",
            "merchant_id": "m_act_micro_vacation_spa",
            "product_id": "prod_micro_spa_two_person",
            "deal_id": "deal_act_micro_spa",
            "time": "14:00",
            "quantity": 2,
            "requires_reservation": True,
        },
        {
            "action_type": "reserve_restaurant",
            "poi_id": "res_spa_light_tea",
            "merchant_id": "m_res_spa_light_tea",
            "product_id": "prod_spa_light_tea_couple_set",
            "deal_id": "deal_res_spa_light_tea",
            "time": "17:30",
            "people": 2,
            "requires_reservation": True,
        },
    ]


def test_execution_commit_normal_case_completes() -> None:
    reset_execution_state()

    result = execution_commit(
        plan_id="execution_slot_alignment",
        user_id="u001",
        action_hints=_execution_slot_alignment_actions(),
        execution_contract={"ready": True, "blocking_reasons": [], "checks": []},
    )

    assert result["success"] is True
    assert result["overall_status"] == "completed"
    assert result["steps"][0]["status"] == "ordered"
    assert result["steps"][0]["order_id"]
    assert result["steps"][1]["status"] == "reserved"
    assert result["steps"][1]["reservation_id"]
    assert result["retry_history"] == []


def test_micro_vacation_route_and_reservation_succeed() -> None:
    reset_execution_state()

    route = route_check(
        from_id="act_micro_vacation_spa",
        to_id="res_spa_light_tea",
        mode="drive",
    )
    result = call_execution_api(
        "/execution/commit",
        {
            "plan_id": "micro_vacation_relaxation",
            "user_id": "u002",
            "action_hints": _micro_vacation_actions(),
        },
    )

    assert route["success"] is True
    assert route["duration_min"] == 9
    assert result["overall_status"] == "completed"
    assert result["steps"][1]["status"] == "reserved"


def test_deal_id_null_skips_coupon_and_continues() -> None:
    reset_execution_state()

    actions = _execution_slot_alignment_actions(deal_id=None)
    result = execution_commit(
        plan_id="deal_null",
        user_id="u003",
        action_hints=actions[:1],
    )

    assert result["overall_status"] == "completed"
    assert result["steps"][0]["coupon_check"]["status"] == "skipped_no_deal"
    assert result["steps"][0]["status"] == "ordered"


def test_execution_commit_supports_addon_service_without_poi() -> None:
    reset_execution_state()

    result = execution_commit(
        plan_id="addon_only",
        user_id="u005",
        action_hints=[
            {
                "action_type": "order_addon_service",
                "addon_type": "cake",
                "address": "home",
                "time": "18:00",
                "notes": ["low_sugar"],
            }
        ],
    )

    assert result["overall_status"] == "completed"
    assert result["steps"][0]["status"] == "ordered"
    assert result["steps"][0]["order_id"].startswith("addon_mock_")
    assert result["steps"][0]["payment_required"] is True


def test_slot_full_returns_alternative_and_commit_retries() -> None:
    reset_execution_state()

    from src.tools.execution_mock_api import _write_state

    _write_state(
        "availability_state.json",
        {
            "slots": {
                "prod_spa_light_tea_couple_set": {
                    "17:30": {
                        "remaining": 0,
                        "requires_reservation": True,
                        "queue_time_min": 20,
                    }
                }
            }
        },
    )

    unavailable = availability_check(
        poi_id="res_spa_light_tea",
        merchant_id="m_res_spa_light_tea",
        product_id="prod_spa_light_tea_couple_set",
        deal_id="deal_res_spa_light_tea",
        time="17:30",
        party_size=2,
    )
    result = execution_commit(
        plan_id="slot_full_retry",
        user_id="u004",
        action_hints=_micro_vacation_actions(),
    )

    assert unavailable["success"] is False
    assert unavailable["failure_reason"] == "slot_full"
    assert unavailable["alternatives"][0]["time"] == "18:30"
    assert result["overall_status"] == "completed"
    assert result["retry_history"]
    assert result["steps"][1]["time"] == "18:30"


def test_availability_check_accepts_dict_available_slots() -> None:
    reset_execution_state()

    result = availability_check(
        poi_id="gaode_act_B0LK3ZZFZ5",
        merchant_id="m_gaode_act_B0LK3ZZFZ5",
        product_id="prod_gaode_act_B0LK3ZZFZ5",
        deal_id=None,
        time="17:00",
        party_size=5,
    )

    assert result["success"] is True
    assert result["status"] == "available"


def test_route_check_estimates_known_poi_pair_when_route_fixture_missing() -> None:
    reset_execution_state()

    result = route_check(
        from_id="gaode_res_B0H17H7L5G",
        to_id="gaode_act_B0LK3ZZFZ5",
        mode="drive",
    )

    assert result["success"] is True
    assert result["traffic_status"] == "estimated"
    assert "duration_min" in result["estimated_fields"]


def test_unknown_ids_return_structured_failures() -> None:
    reset_execution_state()

    unknown_poi = availability_check(
        poi_id="bad_poi",
        merchant_id="m_act_001",
        product_id="prod_act_001_ticket",
        deal_id=None,
        time="14:00",
        party_size=1,
    )
    unknown_product = availability_check(
        poi_id="act_001",
        merchant_id="m_act_001",
        product_id="bad_product",
        deal_id=None,
        time="14:00",
        party_size=1,
    )
    unknown_deal = availability_check(
        poi_id="act_001",
        merchant_id="m_act_001",
        product_id="prod_act_001_ticket",
        deal_id="bad_deal",
        time="14:00",
        party_size=1,
    )

    assert unknown_poi["failure_reason"] == "unknown_poi"
    assert unknown_product["failure_reason"] == "unknown_product"
    assert unknown_deal["failure_reason"] == "unknown_deal"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
