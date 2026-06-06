import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.execution_mock_api import (
    availability_check,
    call_execution_api,
    cancel_lodging_reservation,
    check_lodging_availability,
    execution_state_dir,
    execution_commit,
    reserve_lodging,
    reset_execution_state,
    route_check,
)
from src.nodes.mock_api_layer import mock_api_layer_node

GAODE_SNAPSHOT_DIR = (
    PROJECT_ROOT / "experiments" / "mock_data" / "gaode_supply_shanghai_v2_20260527_full"
)
DEFAULT_EXECUTION_STATE_DIR = PROJECT_ROOT / "experiments" / "mock_data" / "c_execution"


@pytest.fixture(autouse=True)
def _default_mock_env(monkeypatch, tmp_path):
    monkeypatch.delenv("WF_MOCK_DATA_DIR", raising=False)
    monkeypatch.setenv("WF_C_EXECUTION_STATE_DIR", str(tmp_path / "c_state"))


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


def test_execution_commit_continues_after_independent_step_failure() -> None:
    reset_execution_state()

    from src.tools.execution_mock_api import _write_state

    _write_state(
        "coupon_state.json",
        {
            "deals": {
                "deal_act_ceramic_family": {
                    "remaining": 0,
                }
            },
            "purchases": [],
        },
    )

    result = execution_commit(
        plan_id="partial_itinerary",
        user_id="u006",
        action_hints=_execution_slot_alignment_actions(),
    )

    assert result["success"] is False
    assert result["overall_status"] == "partial"
    assert result["failed_step"] == "step_activity_1"
    assert len(result["steps"]) == 2
    assert result["steps"][0]["status"] == "failed"
    assert result["steps"][0]["failure_reason"] == "inventory_empty"
    assert result["steps"][1]["status"] == "reserved"
    assert result["steps"][1]["success"] is True


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


def test_slot_full_retry_does_not_move_reservation_before_planned_time() -> None:
    reset_execution_state()

    from src.tools.execution_mock_api import _write_state

    _write_state(
        "availability_state.json",
        {
            "slots": {
                "prod_spa_light_tea_couple_set": {
                    "18:30": {
                        "remaining": 0,
                        "requires_reservation": True,
                        "queue_time_min": 8,
                    },
                    "17:30": {
                        "remaining": 6,
                        "requires_reservation": True,
                        "queue_time_min": 8,
                    },
                    "19:30": {
                        "remaining": 4,
                        "requires_reservation": True,
                        "queue_time_min": 8,
                    },
                }
            }
        },
    )
    actions = [
        {
            "action_type": "reserve_restaurant",
            "poi_id": "res_spa_light_tea",
            "merchant_id": "m_res_spa_light_tea",
            "product_id": "prod_spa_light_tea_couple_set",
            "deal_id": None,
            "time": "18:30",
            "people": 2,
            "requires_reservation": True,
        }
    ]

    result = execution_commit(
        plan_id="slot_full_later_retry",
        user_id="u004",
        action_hints=actions,
    )

    assert result["overall_status"] == "completed"
    assert result["retry_history"][0]["original_time"] == "18:30"
    assert result["retry_history"][0]["retry_time"] == "19:30"
    assert result["steps"][0]["time"] == "19:30"


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


def test_execution_mock_keeps_default_state_dir_when_fixture_dir_changes(monkeypatch) -> None:
    monkeypatch.setenv("WF_MOCK_DATA_DIR", str(GAODE_SNAPSHOT_DIR))
    monkeypatch.delenv("WF_C_EXECUTION_STATE_DIR", raising=False)

    assert execution_state_dir() == DEFAULT_EXECUTION_STATE_DIR


def test_execution_mock_reads_wf_mock_data_dir_shards(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WF_MOCK_DATA_DIR", str(GAODE_SNAPSHOT_DIR))
    monkeypatch.setenv("WF_C_EXECUTION_STATE_DIR", str(tmp_path / "c_state"))
    reset_execution_state()

    result = availability_check(
        poi_id="gaode_act_B0J1P52IP3",
        merchant_id="m_gaode_act_B0J1P52IP3",
        product_id="prod_gaode_act_B0J1P52IP3",
        time="14:00",
        party_size=3,
    )

    assert result["failure_reason"] != "unknown_poi"
    assert result["success"] is True
    assert "poi_id" in result["verified_fields"]
    assert (tmp_path / "c_state" / "availability_state.json").exists()


def test_execution_commit_accepts_gaode_snapshot_action_hints(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WF_MOCK_DATA_DIR", str(GAODE_SNAPSHOT_DIR))
    monkeypatch.setenv("WF_C_EXECUTION_STATE_DIR", str(tmp_path / "c_state"))
    reset_execution_state()

    result = execution_commit(
        plan_id="test_gaode_snapshot_plan",
        action_hints=[
            {
                "step": 1,
                "action_type": "order_activity_ticket",
                "poi_id": "gaode_act_B0J1P52IP3",
                "merchant_id": "m_gaode_act_B0J1P52IP3",
                "product_id": "prod_gaode_act_B0J1P52IP3",
                "deal_id": "deal_gaode_act_B0J1P52IP3",
                "time": "14:00",
                "party_size": 3,
                "mode": "drive",
            }
        ],
    )

    assert result["failure_reason"] != "unknown_poi"
    assert result["overall_status"] == "completed"


def test_execution_commit_reads_b_rag_data_dir_when_mock_dir_unset(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("WF_MOCK_DATA_DIR", raising=False)
    monkeypatch.setenv("WF_B_RAG_DATA_DIR", str(GAODE_SNAPSHOT_DIR))
    monkeypatch.setenv("WF_C_EXECUTION_STATE_DIR", str(tmp_path / "c_state"))
    reset_execution_state()

    result = execution_commit(
        plan_id="test_b_rag_data_dir_plan",
        action_hints=[
            {
                "step": 1,
                "action_type": "order_activity_ticket",
                "poi_id": "gaode_act_B0KB157SLO",
                "time": "17:00",
                "quantity": 2,
            }
        ],
    )

    assert result["failure_reason"] != "unknown_poi"
    assert result["overall_status"] == "completed"


def test_mock_api_layer_uses_b_rag_metadata_data_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("WF_MOCK_DATA_DIR", raising=False)
    monkeypatch.delenv("WF_B_RAG_DATA_DIR", raising=False)
    monkeypatch.setenv("WF_C_EXECUTION_STATE_DIR", str(tmp_path / "c_state"))
    reset_execution_state()

    result = mock_api_layer_node(
        {
            "user_id": "test_user",
            "selected_plan": {"plan_id": "plan_with_rag_poi"},
            "b_poi_rag_metadata": {"data_dir": str(GAODE_SNAPSHOT_DIR)},
            "action_sequence": [
                {
                    "step": 1,
                    "action_type": "order_activity_ticket",
                    "poi_id": "gaode_act_B0KB157SLO",
                    "time": "17:00",
                    "quantity": 2,
                    "name": "麦悠悠·SPA·推拿(徐家汇地铁站店)",
                }
            ],
            "execution_log": [],
        }
    )

    commit = result["execution_commit_result"]
    assert commit["failure_reason"] != "unknown_poi"
    assert commit["overall_status"] == "completed"


def test_execution_commit_retries_closed_slot_to_available_gaode_slot(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WF_MOCK_DATA_DIR", str(GAODE_SNAPSHOT_DIR))
    monkeypatch.setenv("WF_C_EXECUTION_STATE_DIR", str(tmp_path / "c_state"))
    reset_execution_state()

    result = execution_commit(
        plan_id="test_retry_closed_gaode_slot",
        action_hints=[
            {
                "step": 1,
                "action_type": "order_activity_ticket",
                "poi_id": "gaode_act_B0KB157SLO",
                "time": "10:00",
                "quantity": 2,
            }
        ],
    )

    assert result["overall_status"] == "completed"
    assert result["failure_reason"] is None
    assert result["retry_history"]
    assert result["retry_history"][0]["reason"] == "merchant_closed"
    assert result["steps"][0]["time"] != "10:00"


def test_execution_commit_accepts_dynamic_rag_lodging_after_real_poi(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WF_MOCK_DATA_DIR", str(GAODE_SNAPSHOT_DIR))
    monkeypatch.setenv("WF_C_EXECUTION_STATE_DIR", str(tmp_path / "c_state"))
    reset_execution_state()

    result = execution_commit(
        plan_id="test_dynamic_rag_lodging",
        action_hints=[
            {
                "step": 1,
                "action_type": "reserve_restaurant",
                "poi_id": "gaode_res_B0HBXCR3LD",
                "time": "17:30",
                "people": 2,
            },
            {
                "step": 2,
                "action_type": "reserve_lodging",
                "poi_id": "rag_hotel_rui_jin_intercontinental",
                "name": "上海瑞金洲际酒店",
                "check_in_date": "2026-06-06",
                "check_out_date": "2026-06-07",
                "room_count": 1,
                "people_count": 2,
            },
            {
                "step": 3,
                "action_type": "order_activity_ticket",
                "poi_id": "gaode_act_B0J1P52IP3",
                "time": "17:00",
                "quantity": 2,
            },
        ],
    )

    assert result["overall_status"] == "completed"
    lodging_step = result["steps"][1]
    assert lodging_step["status"] == "success"
    assert lodging_step["route_check"]["success"] is True
    assert "duration_min" in lodging_step["route_check"]["estimated_fields"]
    assert lodging_step["reservation_id"].startswith("H")
    next_activity_step = result["steps"][2]
    assert next_activity_step["success"] is True
    assert next_activity_step["route_check"]["success"] is True
    assert (
        next_activity_step["route_check"]["raw_api_results"]["route"]["route_source"]
        == "c_execution_route_gap_estimate"
    )


def test_lodging_availability_and_reservation_succeed() -> None:
    reset_execution_state()

    availability = check_lodging_availability(
        poi_id="gaode_hotel_disney_family",
        merchant_id="m_gaode_hotel_disney_family",
        check_in_date="2026-06-01",
        check_out_date="2026-06-02",
        room_count=1,
        people_count=2,
        budget=600,
    )
    reservation = reserve_lodging(
        poi_id="gaode_hotel_disney_family",
        merchant_id="m_gaode_hotel_disney_family",
        user_id="user_xxx",
        check_in_date="2026-06-01",
        check_out_date="2026-06-02",
        room_count=1,
        people_count=2,
        room_type=availability["room_type"],
        total_price=availability["total_price"],
        contact_required=True,
    )

    assert availability["success"] is True
    assert availability["available"] is True
    assert availability["room_type"] == "舒适大床房"
    assert availability["total_price"] == 399
    assert reservation["status"] == "success"
    assert reservation["reservation_id"] == "H202606010001"
    assert reservation["payment_required"] is False


def test_lodging_execution_commit_returns_reservation_id_and_skips_guidance_only() -> None:
    reset_execution_state()

    result = execution_commit(
        plan_id="lodging_two_day",
        user_id="user_xxx",
        action_hints=[
            {
                "poi_id": "parking_001",
                "type": "parking",
                "guidance_only": True,
                "name": "停车提醒",
            },
            {
                "action_type": "reserve_lodging",
                "poi_id": "gaode_hotel_disney_family",
                "merchant_id": "m_gaode_hotel_disney_family",
                "name": "迪士尼亲子度假酒店",
                "type": "hotel",
                "supply_domain": "hotel",
                "itinerary_role": "lodging",
                "day": 1,
                "start_time": "21:00",
                "end_time": "次日 09:00",
                "check_in_date": "2026-06-01",
                "check_out_date": "2026-06-02",
                "people_count": 2,
                "room_count": 1,
                "price": 399,
                "available": True,
                "evidence_text": "本地 POI RAG + mock availability",
            },
        ],
    )

    assert result["overall_status"] == "completed"
    assert len(result["steps"]) == 1
    assert result["steps"][0]["action_type"] == "reserve_lodging"
    assert result["steps"][0]["target_poi_id"] == "gaode_hotel_disney_family"
    assert result["steps"][0]["status"] == "success"
    assert result["steps"][0]["reservation_id"] == "H202606010001"
    assert result["steps"][0]["result_id"] == "H202606010001"


def test_lodging_sold_out_and_price_changed_are_structured_failures() -> None:
    reset_execution_state()

    from src.tools.execution_mock_api import _write_state

    _write_state(
        "lodging_state.json",
        {
            "gaode_hotel_disney_family": {
                "poi_id": "gaode_hotel_disney_family",
                "merchant_id": "m_gaode_hotel_disney_family",
                "available": True,
                "room_types": [
                    {
                        "room_type": "舒适大床房",
                        "price_per_night": 399,
                        "rooms_left": 0,
                    }
                ],
                "cancellation_policy": "入住前24小时可取消",
                "failure_modes": [
                    {"type": "sold_out", "message": "当前日期满房"},
                    {"type": "price_changed", "message": "房价发生变化，需要用户二次确认"},
                ],
            },
            "gaode_hotel_price_changed": {
                "poi_id": "gaode_hotel_price_changed",
                "merchant_id": "m_gaode_hotel_price_changed",
                "available": True,
                "active_failure_mode": "price_changed",
                "room_types": [
                    {
                        "room_type": "舒适大床房",
                        "price_per_night": 499,
                        "rooms_left": 2,
                    }
                ],
                "failure_modes": [
                    {"type": "price_changed", "message": "房价发生变化，需要用户二次确认"}
                ],
            },
        },
    )

    sold_out = reserve_lodging(
        poi_id="gaode_hotel_disney_family",
        merchant_id="m_gaode_hotel_disney_family",
        check_in_date="2026-06-01",
        check_out_date="2026-06-02",
        room_count=1,
        people_count=2,
    )
    price_changed = reserve_lodging(
        poi_id="gaode_hotel_price_changed",
        merchant_id="m_gaode_hotel_price_changed",
        check_in_date="2026-06-01",
        check_out_date="2026-06-02",
        room_count=1,
        people_count=2,
    )

    assert sold_out["success"] is False
    assert sold_out["failure_reason"] == "sold_out"
    assert sold_out["retry_history"][0]["reason"] == "sold_out"
    assert price_changed["success"] is False
    assert price_changed["status"] == "need_user_confirm"
    assert price_changed["failure_reason"] == "price_changed"
    assert price_changed["retry_history"][0]["reason"] == "price_changed"


def test_lodging_cancel_reservation_succeeds() -> None:
    reset_execution_state()

    reservation = reserve_lodging(
        poi_id="gaode_hotel_team_building",
        merchant_id="m_gaode_hotel_team_building",
        user_id="user_xxx",
        check_in_date="2026-06-01",
        check_out_date="2026-06-02",
        room_count=1,
        people_count=2,
    )
    cancelled = cancel_lodging_reservation(
        reservation_id=reservation["reservation_id"],
        reason="restaurant_reservation_failed",
    )

    assert cancelled["success"] is True
    assert cancelled["status"] == "success"
    assert cancelled["refund_policy"] == "未支付，无需退款"


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
