import os

from src.city_data_router import (
    apply_route_info,
    infer_city_from_text,
    mock_data_dir_for_city,
    normalize_city,
    resolve_request_city,
    routed_mock_data_dir,
)


def test_resolve_request_city_prefers_explicit_city() -> None:
    assert resolve_request_city(explicit_city="北京", user_input="青岛想吃海鲜") == "北京"
    assert resolve_request_city(explicit_city=None, user_input="青岛想吃海鲜") == "青岛"
    assert resolve_request_city(explicit_city="上海市", user_input="北京烤鸭") == "上海"


def test_infer_city_from_text_supports_known_cities() -> None:
    assert infer_city_from_text("北京周末想吃烤鸭") == "北京"
    assert infer_city_from_text("青岛周末想吃海鲜") == "青岛"
    assert infer_city_from_text("上海今天下午看展") == "上海"
    assert normalize_city("Qingdao") == "青岛"
    assert infer_city_from_text("I want seafood in BEIJING") == "北京"
    assert infer_city_from_text("周末想出去玩") is None


def test_mock_data_dir_for_city_points_to_existing_directories() -> None:
    assert mock_data_dir_for_city("北京").name == "gaode_supply_beijing_v1"
    assert mock_data_dir_for_city("青岛").name == "gaode_supply_qingdao_v1"
    assert mock_data_dir_for_city("上海").name == "gaode_supply_shanghai_v2_20260527_full"
    assert mock_data_dir_for_city("深圳") is None


def test_routed_mock_data_dir_restores_previous_env(monkeypatch) -> None:
    monkeypatch.setenv("WF_MOCK_DATA_DIR", "custom_dir")
    with routed_mock_data_dir(user_input="青岛想吃海鲜") as route_info:
        assert route_info["city"] == "青岛"
        assert route_info["mock_data_dir"]
        assert os.environ["WF_MOCK_DATA_DIR"].endswith("gaode_supply_qingdao_v1")
    assert os.environ["WF_MOCK_DATA_DIR"] == "custom_dir"


def test_routed_mock_data_dir_leaves_default_when_city_unknown(monkeypatch) -> None:
    monkeypatch.delenv("WF_MOCK_DATA_DIR", raising=False)
    with routed_mock_data_dir(user_input="周末想出去玩") as route_info:
        assert route_info["city"] is None
        assert route_info["mock_data_dir"] is None
        assert "WF_MOCK_DATA_DIR" not in os.environ


def test_apply_route_info_updates_state_and_constraints() -> None:
    state = {"constraints": {"route_pattern_hints": {"pace": "slow"}}}
    result = apply_route_info(state, {"city": "青岛", "mock_data_dir": "mock/qingdao"})

    assert result["requested_city"] == "青岛"
    assert result["mock_data_dir"] == "mock/qingdao"
    assert result["constraints"]["city"] == "青岛"
    assert result["route_pattern_hints"]["city"] == "青岛"
