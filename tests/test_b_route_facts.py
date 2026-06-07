import json

from src.nodes.b_route_facts import build_route_facts, build_sequence_route_facts, load_route_overlays


def test_load_route_overlays_accepts_overrides_and_direct_seed_shape(tmp_path):
    routes = {
        "overrides": [
            {"from": "start_point", "to": "act_1", "distance_km": 1.2, "duration_min": 8, "source": "test_override"}
        ],
        "res_1": {"distance_km": 2.4, "duration_min": 13, "source": "direct_seed"},
    }
    (tmp_path / "routes.json").write_text(json.dumps(routes), encoding="utf-8")

    overlays = load_route_overlays(str(tmp_path))

    assert overlays[("start_point", "act_1")]["source"] == "test_override"
    assert overlays[("start_point", "res_1")]["source"] == "direct_seed"


def test_build_route_facts_uses_overlay_then_coordinate_estimate(tmp_path):
    routes = {
        "overrides": [
            {"from": "start_point", "to": "act_1", "distance_km": 1.0, "duration_min": 9, "source": "fixture_overlay"}
        ]
    }
    (tmp_path / "routes.json").write_text(json.dumps(routes), encoding="utf-8")
    activity = {"poi_id": "act_1", "name": "活动", "coordinates": "121.4737,31.2304", "distance_km": 5}
    restaurant = {"poi_id": "res_1", "name": "餐厅", "coordinates": "121.4837,31.2304", "distance_km": 3}

    facts = build_route_facts(
        activity,
        restaurant,
        {"city": "上海"},
        mock_data_dir=tmp_path,
        source_order=["offline_routes_json", "coordinate_estimate"],
    )

    assert facts["legs"][0]["route_source"] == "fixture_overlay"
    assert facts["legs"][1]["route_source"] == "coordinate_estimate"
    assert facts["total_distance_km"] > 1.0
    assert facts["route_sources"] == ["coordinate_estimate", "fixture_overlay"]


def test_build_sequence_route_facts_aggregates_legs(tmp_path):
    nodes = [
        {"poi_id": "act_1", "name": "活动", "distance_km": 2.0},
        {"poi_id": "res_1", "name": "餐厅", "distance_km": 3.0},
    ]

    facts = build_sequence_route_facts(
        nodes,
        {"city": "上海"},
        mock_data_dir=tmp_path,
        source_order=["poi_distance_fallback"],
    )

    assert facts["feasible"] is True
    assert len(facts["legs"]) == 2
    assert facts["total_distance_km"] == 5.0
    assert facts["traffic_status"] == "moderate"
