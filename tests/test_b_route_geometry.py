from src.nodes.b_route_geometry import (
    filter_candidates_by_geo_window,
    haversine_km,
    item_coordinates,
    parse_coordinates,
)


def test_parse_coordinates_accepts_common_shapes():
    assert parse_coordinates("121.50,31.20") == (121.5, 31.2)
    assert parse_coordinates([121.5, 31.2]) == (121.5, 31.2)
    assert parse_coordinates({"lng": 121.5}) is None


def test_item_coordinates_falls_back_to_lng_lat_fields():
    assert item_coordinates({"coordinates": "121.50,31.20"}) == (121.5, 31.2)
    assert item_coordinates({"lng": "121.51", "lat": "31.21"}) == (121.51, 31.21)
    assert item_coordinates({"longitude": 121.52, "latitude": 31.22}) == (121.52, 31.22)


def test_haversine_km_is_reasonable_for_nearby_points():
    distance = haversine_km((121.4737, 31.2304), (121.4837, 31.2304))
    assert 0.8 <= distance <= 1.2


def test_filter_candidates_by_geo_window_keeps_minimum_near_origin():
    constraints = {"city": "上海", "route_origin": "121.4737,31.2304"}
    candidates = [
        {"poi_id": "near", "coordinates": "121.4740,31.2310", "distance_km": 20},
        {"poi_id": "far", "coordinates": "121.7000,31.5000", "distance_km": 1},
        {"poi_id": "unknown"},
    ]

    filtered, meta = filter_candidates_by_geo_window(
        candidates,
        constraints=constraints,
        user_profile={},
        min_keep=1,
    )

    assert meta["applied"] is True
    assert filtered[0]["poi_id"] == "near"
    assert filtered[0]["distance_source"] == "origin_coordinate_estimate"
