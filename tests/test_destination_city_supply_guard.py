from src.nodes.b_utils import item_matches_destination_city


def test_destination_city_guard_rejects_shanghai_coordinate_for_suzhou_trip():
    item = {
        "name": "上海田子坊",
        "coordinates": "121.475,31.210",
        "location": "黄浦区泰康路",
    }

    assert item_matches_destination_city(item, "苏州") is False


def test_destination_city_guard_allows_matching_explicit_city():
    item = {
        "name": "苏州评弹茶馆",
        "cityname": "苏州市",
        "address": "苏州工业园区金鸡湖附近",
    }

    assert item_matches_destination_city(item, "苏州") is True


def test_destination_city_guard_rejects_different_explicit_city():
    item = {
        "name": "鮨士道寿司轻食(日月光中心店)",
        "raw": {
            "cityname": "上海市",
            "address": "上海市黄浦区徐家汇路",
        },
    }

    assert item_matches_destination_city(item, "苏州") is False


def test_destination_city_guard_preserves_shanghai_local_supply():
    item = {
        "name": "自然科学探索馆",
        "coordinates": "121.458,31.233",
        "location": "静安区北京西路",
    }

    assert item_matches_destination_city(item, "上海") is True
