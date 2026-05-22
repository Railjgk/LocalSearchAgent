from src.nodes.poi_cleaning import should_exclude_poi


def test_excludes_closed_restaurant() -> None:
    item = {
        "name": "刃极日料自助(近铁广场店)(暂停营业)",
        "gaode_type": "餐饮服务;外国餐厅;日本料理",
    }

    assert should_exclude_poi(item, "restaurant")


def test_excludes_pure_food_service_activity() -> None:
    item = {
        "name": "老街小食(天津路店)",
        "category": "citywalk",
        "gaode_type": "餐饮服务;中餐厅;上海菜",
        "gaode_keyword": "上海 Citywalk 老街",
    }

    assert should_exclude_poi(item, "activity")


def test_keeps_market_and_hands_on_food_experiences() -> None:
    market = {
        "name": "青杉夜市",
        "category": "local_market",
        "gaode_type": "餐饮服务;中餐厅;中餐厅",
        "gaode_keyword": "上海 夜市 市集",
    }
    diy = {
        "name": "7's cake DIY烘焙教室",
        "category": "handcraft",
        "gaode_type": "餐饮服务;糕饼店;糕饼店",
        "gaode_keyword": "上海 烘焙 DIY",
    }

    assert not should_exclude_poi(market, "activity")
    assert not should_exclude_poi(diy, "activity")
