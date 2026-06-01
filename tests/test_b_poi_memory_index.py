from src.nodes.b_poi_memory_index import (
    build_poi_memory_index,
    load_poi_memory_index_cache,
    retrieve_poi_memory_candidates,
    save_poi_memory_index_cache,
)


def test_poi_memory_index_retrieves_relevant_chinese_poi_before_rerank():
    items = [
        {
            "poi_id": "kid_craft",
            "name": "树屋亲子手作乐园",
            "category": "亲子手作",
            "tags": ["儿童友好", "陶艺", "室内"],
        },
        {
            "poi_id": "yoga_studio",
            "name": "Yogaga优珈家瑜伽普拉提",
            "category": "健身中心",
            "tags": ["室内", "低强度", "放松"],
        },
        {
            "poi_id": "benbang_food",
            "name": "玲珑上海菜馆",
            "category": "本帮菜",
            "tags": ["晚餐", "上海菜"],
        },
    ]

    index = build_poi_memory_index(items)
    candidates, meta = retrieve_poi_memory_candidates(
        index,
        node_terms=["亲子", "手作", "儿童友好"],
        global_terms=[],
        limit=3,
    )

    assert candidates[0]["poi_id"] == "kid_craft"
    assert meta["retriever"] == "local_poi_memory_bm25_v1"
    assert meta["candidate_pool_size"] >= 1


def test_poi_memory_index_separates_meal_semantics():
    items = [
        {
            "poi_id": "light_lunch",
            "name": "喜禾年健康中式轻食",
            "category": "轻食",
            "restaurant_category": "中式轻食",
            "tags": ["午餐", "清淡", "少油"],
        },
        {
            "poi_id": "shanghai_dinner",
            "name": "阿金大白上海菜馆",
            "category": "本帮菜",
            "restaurant_category": "上海菜",
            "tags": ["晚餐", "本帮菜"],
        },
    ]

    index = build_poi_memory_index(items)
    lunch_candidates, _ = retrieve_poi_memory_candidates(
        index,
        node_terms=["午餐", "清淡", "轻食"],
        global_terms=[],
        limit=2,
    )
    dinner_candidates, _ = retrieve_poi_memory_candidates(
        index,
        node_terms=["晚餐", "本帮菜", "上海菜"],
        global_terms=[],
        limit=2,
    )

    assert lunch_candidates[0]["poi_id"] == "light_lunch"
    assert dinner_candidates[0]["poi_id"] == "shanghai_dinner"


def test_poi_memory_index_disk_cache_roundtrip(tmp_path):
    items = [
        {
            "poi_id": "bbq_shop",
            "name": "炭火烤肉小馆",
            "category": "烤肉",
            "tags": ["烧烤", "日式烧肉"],
        }
    ]
    index = build_poi_memory_index(items)
    cache_path = tmp_path / "restaurant_index.pkl"

    assert save_poi_memory_index_cache(cache_path, signature="sig-v1", index=index)
    loaded = load_poi_memory_index_cache(cache_path, expected_signature="sig-v1")

    assert loaded is not None
    assert loaded["doc_count"] == 1
    assert load_poi_memory_index_cache(cache_path, expected_signature="other-sig") is None


def test_poi_memory_index_uses_gaode_raw_business_tags_for_breakfast():
    items = [
        {
            "poi_id": "raw_baozi",
            "name": "城市早餐店",
            "category": "餐厅",
            "raw": {
                "name": "城市早餐店",
                "type": "餐饮服务;中餐厅;中餐厅",
                "biz_ext": {
                    "tag": "鲜肉包子,小馄饨,豆浆,油条",
                    "cost": "9.00",
                },
            },
        },
        {
            "poi_id": "generic_nearby",
            "name": "附近餐厅",
            "category": "餐厅",
            "raw": {
                "name": "附近餐厅",
                "type": "餐饮服务;中餐厅;中餐厅",
                "biz_ext": {"tag": "家常菜,炒菜"},
            },
        },
    ]

    index = build_poi_memory_index(items)
    candidates, meta = retrieve_poi_memory_candidates(
        index,
        node_terms=["早餐", "包子", "馄饨"],
        global_terms=[],
        limit=2,
    )

    assert candidates[0]["poi_id"] == "raw_baozi"
    assert {"包子", "馄饨"}.intersection(set(meta["matched_terms"]))
