from __future__ import annotations

from src.nodes.b_text_match import (
    fast_text_match_score,
    item_matches_terms,
    normalized_query_terms,
    semantic_cache_signature,
    semantic_text_index,
)


FIELDS = ("name", "tags", "category")


def test_normalized_query_terms_dedupes_short_and_duplicate_terms() -> None:
    assert normalized_query_terms(["咖啡", "咖啡", "a", " 下午茶 "]) == ["咖啡", "下午茶"]


def test_semantic_text_index_and_fast_score_distinguish_exact_and_substring() -> None:
    item = {
        "poi_id": "poi_1",
        "name": "安静咖啡馆",
        "tags": ["下午茶"],
        "category": "咖啡甜品",
    }

    blob, values = semantic_text_index(item, fields=FIELDS)

    assert "安静咖啡馆" in blob
    assert "下午茶" in values
    assert fast_text_match_score(["下午茶"], item, fields=FIELDS) == 3.2
    assert fast_text_match_score(["咖啡"], item, fields=FIELDS) == 2.3


def test_item_matches_terms_uses_normalized_tuple_terms() -> None:
    item = {"name": "奈尔宝儿童乐园", "tags": ["亲子", "室内"]}

    assert item_matches_terms(item, ("儿童",), fields=FIELDS) is True
    assert item_matches_terms(item, ("露营",), fields=FIELDS) is False


def test_semantic_cache_signature_tracks_primary_identity_fields() -> None:
    assert semantic_cache_signature({"poi_id": "1", "name": "A"}) != semantic_cache_signature(
        {"poi_id": "1", "name": "B"}
    )
