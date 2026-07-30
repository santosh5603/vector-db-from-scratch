from vectordb.metadata import matches, filter_ids


def test_bare_value_is_equality():
    assert matches({"category": "shoes"}, {"category": "shoes"})
    assert not matches({"category": "shoes"}, {"category": "hats"})


def test_multiple_keys_are_anded():
    md = {"category": "shoes", "price": 50}
    assert matches(md, {"category": "shoes", "price": 50})
    assert not matches(md, {"category": "shoes", "price": 999})


def test_comparison_operators():
    md = {"price": 50}
    assert matches(md, {"price": {"$lt": 100}})
    assert not matches(md, {"price": {"$lt": 10}})
    assert matches(md, {"price": {"$gte": 50}})
    assert matches(md, {"price": {"$ne": 999}})


def test_in_and_nin():
    md = {"category": "shoes"}
    assert matches(md, {"category": {"$in": ["shoes", "hats"]}})
    assert not matches(md, {"category": {"$nin": ["shoes", "hats"]}})


def test_missing_key_fails_comparison_gracefully():
    md = {}
    assert not matches(md, {"price": {"$lt": 100}})


def test_no_filter_matches_everything():
    assert matches({"anything": 1}, None)
    assert matches({"anything": 1}, {})


def test_filter_ids_selects_correct_subset():
    all_md = {
        "a": {"category": "shoes", "price": 20},
        "b": {"category": "hats", "price": 20},
        "c": {"category": "shoes", "price": 200},
    }
    assert filter_ids(all_md, {"category": "shoes"}) == {"a", "c"}
    assert filter_ids(all_md, {"category": "shoes", "price": {"$lt": 100}}) == {"a"}
    assert filter_ids(all_md, None) == {"a", "b", "c"}
