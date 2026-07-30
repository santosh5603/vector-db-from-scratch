import numpy as np
import pytest

from vectordb.db import VectorDB


def _seed(coll, n=200, d=16, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        category = "shoes" if i % 2 == 0 else "hats"
        coll.upsert(f"id{i}", rng.normal(size=d).astype(np.float32),
                    {"category": category, "price": i})


@pytest.mark.parametrize("index_type", ["flat", "ivf", "hnsw"])
def test_search_returns_k_results(index_type):
    db = VectorDB()
    kwargs = {"n_clusters": 10} if index_type == "ivf" else ({"M": 8, "ef_construction": 60} if index_type == "hnsw" else {})
    coll = db.create_collection("c", dim=16, index_type=index_type, **kwargs)
    _seed(coll)
    q = np.random.default_rng(1).normal(size=16).astype(np.float32)
    results = coll.search(q, k=5)
    assert len(results) == 5
    assert all("id" in r and "distance" in r and "metadata" in r for r in results)


def test_selective_prefilter_returns_correct_matches():
    db = VectorDB()
    coll = db.create_collection("c", dim=16, index_type="flat")
    _seed(coll, n=300)
    q = np.random.default_rng(2).normal(size=16).astype(np.float32)
    # price >= 295 matches only a handful of records -> exercises pre-filter path
    results = coll.search(q, k=3, filter={"price": {"$gte": 295}})
    assert len(results) == 3
    assert all(r["metadata"]["price"] >= 295 for r in results)


def test_broad_postfilter_returns_correct_matches():
    db = VectorDB()
    coll = db.create_collection("c", dim=16, index_type="flat")
    _seed(coll, n=300)
    q = np.random.default_rng(3).normal(size=16).astype(np.float32)
    # category filter matches ~50% of records -> exercises post-filter path
    results = coll.search(q, k=5, filter={"category": "shoes"})
    assert len(results) == 5
    assert all(r["metadata"]["category"] == "shoes" for r in results)


def test_delete_removes_from_search_results():
    db = VectorDB()
    coll = db.create_collection("c", dim=8, index_type="flat")
    rng = np.random.default_rng(4)
    for i in range(20):
        coll.upsert(f"id{i}", rng.normal(size=8).astype(np.float32))
    coll.delete("id5")
    results = coll.search(np.zeros(8), k=20)
    assert "id5" not in [r["id"] for r in results]
    assert len(results) == 19


def test_persistence_roundtrip_and_rebuild(tmp_path):
    d = str(tmp_path / "db")
    db = VectorDB(directory=d)
    coll = db.create_collection("c", dim=8, index_type="hnsw", M=8, ef_construction=40)
    rng = np.random.default_rng(5)
    for i in range(50):
        coll.upsert(f"id{i}", rng.normal(size=8).astype(np.float32), {"i": i})
    coll.checkpoint()

    db2 = VectorDB(directory=d)
    coll2 = db2.get_collection("c")
    assert len(coll2) == 50
    results = coll2.search(coll2.get("id0").vector, k=1)
    assert results[0]["id"] == "id0"


def test_create_duplicate_collection_raises():
    db = VectorDB()
    db.create_collection("c", dim=4)
    with pytest.raises(ValueError):
        db.create_collection("c", dim=4)
