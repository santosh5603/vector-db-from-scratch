import numpy as np

from vectordb.index.hnsw import HNSWIndex
from vectordb.index.flat import FlatIndex
from vectordb.distance import Metric


def _random_data(n=500, d=32, seed=0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, d)).astype(np.float32)
    ids = [f"id{i}" for i in range(n)]
    return vectors, ids


def test_hnsw_finds_exact_match():
    vectors, ids = _random_data(n=200)
    index = HNSWIndex(dim=32, metric=Metric.L2, M=16, ef_construction=100, ef_search=50)
    index.build(vectors, ids)

    query = vectors[10]
    results = index.search(query, k=1)
    assert results[0][0] == "id10"


def test_hnsw_recall_against_flat_ground_truth():
    vectors, ids = _random_data(n=1000, d=32)
    flat = FlatIndex(dim=32, metric=Metric.COSINE)
    flat.build(vectors, ids)

    hnsw = HNSWIndex(dim=32, metric=Metric.COSINE, M=16, ef_construction=150, ef_search=80)
    hnsw.build(vectors, ids)

    rng = np.random.default_rng(7)
    n_queries = 30
    k = 10
    hits = 0
    for _ in range(n_queries):
        q = rng.normal(size=32).astype(np.float32)
        truth = {rid for rid, _ in flat.search(q, k)}
        got = {rid for rid, _ in hnsw.search(q, k)}
        hits += len(truth & got)

    recall = hits / (n_queries * k)
    assert recall > 0.85, f"recall too low: {recall}"


def test_hnsw_add_remove_and_graph_stays_connected():
    vectors, ids = _random_data(n=100, d=16)
    index = HNSWIndex(dim=16, M=8, ef_construction=60, ef_search=40)
    index.build(vectors, ids)

    index.remove("id5")
    assert len(index) == 99
    # a fresh nearest-neighbor query should still work after deletion
    results = index.search(vectors[10], k=5)
    assert len(results) == 5
    assert "id5" not in [r[0] for r in results]


def test_hnsw_upsert_overwrites():
    vectors, ids = _random_data(n=20, d=8)
    index = HNSWIndex(dim=8, M=8, ef_construction=40)
    index.build(vectors, ids)

    new_vector = vectors[0] * -1
    index.add("id0", new_vector)
    assert len(index) == 20  # still 20, not 21 -- overwrite not duplicate
    results = index.search(new_vector, k=1)
    assert results[0][0] == "id0"


def test_hnsw_empty_search():
    index = HNSWIndex(dim=8)
    assert index.search(np.zeros(8), k=5) == []
