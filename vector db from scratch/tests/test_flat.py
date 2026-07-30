import numpy as np

from vectordb.index.flat import FlatIndex
from vectordb.distance import Metric


def _random_data(n=200, d=16, seed=0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, d)).astype(np.float32)
    ids = [f"id{i}" for i in range(n)]
    return vectors, ids


def test_flat_search_returns_exact_nearest():
    vectors, ids = _random_data()
    index = FlatIndex(dim=vectors.shape[1], metric=Metric.L2)
    index.build(vectors, ids)

    query = vectors[42] + 1e-4  # near-exact match to a known point
    results = index.search(query, k=5)
    assert results[0][0] == "id42"


def test_flat_add_and_remove():
    vectors, ids = _random_data(n=10)
    index = FlatIndex(dim=vectors.shape[1])
    index.build(vectors[:8], ids[:8])
    index.add(ids[8], vectors[8])
    index.add(ids[9], vectors[9])
    assert len(index) == 10

    index.remove(ids[0])
    assert len(index) == 9
    assert all(rid != ids[0] for rid, _ in index.search(vectors[8], k=9))


def test_flat_k_larger_than_dataset():
    vectors, ids = _random_data(n=3)
    index = FlatIndex(dim=vectors.shape[1])
    index.build(vectors, ids)
    results = index.search(vectors[0], k=100)
    assert len(results) == 3


def test_flat_empty_index_search():
    index = FlatIndex(dim=4)
    assert index.search(np.zeros(4), k=5) == []
