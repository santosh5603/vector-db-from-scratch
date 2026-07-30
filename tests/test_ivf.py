import numpy as np

from vectordb.index.ivf import IVFIndex, kmeans
from vectordb.index.flat import FlatIndex
from vectordb.distance import Metric


def _random_data(n=500, d=16, seed=0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, d)).astype(np.float32)
    ids = [f"id{i}" for i in range(n)]
    return vectors, ids


def test_kmeans_assigns_every_point():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 4)).astype(np.float32)
    centroids, assignments = kmeans(data, k=5, iters=10)
    assert centroids.shape == (5, 4)
    assert assignments.shape == (100,)
    assert set(assignments.tolist()) <= set(range(5))


def test_ivf_recall_against_flat_ground_truth():
    vectors, ids = _random_data(n=1000, d=32)
    flat = FlatIndex(dim=32, metric=Metric.L2)
    flat.build(vectors, ids)

    ivf = IVFIndex(dim=32, metric=Metric.L2, n_clusters=20, n_probe=8)
    ivf.build(vectors, ids)

    rng = np.random.default_rng(42)
    hits = 0
    n_queries = 30
    k = 10
    for _ in range(n_queries):
        q = rng.normal(size=32).astype(np.float32)
        truth = {rid for rid, _ in flat.search(q, k)}
        got = {rid for rid, _ in ivf.search(q, k)}
        hits += len(truth & got)

    recall = hits / (n_queries * k)
    # approximate index -- should get most of the true neighbors with n_probe=8/20
    assert recall > 0.7, f"recall too low: {recall}"


def test_ivf_add_and_remove():
    vectors, ids = _random_data(n=50)
    ivf = IVFIndex(dim=16, n_clusters=5)
    ivf.build(vectors, ids)
    assert len(ivf) == 50

    new_vec = np.random.default_rng(1).normal(size=16).astype(np.float32)
    ivf.add("new_id", new_vec)
    assert len(ivf) == 51
    results = ivf.search(new_vec, k=1)
    assert results[0][0] == "new_id"

    ivf.remove("new_id")
    assert len(ivf) == 50


def test_ivf_empty_search():
    ivf = IVFIndex(dim=8)
    assert ivf.search(np.zeros(8), k=5) == []
