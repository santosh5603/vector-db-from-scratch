import numpy as np
import pytest

from vectordb.distance import Metric, pairwise, batch_pairwise, single


def test_l2_distance_matches_naive():
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    got = pairwise(q, c, Metric.L2)
    expected = np.array([0.0, 2.0, 1.0])
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_cosine_distance_identical_vectors_is_zero():
    q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    c = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    got = pairwise(q, c, Metric.COSINE)
    assert got[0] == pytest.approx(0.0, abs=1e-5)


def test_cosine_distance_orthogonal_is_one():
    q = np.array([1.0, 0.0], dtype=np.float32)
    c = np.array([[0.0, 1.0]], dtype=np.float32)
    got = pairwise(q, c, Metric.COSINE)
    assert got[0] == pytest.approx(1.0, abs=1e-5)


def test_dot_distance_orders_by_magnitude():
    q = np.array([1.0, 1.0], dtype=np.float32)
    c = np.array([[1.0, 1.0], [2.0, 2.0], [0.1, 0.1]], dtype=np.float32)
    got = pairwise(q, c, Metric.DOT)
    # higher dot product => more negative distance => should sort as index 1, 0, 2
    order = np.argsort(got)
    assert list(order) == [1, 0, 2]


def test_batch_pairwise_matches_single_pairwise():
    rng = np.random.default_rng(0)
    queries = rng.normal(size=(5, 8)).astype(np.float32)
    candidates = rng.normal(size=(20, 8)).astype(np.float32)

    for metric in [Metric.L2, Metric.COSINE, Metric.DOT]:
        batch = batch_pairwise(queries, candidates, metric)
        for i in range(queries.shape[0]):
            row = pairwise(queries[i], candidates, metric)
            np.testing.assert_allclose(batch[i], row, atol=1e-4)


def test_single_matches_pairwise():
    rng = np.random.default_rng(1)
    a = rng.normal(size=8).astype(np.float32)
    b = rng.normal(size=8).astype(np.float32)
    for metric in [Metric.L2, Metric.COSINE, Metric.DOT]:
        s = single(a, b, metric)
        p = pairwise(a, b.reshape(1, -1), metric)[0]
        assert s == pytest.approx(p, abs=1e-4)


def test_empty_candidates_returns_empty():
    q = np.zeros(4, dtype=np.float32)
    c = np.empty((0, 4), dtype=np.float32)
    assert pairwise(q, c, Metric.L2).shape == (0,)
