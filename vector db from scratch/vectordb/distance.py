"""
Distance / similarity functions used across all index types.

All functions operate on numpy arrays and are vectorized: given a single
query vector of shape (d,) and a matrix of candidates of shape (n, d),
they return a (n,) array of distances (lower = more similar) so that
every index implementation can call `argsort` / `argpartition` the same
way regardless of which metric was chosen.

Cosine and dot-product are converted to a "distance" (lower-is-better)
by negating, so the rest of the codebase never has to branch on
"is this a similarity or a distance metric".
"""
from __future__ import annotations

from enum import Enum
import numpy as np


class Metric(str, Enum):
    COSINE = "cosine"
    L2 = "l2"
    DOT = "dot"


def _normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.clip(norms, eps, None)


def l2_distance(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance, query (d,) vs candidates (n, d) -> (n,)."""
    diff = candidates - query
    return np.einsum("ij,ij->i", diff, diff)


def cosine_distance(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """1 - cosine similarity, so 0 = identical direction, 2 = opposite."""
    q = query / max(np.linalg.norm(query), 1e-12)
    c = _normalize_rows(candidates)
    sims = c @ q
    return 1.0 - sims


def dot_distance(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Negative dot product, so that "lower is better" holds like the
    other two metrics (higher raw dot product = more similar)."""
    return -(candidates @ query)


_DISPATCH = {
    Metric.L2: l2_distance,
    Metric.COSINE: cosine_distance,
    Metric.DOT: dot_distance,
}


def pairwise(query: np.ndarray, candidates: np.ndarray, metric: Metric | str) -> np.ndarray:
    """Compute distances from `query` to every row in `candidates` under `metric`."""
    metric = Metric(metric)
    if candidates.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    return _DISPATCH[metric](query, candidates)


def batch_pairwise(queries: np.ndarray, candidates: np.ndarray, metric: Metric | str) -> np.ndarray:
    """Same as `pairwise` but for a matrix of queries (m, d) -> (m, n) distance matrix.

    Kept separate (rather than just looping `pairwise`) because the matrix form
    lets numpy do a single BLAS matmul instead of m separate ones -- this is the
    difference that matters once you're benchmarking the flat index.
    """
    metric = Metric(metric)
    if candidates.shape[0] == 0:
        return np.empty((queries.shape[0], 0), dtype=np.float32)

    if metric == Metric.L2:
        # ||q - c||^2 = ||q||^2 - 2 q.c + ||c||^2, computed as one matmul.
        q_sq = np.einsum("ij,ij->i", queries, queries)[:, None]
        c_sq = np.einsum("ij,ij->i", candidates, candidates)[None, :]
        cross = queries @ candidates.T
        return q_sq - 2 * cross + c_sq
    elif metric == Metric.COSINE:
        q = _normalize_rows(queries)
        c = _normalize_rows(candidates)
        return 1.0 - (q @ c.T)
    elif metric == Metric.DOT:
        return -(queries @ candidates.T)
    else:  # pragma: no cover - exhaustive via Metric enum
        raise ValueError(f"Unknown metric: {metric}")


def single(a: np.ndarray, b: np.ndarray, metric: Metric | str) -> float:
    """Distance between two individual vectors. Used by graph-traversal
    indexes (HNSW) that compare one node to another rather than one query
    against a whole matrix."""
    metric = Metric(metric)
    if metric == Metric.L2:
        diff = a - b
        return float(np.dot(diff, diff))
    elif metric == Metric.COSINE:
        na = max(np.linalg.norm(a), 1e-12)
        nb = max(np.linalg.norm(b), 1e-12)
        return 1.0 - float(np.dot(a, b) / (na * nb))
    elif metric == Metric.DOT:
        return -float(np.dot(a, b))
    else:  # pragma: no cover
        raise ValueError(f"Unknown metric: {metric}")
