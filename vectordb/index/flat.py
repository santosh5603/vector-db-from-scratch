"""
Flat (brute-force) index: the ground-truth baseline every other index is
benchmarked against.

Search is O(n*d) -- compute the distance from the query to every stored
vector and take the k smallest. No approximation, no build step, always
100% recall. This is deliberately the simplest possible correct
implementation so it can double as the "ground truth" used to measure
recall@k for IVF and HNSW.
"""
from __future__ import annotations

import numpy as np

from ..distance import Metric, pairwise


class FlatIndex:
    name = "flat"

    def __init__(self, dim: int, metric: Metric | str = Metric.COSINE):
        self.dim = dim
        self.metric = Metric(metric)
        self._ids: list[str] = []
        self._vectors = np.empty((0, dim), dtype=np.float32)
        self._id_to_pos: dict[str, int] = {}

    def build(self, vectors: np.ndarray, ids: list[str]):
        self._vectors = np.asarray(vectors, dtype=np.float32).copy()
        self._ids = list(ids)
        self._id_to_pos = {rid: i for i, rid in enumerate(self._ids)}

    def add(self, record_id: str, vector: np.ndarray):
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if record_id in self._id_to_pos:
            self._vectors[self._id_to_pos[record_id]] = vector
            return
        self._vectors = np.vstack([self._vectors, vector])
        self._id_to_pos[record_id] = len(self._ids)
        self._ids.append(record_id)

    def remove(self, record_id: str):
        if record_id not in self._id_to_pos:
            return
        pos = self._id_to_pos.pop(record_id)
        last = len(self._ids) - 1
        if pos != last:
            self._vectors[pos] = self._vectors[last]
            moved_id = self._ids[last]
            self._ids[pos] = moved_id
            self._id_to_pos[moved_id] = pos
        self._ids.pop()
        self._vectors = self._vectors[:-1]

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if len(self._ids) == 0:
            return []
        dists = pairwise(np.asarray(query, dtype=np.float32), self._vectors, self.metric)
        k = min(k, len(self._ids))
        top = np.argpartition(dists, k - 1)[:k]
        top = top[np.argsort(dists[top])]
        return [(self._ids[i], float(dists[i])) for i in top]

    def __len__(self):
        return len(self._ids)
