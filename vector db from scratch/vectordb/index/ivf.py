"""
IVF (Inverted File) index: a coarse quantizer + inverted lists.

Idea: cluster the dataset into `n_clusters` groups with k-means, so each
vector lives in exactly one "bucket". At search time, instead of scanning
every vector (like FlatIndex), we only compare the query against the
centroids (cheap: n_clusters comparisons), then do an exact brute-force
scan inside just the `n_probe` closest buckets. This trades a small amount
of recall for a large reduction in the number of distance computations,
which is the whole point of an approximate index.

K-means is implemented from scratch (Lloyd's algorithm with k-means++
seeding) rather than pulled from sklearn, since the point of this project
is understanding the index internals, not wrapping a library.
"""
from __future__ import annotations

import numpy as np

from ..distance import Metric, pairwise, batch_pairwise


def _kmeans_plus_plus_init(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = data.shape[0]
    centroids = np.empty((k, data.shape[1]), dtype=np.float32)
    first = rng.integers(0, n)
    centroids[0] = data[first]
    closest_sq_dist = np.sum((data - centroids[0]) ** 2, axis=1)

    for i in range(1, k):
        probs = closest_sq_dist / max(closest_sq_dist.sum(), 1e-12)
        chosen = rng.choice(n, p=probs)
        centroids[i] = data[chosen]
        new_sq_dist = np.sum((data - centroids[i]) ** 2, axis=1)
        closest_sq_dist = np.minimum(closest_sq_dist, new_sq_dist)

    return centroids


def kmeans(data: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Plain Lloyd's-algorithm k-means. Returns (centroids, assignments)."""
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    k = min(k, n)
    centroids = _kmeans_plus_plus_init(data, k, rng)
    assignments = np.zeros(n, dtype=np.int64)

    for _ in range(iters):
        # Assign step: nearest centroid to each point (squared L2, via the
        # same expansion trick used in distance.batch_pairwise).
        d_sq = batch_pairwise(data, centroids, Metric.L2)
        new_assignments = np.argmin(d_sq, axis=1)
        if np.array_equal(new_assignments, assignments) and _ > 0:
            break
        assignments = new_assignments

        # Update step: recompute each centroid as the mean of its members.
        for c in range(k):
            members = data[assignments == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)
            # empty clusters keep their previous centroid rather than
            # collapsing to the origin

    return centroids, assignments


class IVFIndex:
    name = "ivf"

    def __init__(self, dim: int, metric: Metric | str = Metric.COSINE,
                 n_clusters: int = 32, n_probe: int = 4):
        self.dim = dim
        self.metric = Metric(metric)
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.centroids: np.ndarray | None = None
        # cluster_id -> {record_id: vector}, kept as parallel arrays for speed
        self._buckets_ids: list[list[str]] = []
        self._buckets_vecs: list[np.ndarray] = []
        self._id_to_cluster: dict[str, int] = {}

    def build(self, vectors: np.ndarray, ids: list[str]):
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(ids) == 0:
            self.centroids = np.empty((0, self.dim), dtype=np.float32)
            self._buckets_ids, self._buckets_vecs = [], []
            return

        self.centroids, assignments = kmeans(vectors, self.n_clusters)
        n_clusters = self.centroids.shape[0]
        self._buckets_ids = [[] for _ in range(n_clusters)]
        self._buckets_vecs = [np.empty((0, self.dim), dtype=np.float32) for _ in range(n_clusters)]
        self._id_to_cluster = {}

        for c in range(n_clusters):
            mask = assignments == c
            self._buckets_ids[c] = [ids[i] for i in np.nonzero(mask)[0]]
            self._buckets_vecs[c] = vectors[mask]
            for rid in self._buckets_ids[c]:
                self._id_to_cluster[rid] = c

    def _nearest_cluster(self, vector: np.ndarray) -> int:
        d = pairwise(vector, self.centroids, Metric.L2)
        return int(np.argmin(d))

    def add(self, record_id: str, vector: np.ndarray):
        """Assign to the nearest existing centroid. Centroids themselves are
        only recomputed on the next full `build()` / rebalance -- this mirrors
        how real IVF indexes (e.g. FAISS) treat clustering as an offline
        "training" step separate from cheap incremental inserts."""
        vector = np.asarray(vector, dtype=np.float32)
        if self.centroids is None or len(self.centroids) == 0:
            self.build(vector.reshape(1, -1), [record_id])
            return
        if record_id in self._id_to_cluster:
            self.remove(record_id)
        c = self._nearest_cluster(vector)
        self._buckets_ids[c].append(record_id)
        self._buckets_vecs[c] = np.vstack([self._buckets_vecs[c], vector.reshape(1, -1)])
        self._id_to_cluster[record_id] = c

    def remove(self, record_id: str):
        if record_id not in self._id_to_cluster:
            return
        c = self._id_to_cluster.pop(record_id)
        pos = self._buckets_ids[c].index(record_id)
        self._buckets_ids[c].pop(pos)
        self._buckets_vecs[c] = np.delete(self._buckets_vecs[c], pos, axis=0)

    def search(self, query: np.ndarray, k: int = 10, n_probe: int | None = None) -> list[tuple[str, float]]:
        if self.centroids is None or len(self.centroids) == 0:
            return []
        n_probe = n_probe or self.n_probe
        query = np.asarray(query, dtype=np.float32)

        centroid_dists = pairwise(query, self.centroids, Metric.L2)
        n_probe = min(n_probe, len(self.centroids))
        probe_clusters = np.argsort(centroid_dists)[:n_probe]

        candidate_ids: list[str] = []
        candidate_vecs = []
        for c in probe_clusters:
            candidate_ids.extend(self._buckets_ids[c])
            if len(self._buckets_vecs[c]) > 0:
                candidate_vecs.append(self._buckets_vecs[c])

        if not candidate_ids:
            return []
        vecs = np.vstack(candidate_vecs)
        dists = pairwise(query, vecs, self.metric)
        k = min(k, len(candidate_ids))
        top = np.argpartition(dists, k - 1)[:k]
        top = top[np.argsort(dists[top])]
        return [(candidate_ids[i], float(dists[i])) for i in top]

    def __len__(self):
        return sum(len(b) for b in self._buckets_ids)
