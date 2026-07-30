"""
HNSW (Hierarchical Navigable Small World) index, implemented from the
Malkov & Yashunin algorithm.

Intuition: build a multi-layer graph where layer 0 contains every vector
and each layer above it contains an exponentially shrinking random subset,
like a skip list. Layer counts thin out geometrically, so a search starts
at the sparse top layer (a few big hops to get roughly close) and descends
into denser layers (many small hops to refine), giving O(log n)-ish search
instead of the O(n) flat scan.

Two knobs matter most:
  - M: max neighbors per node per layer. Bigger M = better recall, more
    memory, slower inserts.
  - ef_construction / ef_search: size of the candidate list explored at
    build time / query time. Bigger ef = better recall, slower search.

This is the index most resume readers will ask about, so the
`_search_layer` method (the actual greedy beam search) is the part worth
being able to explain line-by-line.
"""
from __future__ import annotations

import heapq
import math
import random
from typing import Optional

import numpy as np

from ..distance import Metric, single as dist_single


class HNSWIndex:
    name = "hnsw"

    def __init__(self, dim: int, metric: Metric | str = Metric.COSINE,
                 M: int = 16, ef_construction: int = 200, ef_search: int = 50, seed: int = 0):
        self.dim = dim
        self.metric = Metric(metric)
        self.M = M
        self.M0 = 2 * M  # layer 0 gets more neighbors since it must stay fully connected
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.mL = 1.0 / math.log(M)
        self._rng = random.Random(seed)

        self.vectors: dict[int, np.ndarray] = {}
        self.levels: dict[int, int] = {}
        # graph[layer][node] -> list of neighbor node ids
        self.graph: list[dict[int, list[int]]] = [{}]
        self.entry_point: Optional[int] = None
        self.max_level = -1

        self._id_to_internal: dict[str, int] = {}
        self._internal_to_id: dict[int, str] = {}
        self._next_internal = 0

    # -- distance helper -------------------------------------------------

    def _d(self, a: int, b: int) -> float:
        return dist_single(self.vectors[a], self.vectors[b], self.metric)

    def _d_query(self, q: np.ndarray, node: int) -> float:
        return dist_single(q, self.vectors[node], self.metric)

    # -- core greedy search over a single layer --------------------------

    def _search_layer(self, query: np.ndarray, entry_points: list[int], ef: int, layer: int) -> list[tuple[float, int]]:
        """Beam search within one layer. Returns up to `ef` (distance, node)
        pairs, sorted by distance ascending.

        `candidates` is a min-heap (explore closest-first); `results` is a
        max-heap of size <= ef (we track the *worst* kept result so we can
        cheaply decide whether a newly found node is worth keeping).
        """
        visited = set(entry_points)
        candidates = [(self._d_query(query, ep), ep) for ep in entry_points]
        heapq.heapify(candidates)
        results = [(-d, ep) for d, ep in candidates]
        heapq.heapify(results)

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            worst_result_dist = -results[0][0]
            if dist_c > worst_result_dist and len(results) >= ef:
                break  # nothing closer left in the frontier -- stop expanding

            for neighbor in self.graph[layer].get(c, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                d = self._d_query(query, neighbor)
                worst_result_dist = -results[0][0]
                if len(results) < ef or d < worst_result_dist:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(results, (-d, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted([(-d, n) for d, n in results], key=lambda x: x[0])

    def _select_neighbors(self, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """Simple heuristic: just take the m closest. (The paper's full
        heuristic also tries to keep the graph diverse/spread-out rather
        than picking m near-duplicates, but closest-m is the standard
        simplification and is what most from-scratch implementations use.)"""
        return [n for _, n in sorted(candidates, key=lambda x: x[0])[:m]]

    # -- insertion ---------------------------------------------------------

    def _random_level(self) -> int:
        return int(-math.log(self._rng.random() + 1e-12) * self.mL)

    def add(self, record_id: str, vector: np.ndarray):
        vector = np.asarray(vector, dtype=np.float32)

        if record_id in self._id_to_internal:
            self.remove(record_id)

        node = self._next_internal
        self._next_internal += 1
        self._id_to_internal[record_id] = node
        self._internal_to_id[node] = record_id
        self.vectors[node] = vector
        level = self._random_level()
        self.levels[node] = level

        while len(self.graph) <= level:
            self.graph.append({})
        for l in range(level + 1):
            self.graph[l].setdefault(node, [])

        if self.entry_point is None:
            self.entry_point = node
            self.max_level = level
            return

        ep = self.entry_point
        # Phase 1: descend from the top layer down to `level+1` with ef=1,
        # just to get a good entry point close to the query.
        for l in range(self.max_level, level, -1):
            nearest = self._search_layer(vector, [ep], ef=1, layer=l)
            ep = nearest[0][1]

        # Phase 2: from min(level, max_level) down to 0, do a real
        # ef_construction-width search and wire up bidirectional edges.
        for l in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(vector, [ep], ef=self.ef_construction, layer=l)
            m = self.M0 if l == 0 else self.M
            neighbors = self._select_neighbors(candidates, m)
            self.graph[l][node] = neighbors

            for nb in neighbors:
                self.graph[l].setdefault(nb, [])
                self.graph[l][nb].append(node)
                # prune the neighbor's list if it grew past the cap
                if len(self.graph[l][nb]) > (self.M0 if l == 0 else self.M):
                    ranked = sorted(self.graph[l][nb], key=lambda n: self._d(nb, n))
                    self.graph[l][nb] = ranked[: (self.M0 if l == 0 else self.M)]

            if candidates:
                ep = candidates[0][1]

        if level > self.max_level:
            self.max_level = level
            self.entry_point = node

    # -- deletion ---------------------------------------------------------

    def remove(self, record_id: str):
        """Remove a node and stitch its neighbors together so the graph
        stays connected (rather than leaving dangling edges or holes).

        Note: because `_select_neighbors` + the per-insert pruning step can
        leave *asymmetric* edges (A lists B as a neighbor, but B's list
        later got pruned down and dropped A), we can't rely on `node`'s own
        outgoing list to find everyone pointing at it. We do a full scan of
        the layer to strip any remaining reference to `node`, then bridge
        node's former (outgoing) neighbors to each other so local
        connectivity isn't lost. This is O(n) per delete -- fine for an
        educational implementation, but a production version would keep a
        reverse-adjacency index to make deletes O(degree) instead.
        """
        if record_id not in self._id_to_internal:
            return
        node = self._id_to_internal.pop(record_id)
        del self._internal_to_id[node]
        level = self.levels.pop(node)
        del self.vectors[node]

        for l in range(level + 1):
            neighbors = self.graph[l].pop(node, [])

            # strip every remaining reference to `node`, even asymmetric ones
            for other, other_neighbors in self.graph[l].items():
                if node in other_neighbors:
                    other_neighbors.remove(node)

            # bridge the neighbors to each other so removing `node` doesn't
            # disconnect the local neighborhood
            for i, a in enumerate(neighbors):
                for b in neighbors[i + 1:]:
                    if a not in self.graph[l] or b not in self.graph[l]:
                        continue
                    cap = self.M0 if l == 0 else self.M
                    if b not in self.graph[l][a] and len(self.graph[l][a]) < cap:
                        self.graph[l][a].append(b)
                    if a not in self.graph[l][b] and len(self.graph[l][b]) < cap:
                        self.graph[l][b].append(a)

        if self.entry_point == node:
            remaining = list(self.vectors.keys())
            if remaining:
                self.entry_point = max(remaining, key=lambda n: self.levels[n])
                self.max_level = self.levels[self.entry_point]
            else:
                self.entry_point = None
                self.max_level = -1

    # -- search ---------------------------------------------------------

    def search(self, query: np.ndarray, k: int = 10, ef_search: int | None = None) -> list[tuple[str, float]]:
        if self.entry_point is None:
            return []
        query = np.asarray(query, dtype=np.float32)
        ef = ef_search or self.ef_search
        ef = max(ef, k)

        ep = self.entry_point
        for l in range(self.max_level, 0, -1):
            nearest = self._search_layer(query, [ep], ef=1, layer=l)
            ep = nearest[0][1]

        results = self._search_layer(query, [ep], ef=ef, layer=0)
        return [(self._internal_to_id[n], d) for d, n in results[:k]]

    def build(self, vectors: np.ndarray, ids: list[str]):
        """Bulk build by inserting one at a time (HNSW has no batch-build
        shortcut -- graph position depends on insertion order)."""
        for i, rid in enumerate(ids):
            self.add(rid, vectors[i])

    def __len__(self):
        return len(self.vectors)
