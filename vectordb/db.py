"""
The public-facing API: `VectorDB` manages named `Collection`s, and each
`Collection` glues together a `VectorStore` (durable storage), an index
(Flat / IVF / HNSW), and metadata filtering into one `upsert` / `search` /
`delete` surface.

This is the layer you'd import directly in Python (`from vectordb import
VectorDB`), and it's also what `server.py` wraps with a REST API.
"""
from __future__ import annotations

import json
import os
from typing import Any, Literal

import numpy as np

from .distance import Metric
from .index.flat import FlatIndex
from .index.ivf import IVFIndex
from .index.hnsw import HNSWIndex
from .metadata import matches, filter_ids
from .storage import VectorStore

IndexType = Literal["flat", "ivf", "hnsw"]

_INDEX_CLASSES = {"flat": FlatIndex, "ivf": IVFIndex, "hnsw": HNSWIndex}

# Above this many candidates, prefer exact pre-filtering (build a small flat
# scan over just the matching ids) instead of oversampling the ANN index --
# see Collection.search() docstring for the reasoning.
_PREFILTER_THRESHOLD = 0.15


class Collection:
    def __init__(self, name: str, dim: int, metric: Metric | str = Metric.COSINE,
                 index_type: IndexType = "hnsw", directory: str | None = None,
                 index_kwargs: dict[str, Any] | None = None):
        self.name = name
        self.dim = dim
        self.metric = Metric(metric)
        self.index_type = index_type
        self.directory = directory
        self.index_kwargs = index_kwargs or {}

        self.store = VectorStore(dim, directory=directory)
        self.index = _INDEX_CLASSES[index_type](dim, metric=self.metric, **self.index_kwargs)

        if len(self.store) > 0:
            vectors, ids = self.store.dense_view()
            self.index.build(vectors, ids)

    # -- mutation ---------------------------------------------------------

    def upsert(self, record_id: str, vector: np.ndarray, metadata: dict[str, Any] | None = None):
        self.store.upsert(record_id, vector, metadata)
        self.index.add(record_id, np.asarray(vector, dtype=np.float32))

    def upsert_batch(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict] | None = None):
        metadatas = metadatas or [{}] * len(ids)
        for rid, vec, md in zip(ids, vectors, metadatas):
            self.upsert(rid, vec, md)

    def delete(self, record_id: str) -> bool:
        removed = self.store.delete(record_id)
        if removed:
            self.index.remove(record_id)
        return removed

    def get(self, record_id: str):
        return self.store.get(record_id)

    # -- search ---------------------------------------------------------

    def search(self, query: np.ndarray, k: int = 10, filter: dict[str, Any] | None = None,
               oversample: int = 10, max_oversample_rounds: int = 4) -> list[dict[str, Any]]:
        """Vector search, optionally restricted by a metadata filter.

        Strategy:
          - No filter: query the ANN index directly.
          - Filter present: estimate selectivity by counting matches. If the
            filter is selective (matches < ~15% of the collection), pre-filter
            -- gather the matching vectors and run an exact brute-force scan
            over just that (small) candidate set, since that's cheap and
            guarantees k correct results.
          - Otherwise (filter matches most of the collection), post-filter --
            ask the ANN index for oversample*k results and filter those,
            widening the ask geometrically if we don't get k matches back
            (bounded by max_oversample_rounds so a pathological filter can't
            spin forever).
        """
        query = np.asarray(query, dtype=np.float32)

        if not filter:
            hits = self.index.search(query, k)
            return [self._to_result(rid, dist) for rid, dist in hits]

        all_metadata = {rid: self.store.metadata_for(rid) for rid in self.store.all_ids()}
        total = len(all_metadata)
        if total == 0:
            return []

        candidate_ids = filter_ids(all_metadata, filter)
        selectivity = len(candidate_ids) / total

        if selectivity <= _PREFILTER_THRESHOLD:
            return self._prefiltered_search(query, k, candidate_ids)

        # post-filter with widening oversample
        want = k
        attempt_k = k * oversample
        for _ in range(max_oversample_rounds):
            hits = self.index.search(query, min(attempt_k, total))
            filtered = [(rid, d) for rid, d in hits if rid in candidate_ids]
            if len(filtered) >= want or attempt_k >= total:
                return [self._to_result(rid, d) for rid, d in filtered[:want]]
            attempt_k *= oversample
        return [self._to_result(rid, d) for rid, d in filtered[:want]]

    def _prefiltered_search(self, query: np.ndarray, k: int, candidate_ids: set[str]) -> list[dict[str, Any]]:
        if not candidate_ids:
            return []
        scratch = FlatIndex(self.dim, metric=self.metric)
        ids = list(candidate_ids)
        vecs = np.stack([self.store.get(rid).vector for rid in ids])
        scratch.build(vecs, ids)
        hits = scratch.search(query, k)
        return [self._to_result(rid, d) for rid, d in hits]

    def _to_result(self, record_id: str, distance: float) -> dict[str, Any]:
        return {
            "id": record_id,
            "distance": distance,
            "metadata": self.store.metadata_for(record_id),
        }

    # -- persistence ---------------------------------------------------------

    def checkpoint(self):
        if not self.directory:
            raise RuntimeError("checkpoint() requires a directory-backed collection")
        self.store.checkpoint()
        with open(os.path.join(self.directory, "config.json"), "w") as fh:
            json.dump({
                "name": self.name, "dim": self.dim, "metric": self.metric.value,
                "index_type": self.index_type, "index_kwargs": self.index_kwargs,
            }, fh)

    def rebuild_index(self):
        """Rebuild the ANN index from the current store contents. The index
        structures themselves aren't serialized to disk (HNSW's graph and
        IVF's clusters are cheap to regenerate deterministically from the
        vectors) -- only the vectors + metadata are durable, and the index
        is rebuilt in memory whenever a collection is (re)loaded."""
        vectors, ids = self.store.dense_view()
        self.index = _INDEX_CLASSES[self.index_type](self.dim, metric=self.metric, **self.index_kwargs)
        if len(ids) > 0:
            self.index.build(vectors, ids)

    def __len__(self):
        return len(self.store)

    def close(self):
        self.store.close()


class VectorDB:
    """Top-level handle managing multiple named collections, each with its
    own directory (when persisted) so collections don't share WAL/snapshot
    files."""

    def __init__(self, directory: str | None = None):
        self.directory = directory
        self._collections: dict[str, Collection] = {}
        if directory:
            os.makedirs(directory, exist_ok=True)
            self._load_existing_collections()

    def _load_existing_collections(self):
        for entry in os.listdir(self.directory):
            coll_dir = os.path.join(self.directory, entry)
            config_path = os.path.join(coll_dir, "config.json")
            if os.path.isdir(coll_dir) and os.path.exists(config_path):
                with open(config_path) as fh:
                    cfg = json.load(fh)
                self._collections[cfg["name"]] = Collection(
                    name=cfg["name"], dim=cfg["dim"], metric=cfg["metric"],
                    index_type=cfg["index_type"], directory=coll_dir,
                    index_kwargs=cfg.get("index_kwargs", {}),
                )

    def create_collection(self, name: str, dim: int, metric: Metric | str = Metric.COSINE,
                           index_type: IndexType = "hnsw", **index_kwargs) -> Collection:
        if name in self._collections:
            raise ValueError(f"collection '{name}' already exists")
        coll_dir = os.path.join(self.directory, name) if self.directory else None
        coll = Collection(name, dim, metric=metric, index_type=index_type,
                           directory=coll_dir, index_kwargs=index_kwargs)
        self._collections[name] = coll
        return coll

    def get_collection(self, name: str) -> Collection:
        if name not in self._collections:
            raise KeyError(f"no such collection: {name}")
        return self._collections[name]

    def list_collections(self) -> list[str]:
        return list(self._collections.keys())

    def delete_collection(self, name: str):
        coll = self._collections.pop(name, None)
        if coll:
            coll.close()

    def checkpoint_all(self):
        for coll in self._collections.values():
            if coll.directory:
                coll.checkpoint()
