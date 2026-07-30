"""
Durable storage layer: holds raw vectors + metadata for a collection, and is
responsible for surviving a crash.

Design:
  - In memory, vectors live in a single (n, d) numpy array that grows in
    chunks (like a Python list's over-allocation) so appends are amortized
    O(1) instead of reallocating on every insert.
  - Every mutation (upsert/delete) is first appended to a write-ahead log
    (WAL) file on disk *before* it's applied in memory. On startup, we load
    the last snapshot and replay the WAL on top of it, so a crash between
    two `save()` calls can never lose committed writes.
  - `checkpoint()` writes a full snapshot (vectors as .npy, metadata/ids as
    .json) and truncates the WAL, bounding replay time.
"""
from __future__ import annotations

import json
import os
import struct
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

_GROWTH_FACTOR = 2
_INITIAL_CAPACITY = 64


@dataclass
class Record:
    id: str
    vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class WriteAheadLog:
    """Append-only JSONL log of mutations, used to recover uncheckpointed writes."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "a", buffering=1)  # line-buffered
        self._lock = threading.Lock()

    def append(self, op: str, record_id: str, vector: np.ndarray | None, metadata: dict | None):
        entry = {
            "op": op,
            "id": record_id,
            "vector": vector.tolist() if vector is not None else None,
            "metadata": metadata,
        }
        with self._lock:
            self._fh.write(json.dumps(entry) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def replay(self) -> Iterable[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def truncate(self):
        with self._lock:
            self._fh.close()
            open(self.path, "w").close()
            self._fh = open(self.path, "a", buffering=1)

    def close(self):
        self._fh.close()


class VectorStore:
    """In-memory vector + metadata store with WAL-backed durability.

    External record ids are arbitrary strings, chosen by the caller.
    Internally we map them to dense row indices in `self._vectors` so
    indexes can work with plain numpy arrays; deletions leave a tombstone
    (row is zeroed and the slot is recycled on the next insert) rather than
    shifting every later row.
    """

    def __init__(self, dim: int, directory: str | None = None):
        self.dim = dim
        self.directory = directory
        self._vectors = np.zeros((_INITIAL_CAPACITY, dim), dtype=np.float32)
        self._size = 0  # number of occupied+tombstoned rows
        self._id_to_row: dict[str, int] = {}
        self._row_to_id: dict[int, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._free_rows: list[int] = []
        self._wal: WriteAheadLog | None = None

        if directory:
            os.makedirs(directory, exist_ok=True)
            self._wal = WriteAheadLog(os.path.join(directory, "wal.log"))
            self._load_snapshot_and_replay()

    # -- capacity management -------------------------------------------------

    def _ensure_capacity(self, extra: int = 1):
        needed = self._size + extra
        if needed <= self._vectors.shape[0]:
            return
        new_capacity = max(needed, self._vectors.shape[0] * _GROWTH_FACTOR)
        grown = np.zeros((new_capacity, self.dim), dtype=np.float32)
        grown[: self._vectors.shape[0]] = self._vectors
        self._vectors = grown

    # -- mutation API ---------------------------------------------------------

    def upsert(self, record_id: str, vector: np.ndarray, metadata: dict[str, Any] | None = None,
               _skip_wal: bool = False):
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.dim,):
            raise ValueError(f"expected vector of shape ({self.dim},), got {vector.shape}")
        metadata = metadata or {}

        if not _skip_wal and self._wal:
            self._wal.append("upsert", record_id, vector, metadata)

        if record_id in self._id_to_row:
            row = self._id_to_row[record_id]
        elif self._free_rows:
            row = self._free_rows.pop()
        else:
            self._ensure_capacity(1)
            row = self._size
            self._size += 1

        self._vectors[row] = vector
        self._id_to_row[record_id] = row
        self._row_to_id[row] = record_id
        self._metadata[record_id] = metadata

    def delete(self, record_id: str, _skip_wal: bool = False) -> bool:
        if record_id not in self._id_to_row:
            return False
        if not _skip_wal and self._wal:
            self._wal.append("delete", record_id, None, None)

        row = self._id_to_row.pop(record_id)
        del self._row_to_id[row]
        del self._metadata[record_id]
        self._vectors[row] = 0
        self._free_rows.append(row)
        return True

    # -- read API ---------------------------------------------------------

    def get(self, record_id: str) -> Record | None:
        if record_id not in self._id_to_row:
            return None
        row = self._id_to_row[record_id]
        return Record(id=record_id, vector=self._vectors[row].copy(), metadata=self._metadata[record_id])

    def __len__(self) -> int:
        return len(self._id_to_row)

    def all_ids(self) -> list[str]:
        return list(self._id_to_row.keys())

    def dense_view(self) -> tuple[np.ndarray, list[str]]:
        """Return (vectors, ids) for every *live* row, in a stable row order.

        Indexes that want to do a single matmul over the whole collection
        (the flat index, IVF cluster assignment, etc.) use this rather than
        touching `_vectors` directly, so tombstoned rows never leak in.
        """
        rows = sorted(self._row_to_id.keys())
        if not rows:
            return np.empty((0, self.dim), dtype=np.float32), []
        ids = [self._row_to_id[r] for r in rows]
        return self._vectors[rows], ids

    def metadata_for(self, record_id: str) -> dict[str, Any]:
        return self._metadata.get(record_id, {})

    # -- persistence ---------------------------------------------------------

    def checkpoint(self):
        """Write a full snapshot to disk and truncate the WAL."""
        if not self.directory:
            raise RuntimeError("checkpoint() requires a directory-backed store")
        vectors, ids = self.dense_view()
        np.save(os.path.join(self.directory, "vectors.npy"), vectors)
        with open(os.path.join(self.directory, "meta.json"), "w") as fh:
            json.dump({"ids": ids, "metadata": self._metadata, "dim": self.dim}, fh)
        if self._wal:
            self._wal.truncate()

    def _load_snapshot_and_replay(self):
        vec_path = os.path.join(self.directory, "vectors.npy")
        meta_path = os.path.join(self.directory, "meta.json")
        if os.path.exists(vec_path) and os.path.exists(meta_path):
            vectors = np.load(vec_path)
            with open(meta_path) as fh:
                snap = json.load(fh)
            for i, rid in enumerate(snap["ids"]):
                self.upsert(rid, vectors[i], snap["metadata"].get(rid, {}), _skip_wal=True)

        for entry in self._wal.replay():
            if entry["op"] == "upsert":
                self.upsert(entry["id"], np.array(entry["vector"], dtype=np.float32),
                            entry["metadata"], _skip_wal=True)
            elif entry["op"] == "delete":
                self.delete(entry["id"], _skip_wal=True)

    def close(self):
        if self._wal:
            self._wal.close()
