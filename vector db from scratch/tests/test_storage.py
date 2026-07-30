import numpy as np

from vectordb.storage import VectorStore


def test_upsert_get_delete_roundtrip(tmp_path):
    store = VectorStore(dim=4)
    store.upsert("a", np.array([1, 2, 3, 4], dtype=np.float32), {"tag": "x"})
    record = store.get("a")
    assert record is not None
    np.testing.assert_array_equal(record.vector, [1, 2, 3, 4])
    assert record.metadata == {"tag": "x"}

    assert store.delete("a")
    assert store.get("a") is None
    assert not store.delete("a")  # already gone


def test_dense_view_excludes_deleted_rows():
    store = VectorStore(dim=2)
    store.upsert("a", np.array([1, 1], dtype=np.float32))
    store.upsert("b", np.array([2, 2], dtype=np.float32))
    store.delete("a")
    vectors, ids = store.dense_view()
    assert ids == ["b"]
    assert vectors.shape == (1, 2)


def test_checkpoint_and_reload(tmp_path):
    d = str(tmp_path / "coll")
    store = VectorStore(dim=3, directory=d)
    store.upsert("a", np.array([1, 2, 3], dtype=np.float32), {"k": 1})
    store.upsert("b", np.array([4, 5, 6], dtype=np.float32), {"k": 2})
    store.checkpoint()
    store.close()

    reloaded = VectorStore(dim=3, directory=d)
    assert len(reloaded) == 2
    np.testing.assert_array_equal(reloaded.get("a").vector, [1, 2, 3])
    assert reloaded.get("b").metadata == {"k": 2}


def test_wal_replays_uncheckpointed_writes(tmp_path):
    """Simulates a crash: writes happen, but checkpoint() is never called.
    A fresh VectorStore pointed at the same directory should still recover
    every write by replaying the WAL."""
    d = str(tmp_path / "coll")
    store = VectorStore(dim=2, directory=d)
    store.upsert("a", np.array([1, 1], dtype=np.float32))
    store.upsert("b", np.array([2, 2], dtype=np.float32))
    store.delete("a")
    store.upsert("c", np.array([3, 3], dtype=np.float32))
    store.close()  # no checkpoint() call -- only the WAL is on disk

    recovered = VectorStore(dim=2, directory=d)
    assert recovered.get("a") is None  # deleted
    np.testing.assert_array_equal(recovered.get("b").vector, [2, 2])
    np.testing.assert_array_equal(recovered.get("c").vector, [3, 3])
    assert len(recovered) == 2


def test_capacity_growth_beyond_initial_size():
    store = VectorStore(dim=2)
    for i in range(200):  # exceeds _INITIAL_CAPACITY
        store.upsert(f"id{i}", np.array([i, i], dtype=np.float32))
    assert len(store) == 200
    np.testing.assert_array_equal(store.get("id150").vector, [150, 150])
