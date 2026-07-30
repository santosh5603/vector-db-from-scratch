"""
Benchmark suite: compares Flat, IVF, and HNSW on synthetic data across
build time, query latency, recall@k (against the Flat index as ground
truth), and approximate memory footprint.

Run with:
    python benchmarks/bench.py

Produces:
    benchmarks/results.csv
    benchmarks/results.png  (recall vs. latency tradeoff plot, if matplotlib available)

Why these particular metrics: build time and memory matter when you're
deciding whether an index is even feasible to construct/host; query
latency and recall are the actual product of an ANN index and are always
in tension with each other -- that tradeoff curve is the single most
useful thing to be able to show and explain about this project.
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectordb.distance import Metric
from vectordb.index.flat import FlatIndex
from vectordb.index.ivf import IVFIndex
from vectordb.index.hnsw import HNSWIndex

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.csv")
PLOT_PATH = os.path.join(os.path.dirname(__file__), "results.png")


def make_dataset(n: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, d)).astype(np.float32)
    ids = [f"id{i}" for i in range(n)]
    return vectors, ids


def _deep_size(obj, seen=None) -> int:
    """Single-pass approximate memory footprint via sys.getsizeof, walking
    containers recursively. Deliberately NOT using tracemalloc here --
    tracemalloc instruments every allocation and slows the pure-Python HNSW
    build by several times, which matters when you're timing that same
    build. A one-time post-hoc size walk is much cheaper and good enough
    for a relative (flat vs ivf vs hnsw) memory comparison."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, np.ndarray):
        return obj.nbytes

    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(_deep_size(k, seen) + _deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(_deep_size(i, seen) for i in obj)
    return size


def estimate_memory_mb(index) -> float:
    return _deep_size(index.__dict__) / (1024 * 1024)


def measure_build(index, vectors, ids):
    t0 = time.perf_counter()
    index.build(vectors, ids)
    build_time = time.perf_counter() - t0
    mem_mb = estimate_memory_mb(index)
    return build_time, mem_mb


def measure_query(index, queries, k, **search_kwargs):
    t0 = time.perf_counter()
    all_results = []
    for q in queries:
        all_results.append(index.search(q, k=k, **search_kwargs))
    total_time = time.perf_counter() - t0
    avg_latency_ms = (total_time / len(queries)) * 1000
    return avg_latency_ms, all_results


def recall_at_k(approx_results, ground_truth):
    hits = 0
    for approx, truth in zip(approx_results, ground_truth):
        approx_ids = {rid for rid, _ in approx}
        truth_ids = {rid for rid, _ in truth}
        hits += len(approx_ids & truth_ids)
    return hits / sum(len(t) for t in ground_truth)


def run_benchmark(n=5000, d=64, n_queries=100, k=10, seed=0):
    print(f"\n=== Dataset: n={n}, d={d}, k={k} ===")
    vectors, ids = make_dataset(n, d, seed=seed)
    rng = np.random.default_rng(seed + 1)
    queries = [rng.normal(size=d).astype(np.float32) for _ in range(n_queries)]

    rows = []

    # -- Flat (ground truth) --
    flat = FlatIndex(dim=d, metric=Metric.COSINE)
    build_time, mem_mb = measure_build(flat, vectors, ids)
    latency_ms, ground_truth = measure_query(flat, queries, k)
    rows.append({
        "index": "flat", "n": n, "build_s": round(build_time, 4),
        "query_ms": round(latency_ms, 4), "recall": 1.0, "peak_mb": round(mem_mb, 2),
    })
    print(f"flat : build={build_time:.3f}s  query={latency_ms:.3f}ms  recall=1.000  mem={mem_mb:.1f}MB")

    # -- IVF --
    n_clusters = max(8, int(np.sqrt(n)))
    ivf = IVFIndex(dim=d, metric=Metric.COSINE, n_clusters=n_clusters, n_probe=max(2, n_clusters // 3))
    build_time, mem_mb = measure_build(ivf, vectors, ids)
    latency_ms, ivf_results = measure_query(ivf, queries, k)
    recall = recall_at_k(ivf_results, ground_truth)
    rows.append({
        "index": "ivf", "n": n, "build_s": round(build_time, 4),
        "query_ms": round(latency_ms, 4), "recall": round(recall, 4), "peak_mb": round(mem_mb, 2),
    })
    print(f"ivf  : build={build_time:.3f}s  query={latency_ms:.3f}ms  recall={recall:.3f}  mem={mem_mb:.1f}MB")

    # -- HNSW --
    hnsw = HNSWIndex(dim=d, metric=Metric.COSINE, M=10, ef_construction=60, ef_search=40)
    build_time, mem_mb = measure_build(hnsw, vectors, ids)
    latency_ms, hnsw_results = measure_query(hnsw, queries, k)
    recall = recall_at_k(hnsw_results, ground_truth)
    rows.append({
        "index": "hnsw", "n": n, "build_s": round(build_time, 4),
        "query_ms": round(latency_ms, 4), "recall": round(recall, 4), "peak_mb": round(mem_mb, 2),
    })
    print(f"hnsw : build={build_time:.3f}s  query={latency_ms:.3f}ms  recall={recall:.3f}  mem={mem_mb:.1f}MB")

    return rows


def main():
    all_rows = []
    # NOTE: the HNSW build here is pure Python (no C extension / SIMD), so its
    # constant factor is high -- n=5000 already takes ~20s to build with these
    # params. That's fine for demonstrating the recall/latency/build-time
    # tradeoff, but it's also exactly the reason production vector DBs (FAISS,
    # hnswlib, Milvus, etc.) reimplement this in C++/Rust: the *algorithm* is
    # the same, the win is in the constant factor. Feel free to add larger n
    # here if you don't mind a longer run.
    for n in [500, 2000, 5000]:
        all_rows.extend(run_benchmark(n=n, d=32, n_queries=30, k=10))

    with open(RESULTS_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["index", "n", "build_s", "query_ms", "recall", "peak_mb"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {RESULTS_PATH}")

    try:
        _plot(all_rows)
    except ImportError:
        print("matplotlib not installed -- skipping plot, CSV results still written")


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"flat": "tab:gray", "ivf": "tab:orange", "hnsw": "tab:blue"}

    ns = sorted(set(r["n"] for r in rows))
    for idx_name in ["flat", "ivf", "hnsw"]:
        sub = [r for r in rows if r["index"] == idx_name]
        axes[0].plot([r["n"] for r in sub], [r["query_ms"] for r in sub], marker="o",
                     label=idx_name, color=colors[idx_name])
        axes[1].scatter([r["recall"] for r in sub], [r["query_ms"] for r in sub],
                         label=idx_name, color=colors[idx_name], s=60)

    axes[0].set_xlabel("dataset size (n)")
    axes[0].set_ylabel("avg query latency (ms)")
    axes[0].set_title("Query latency vs. dataset size")
    axes[0].legend()
    axes[0].set_yscale("log")

    axes[1].set_xlabel("recall@10")
    axes[1].set_ylabel("avg query latency (ms)")
    axes[1].set_title("Recall / latency tradeoff")
    axes[1].legend()
    axes[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"Wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
