"""
Metadata filtering: lets a search be restricted to records whose metadata
matches a filter, e.g. {"category": "shoes", "price": {"$lt": 100}}.

Supported operators: $eq (default for bare values), $ne, $gt, $gte, $lt,
$lte, $in, $nin. Top-level keys are ANDed together.

Two strategies are exposed because they have very different cost profiles
depending on how selective the filter is:
  - post_filter: run vector search first, then drop results that fail the
    filter. Cheap when the filter matches most records, but can return
    fewer than k results if the filter is very selective.
  - pre_filter: compute the set of ids that pass the filter first, then
    restrict the vector search to just those candidates. Correct even for
    highly selective filters, at the cost of a metadata scan up front.
"""
from __future__ import annotations

from typing import Any

_OPS = {
    "$eq": lambda v, target: v == target,
    "$ne": lambda v, target: v != target,
    "$gt": lambda v, target: v is not None and v > target,
    "$gte": lambda v, target: v is not None and v >= target,
    "$lt": lambda v, target: v is not None and v < target,
    "$lte": lambda v, target: v is not None and v <= target,
    "$in": lambda v, target: v in target,
    "$nin": lambda v, target: v not in target,
}


def matches(metadata: dict[str, Any], filter_: dict[str, Any] | None) -> bool:
    if not filter_:
        return True
    for key, condition in filter_.items():
        value = metadata.get(key)
        if isinstance(condition, dict):
            for op, target in condition.items():
                if op not in _OPS:
                    raise ValueError(f"Unsupported filter operator: {op}")
                try:
                    if not _OPS[op](value, target):
                        return False
                except TypeError:
                    return False  # e.g. comparing None with $gt
        else:
            if value != condition:
                return False
    return True


def filter_ids(all_metadata: dict[str, dict[str, Any]], filter_: dict[str, Any] | None) -> set[str]:
    """Pre-filter: return the set of record ids whose metadata matches."""
    if not filter_:
        return set(all_metadata.keys())
    return {rid for rid, md in all_metadata.items() if matches(md, filter_)}
