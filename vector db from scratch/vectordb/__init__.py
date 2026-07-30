"""
vectordb: a small vector database built from scratch.

Exposes the main VectorDB class and the index implementations.
"""

from .db import VectorDB, Collection
from .distance import Metric

__all__ = ["VectorDB", "Collection", "Metric"]
__version__ = "0.1.0"
