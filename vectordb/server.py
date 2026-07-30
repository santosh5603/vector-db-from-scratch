"""
Thin REST wrapper around VectorDB, using FastAPI.

Run with:
    uvicorn vectordb.server:app --reload

Endpoints:
    POST   /collections                          create a collection
    GET    /collections                           list collections
    DELETE /collections/{name}                     delete a collection
    POST   /collections/{name}/vectors             upsert one or many vectors
    GET    /collections/{name}/vectors/{id}        fetch a vector by id
    DELETE /collections/{name}/vectors/{id}        delete a vector by id
    POST   /collections/{name}/search              vector search (+ optional filter)
    POST   /collections/{name}/checkpoint          flush to disk
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .db import VectorDB

DATA_DIR = os.environ.get("VECTORDB_DATA_DIR", "./vectordb_data")

app = FastAPI(title="vectordb", version="0.1.0")
db = VectorDB(directory=DATA_DIR)


class CreateCollectionRequest(BaseModel):
    name: str
    dim: int
    metric: str = "cosine"
    index_type: str = "hnsw"
    index_kwargs: dict[str, Any] = Field(default_factory=dict)


class UpsertItem(BaseModel):
    id: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpsertRequest(BaseModel):
    items: list[UpsertItem]


class SearchRequest(BaseModel):
    vector: list[float]
    k: int = 10
    filter: Optional[dict[str, Any]] = None


@app.post("/collections")
def create_collection(req: CreateCollectionRequest):
    try:
        db.create_collection(req.name, req.dim, metric=req.metric,
                              index_type=req.index_type, **req.index_kwargs)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"status": "created", "name": req.name}


@app.get("/collections")
def list_collections():
    return {"collections": db.list_collections()}


@app.delete("/collections/{name}")
def delete_collection(name: str):
    db.delete_collection(name)
    return {"status": "deleted", "name": name}


@app.post("/collections/{name}/vectors")
def upsert_vectors(name: str, req: UpsertRequest):
    coll = _get_collection_or_404(name)
    for item in req.items:
        coll.upsert(item.id, np.array(item.vector, dtype=np.float32), item.metadata)
    return {"status": "ok", "count": len(req.items)}


@app.get("/collections/{name}/vectors/{record_id}")
def get_vector(name: str, record_id: str):
    coll = _get_collection_or_404(name)
    record = coll.get(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="record not found")
    return {"id": record.id, "vector": record.vector.tolist(), "metadata": record.metadata}


@app.delete("/collections/{name}/vectors/{record_id}")
def delete_vector(name: str, record_id: str):
    coll = _get_collection_or_404(name)
    removed = coll.delete(record_id)
    if not removed:
        raise HTTPException(status_code=404, detail="record not found")
    return {"status": "deleted", "id": record_id}


@app.post("/collections/{name}/search")
def search(name: str, req: SearchRequest):
    coll = _get_collection_or_404(name)
    results = coll.search(np.array(req.vector, dtype=np.float32), k=req.k, filter=req.filter)
    return {"results": results}


@app.post("/collections/{name}/checkpoint")
def checkpoint(name: str):
    coll = _get_collection_or_404(name)
    if not coll.directory:
        raise HTTPException(status_code=400, detail="collection is not persisted to disk")
    coll.checkpoint()
    return {"status": "checkpointed"}


def _get_collection_or_404(name: str):
    try:
        return db.get_collection(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no such collection: {name}")
