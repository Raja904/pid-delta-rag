"""
chat/index.py
Builds and queries a ChromaDB vector index over:
  - Chunks from PID A
  - Chunks from PID B
  - Delta report entries
Each chunk carries metadata so citations can point back to source + location.
"""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any

from src.canonical.model import CanonicalDocument
from src.delta.engine import DeltaItem
from src.observability.logger import log

CHROMA_DIR = Path(__file__).parent.parent.parent / ".chroma_db"


def _get_collection(collection_name: str = "delta_chat"):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(
        name              = collection_name,
        metadata          = {"hnsw:space": "cosine"},
    )
    return col


def build_index(
    doc_a: CanonicalDocument,
    doc_b: CanonicalDocument,
    delta_items: list[DeltaItem],
    collection_name: str = "delta_chat",
    reset: bool = True,
) -> Any:
    """Index all content. If reset=True, clears old data first."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    col = client.get_or_create_collection(
        name     = collection_name,
        metadata = {"hnsw:space": "cosine"},
    )

    docs, metas, ids = [], [], []

    # Index PID A blocks
    for block in doc_a.all_blocks():
        if len(block.text.strip()) < 5:
            continue
        docs.append(block.text)
        metas.append({
            "source": "pid_a",
            "pid":    doc_a.pid,
            "page":   block.page,
            "block_id": block.block_id,
            "block_type": block.block_type,
            "confidence": block.confidence,
        })
        ids.append(f"a_{block.block_id}")

    # Index PID B blocks
    for block in doc_b.all_blocks():
        if len(block.text.strip()) < 5:
            continue
        docs.append(block.text)
        metas.append({
            "source": "pid_b",
            "pid":    doc_b.pid,
            "page":   block.page,
            "block_id": block.block_id,
            "block_type": block.block_type,
            "confidence": block.confidence,
        })
        ids.append(f"b_{block.block_id}")

    # Index delta report entries
    for item in delta_items:
        text = f"{item.change_type.upper()} ({item.content_type}): {item.description}"
        if item.old_value:
            text += f" | Before: {item.old_value[:200]}"
        if item.new_value:
            text += f" | After: {item.new_value[:200]}"
        docs.append(text)
        metas.append({
            "source":      "delta_report",
            "change_id":   item.change_id,
            "change_type": item.change_type,
            "content_type":item.content_type,
            "page_a":      str(item.page_a or ""),
            "page_b":      str(item.page_b or ""),
            "confidence":  item.confidence,
        })
        ids.append(f"delta_{item.change_id}")

    if docs:
        # Batch in chunks of 100 to avoid memory spikes
        batch = 100
        for start in range(0, len(docs), batch):
            col.add(
                documents = docs[start:start+batch],
                metadatas = metas[start:start+batch],
                ids       = ids[start:start+batch],
            )
        log.info("Index built", extra={"total_chunks": len(docs)})

    return col


def retrieve(
    query: str,
    collection_name: str = "delta_chat",
    n_results: int = 6,
) -> list[dict]:
    """
    Query the index. Returns list of dicts with keys:
      text, source, metadata, distance
    """
    col = _get_collection(collection_name)
    results = col.query(query_texts=[query], n_results=min(n_results, col.count() or 1))

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":     doc,
            "source":   meta.get("source", "unknown"),
            "metadata": meta,
            "distance": dist,
        })
    return chunks
