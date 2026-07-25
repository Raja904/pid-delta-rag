"""
chat/tools.py
Defines the LangChain tools available to the ReAct agent for querying document data.
"""
from typing import Optional, List
from langchain_core.tools import tool
from src.chat.index import retrieve, CHROMA_DIR
import chromadb

@tool
def semantic_search(query: str) -> str:
    """
    Use this tool for general questions about the documents, how things work, or broad conceptual changes.
    It performs a fuzzy semantic search across the original documents and the delta report.
    Returns the top matching chunks of text with their citation references.
    """
    try:
        chunks = retrieve(query, collection_name="delta_chat", n_results=6)
        if not chunks:
            return "No relevant information found in the documents."
        
        context_parts = []
        for chunk in chunks:
            meta = chunk["metadata"]
            src = chunk["source"]
            
            if src in ("pid_a", "pid_b"):
                label = "PID_A" if src == "pid_a" else "PID_B"
                page = meta.get("page", "?")
                bid = meta.get("block_id", "?")
                ref = f"{label}·p.{page}·{bid}"
            else:
                cid = meta.get("change_id", "?")
                ref = f"DELTA·{cid}"
                
            context_parts.append(f"[{ref}]: {chunk['text']}")
            
        return "\n\n".join(context_parts)
    except Exception as e:
        return f"Error performing semantic search: {str(e)}"


@tool
def exact_tag_search(tag_id: str) -> str:
    """
    Use this tool when the user asks about a specific Component ID, Valve Tag, Line Number, Dimension, or Work Pack.
    It performs an exact keyword match across the documents and the delta report to find EXACT references to the given tag.
    Returns the blocks of text where this exact tag appears, along with citation references.
    """
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col = client.get_collection(name="delta_chat")
        
        # Use chromadb's where_document filter for a substring match
        results = col.get(
            where_document={"$contains": tag_id},
            include=["documents", "metadatas"]
        )
        
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        
        if not docs:
            return f"No exact matches found for tag ID '{tag_id}' in the documents."
            
        context_parts = []
        for i in range(len(docs)):
            doc = docs[i]
            meta = metas[i]
            src = meta.get("source", "unknown")
            
            if src in ("pid_a", "pid_b"):
                label = "PID_A" if src == "pid_a" else "PID_B"
                page = meta.get("page", "?")
                bid = meta.get("block_id", "?")
                ref = f"{label}·p.{page}·{bid}"
            else:
                cid = meta.get("change_id", "?")
                ref = f"DELTA·{cid}"
                
            context_parts.append(f"[{ref}]: {doc}")
            
        return "\n\n".join(context_parts)
    except Exception as e:
        return f"Error performing exact tag search: {str(e)}"
