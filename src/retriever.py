# src/retriever.py
"""
LangChain-style retriever wrapper that queries Chroma and returns documents suitable
for using as context in LLM chains.

This module provides:
 - a Retriever class with get_relevant_documents(query, k, filters)
"""

from typing import List, Dict, Any, Optional
from src.vectorstore import query, create_or_get_collection
from src.embeddings import embed_text
from src.observability.langfuse_client import get_langfuse
import time


class SimpleRetriever:
    def __init__(self, collection_name="news", k=5):
        self.collection_name = collection_name
        self.k = k
        # lazy: collection handled by vectorstore module

    def get_relevant_documents(self, query_text: str, k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None):
        lf = get_langfuse()
        start = time.time()

        with lf.trace(
            name="news_retrieval",
            input={
                "query": query_text,
                "filters": filters,
                "top_k": k or self.k,
            },
            metadata={"component": "retriever"}
        ) as trace:

            docs = query(
                collection_name=self.collection_name,
                query_text=query_text,
                n_results=k or self.k,
                where=filters
            )

            latency = time.time() - start

            trace.output = {
                "documents_returned": len(docs.get("documents", [[]])[0])
            }
            trace.metadata["latency_sec"] = latency

            return docs  # existing logic


# convenience function to create retriever
def get_default_retriever(k=5):
    return SimpleRetriever(k=k)
