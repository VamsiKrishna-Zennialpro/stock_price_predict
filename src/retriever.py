# src/retriever.py
from typing import List, Dict, Any, Optional
from src.vectorstore import get_collection
from src.observability.langfuse_client import get_langfuse
import time


class NewsRetriever:
    def __init__(self, collection_name: str = "news", k: int = 5):
        self.collection = get_collection(collection_name)
        self.k = k

    def get_relevant_documents(
        self,
        query_text: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        lf = get_langfuse()
        start = time.time()

        with lf.trace(
            name="news_retrieval",
            input={
                "query": query_text,
                "filters": filters,
                "top_k": k or self.k,
            },
            metadata={"retriever": "chroma+gemini"},
        ) as trace:
            result = self.collection.query(
                query_texts=[query_text],
                n_results=k or self.k,
                where=filters,
            )

            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            dists = result.get("distances", [[]])[0]

            output = []
            for i in range(len(docs)):
                output.append(
                    {
                        "text": docs[i],
                        "metadata": metas[i],
                        "distance": dists[i],
                    }
                )

            trace.output = {"results": len(output)}
            trace.metadata["latency_sec"] = time.time() - start

            return output


def get_default_retriever(k: int = 5) -> NewsRetriever:
    return NewsRetriever(k=k)
