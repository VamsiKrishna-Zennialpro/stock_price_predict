# src/embeddings.py
"""
Gemini embeddings for Chroma vector store.
"""

import os
import google.generativeai as genai
from typing import List

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "models/embedding-001"

def embed_texts(texts: List[str]) -> List[List[float]]:
    embeddings = []

    for text in texts:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
        )
        embeddings.append(result["embedding"])

    return embeddings
