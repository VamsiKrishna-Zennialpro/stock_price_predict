# src/embeddings.py
"""
Embedding provider abstraction.

Defaults to a local sentence-transformers model (all-MiniLM-L6-v2).
Switch to OpenAI by setting EMBEDDING_PROVIDER=openai and providing OPENAI_API_KEY in .env.
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "hf-mini")  # "hf-mini"
HF_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)

# Lazy import to avoid heavy deps if not needed
def get_hf_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(HF_MODEL)
    return model

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Return list of vectors for provided texts.
    """
    if EMBEDDING_PROVIDER == "genai":
        if not GEMINI_API_KEY:
            raise ValueError("GENAI_API_KEY not set in .env but EMBEDDING_PROVIDER=genai")
        from google import genai
        genai.api_key = GEMINI_API_KEY
        # OpenAI's embeddings API supports batching; keep simple here
        resp = genai.Embedding.create(model="text-embedding-3-small", input=texts)
        vectors = [r["embedding"] for r in resp["data"]]
        return vectors
    else:
        # default: local HF sentence-transformers
        model = get_hf_embedder()
        # model.encode returns a numpy array
        vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

def embed_text(text: str):
    return embed_texts([text])[0]
