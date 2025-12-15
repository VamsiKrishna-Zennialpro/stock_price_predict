# src/observability/langfuse_client.py
from langfuse import Langfuse
import os

_langfuse = None

def get_langfuse():
    global _langfuse
    if _langfuse is None:
        _langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST"),
        )
    return _langfuse
