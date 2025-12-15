# src/vectorstore.py
"""
Build and manage a Chroma vectorstore for clean_news.

Chroma will persist to local directory `data/chroma`. Each document stored with metadata:
 - clean_id (int)
 - ticker (str)
 - published_at (ISO str)
 - title (str)
 - source (optional)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")

from chromadb.config import Settings
import chromadb

CHROMA_DIR = os.getenv("CHROMA_DIR", str(Path(__file__).parents[1] / "data" / "chroma"))

# initialize client (persistent)
def get_chroma_client():
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
    return client

def create_or_get_collection(name="news"):
    client = get_chroma_client()
    # create if not exists
    try:
        collection = client.get_collection(name)
    except Exception:
        collection = client.create_collection(name)
    return collection

def add_documents(documents: List[str], metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None, collection_name="news"):
    """
    Add documents to Chroma collection.
    - documents: list of text strings
    - metadatas: list of dicts (same length)
    - ids: optional list of ids (strings)
    """
    collection = create_or_get_collection(collection_name)
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    # persist is automatic with duckdb+parquet; but call persist for clarity
    collection.client.persist()

def query(collection_name="news", query_text="", n_results=5, where: Optional[Dict[str, Any]] = None):
    collection = create_or_get_collection(collection_name)
    # chroma's query supports 'where' filters (equality matches)
    res = collection.query(query_texts=[query_text], n_results=n_results, where=where or {})
    # returns dict with 'ids','distances','metadatas','documents'
    return res

def count_documents(collection_name="news"):
    collection = create_or_get_collection(collection_name)
    return collection.count()
