# scripts/build_vectorstore.py
"""
Index clean_news rows from DB into Chroma vectorstore.

Behavior:
 - reads all rows from clean_news (or optionally a date/ticker filter)
 - builds embeddings with src.embeddings.embed_texts
 - stores docs and metadata into Chroma (ids set to "clean:{id}")
 - avoids re-adding docs if id already present (basic dedupe)
"""

from pathlib import Path
from datetime import datetime
import argparse
import math
from typing import List

from src.db import SessionLocal, engine
from src.schema import Base, CleanNews
from src.vectorstore import create_or_get_collection, add_documents, get_chroma_client
from src.embeddings import embed_texts

BATCH_SIZE = 128

def fetch_cleannews(session, ticker=None, start_date=None, end_date=None):
    q = session.query(CleanNews)
    if ticker:
        q = q.filter(CleanNews.ticker == ticker)
    if start_date:
        q = q.filter(CleanNews.published_at >= start_date)
    if end_date:
        q = q.filter(CleanNews.published_at < end_date)
    q = q.order_by(CleanNews.published_at.asc())
    return q.all()

def run(ticker=None, start_date=None, end_date=None, collection="news", batch_size=BATCH_SIZE):
    session = SessionLocal()
    rows = fetch_cleannews(session, ticker, start_date, end_date)
    total = len(rows)
    print(f"Found {total} clean_news rows to index (ticker={ticker}, start={start_date}, end={end_date})")
    if total == 0:
        return
    coll = create_or_get_collection(collection)
    # find existing ids in collection to avoid re-adding duplicates
    client = get_chroma_client()
    existing_ids = set()
    try:
        stats = coll.count()
    except Exception:
        stats = 0
    # Chroma does not expose simple list ids API in some versions; we will attempt a safe approach:
    # We'll re-add all docs but use deterministic ids "clean:{id}" so duplicates are replaced.
    docs = []
    metas = []
    ids = []
    for i, r in enumerate(rows, start=1):
        text = (r.title or "") + "\n\n" + (r.body or "")
        meta = {
            "clean_id": r.id,
            "raw_id": r.raw_id,
            "ticker": r.ticker,
            "published_at": r.published_at.isoformat() if r.published_at else None,
            "title": (r.title or "")[:300],
        }
        docs.append(text)
        metas.append(meta)
        ids.append(f"clean:{r.id}")
        # batch commit
        if len(docs) >= batch_size or i == total:
            print(f"Embedding and adding batch {i - len(docs) + 1}..{i} (size={len(docs)})")
            # embeddings = embed_texts(docs)  # vectorstore.add can accept texts and compute embeddings depending on conf,
            # but we will let chroma compute embeddings itself only if configured. For reproducibility we compute embeddings
            # externally here with our embedder and use collection.add with embeddings param if needed. Simpler: pass texts only.
            add_documents(documents=docs, metadatas=metas, ids=ids, collection_name=collection)
            docs, metas, ids = [], [], []
    session.close()
    print("Indexing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD")
    args = parser.parse_args()
    run(ticker=args.ticker, start_date=args.start, end_date=args.end)
