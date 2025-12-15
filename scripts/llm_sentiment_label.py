# scripts/llm_sentiment_label.py
"""
Apply LLM sentiment classifier to clean_news rows and write results to sentiment_scores table.

Behavior:
- Iterates over clean_news rows (optionally filtered by ticker/date)
- Calls src.sentiment_chain.classify_text for each item
- Inserts SentimentScore rows with model_version 'llm-v1'
- Skips items that already have a sentiment_scores entry with model_version 'llm-v1' unless --force
"""

import argparse
from pathlib import Path
from datetime import datetime
from sqlalchemy import and_
from src.db import SessionLocal, engine
from src.schema import Base, CleanNews, SentimentScore
from src.sentiment_chain import classify_text

DATA_DIR = Path(__file__).parents[1] / "data"

def create_tables():
    Base.metadata.create_all(bind=engine)

def already_labeled(session, clean_id, model_version="llm-v1"):
    return session.query(SentimentScore).filter(SentimentScore.clean_id == clean_id, SentimentScore.model_version == model_version).first() is not None

def run(ticker=None, start=None, end=None, force=False, limit=None):
    create_tables()
    session = SessionLocal()
    q = session.query(CleanNews)
    if ticker:
        q = q.filter(CleanNews.ticker == ticker)
    if start:
        q = q.filter(CleanNews.published_at >= start)
    if end:
        q = q.filter(CleanNews.published_at < end)
    q = q.order_by(CleanNews.published_at.asc())
    if limit:
        q = q.limit(limit)
    rows = q.all()
    print(f"Found {len(rows)} clean_news rows to classify (ticker={ticker}, start={start}, end={end})")
    inserted = 0
    for r in rows:
        if not force and already_labeled(session, r.id, model_version="llm-v1"):
            continue
        try:
            title = r.title or ""
            body = r.body or ""
            res = classify_text(title, body)
            ss = SentimentScore(
                clean_id=r.id,
                raw_id=r.raw_id,
                ticker=r.ticker,
                published_at=r.published_at,
                neg=None,
                neu=None,
                pos=None,
                compound=None,
                label=res.get("label"),
                model_version="llm-v1",
            )
            session.add(ss)
            session.commit()
            inserted += 1
        except Exception as ex:
            session.rollback()
            print(f"Error classifying clean_id={r.id}: {ex}")
    session.close()
    print(f"Inserted {inserted} llm sentiment rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (exclusive)")
    parser.add_argument("--force", action="store_true", help="Recompute and overwrite existing llm-v1 entries")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(ticker=args.ticker, start=args.start, end=args.end, force=args.force, limit=args.limit)
