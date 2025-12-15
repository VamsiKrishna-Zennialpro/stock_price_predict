# scripts/clean_and_label.py
"""
Load rows from raw_news (DB) and/or data/raw_news_sample.csv, clean, dedupe,
map tickers, label sentiment (VADER), and write to clean_news and sentiment_scores.
"""

import csv
from pathlib import Path
from datetime import datetime
from typing import List
import argparse

from src.config import DATA_DIR
from src.db import SessionLocal, engine
from src.schema import Base, RawNews, CleanNews, SentimentScore
from src.cleaning import (
    strip_html,
    normalize_timestamp,
    dedupe_records,
    map_tickers,
    label_sentiment,
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_NEWS_CSV = DATA_DIR / "raw_news_sample.csv"

def create_tables():
    Base.metadata.create_all(bind=engine)

def load_csv_rows(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "url": row.get("url"),
                    "title": row.get("title"),
                    "body": row.get("body"),
                    "published_at": row.get("published_at"),
                    "source": row.get("source"),
                }
            )
    return rows

def fetch_rawnews_from_db(session):
    return session.query(RawNews).all()

def process_and_store(rows):
    session = SessionLocal()
    inserted_clean = 0
    inserted_sent = 0

    # dedupe incoming batch first (CSV-level)
    rows = dedupe_records(rows)

    for r in rows:
        raw_id = None
        # If raw news exists in DB already, we may want raw_id mapping. We'll try to insert into raw_news if not present.
        try:
            # find if raw exists by url (if url present)
            if r.get("url"):
                existing = session.query(RawNews).filter(RawNews.url == r["url"]).first()
            else:
                existing = None
            if existing:
                raw_id = existing.id
            else:
                rn = RawNews(
                    url=r.get("url"),
                    title=r.get("title") or "",
                    body=r.get("body") or "",
                    published_at=normalize_timestamp(r.get("published_at")),
                    source=r.get("source"),
                )
                session.add(rn)
                session.commit()
                raw_id = rn.id
        except Exception:
            session.rollback()
            # skip problematic row
            continue

        # Clean text
        clean_title = strip_html(r.get("title") or "")
        clean_body = strip_html(r.get("body") or "")

        # Map tickers (may return many; we store one CleanNews per matched ticker or as ticker=NULL)
        tickers = map_tickers(clean_title + " " + clean_body)
        if not tickers:
            tickers = [None]  # we'll still store the cleaned article, but ticker null

        published_at = normalize_timestamp(r.get("published_at"))

        for t in tickers:
            # insert CleanNews
            try:
                cn = CleanNews(
                    raw_id=raw_id,
                    ticker=t,
                    title=clean_title,
                    body=clean_body,
                    published_at=published_at,
                )
                session.add(cn)
                session.commit()
                inserted_clean += 1
            except Exception:
                session.rollback()
                continue

            # label sentiment on title+body (concatenate; you may want to weight title more later)
            try:
                text_for_sent = (clean_title + ". " + (clean_body or ""))[:10000]
                sent = label_sentiment(text_for_sent)
                ss = SentimentScore(
                    clean_id=cn.id,
                    raw_id=raw_id,
                    ticker=t,
                    published_at=published_at,
                    neg=sent["neg"],
                    neu=sent["neu"],
                    pos=sent["pos"],
                    compound=sent["compound"],
                    label=sent["label"],
                    model_version="vader-v1",
                )
                session.add(ss)
                session.commit()
                inserted_sent += 1
            except Exception:
                session.rollback()
                continue

    session.close()
    print(f"Inserted {inserted_clean} clean_news rows and {inserted_sent} sentiment_scores rows.")

def main(use_db: bool, csv_path: Path):
    create_tables()
    rows = []
    if use_db:
        session = SessionLocal()
        raws = fetch_rawnews_from_db(session)
        session.close()
        for r in raws:
            rows.append(
                {
                    "url": r.url,
                    "title": r.title,
                    "body": r.body,
                    "published_at": r.published_at.isoformat() if r.published_at else None,
                    "source": r.source,
                }
            )
    if csv_path and csv_path.exists():
        rows.extend(load_csv_rows(csv_path))
    if not rows:
        print("No rows to process. Provide --from-db or --csv data.")
        return
    process_and_store(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-db", action="store_true", dest="from_db", help="Read raw_news from DB")
    parser.add_argument("--csv", dest="csv", default=str(RAW_NEWS_CSV), help="CSV file to load (default data/raw_news_sample.csv)")
    args = parser.parse_args()
    main(use_db=args.from_db, csv_path=Path(args.csv))
