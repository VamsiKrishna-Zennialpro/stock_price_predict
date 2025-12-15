# scripts/aggregate_sentiment.py
"""
Compute daily aggregated sentiment per ticker from sentiment_scores table,
and write into daily_sentiment table.
"""

from datetime import datetime
from pathlib import Path
import argparse
import pandas as pd
from sqlalchemy import func
from src.db import SessionLocal, engine
from src.schema import Base, SentimentScore, DailySentiment
from sqlalchemy import select
from sqlalchemy.orm import Session

def create_tables():
    Base.metadata.create_all(bind=engine)

def compute_daily(session: Session, start_date=None, end_date=None, model_version="vader-v1"):
    """
    start_date/end_date are date strings 'YYYY-MM-DD' or None (all)
    """
    q = session.query(
        SentimentScore.ticker,
        func.date_trunc('day', SentimentScore.published_at).label("date"),
        func.avg(SentimentScore.compound).label("avg_compound"),
        func.count(SentimentScore.id).label("article_count"),
        func.sum(func.case([(SentimentScore.label == 'positive', 1)], else_=0)).label("n_pos"),
        func.sum(func.case([(SentimentScore.label == 'negative', 1)], else_=0)).label("n_neg"),
    ).filter(SentimentScore.ticker != None).group_by(SentimentScore.ticker, func.date_trunc('day', SentimentScore.published_at))

    if start_date:
        q = q.filter(SentimentScore.published_at >= start_date)
    if end_date:
        q = q.filter(SentimentScore.published_at < end_date)

    rows = q.all()
    inserted = 0
    for r in rows:
        ticker = r[0]
        dt = r[1].date() if hasattr(r[1], "date") else r[1]
        avg_comp = float(r[2] or 0.0)
        article_count = int(r[3] or 0)
        n_pos = int(r[4] or 0)
        n_neg = int(r[5] or 0)
        pct_pos = (n_pos / article_count) if article_count > 0 else 0.0
        pct_neg = (n_neg / article_count) if article_count > 0 else 0.0

        # upsert: remove existing for same ticker+date and re-insert (simple approach)
        session.query(DailySentiment).filter(DailySentiment.ticker == ticker, DailySentiment.date == dt).delete()
        ds = DailySentiment(
            ticker=ticker,
            date=dt,
            avg_compound=avg_comp,
            article_count=article_count,
            pct_positive=pct_pos,
            pct_negative=pct_neg,
            model_version=model_version,
        )
        session.add(ds)
        try:
            session.commit()
            inserted += 1
        except Exception:
            session.rollback()
    return inserted

def main(start_date=None, end_date=None):
    create_tables()
    session = SessionLocal()
    n = compute_daily(session, start_date=start_date, end_date=end_date)
    session.close()
    print(f"Inserted/updated {n} daily_sentiment rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", dest="start", default=None, help="start date YYYY-MM-DD")
    parser.add_argument("--end", dest="end", default=None, help="end date YYYY-MM-DD (exclusive)")
    args = parser.parse_args()
    main(start_date=args.start, end_date=args.end)
