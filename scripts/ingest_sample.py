"""
Day 1 sample ingestion with PostgreSQL.
"""

import csv
from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf
from sqlalchemy.exc import IntegrityError

from src.config import DATA_DIR
from src.db import engine, SessionLocal
from src.schema import Base, RawNews, PriceHistory

DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_NEWS_CSV = DATA_DIR / "raw_news_sample.csv"
PRICE_HISTORY_CSV = DATA_DIR / "price_history_sample.csv"


def create_tables():
    Base.metadata.create_all(bind=engine)
    print("DB tables created.")


def load_news_csv_to_db(csv_path: Path):
    session = SessionLocal()
    inserted = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                published_at = datetime.fromisoformat(row.get("published_at"))
            except Exception:
                published_at = None

            rn = RawNews(
                url=row.get("url"),
                title=row.get("title") or "",
                body=row.get("body") or "",
                published_at=published_at,
                source=row.get("source"),
            )

            session.add(rn)
            try:
                session.commit()
                inserted += 1
            except IntegrityError:
                session.rollback()

    session.close()
    print(f"Inserted {inserted} news rows.")


def fetch_price_history(ticker="RELIANCE.NS", period_days=30):
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=period_days)

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        print("No data from yfinance.")
        return None

    df.reset_index(inplace=True)
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })
    df["ticker"] = ticker

    df.to_csv(PRICE_HISTORY_CSV, index=False)
    print("Saved price CSV.")
    return df


def load_price_csv_to_db(df):
    session = SessionLocal()
    inserted = 0

    for _, r in df.iterrows():
        ph = PriceHistory(
            ticker=r["ticker"],
            date=pd.to_datetime(r["date"]).to_pydatetime(),
            open=float(r["open"] or 0),
            high=float(r["high"] or 0),
            low=float(r["low"] or 0),
            close=float(r["close"] or 0),
            adj_close=float(r["adj_close"] or 0),
            volume=float(r["volume"] or 0),
        )
        session.add(ph)
        try:
            session.commit()
            inserted += 1
        except IntegrityError:
            session.rollback()

    session.close()
    print(f"Inserted {inserted} price rows.")


if __name__ == "__main__":
    create_tables()

    if RAW_NEWS_CSV.exists():
        load_news_csv_to_db(RAW_NEWS_CSV)
    else:
        print("Raw news CSV not found.")

    df = fetch_price_history("RELIANCE.NS", period_days=60)
    if df is not None:
        load_price_csv_to_db(df)
