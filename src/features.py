# src/features.py
"""
Feature engineering: merge price_history + daily_sentiment (vader) + llm sentiment.
Creates daily supervised learning dataset for next-day price direction prediction.
"""

import pandas as pd
from sqlalchemy import create_engine
from src.config import DATABASE_URL
from sqlalchemy import text

engine = create_engine(DATABASE_URL)

def load_price_history():
    q = """
    SELECT ticker, date, open, high, low, close, adj_close, volume
    FROM price_history
    ORDER BY date ASC
    """
    return pd.read_sql(q, engine, parse_dates=["date"])

def load_daily_sentiment():
    q = """
    SELECT ticker, date, avg_compound, article_count, pct_positive, pct_negative, model_version
    FROM daily_sentiment
    ORDER BY date ASC
    """
    return pd.read_sql(q, engine, parse_dates=["date"])

def make_features(ticker="RELIANCE.NS", sentiment_model="vader-v1"):
    prices = load_price_history()
    prices = prices[prices["ticker"] == ticker].copy()

    # returns
    prices["return_1d"] = prices["close"].pct_change()
    prices["return_5d"] = prices["close"].pct_change(5)
    prices["return_10d"] = prices["close"].pct_change(10)
    prices["vol_change"] = prices["volume"].pct_change()

    # sentiment
    sent = load_daily_sentiment()
    sent = sent[(sent["ticker"] == ticker) & (sent["model_version"] == sentiment_model)]

    df = prices.merge(sent, on=["ticker", "date"], how="left")

    # Fill NA sentiment as neutral
    df["avg_compound"] = df["avg_compound"].fillna(0)
    df["pct_positive"] = df["pct_positive"].fillna(0)
    df["pct_negative"] = df["pct_negative"].fillna(0)
    df["article_count"] = df["article_count"].fillna(0)

    # Target variable: next-day direction
    df["target"] = df["return_1d"].shift(-1)
    df["target_class"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    df = df.dropna(subset=["return_1d", "return_5d", "return_10d", "vol_change", "target_class"])
    return df
