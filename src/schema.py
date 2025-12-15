# src/schema.py
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Float,
    Text,
    UniqueConstraint,
    Index,
    func,
    Table,
    MetaData,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func as sqlfunc

Base = declarative_base()

class RawNews(Base):
    __tablename__ = "raw_news"
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=False, nullable=True)
    title = Column(String, nullable=False)
    body = Column(Text, nullable=True)
    published_at = Column(DateTime, nullable=True, index=True)
    source = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=sqlfunc.now())

    __table_args__ = (Index("ix_rawnews_published", "published_at"),)

class CleanNews(Base):
    __tablename__ = "clean_news"
    id = Column(Integer, primary_key=True, index=True)
    raw_id = Column(Integer, nullable=True, index=True)
    ticker = Column(String, index=True, nullable=True)  # simple mapping column
    title = Column(String, nullable=False)
    body = Column(Text, nullable=True)
    published_at = Column(DateTime, nullable=True, index=True)
    normalized_at = Column(DateTime(timezone=True), server_default=sqlfunc.now())

    __table_args__ = (Index("ix_cleannews_ticker_pub", "ticker", "published_at"),)

class PriceHistory(Base):
    __tablename__ = "price_history"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    adj_close = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)

    __table_args__ = (UniqueConstraint("ticker", "date", name="uix_ticker_date"),)

class SentimentScore(Base):
    """
    One row per processed news item. Baseline: polarity float (-1..1), compound from VADER,
    label: positive/neutral/negative
    """
    __tablename__ = "sentiment_scores"
    id = Column(Integer, primary_key=True, index=True)
    clean_id = Column(Integer, nullable=True, index=True)
    raw_id = Column(Integer, nullable=True, index=True)
    ticker = Column(String, index=True, nullable=True)
    published_at = Column(DateTime, nullable=True, index=True)
    neg = Column(Float, nullable=True)
    neu = Column(Float, nullable=True)
    pos = Column(Float, nullable=True)
    compound = Column(Float, nullable=True)  # -1..1
    label = Column(String, nullable=True)  # 'positive'|'neutral'|'negative'
    model_version = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=sqlfunc.now())

    __table_args__ = (Index("ix_sentiment_ticker_date", "ticker", "published_at"),)

class DailySentiment(Base):
    """
    Aggregated daily sentiment for ticker (computed from sentiment_scores)
    """
    __tablename__ = "daily_sentiment"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    date = Column(DateTime, nullable=False, index=True)  # date only (time normalized to midnight)
    avg_compound = Column(Float, nullable=True)
    article_count = Column(Integer, nullable=True)
    pct_positive = Column(Float, nullable=True)
    pct_negative = Column(Float, nullable=True)
    model_version = Column(String, nullable=True)
    computed_at = Column(DateTime(timezone=True), server_default=sqlfunc.now())

    __table_args__ = (UniqueConstraint("ticker", "date", name="uix_ticker_date_daily"),)
